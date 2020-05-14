import logging
import requests
from urllib.parse import urlparse

import regex as re
import spacy
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.pipeline import make_pipeline
from sklearn import compose as cp, dummy, feature_extraction as fe, impute, multioutput, neighbors, preprocessing as pp

# FIXME: Remove me.
from util.caching import cache_parquet_today, cache_today
from util.performance import instrument_latency

class AutoTagger(multioutput.MultiOutputClassifier):
    def __init__(self, featurizer=None, estimator=None):
        if featurizer is None:
            featurizer = self._build_featurizer()
        if estimator is None:
            estimator = self._build_estimator()
        super().__init__(estimator=estimator)
        self.set_params(featurizer=featurizer)

    # FIXME: Remove me.
    @instrument_latency
    def fit(self, X, y):
        self.binarizer_ = pp.MultiLabelBinarizer().fit(y)
        self.featurizer_ = self.featurizer.fit(X)

        Yt = pd.DataFrame(self.binarizer_.transform(y), index=X.index, columns=self.binarizer_.classes_)
        Xt = self.featurizer_.transform(X)
        return super().fit(Xt, Yt)

    # FIXME: Remove me.
    @instrument_latency
    def predict(self, untagged_df):
        untagged_df = untagged_df[_get_ivars(untagged_df)]
        Xt = self.featurizer_.transform(untagged_df)
        return super().predict(Xt)

    def transform(self, untagged_df):
        # classification-based
        tagged_df = pd.DataFrame({ "classifier_tags": self.binarizer_.inverse_transform(self.predict(untagged_df)) },
                                 index=untagged_df.index)
        tagged_df.index.name = "item_id"

        # domain-based
        tagged_df["domain_tags"] = untagged_df["resolved_domain"].apply(lambda x: (x,))

        tagged_df["tags"] = tagged_df.sum(axis=1)
        return tagged_df

    def _build_featurizer(self):
        raise NotImplementedError()

    def _build_estimator(self):
        raise NotImplementedError()

    def _protect_nullable(self, featurizer, fill_value=""):
        return make_pipeline(pp.FunctionTransformer(lambda df: df.fillna(fill_value)),
                             featurizer)

# class NullableTransformer(object):
#     def __init__(self, transformer, fill_value=""):
#         self.transformer = transformer
#         self.fill_value = fill_value

#     def fit(self, *nargs, **kwargs):
#         return self.transformer.fit(*nargs, **kwargs)

#     def transform(self, X):
#         Xt = self.transformer.transform(X.fillna(self.fill_value))
#         Xt[pd.isnull(X)] = pd.NA
#         return Xt

def _protect_nullable(featurizer, fill_value=""):
    return NullableTransformer(featurizer, fill_value=fill_value)

class AutoTagger__Dummy(AutoTagger):
    def _build_estimator(self):
        return dummy.DummyClassifier(strategy="stratified")

    def _build_featurizer(self):
        return cp.ColumnTransformer([
            ("title_bow", self._protect_nullable(fe.text.CountVectorizer()), "resolved_title"),
        ], remainder="drop")

class AutoTagger__KNN(AutoTagger):
    def _build_estimator(self):
        return neighbors.KNeighborsClassifier()

    def _build_featurizer(self):
        def __binary_tfidf():
            return fe.text.TfidfVectorizer(strip_accents="unicode", binary=True)

        # FIXME: Too slow?
        def __spacy_embeddings():
            nlp = spacy.load("en_core_web_md")

            # FIXME: Remove me.
            @instrument_latency
            def __transform(texts_s):
                return np.vstack([doc.vector for doc in nlp.pipe(texts_s, disable=["tagger", "parser", "ner"])])
            return pp.FunctionTransformer(__transform)

        # from sparknlp.annotator import SentenceEmbeddings
        # from sparknlp.pretrained import PretrainedPipeline
        # import sparknlp

        # spark = sparknlp.start()
        def __sparknlp_embeddings():
            pipeline = PretrainedPipeline("explain_document_dl", lang="en")
            embedder = SentenceEmbeddings().setInputCols(["document", "embeddings"]) \
                                           .setOutputCol("sentence_embeddings") \
                                           .setPoolingStrategy("AVERAGE")

            # FIXME: Remove me.
            @instrument_latency
            def __transform(texts_s):
                text_df = spark.createDataFrame(pd.DataFrame({ "text": texts_s }))
                results_df = embedder.transform(pipeline.transform(text_df))

                # FIXME: Hacky. There's surely a better way to do this.
                embeddings_ndy = np.vstack([np.array(x.embeddings).reshape(-1, 100).mean(axis=0)
                                            for x in results_df.select("sentence_embeddings.embeddings").collect()])

                nulled_ids = texts_s.iloc[np.where((~np.isfinite(embeddings_ndy)).any(axis=1))].index
                if len(nulled_ids):
                    logging.info("Backfilling %d (%.1f%%) documents without embeddings:\n%s", len(nulled_ids),
                                 len(nulled_ids) / len(texts_s) * 100, texts_s.loc[nulled_ids].to_string())
                    embeddings_ndy = np.nan_to_num(embeddings_ndy)
                return embeddings_ndy
            return pp.FunctionTransformer(__transform)

        # FIXME: Try unsupervised techniques.
        return cp.ColumnTransformer([
            # ("emb_resolved_title", self._protect_nullable(__sparknlp_embeddings()), "resolved_title"),
            # ("emb_excerpt", self._protect_nullable(__sparknlp_embeddings()), "excerpt"),
            # ("emb_text", self._protect_nullable(__sparknlp_embeddings()), "text"),

            # NOTE: Distracts from more effective content-based features. When used, need one-hot encoding
            # since ordinal encoding breaks KNN.
            # ("url_domain", self._protect_nullable(fe.text.CountVectorizer(min_df=1)), "resolved_domain"),
            ("url_categories", self._protect_nullable(fe.text.CountVectorizer(min_df=5, binary=True)), "resolved_path"),
        ], remainder="drop")

def build_auto_tagger(tagged_df, model_cls=AutoTagger__KNN):
    tagger = model_cls().fit(tagged_df[_get_ivars(tagged_df)], tagged_df["tags"])
    return tagger

def augment_dataset(saved_df):
    # domain
    def __extract_domain(url):
        if pd.isnull(url):
            return pd.NA
        domain = urlparse(url).hostname
        domain = ".".join(domain.rsplit(".")[-2:])
        return domain
    saved_df["resolved_domain"] = saved_df["resolved_url"].apply(__extract_domain)

    def __extract_path(url):
        if pd.isnull(url):
            return pd.NA

        path = urlparse(url).path
        path = re.sub(r"\b\d+\b", "", path)  # dates, post-ids, etc.
        path = re.sub(r"(^/r\b|/comments/.*$)", "", path)  # reddit
        path = re.sub(r"//", "/", path)
        path = path.strip("/")
        return path
    saved_df["resolved_path"] = saved_df["resolved_url"].apply(__extract_path)

    # text
    @cache_today
    def __retrieve_html(url):
        try:
            return requests.get(url, timeout=3.000).text
        except:
            logging.warn("Couldn't retrieve URL: %s", url, exc_info=True)
            return pd.NA

    def __extract_text(html):
        if pd.isnull(html):
            return pd.NA

        soup = BeautifulSoup(html)
        _ = [s.extract() for s in soup(['style', 'script', 'head', 'title'])]  # discard
        text = soup.getText()
        return text

    # from joblib import Parallel, delayed, parallel_backend
    # with parallel_backend("threading", n_jobs=10):
    #     saved_df["html"] = Parallel()(delayed(__retrieve_html)(url) for url in saved_df["resolved_url"])
    # with parallel_backend("loky", n_jobs=2):
    #     saved_df["text"] = Parallel()(delayed(__extract_text)(html) for html in saved_df["html"])
    # saved_df["text"] = saved_df["text"].str.strip().str.replace(r"\s{2,}", " ")
    # saved_df.loc[saved_df["text"].str.len() == 0, "text"] = pd.NA

    return saved_df

TAGGABLE_IVARS = [
    "resolved_title",
    "excerpt",
    "text",
    "resolved_url",
    "resolved_domain",
    "resolved_path",
    "time_added",
]
def _get_ivars(tagged_df):
    # FIXME: Brittle.
    # return [col for col in tagged_df.columns if col not in ["tags", "videos", "amp_url", "given_title"]]
    return [col for col in TAGGABLE_IVARS if col in tagged_df.columns]
