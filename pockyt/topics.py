import logging

import spacy
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import compose, dummy, feature_extraction, impute, multioutput, neighbors, preprocessing

# FIXME: Remove me.
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
        self.binarizer_ = preprocessing.MultiLabelBinarizer().fit(y)
        self.featurizer_ = self.featurizer.fit(X)

        Yt = pd.DataFrame(self.binarizer_.transform(y), index=X.index, columns=self.binarizer_.classes_)
        Xt = self.featurizer_.transform(X)
        return super().fit(Xt, Yt)

    # FIXME: Remove me.
    @instrument_latency
    def transform(self, untagged_df):
        untagged_df = untagged_df[_get_ivars(untagged_df)]
        Xt = self.featurizer_.transform(untagged_df)

        tagged_df = pd.DataFrame({ "tags": self.binarizer_.inverse_transform(self.predict(Xt)) }, index=untagged_df.index)
        tagged_df.index.name = "item_id"
        return tagged_df

    def _build_featurizer(self):
        raise NotImplementedError()

    def _build_estimator(self):
        raise NotImplementedError()

    def _protect_nullable(self, featurizer, fill_value=""):
        return make_pipeline(preprocessing.FunctionTransformer(lambda df: df.fillna(fill_value)),
                             featurizer)

class AutoTagger__Dummy(AutoTagger):
    def _build_estimator(self):
        return dummy.DummyClassifier(strategy="stratified")

    def _build_featurizer(self):
        return compose.ColumnTransformer([
            ("title_bow", self._protect_nullable(feature_extraction.text.CountVectorizer()), "resolved_title"),
        ], remainder="drop")

class AutoTagger__KNN(AutoTagger):
    def _build_estimator(self):
        return neighbors.KNeighborsClassifier()

    def _build_featurizer(self):
        def __binary_tfidf():
            return feature_extraction.text.TfidfVectorizer(strip_accents="unicode", binary=True)

        # FIXME: Too slow?
        def __spacy_embeddings():
            nlp = spacy.load("en_core_web_md")

            # FIXME: Remove me.
            @instrument_latency
            def __transform(texts_s):
                return np.vstack([doc.vector for doc in nlp.pipe(texts_s, disable=["tagger", "parser", "ner"])])
            return preprocessing.FunctionTransformer(__transform)

        def __sparknlp_embeddings():
            from sparknlp.annotator import SentenceEmbeddings
            from sparknlp.pretrained import PretrainedPipeline
            import sparknlp

            spark = sparknlp.start()
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
            return preprocessing.FunctionTransformer(__transform)

        # FIXME: Try domains.
        # FIXME: Try unsupervised techniques.
        # FIXME: Try crawling/parsing raw HTML.
        return compose.ColumnTransformer([
            # ("bow_resolved_title", self._protect_nullable(__binary_tfidf()), "resolved_title"),
            # ("bow_excerpt", self._protect_nullable(__binary_tfidf()), "excerpt"),
            ("emb_resolved_title", self._protect_nullable(__sparknlp_embeddings()), "resolved_title"),
        ], remainder="drop")

def build_auto_tagger(tagged_df, model_cls=AutoTagger__KNN):
    tagger = model_cls().fit(tagged_df[_get_ivars(tagged_df)], tagged_df["tags"])
    return tagger

def _get_ivars(tagged_df):
    # FIXME: Brittle.
    # return [col for col in tagged_df.columns if col not in ["tags", "videos", "amp_url", "given_title"]]
    return ["resolved_title", "excerpt", "resolved_url", "time_added"]
