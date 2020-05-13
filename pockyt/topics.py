import spacy
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import compose, dummy, feature_extraction, impute, multioutput, neighbors, preprocessing

class AutoTagger(multioutput.MultiOutputClassifier):
    def __init__(self, featurizer=None, estimator=None):
        if featurizer is None:
            featurizer = self._build_featurizer()
        if estimator is None:
            estimator = self._build_estimator()
        super().__init__(estimator=estimator)
        self.set_params(featurizer=featurizer)

    def fit(self, X, y):
        self.binarizer_ = preprocessing.MultiLabelBinarizer().fit(y)
        self.featurizer_ = self.featurizer.fit(X)

        Yt = pd.DataFrame(self.binarizer_.transform(y), index=X.index, columns=self.binarizer_.classes_)
        Xt = self.featurizer_.transform(X)
        return super().fit(Xt, Yt)

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

        def __spacy_embeddings():
            nlp = spacy.load("en_core_web_md")
            def __spacy_transform(texts):
                return np.vstack([doc.vector for doc in nlp.pipe(texts, disable=["tagger", "parser", "ner"])])
            return preprocessing.FunctionTransformer(__spacy_transform)

        # FIXME: Try domains.
        # FIXME: Try embeddings.
        # FIXME: Try unsupervised techniques.
        # FIXME: Try crawling/parsing raw HTML.
        return compose.ColumnTransformer([
            # ("bow_resolved_title", self._protect_nullable(__binary_tfidf()), "resolved_title"),
            # ("bow_excerpt", self._protect_nullable(__binary_tfidf()), "excerpt"),
            ("emb_resolved_title", self._protect_nullable(__spacy_embeddings()), "resolved_title"),
        ], remainder="drop")

def build_auto_tagger(tagged_df, model_cls=AutoTagger__KNN):
    tagger = model_cls().fit(tagged_df[_get_ivars(tagged_df)], tagged_df["tags"])
    return tagger

def _get_ivars(tagged_df):
    # FIXME: Brittle.
    # return [col for col in tagged_df.columns if col not in ["tags", "videos", "amp_url", "given_title"]]
    return ["resolved_title", "excerpt", "resolved_url", "time_added"]
