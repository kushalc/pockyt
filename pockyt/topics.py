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
        Xt = X.dropna()  # FIXME: Irritating, but good enough for now.
        yt = y.loc[Xt.index]

        self.binarizer_ = preprocessing.MultiLabelBinarizer().fit(yt)
        self.featurizer_ = self.featurizer.fit(Xt)

        Yt = pd.DataFrame(self.binarizer_.transform(yt), index=Xt.index, columns=self.binarizer_.classes_)
        Xt = self.featurizer_.transform(Xt)
        return super().fit(Xt, Yt)

    def transform(self, untagged_df):
        untagged_df = untagged_df[_get_ivars(untagged_df)].dropna()  # FIXME
        Xt = self.featurizer_.transform(untagged_df)

        tagged_df = pd.DataFrame({ "tags": self.binarizer_.inverse_transform(self.predict(Xt)) }, index=untagged_df.index)
        tagged_df.index.name = "item_id"
        return tagged_df

    def _build_featurizer(self):
        raise NotImplementedError()

    def _build_estimator(self):
        raise NotImplementedError()

class AutoTagger__Dummy(AutoTagger):
    def _build_estimator(self):
        return dummy.DummyClassifier(strategy="stratified")

    def _build_featurizer(self):
        return compose.ColumnTransformer([
            # FIXME: Make this not break.
            # ("title_bow", make_pipeline(impute.SimpleImputer(strategy="constant"),
            #                             feature_extraction.text.CountVectorizer()), ["resolved_title"]),
            ("title_bow", feature_extraction.text.CountVectorizer(), "resolved_title"),
        ], remainder="drop")

class AutoTagger__KNN(AutoTagger):
    def _build_estimator(self):
        return neighbors.KNeighborsClassifier()

    def _build_featurizer(self):
        # FIXME: Try embeddings.
        # FIXME: Try domains.
        # FIXME: Try unsupervised techniques.
        return compose.ColumnTransformer([
            ("title_bow", feature_extraction.text.TfidfVectorizer(strip_accents="unicode", binary=True), "resolved_title"),
            ("excerpt_bow", feature_extraction.text.CountVectorizer(), "excerpt"),
        ], remainder="drop")

def build_auto_tagger(tagged_df, model_cls=AutoTagger__KNN):
    tagger = model_cls().fit(tagged_df[_get_ivars(tagged_df)], tagged_df["tags"])
    return tagger

def _get_ivars(tagged_df):
    # FIXME: Brittle.
    # return [col for col in tagged_df.columns if col not in ["tags", "videos", "amp_url", "given_title"]]
    return ["resolved_title", "excerpt", "resolved_url", "time_added"]
