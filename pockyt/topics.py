import pandas as pd
from sklearn import dummy, multioutput, neighbors, preprocessing

class AutoTagger(multioutput.MultiOutputClassifier):
    def fit(self, X, y):
        self.binarizer_ = preprocessing.MultiLabelBinarizer().fit(y)
        Y = pd.DataFrame(self.binarizer_.transform(y), index=X.index, columns=self.binarizer_.classes_)

        X = X.dropna()
        Y = Y.loc[X.index]

        return super().fit(X, Y)

    def transform(self, untagged_df):
        tag_proba_df = pd.DataFrame(self.predict_proba(untagged_df[_get_ivars(untagged_df)]),
                                    columns=self.classes_, index=untagged_df.index)
        tagged_df = tag_proba_df.where(tag_proba_df > 0.500) \
                                .reset_index() \
                                .rename(columns={ "index": "item_id" }) \
                                .melt(id_vars="item_id", var_name="tags", value_name="proba") \
                                .dropna() \
                                .sort_values(["item_id", "proba"], ascending=[True, False]) \
                                .groupby("item_id")["tags"] \
                                .apply(list) \
                                .to_frame()
        return tagged_df

class AutoTagger__Dummy(AutoTagger):
    def __init__(self):
        # FIXME: Implement pipeline.
        super().__init__(estimator=dummy.DummyClassifier(strategy="stratified"))

class AutoTagger__Chained(AutoTagger):
    def __init__(self):
        # FIXME: Implement pipeline.
        super().__init__(estimator=neighbors.KNeighborsClassifier(strategy="stratified"))

    def transform(self, untagged_df):
        tag_proba_df = pd.DataFrame(self.predict_proba(untagged_df[_get_ivars(untagged_df)]),
                                    columns=self.classes_, index=untagged_df.index)
        tagged_df = tag_proba_df.where(tag_proba_df > 0.500) \
                                .reset_index() \
                                .rename(columns={ "index": "item_id" }) \
                                .melt(id_vars="item_id", var_name="tags", value_name="proba") \
                                .dropna() \
                                .groupby("item_id")["tags"] \
                                .sum() \
                                .to_frame()
        return tagged_df

def build_auto_tagger(tagged_df, model_cls=AutoTagger__Dummy):
    tagger = model_cls().fit(tagged_df[_get_ivars(tagged_df)], tagged_df["tags"])
    return tagger

def _get_ivars(tagged_df):
    return [col for col in tagged_df.columns if col not in ["tags", "videos", "amp_url", "given_title"]]
