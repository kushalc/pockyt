import pandas as pd
from sklearn import dummy, pipeline, ensemble

class AutoTagger(object):
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

class AutoTagger__Dummy(AutoTagger, dummy.DummyClassifier):
    pass

def build_auto_tagger(tagged_df, model_cls=AutoTagger__Dummy, model_kwargs={ "strategy": "stratified" }):
    tagger = model_cls().fit(tagged_df[_get_ivars(tagged_df)], tagged_df["tags"])
    return tagger

def _get_ivars(tagged_df):
    return [col for col in tagged_df.columns if col not in ["tags"]]
