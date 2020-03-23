import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import CategoricalDtype

def json_helper(x):
    if type(x) == str:
        return json.loads(x)
    else:
        return {}
    
def normalize_json_column(df, column):
    column_normalized_df = json_normalize(df[column].apply(json_helper))
    column_normalized_df.columns = list(map(lambda x: "{}.{}".format(column,x),
                                            column_normalized_df.columns))
    result = pd.concat([df, column_normalized_df], axis=1, sort=False)
    del result[column]
    return result

class KickstarterModel:

    def __init__(self):

        self.model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

    def common_preprocess_steps(self, df):
        normalized_df = df
        normalized_df = normalize_json_column(normalized_df, 'location')
        normalized_df = normalize_json_column(normalized_df, 'category')

        selected_features = ['name', 'blurb', 'goal', 'slug', 'country', 'currency',
                     'deadline', 'created_at', 'launched_at', 'static_usd_rate',
                     'location.slug', 'location.state', 'location.type',
                     'category.id', 'category.parent_id',
                     'state']

        selected_df = normalized_df[selected_features]

        transformed_df = selected_df.copy()
        transformed_df['name'] = transformed_df['name'].fillna('').apply(lambda x: len(x))
        transformed_df['blurb'] = transformed_df['blurb'].fillna('').apply(lambda x: len(x))
        transformed_df['goal_in_usd'] = transformed_df['goal'] * transformed_df['static_usd_rate']
        del transformed_df['goal']
        del transformed_df['static_usd_rate']
        transformed_df['slug'] = transformed_df['slug'].fillna('').apply(lambda x: len(x))
        del transformed_df['currency']
        transformed_df['deadline-created_at'] = transformed_df['deadline'] - transformed_df['created_at']
        transformed_df['deadline-launched_at'] = transformed_df['deadline'] - transformed_df['launched_at']
        transformed_df['launched_at-created_at'] = transformed_df['launched_at'] - transformed_df['created_at']
        del transformed_df['deadline']
        del transformed_df['launched_at']
        del transformed_df['created_at']
        del transformed_df['location.state']

        selected_countries = ['US', 'GB', 'CA','AU']
        for country in selected_countries:
            transformed_df['country_{}'.format(country)] = transformed_df['country'].apply(lambda x: 1 if x == country else 0)
        del transformed_df['country']

        selected_loc_slugs = ['los-angeles-ca', 'new-york-ny',
                            'london-gb', 'chicago-il']
        for slug in selected_loc_slugs:
            transformed_df['location.slug_{}'.format(slug)] = transformed_df['location.slug'].apply(lambda x: 1 if x == slug else 0)
        del transformed_df['location.slug']
            
        selected_loc_types = ['Town', 'County', 'Suburb']
        for loctype in selected_loc_types:
            transformed_df['location.type_{}'.format(loctype)] = transformed_df['location.type'].apply(lambda x: 1 if x == loctype else 0)
        del transformed_df['location.type']

        return transformed_df


    def preprocess_training_data(self, df):
        df = self.common_preprocess_steps(df)

        state_transform_func = lambda x: 1 if x == 'successful' else 0
        df['target'] = df['state'].apply(state_transform_func)
        del df['state']

        y = df['target']
        X = df.copy()
        del X['target']

        return (X,y)

    def fit(self, X, y):
        self.model.fit(X,y)

    def preprocess_unseen_data(self, df):
        df['state'] = None
        df = self.common_preprocess_steps(df)
        del df['state']
        
        return df

    def predict(self, X):
        return self.model.predict(X)
