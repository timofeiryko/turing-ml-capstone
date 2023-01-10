import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import spacy

SEED = 42

# Transformer which adds ratio features

class RatioFeaturesGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, ratio_columns):
        self.ratio_columns = ratio_columns
        
    def fit(self, data, y=None):
        return self
    
    def transform(self, data, y=None):
        X = data.copy()
        for col1, col2 in self.ratio_columns:
            
            ratio_col = f'{col1}_{col2}_ratio'
            X[ratio_col] = X[col1] / X[col2]

            # Replace infinities with nan
            X[ratio_col] = X[ratio_col].replace([np.inf, -np.inf], np.nan)

            # Replace nans with the mean of the column
            X[ratio_col] = X[ratio_col].fillna(X[ratio_col].mean())

        return X

    def get_feature_names_out(self, input_features=None):
    
        if input_features is None:
            input_features = self.X.columns.tolist()

        return input_features

# Create a function which creates a cluster column to add to the pipeline

def create_cluster_column(column, n_clusters=5, random_state=SEED):

        # convert to pandas series and get unique values
        column = pd.Series(column, name='COLUMN')
        words = column.unique()

        # tokenize and vectorize with SpaCy
        nlp = spacy.load('en_core_web_sm')
        tokens = [nlp(str(elem)) for elem in words]
        vectors = [token.vector for token in tokens]
        
        # cluster with kmeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(vectors)
        clusters = kmeans.labels_

        # create a dataframe with the clusters
        clusters_df = pd.DataFrame({'COLUMN': column.unique(), 'cluster': clusters})

        # when value is "Not provided", make cluster = -1
        clusters_df.loc[clusters_df['COLUMN'] == 'Not provided', 'cluster'] = -1

        # merge the clusters with the original dataframe
        merged_df = column.to_frame().merge(clusters_df, on='COLUMN', how='left')
    
        return merged_df.cluster

# Create a custom transformer using the function above
class Cluster(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=5, random_state=SEED, col_names=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.col_names = col_names

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        data = data.copy()

        for col in self.col_names:
            cluster_name = col + '_cluster'
            data[cluster_name] = create_cluster_column(data[col], self.n_clusters, self.random_state)
            data[cluster_name] = data[cluster_name].astype('str')

        return data

    def get_feature_names_out(self, input_features=None):
        
        if input_features is None:
            input_features = self.data.columns.tolist()

        return input_features

# Final pipeline

def get_ready_pipeline():

    ratio_columns = [
        ('amt_income_total', 'amt_credit'),
        ('amt_income_total', 'amt_annuity'),
        ('amt_credit', 'amt_annuity'),
        ('days_birth', 'days_employed'),
        ('amt_credit', 'amt_goods_price'),
        ('amt_income_total', 'amt_goods_price'),
        ('amt_annuity', 'amt_goods_price'),
        ('cnt_fam_members', 'cnt_children'),
        ('amt_credit', 'amt_goods_price'),
        ('amt_req_credit_bureau_year', 'amt_req_credit_bureau_mon'),
        ('amt_credit', 'amt_req_credit_bureau_year'),
        ('amt_income_total', 'amt_req_credit_bureau_year'),
    ]

    not_treated = ('occupation_type', 'organization_type')

    status_df = pd.read_csv(os.path.join('data', 'downsampled_data', 'part-00000-2ccbea9d-88f8-498a-b0b9-bc5808ac195f-c000.csv'))
    X = status_df.drop(['ext_source_1', 'ext_source_2', 'ext_source_3', 'target'], axis='columns')
    y, amt_credit = status_df.target, status_df.amt_credit


    X_train, X_test, y_train, y_test, amt_credit_train, amt_credit_test = train_test_split(
        X, y, amt_credit, test_size=0.2, random_state=SEED
    )

    # numerical pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ratios', RatioFeaturesGenerator(ratio_columns))
    ])

    # categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Not provided')),
        ('clustering', Cluster(col_names=not_treated)),
        ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # create pipeline for predictors
    encoding_col_transformer = ColumnTransformer([
        ('numeric_transformer', numeric_transformer, selector(dtype_include ='number')),
        ('categorical_transformer', categorical_transformer, selector(dtype_exclude='number'))
    ], remainder='passthrough')

    encoding_col_transformer.set_output(transform='pandas')

    # Fit on the training data
    encoding_col_transformer.fit(X_train)

    return encoding_col_transformer