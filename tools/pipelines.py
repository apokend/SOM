#---------------------------+
#        Version:  1.01     +
#   Status: Ready to Test   +
#   Author: Shevchenko A.A. +
#-------------------------- +

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Preprocessing pipline
# Step 1. Imput miss value by median
# Step 2. Scale our data
# Step 3. Use PCA with 4 components
prep_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=4))
])

# Pattern in which will be add SOM model
full_pipeline_with_predictor = Pipeline([
    ('preproc', prep_pipeline)
])

