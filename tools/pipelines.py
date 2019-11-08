from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

prep_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=4))
])


full_pipeline_with_predictor = Pipeline([
    ('preproc', prep_pipeline)
    #('kohonen', None)
])


# full_pipeline_with_predictor.steps.append(['kohonen',None])
