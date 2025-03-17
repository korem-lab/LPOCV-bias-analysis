import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from rebalancedcv import RebalancedLeaveOneOut
from ucimlrepo import fetch_ucirepo, list_available_datasets
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, cross_val_predict

# Define the pipeline
pipeline_pca = Pipeline([
    ('pca', PCA(n_components=2)),
    ('logistic', LogisticRegressionCV(solver='newton-cg', 
                                      cv=LeaveOneOut()))
])


# Define the pipeline
pipeline = Pipeline([
    ('logistic', LogisticRegressionCV(solver='newton-cg', 
                                      cv=LeaveOneOut()
                                       )) ])
# Define the pipeline
pipeline_pca_rloocv = Pipeline([
    ('pca', PCA(n_components=2)),
    ('logistic', LogisticRegressionCV(solver='newton-cg', 
                                      cv=RebalancedLeaveOneOut()))
])

# Define the pipeline
pipeline_rloocv = Pipeline([
    ('logistic', LogisticRegressionCV(solver='newton-cg', 
                                      cv=RebalancedLeaveOneOut()
                                       )) ])




def test_loo_variations(X, 
                        y, 
                        seed=42):
    
    np.random.seed(seed)
    v1 = cross_val_predict(pipeline, 
                           X, 
                           y, 
                           cv=LeaveOneOut(), 
                           method='predict_proba'
                           )
    
    np.random.seed(seed)
    v2 = cross_val_predict(pipeline_rloocv, 
                           X, 
                           y, 
                           cv=RebalancedLeaveOneOut(), 
                           method='predict_proba'
                           )
    
    return(roc_auc_score(y, v1[:, 1]), 
           roc_auc_score(y, v2[:, 1]), 
           )

def test_loo_variations_pca(X, 
                            y, 
                            seed=42):
    
    np.random.seed(seed)
    v1 = cross_val_predict(pipeline_pca, 
                           X, 
                           y, 
                           cv=LeaveOneOut(), 
                           method='predict_proba'
                           )
    
    np.random.seed(seed)
    v2 = cross_val_predict(pipeline_pca_rloocv, 
                           X, 
                           y, 
                           cv=RebalancedLeaveOneOut(), 
                           method='predict_proba'
                           )
    
    return(roc_auc_score(y, v1[:, 1]), 
           roc_auc_score(y, v2[:, 1]), 
           )
def main():
    useable_ids=[  1,   2,   3,  12,  15,  16,  17,  20,  23,  27,  28,  30,  31,
                   32,  33,  39,  42,  43,  44,  45,  46,  47,  50,  52,  53,  54,
                   59,  62,  63,  70,  74,  75,  80,  81,  83,  90,  91,  94,  95,
                   96, 107, 109, 110, 111, 143, 144, 145, 146, 147]


    loo_aurocs=[]
    rloo_aurocs=[]
    loo_aurocs_pca=[]
    rloo_aurocs_pca=[]
    result_ids = []

    ss=StandardScaler()
    for ii in useable_ids:
        ds=fetch_ucirepo(id=ii)
        # access data
        X = ds.data.features
        y = ds.data.targets

        X=X.select_dtypes(include=numerics)

        if X.shape[1] >=2 and y is not None:
            col=y.columns[0]
            y=y[y[col].isin(y[col].value_counts().head(2).index)]
            
            np.random.seed(42)
            y=y.sample(n=min(y.shape[0], 50), replace=False)

            X=X.loc[y.index]

            y = y.values[:, 0]==y.values[0, 0]
            X = pd.DataFrame( ss.fit_transform(X) ).fillna(0).values

            
            if np.unique(y).shape[0]==2 and pd.Series(y).value_counts().min() > 5 :

                looau, rlooau = test_loo_variations(X, y, seed=42)
                looaupca, rlooaupca = test_loo_variations_pca(X, y, seed=42)

                loo_aurocs.append(looau)
                rloo_aurocs.append(rlooau)
                loo_aurocs_pca.append(looaupca)
                rloo_aurocs_pca.append( rlooaupca )
                result_ids.append(ii)

                print(loo_aurocs)
                print(rloo_aurocs)
                print(loo_aurocs_pca)
                print(rloo_aurocs_pca)


    pd.DataFrame({'auROC':loo_aurocs + rloo_aurocs+\
                              loo_aurocs_pca+rloo_aurocs_pca, 
                  'cv':['LOOCV']*len(loo_aurocs) + \
                          ['RLOOCV']*len(rloo_aurocs) + \
                          ['LOOCV PCA']*len(loo_aurocs_pca) + \
                          ['RLOOCV PCA']*len(rloo_aurocs_pca),
                   'dataset_id':result_ids*4
                   }).to_csv('results/hloocv_ucirvine_classif_results_all.csv')


if __name__=='__main__':
    main()






