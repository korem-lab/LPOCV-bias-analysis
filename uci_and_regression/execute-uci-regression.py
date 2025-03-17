import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from rebalancedcv import RebalancedLeaveOneOut
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from scipy.stats import normaltest

from loocv_regression import RebalancedLeaveOneOutRegression
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import RidgeCV, Ridge
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from ucimlrepo import fetch_ucirepo, list_available_datasets

from loocv_regression import RebalancedLeaveOneOutRegression

from sklearn.linear_model import RidgeCV
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

pipeline_regression = Pipeline([
    ('logistic', RidgeCV(alphas = np.logspace(-6, 6, 13)))
])


pipeline_pca_regression = Pipeline([
    ('pca', PCA(n_components=2)),
    ('logistic', RidgeCV(alphas = np.logspace(-6, 6, 13)))
])


def test_loo_variations_regression(X, 
                                   y, 
                                   pipe=pipeline_regression,
                                   seed=42
                                   ):
    
    np.random.seed(seed)
    v1 = cross_val_predict(pipe, 
                           X, 
                           y, 
                           cv=LeaveOneOut(), 
                           method='predict'
                           )
    
    np.random.seed(seed)
    v2 = cross_val_predict(pipe, 
                           X, 
                           y, 
                           cv=RebalancedLeaveOneOutRegression(), 
                           method='predict'
                           )
    
    return(r2_score(y, v1), 
           r2_score(y, v2)
           )

def main():
    possible_ids = [ 
                     9,  10,  60,  87,  89, 162, 165, 183, 
                     189, 211, 275, 291, 294,
                     368, 374, 390, 409, 464, 
                     477, 492, 849, 925
                   ]

    result_ids = []
    loo_r2s = []
    rloo_r2s = []
    loo_pca_r2s = []
    rloo_pca_r2s = []

    ss=StandardScaler()

    seed=42
    sample_num=0
    for ii in possible_ids:
        dataset = fetch_ucirepo(id=int(ii))
        ds=dataset
        if(('Regression' in dataset['metadata']['tasks'] \
            and 'Classification' not in dataset['metadata']['tasks'] )\
                or 'Regression' == dataset['metadata']['tasks']
                        ):
            
            for sample_num in range(5):

                X = dataset['data']['features']

                # access data
                X = ds.data.features
                if ds.data.targets is not None:

                    np.random.seed(seed+sample_num)

                    y = ds.data.targets
                    y=y.loc[y.iloc[:, 0] > y.iloc[:, 0].min()]
                    y=y.loc[y.iloc[:, 0] < y.iloc[:, 0].max()]
                    
                    y = y.dropna(axis=0).sample(n=min(y.shape[0], 50)) ## just use 50
                    
                    X=X.select_dtypes(include=numerics)
                    if X.shape[1] >=2 and y is not None:
                        col=y.columns[0]

                        X=X.loc[y.index]

                        try:
                            y.iloc[:, 0]=y.iloc[:, 0].str.rstrip('%').astype(float)
                        except:
                            pass

                        if y.iloc[0].dtype!='O':

                            y_other = y.values[:, 0].astype(float)
                            y = ss.fit_transform( y.values[:, 0].astype(float).reshape(1, -1).T )[:, 0]
                            X = pd.DataFrame( ss.fit_transform(X) ).fillna(0).values

                            if np.unique(y).shape[0]>=10 :
                                loo, rloo = test_loo_variations_regression(X, y, seed=seed)

                                loo_pca, rlooa_pca = test_loo_variations_regression(X, 
                                                                                    y, 
                                                                                    pipe=pipeline_pca_regression,
                                                                                    seed=seed)

                                loo_r2s.append(loo)
                                rloo_r2s.append(rloo)

                                loo_pca_r2s.append(loo_pca)
                                rloo_pca_r2s.append(rlooa_pca)
                                result_ids.append(ii)



    plot_data = \
        pd.DataFrame({'$R^2$':loo_pca_r2s + rloo_pca_r2s +\
                                  loo_r2s + rloo_r2s, 
                      'Method':['LOOCV --> PCA']*len(loo_pca_r2s) + \
                                  ['RLOOCV --> PCA']*len(rloo_pca_r2s) + \
                                  ['LOOCV']*len(loo_r2s) + \
                                  ['RLOOCV']*len(rloo_r2s),
                      'id':result_ids*4
                      })

    plot_data.to_csv('results/uci-regression-run.csv')
    
    
if __name__=='__main__':
    main()