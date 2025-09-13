

import numpy as np
import pandas as pd
from rebalancedcv import RebalancedLeaveOneOut
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black', 
            'axes.grid': False
            }, 
        
        style='ticks',
        font_scale=2
        )



def run_analysis(
            n_samples=50,
            n_features=10,
            penalty='l2',
            noise_strength = 3,
            weight_std=1,
            noise_std=1, 
            seed=1, 
            max_depth=5,
            n_estimators=100, 
            n_classes=2
            ):
    np.random.seed(seed)
    
    model=LogisticRegressionCV(solver='newton-cg', max_iter=1000)
    
    ## set up data
    X=np.random.normal(size=(n_samples, n_features))
    
    ## assume some measurement noise on the variables
    measurement_noise = np.random.normal(scale=noise_std, 
                                         size=X.shape
                                         )
    
    ## need minimum 6 samples in each class to run LogisticRegressionCV nested 5-fold
    done=False
    counter=0
    while not done:

        true_weights = np.random.normal(scale=weight_std,
                                        size=(n_features, n_classes)
                                        ) 
        ## zero out 80% of these rows, making only 20% of features relevant
        true_weights *= np.random.rand(n_features, 1) > 0.8

        noise=np.random.normal(scale=noise_std,
                               size=(n_samples, n_classes)
                               )


        y_float = ( X@true_weights )+noise
        y=y_float.argmax(axis=1)
        
        if pd.Series(y).value_counts().min()>=6:
            done=True
            
        counter+=1
        if counter>100:
            raise(ValueError('Could not set up y labels'))

    loocv_auroc = roc_auc_score(y, 
                    cross_val_predict(model, 
                                      X=X + measurement_noise, 
                                      y=y,
                                      cv=LeaveOneOut(), 
                                      method='predict_proba')[:, 1]
                     )


    rloocv_auroc = \
        roc_auc_score(y, 
                    cross_val_predict(model, 
                                      X=X + measurement_noise, 
                                      y=y,
                                      cv=RebalancedLeaveOneOut(), 
                                      method='predict_proba')[:, 1]
                     )
    return(loocv_auroc, rloocv_auroc)


def main(out_path='general_signal_variation.csv'):
    
    default_setting = {'noise':0.75, 
                       'nsamp':40, 
                       'nfeats':5, 
                       }
    
    loocv_aurocs=[]
    rloocv_aurocs=[]
    noise_stds=[]
    n_samples=[]
    all_nfeats=[]

    for noise_std in [0.25, 0.5, 0.75, 1, 2, 4]:
#     for noise_std in [1,2,4,6,8,10]:
        for n_sample in [20,40,60,80,100]:#[25,50,100,250]:
            for nfeats in [1,2,5,15,25, 50, 100, 200]:  #10,25,50,100]:
                ## only run scenarios that get ploted
                if ((noise_std == default_setting['noise'] ) + \
                        ( n_sample == default_setting['nsamp'] ) + \
                                     ( nfeats == default_setting['nfeats']) )>=2:

                    for seed in range(25):
                        loocv_auroc, rloocv_auroc = \
                            run_analysis(noise_std=noise_std, 
                                         n_samples=n_sample, 
                                         n_features=nfeats,
                                         seed=seed
                                         )

                        loocv_aurocs.append(loocv_auroc)
                        rloocv_aurocs.append(rloocv_auroc)
                        noise_stds.append(noise_std)
                        n_samples.append(n_sample)
                        all_nfeats.append(nfeats)

    plot_df_main = pd.DataFrame({'Signal noise':noise_stds*2, 
                                 'N samples':n_samples*2, 
                                 'N features':all_nfeats*2,
                                 'auROC':loocv_aurocs + rloocv_aurocs, 
                                 'Group':['LOOCV']*len(loocv_aurocs) + \
                                         ['RLOOCV']*len(rloocv_aurocs)
                                })
    
    plot_df_main.to_csv('general_signal_variation.csv')
    
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    

