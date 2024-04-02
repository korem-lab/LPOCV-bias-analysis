#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import LeaveOneOut
import sys
from scipy.stats import wilcoxon
from sklearn.decomposition import KernelPCA
from skbio.stats.composition import clr
import warnings
warnings.filterwarnings('ignore')
sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black'
            }, 
       font_scale=2)
np.random.seed(0)
from rebalancedleaveoneout import RebalancedLeaveOneOut
from sklearn.linear_model import LogisticRegression

def rescale(xx):
    return(xx/xx.sum(axis=1)[:, np.newaxis] )


def test_pipelines_subsampled_reg(prediction_task, 
                   df, 
                   md, 
                   seed=0, 
                   C_reg=1,
                   pca_components=50, 
                   use_subsampling=False, 
                   return_roc_curve=False
                   ):
    val_inds=md.disease_type_consol.isin(prediction_task)
    val_split = df.loc[val_inds]
    pc=KernelPCA(n_components=pca_components, 
                 kernel='cosine'
                )
    val_split= pd.DataFrame( pc.fit_transform( val_split ), 
                             index=val_split.index
                           )

    y_tt = md.loc[val_inds].disease_type_consol != prediction_task[0]
    
    np.random.seed(seed)
    db_preds=[]
    
    if use_subsampling:
        loo_method=RebalancedLeaveOneOut()
    else:
        loo_method=LeaveOneOut()
    
    for train_inds, test_inds in loo_method.split(val_split.values, y_tt.values):
        
        XTr = val_split.values[train_inds]
        ytr = y_tt.values[train_inds]
        
        
        lr=LogisticRegression(
                              max_iter=5000, 
                              C=C_reg, 
                              solver='liblinear'
                              )
        
        lr.fit(XTr, ytr)
        
        db_preds.append( lr.predict_proba( val_split.values[test_inds,] )[:, 1][0] )

    v1= roc_auc_score(y_tt, db_preds)
    db_preds_actual=db_preds.copy()
    
    if return_roc_curve:
        return(roc_curve(y_tt, db_preds_actual, drop_intermediate=False), 
               roc_curve(y_tt, db_preds_actual, drop_intermediate=False) )
    
    else:
        return(v1, v1)

def main():
    data=pd.read_csv('scrubbed-plasma.csv', index_col=0)
    md=pd.read_csv('metadata_169samples.tsv', sep='\t', index_col=0)
    data=data.loc[md.index]
    data=data.loc[:, (data > 0 ).mean(axis=0) > 0.01]
    np.random.seed(0)

    # setting up the clr inputs
    df =  pd.DataFrame( clr(rescale(1e-4+rescale(data))), 
                                              index=data.index, 
                                              columns=data.columns 
                                            )

    ## considering the batch-corrected representation of the microbiome data, in addition to the raw data
    transformed=pd.read_csv('batch-corrected-relabund.csv', index_col=0)
    transformed=pd.DataFrame( clr(rescale(1e-4+transformed)), 
                                              index=transformed.index, 
                                              columns=transformed.columns 
                                            ).loc[md.index]

    ## some prevalence filter
    inds = ( data>0 ).mean( axis=0 ) > .01
    all_runs = []


    n_pc_comps=50
    all_runs = []
    ret_roc=False
    for reg_level_C in [1e-4, 1e-3, 1e-2,
                        1e-1, 1, 10, 1e2,1e3,1e4,1e5, 1e6]:

        print(reg_level_C)
        for use_transformed_data in [True, False]:
            for use_subsampling in [True, False]:
    #             if use_subsampling:
                v1,v2=test_pipelines_subsampled_reg(['Control', 'SKCM'],
                                     [transformed if use_transformed_data else df][0].loc[:,inds],
                                     md,
                                     seed=0, 
                                     pca_components=n_pc_comps, 
                                     use_subsampling=use_subsampling,
                                     return_roc_curve=ret_roc, 
                                     C_reg=reg_level_C
                          )
                all_runs.append(('Control vs SKCM', use_transformed_data, use_subsampling, v1, v2, n_pc_comps, reg_level_C))


                v1,v2=test_pipelines_subsampled_reg(['Control', 'NSCLC'],
                                     [transformed if use_transformed_data else df][0].loc[:,inds],
                                     md,
                                     seed=0, 
                                     pca_components=n_pc_comps, 
                                     use_subsampling=use_subsampling, 
                                     return_roc_curve=ret_roc, 
                                     C_reg=reg_level_C
                                     )

                all_runs.append(('Control vs NSCLC', use_transformed_data, use_subsampling, v1, v2, n_pc_comps, reg_level_C))

                ## subsetting a bechmark population for prad
                iii=(md.sex=='male')&(md.host_age<65)

                v1,v2=test_pipelines_subsampled_reg(['Control', 'PRAD'],
                                     [transformed if use_transformed_data else df][0].loc[:,inds].loc[iii].loc[:, inds], 
                                     md.loc[iii],
                                     seed=0, 
                                     pca_components=n_pc_comps, 
                                     use_subsampling=use_subsampling,
                                     return_roc_curve=ret_roc, 
                                     C_reg=reg_level_C
                                     )

                all_runs.append(('Control vs PRAD, male aged 40-65', use_transformed_data, use_subsampling, v1, v2, n_pc_comps, reg_level_C))


    results=pd.DataFrame(np.vstack([np.array(a) for a in all_runs]), 
                     columns = ['Task', 'Batch-corrected', 'Subsampling', 'ROC', 
                                'Randomized ROC', 'PCA components', 'C regularization']
                    ).sort_values('Task')


    results['Group'] = results['Task'] + ' ' +                    results['Batch-corrected'].str.replace('True', 
                                                                'Batch-corrected')\
                                                   .str.replace('False', 
                                                                '')+ ' ' +\
                        results['Subsampling'].str.replace('True', 
                                                                'Subsampled')\
                                                   .str.replace('False', 
                                                                '')+ ' '

    pd.set_option('display.precision', 3)
    qqq=results            .pivot(index='Group', 
                                   columns='C regularization', 
                                   values='ROC'
                                   ).astype(float)\
            .head(4)
    hhtml=qqq.round(3).style.background_gradient(cmap ='coolwarm', 
                                       vmin=qqq.values.min(), 
                                      vmax=qqq.values.max(), 
                                      )\
            .set_properties(**{'font-size': '20px'})

    hhtml.to_excel('../plots-latest/plasma-nsclc.xlsx')

    hhtml



    pd.set_option('display.precision', 3)
    qqq=results            .pivot(index='Group', 
                                   columns='C regularization', 
                                   values='ROC'
                                   ).astype(float)\
            .head(8).tail(4)
    hhtml=qqq.round(3).style.background_gradient(cmap ='coolwarm', 
                                       vmin=qqq.values.min(), 
                                      vmax=qqq.values.max(), 
                                      )\
            .set_properties(**{'font-size': '20px'})


    hhtml.to_excel('../plots-latest/plasma-prad.xlsx')

    hhtml

    pd.set_option('display.precision', 3)
    qqq=results            .pivot(index='Group', 
                                   columns='C regularization', 
                                   values='ROC'
                                   ).astype(float)\
            .tail(4)


    hhtml = qqq.round(3).style.background_gradient(cmap ='coolwarm', 
                                       vmin=qqq.values.min(), 
                                      vmax=qqq.values.max(), 
                                      )\
            .set_properties(**{'font-size': '20px'})\

    hhtml.to_excel('../plots-latest/plasma-skcm.xlsx')

    hhtml


    print( wilcoxon( results.loc[results.Subsampling!='True'].ROC.astype(float),
              results.loc[results.Subsampling=='True'].ROC.astype(float)
            ) )


    print( [[ wilcoxon( results.loc[(results.Subsampling!='True')&                        (results.Task==t)&                         (results['Batch-corrected']==bc)
                                ].ROC.astype(float),
                 results.loc[(results.Subsampling=='True')&\
                             (results.Task==t)&\
                             (results['Batch-corrected']==bc)
                                ].ROC.astype(float)
               ) for t in results.Task.unique() 
         ] for bc in results['Batch-corrected'].unique() ] )


if __name__=='__main__':
    main()

