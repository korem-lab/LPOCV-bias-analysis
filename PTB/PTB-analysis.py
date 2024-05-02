#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegressionCV
from skbio.stats.composition import clr
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('..')
import delong


def main():
    md = pd.read_csv('ptb_metadata.csv', 
               index_col=0)
    md=md.loc[md.Abbr == 'Fe']
    df = pd.read_csv('ptb_data.csv', 
                    index_col=0).loc[md.index]

    y=md.Preterm.values
    X=df.values
    np.random.seed(1)
    df=df.loc[:, (df>0).mean(axis=0)>0.25]
    X = clr(1e-6 + df.values )

    loo=LeaveOneOut()
    vals = [ LogisticRegressionCV(Cs=np.logspace(-3, 3, 7),
                                  max_iter=100, 
                                  solver='newton-cg'
                                 )\
                        .fit(X[train_index], y[train_index])\
                        .predict_proba(X[test_index])[:, 1][0]
                  for train_index, test_index in loo.split(X) ]
    roc_auc_score(y, vals)

    vals_corrected=[]
    loo=LeaveOneOut()
    for train_index, test_index in loo.split(X):    
        lr = LogisticRegressionCV(Cs=np.logspace(-3, 3, 7),
                                  max_iter=100,  
                                  solver='newton-cg',
                                  )

        inds = train_index != train_index[ np.random.choice( np.where( y[train_index] 
                                                             != y[test_index][0])[0] ) ]

        train_index=train_index[inds]


        vals_corrected.append( lr.fit( X[train_index], y[train_index] )                    .predict_proba(X[test_index])[:, 1][0]
                   )


    roc_auc_score(y, vals_corrected)


    from sklearn.metrics import roc_curve, auc
    sns.set(rc={'axes.facecolor':'white', 
                'figure.facecolor':'white', 
                'axes.edgecolor':'black', 
                'grid.color': 'black'
                }, 
           font_scale=3)
    fpr, tpr, _ = roc_curve(y, vals, drop_intermediate=False)
    fpr2, tpr2, _ = roc_curve(y, vals_corrected, drop_intermediate=False)


    rocplotdf = pd.concat([
                    pd.DataFrame({'FPR':fpr, 
                                  'TPR':tpr, 
            'Group':'LOOCV (original; auROC = {:.3f})'.format(auc(fpr, tpr))}), 
                    pd.DataFrame({'FPR':fpr2, 
                                  'TPR':tpr2, 
            'Group':'RLOOCV (auROC = {:.3f})'.format(auc(fpr2, tpr2))}),
                ], axis=0
            ).reset_index(drop=True)

    plt.figure(figsize=(12, 12))
    ax = sns.lineplot(x='FPR', 
                 y='TPR', 
                 hue='Group',
                 linewidth=5, 
                 data=rocplotdf, 
                 ci=0
                 )
    ax.legend_.set_title(None)

    from sklearn.metrics import roc_curve, auc
    sns.set(rc={'axes.facecolor':'white', 
                'figure.facecolor':'white', 
                'axes.edgecolor':'black', 
                'grid.color': 'black'
                }, 
           font_scale=3)
    fpr, tpr, _ = roc_curve(y, vals, drop_intermediate=False)
    fpr2, tpr2, _ = roc_curve(y, vals_corrected, drop_intermediate=False)


    rocplotdf = pd.concat([
                    pd.DataFrame({'FPR':fpr, 
                                  'TPR':tpr, 
            'Group':'LOOCV (original; auROC = {:.3f})'.format(auc(fpr, tpr))}), 
                    pd.DataFrame({'FPR':fpr2, 
                                  'TPR':tpr2, 
            'Group':'RLOOCV (auROC = {:.3f})'.format(auc(fpr2, tpr2))}),
                ], axis=0
            ).reset_index(drop=True)

    plt.figure(figsize=(12, 12))
    ax = sns.lineplot(x='FPR', 
                 y='TPR', 
                 hue='Group',
                 linewidth=5, 
                 data=rocplotdf, 
                 ci=0
                 )
    plt.ylim(0,1)
    plt.xlim(0,1)
    ax.set(xlabel=None,
           ylabel=None, 
           xticklabels=[],
           yticklabels=[]
          )
    ax.legend_.set_title(None)
    plt.savefig('../plots-latest/PTB-roc.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')
    
    np.random.seed(1)
    all_group_preds = []
    for __ in range(10):
        vals_corrected=[]
        loo=LeaveOneOut()
        for train_index, test_index in loo.split(X):    
            lr = LogisticRegressionCV(Cs=np.logspace(-3, 3, 7),
                                      max_iter=100,  
                                      solver='newton-cg'
                                      )

            inds = train_index != train_index[ np.random.choice( np.where( y[train_index] 
                                                                 != y[test_index][0])[0] ) ]

            train_index=train_index[inds]


            vals_corrected.append( lr.fit( X[train_index], y[train_index] )                        .predict_proba(X[test_index])[:, 1][0]
                       )

        all_group_preds.append(vals_corrected)

        print( roc_auc_score(y, vals_corrected) )
        
    np.random.seed(1)
    all_group_preds_base = []
    for __ in range(10):
        vals_base=[]
        loo=LeaveOneOut()
        for train_index, test_index in loo.split(X):    
            lr = LogisticRegressionCV(Cs=np.logspace(-3, 3, 7),
                                      max_iter=100,  
                                      solver='newton-cg'
                                      )

            vals_base.append( lr.fit( X[train_index], y[train_index] )                        
                             .predict_proba(X[test_index])[:, 1][0]
                       )

        all_group_preds_base.append(vals_base)

        print( roc_auc_score(y, vals_base) )
        
        
        
    print( np.power(10, 
                 delong.delong_roc_test(
                        y,
                        np.vstack( all_group_preds_base ).mean(axis=0),
                        np.vstack( all_group_preds ).mean(axis=0) ,
                         )[0] 
                ) )


    def make_roc_df(df_tmp, group_num, run):
        fpr, tpr, _ = roc_curve(df_tmp.labels, 
                                df_tmp.predictions, 
                                drop_intermediate=False)
        return( pd.DataFrame({'FPR':fpr, 
                              'TPR':tpr, 
                              'Group':group_num, 
                              'run':run})
              )
    
    

    def format_rocs(summary):
        ## this function ensures all unceratainty ranges along the roc curve
        ## are composed of results across all bootstrap runs

        summary['key']=0
        ss=summary.merge( pd.DataFrame( {'full_FPR':np.sort( summary.FPR.unique() ), 
                                 'key':0} ), 
                     on='key')

        ss=ss.loc[ss.FPR<=ss.full_FPR]        .groupby(['full_FPR', 'Group', 'run'])['TPR'].max().reset_index()

        ss=ss[['full_FPR', 'TPR', 'Group', 'run']]
        ss.columns=['FPR', 'TPR', 'Group', 'run']
        return(ss)


    rocs = format_rocs( pd.concat([ make_roc_df( pd.DataFrame({'labels':y, 
                    'predictions':a
                    }), 
                     'RLOOCV (auROC: {:.3f})'.format(
                                     np.median( [roc_auc_score(y, a) 
                                                 for a in all_group_preds] )), 
                     i
                     )
          for i,a in enumerate(all_group_preds) ]
                          )
               )


    rocs_base = format_rocs( pd.concat([ make_roc_df( pd.DataFrame({'labels':y, 
                    'predictions':a
                    }), 
                     'LOOCV (auROC: {:.3f})'.format(
                                     np.median( [roc_auc_score(y, a) 
                                                 for a in all_group_preds_base] )), 
                     i
                     )
          for i,a in enumerate(all_group_preds_base) ]
                          )
               )

    rocs_all = pd.concat([
                          rocs_base, 
                          rocs ], 
                    axis=0
                    ).reset_index(drop=True)

    pal = sns.color_palette()

    plt.figure(figsize=(12, 12))

    ax = sns.lineplot(x='FPR', 
                      y='TPR', 
                      hue='Group',
                      linewidth=5, 
                      data=rocs_all, 
                      ci=95, 
                      )

    plt.plot([0,1], 
             [0,1], 
             color='black', 
             linestyle = '--', 
             linewidth=5)
    plt.ylim([0,1])
    plt.xlim([0,1])
    ax.legend().set_title(None)
    plt.legend(loc='lower right')
    

    plt.savefig('../plots-latest/PTB-roc-with-bootstrap.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')


if __name__=='__main__':
    main()

