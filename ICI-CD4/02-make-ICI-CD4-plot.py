#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from rebalancedleaveoneout import RebalancedLeaveOneOut
from sklearn.linear_model import LogisticRegression

from scipy.stats import combine_pvalues

import sys
sys.path.append('../')
import delong

def main():
    seed=1
    x1 = pd.read_csv('no-correction-results.csv', index_col=0)
    x2 = pd.read_csv('with-correction-results.csv', index_col=0)

    roc_auc_score(x1.Severe_irAE, x1.CompositeModelLOOCV)


    roc_auc_score(x2.Severe_irAE, x2.CompositeModelLOOCV)

    fpr, tpr, _ = roc_curve(x1.Severe_irAE, 
                            x1.CompositeModelLOOCV, 
                            drop_intermediate=False
                           )
    fpr2, tpr2, _ = roc_curve(x2.Severe_irAE, 
                              x2.CompositeModelLOOCV, 
                              drop_intermediate=False
                              )


    df = pd.concat([
                    pd.DataFrame({'FPR':fpr, 
                                  'TPR':tpr, 
            'Group':'LOOCV (original; auROC = {:.3f})'.format(auc(fpr, tpr))}), 
                    pd.DataFrame({'FPR':fpr2, 
                                  'TPR':tpr2, 
            'Group':'RLOOCV (auROC = {:.3f})'.format(auc(fpr2, tpr2))}),
                    ], axis=0
            ).reset_index(drop=True)

    sns.set(rc={'axes.facecolor':'white', 
                'figure.facecolor':'white', 
                'axes.edgecolor':'black', 
                'grid.color': 'black'
                }, 
           font_scale=3)


    plt.figure(figsize=(12, 12))
    ax = sns.lineplot(x='FPR', 
                 y='TPR', 
                 hue='Group',
                 linewidth=5, 
                 data=df, 
                 ci=0
                 )
    ax.legend_.set_title(None)

    plt.title('Correcting for LOOCV distributional bias\n'+             'slightly improves prediction of ICI adverse outcomes\n'+             'from T Cell abundance and diversity')


    plt.figure(figsize=(12, 12))
    ax = sns.lineplot(x='FPR', 
                 y='TPR', 
                 hue='Group',
                 linewidth=5, 
                 data=df, 
                 ci=0
                 )
    ax.legend_.set_title(None)
    plt.ylim(0,1)
    plt.xlim(0,1)
    ax.set(xlabel=None,
           ylabel=None, 
           xticklabels=[],
           yticklabels=[]
          )
    ax.legend_.set_title(None)
    plt.savefig('../plots-latest/Lozano-roc.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')


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


    bootstrapped_preds = pd.read_csv('bootstrapped-with-correction-results.csv', 
                                     index_col=0)


    bootstrapped_preds['labels']=bootstrapped_preds['Severe_irAE']
    bootstrapped_preds['predictions']=bootstrapped_preds['CompositeModelLOOCV']


    rocs = format_rocs( pd.concat( [ make_roc_df(bootstrapped_preds.loc[bootstrapped_preds.run_num==rr], 
                  'RLOOCV (auROC = {:.3f})'.format(
                      np.median( [ roc_auc_score(bootstrapped_preds.loc[bootstrapped_preds.run_num==rr].labels,
                             bootstrapped_preds.loc[bootstrapped_preds.run_num==rr].predictions
                             ) 
              for rr in bootstrapped_preds.run_num.unique() ] )
                       ), 
                  rr
                  )
                for rr in bootstrapped_preds.run_num.unique() 
               ]) ).drop('run', axis=1)
    
    
    bootstrapped_preds_baseline = pd.read_csv('bootstrapped-no-correction-results.csv', 
                                             index_col=0)
    bootstrapped_preds_baseline['labels'] =    bootstrapped_preds_baseline['Severe_irAE']
    bootstrapped_preds_baseline['predictions']=bootstrapped_preds_baseline['CompositeModelLOOCV']

    rocs_base = format_rocs( pd.concat( [ 
        make_roc_df(bootstrapped_preds_baseline.loc[bootstrapped_preds_baseline.run_num==rr], 
                  'LOOCV (auROC = {:.3f})'.format(
                      np.median( [ roc_auc_score(bootstrapped_preds_baseline.loc[bootstrapped_preds_baseline.run_num==rr].labels,
                             bootstrapped_preds_baseline.loc[bootstrapped_preds_baseline.run_num==rr].predictions
                             ) 
              for rr in bootstrapped_preds_baseline.run_num.unique() ] )
                       ), 
                  rr
                  )
                for rr in bootstrapped_preds_baseline.run_num.unique() 
               ]) ).drop('run', axis=1)


    rocs_all = pd.concat([
                      rocs_base, 
                      rocs 
                    ], axis=0
                ).reset_index(drop=True)

    
    
    n_tot=bootstrapped_preds.shape[0]/10 ## total # of patients, since there are 10 iters
    bootstrapped_preds['pat_num'] = bootstrapped_preds.index % n_tot
    bootstrapped_preds_baseline['pat_num'] = bootstrapped_preds_baseline.index % n_tot


    # bootstrapped_preds
    print( np.power(10, delong.delong_roc_test(
                bootstrapped_preds.groupby('pat_num')[['predictions','labels']].mean().labels, 
                bootstrapped_preds_baseline.groupby('pat_num')[['predictions','labels']].mean().predictions,
                bootstrapped_preds.groupby('pat_num')[['predictions','labels']].mean().predictions
            ) )
         )


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
    
#     plt.legend(loc='lower center')

    plt.title('Correcting for LOOCV distributional bias\n'+             'slightly improves prediction of ICI adverse outcomes\n'+             'from T Cell abundance and diversity')

    plt.savefig('../plots-latest/Lozano-roc-with-rloocv-bootstrap.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')
    
    x1['AMCD4']=x1['Activated_CD4_memory_T_CSx']
    x1.loc[x1['Activated_CD4_memory_T_CSx']>.1, 'AMCD4']= \
                x1.loc[x1['Activated_CD4_memory_T_CSx']<.1, 'Activated_CD4_memory_T_CSx'].max()

    X = StandardScaler().fit_transform(
                    x1[['AMCD4','TCR_clonotype_diversity_ShannonEntropy']]
                            )
    y = x1.Severe_irAE.values

    np.random.seed(seed)
    loo=LeaveOneOut()
    regs=np.logspace(-10, 6, 17)
    regs=np.logspace(-6, 6, 13)
    baseline_perfs = [ roc_auc_score(y, 
                                     [ LogisticRegression(C=reg_level, 
                                                         )\
                                        .fit(X[train_index], y[train_index])\
                                        .predict_proba(X[test_index]
                                                    )[:, 1][0]
                                  for train_index, test_index in loo.split(X, y) ]
                                    )
                        for reg_level in regs ]

    np.random.seed(seed)
    regs=np.logspace(-6, 6, 13)
    rloo=RebalancedLeaveOneOut()
    rloocv_perfs = [ roc_auc_score(y, 
                                     [ LogisticRegression(C=reg_level,
                                                         )\
                                        .fit(X[train_index], y[train_index])\
                                        .predict_proba(X[test_index]
                                                    )[:, 1][0]
                                  for train_index, test_index in rloo.split(X, 
                                                                            y, 
                                                                            seed=seed) ]
                                    )
                        for reg_level in regs ]


    pd.set_option('display.precision', 3)
    qqq=pd.DataFrame({'LOOCV':baseline_perfs, 
                  'RLOOCV':rloocv_perfs}, 
                   index=regs
                 ).T

    qqq.style.background_gradient(cmap ='coolwarm', 
                                           vmin=qqq.values.min(), 
                                          vmax=qqq.values.max(), 
                                          )\
                .set_properties(**{'font-size': '20px'})

    qqq.round(3).style.background_gradient(cmap ='coolwarm', 
                                           vmin=qqq.values.min(), 
                                          vmax=qqq.values.max(), 
                                          )\
                .set_properties(**{'font-size': '20px'}
                            ).to_excel('../plots-latest/ICI-regularization.xlsx')
    
    
    ### looking at hte combination of pvalues from all four delong tests
    print( combine_pvalues([0.12437051, 
                            0.04828611, 
                             0.03233037,
                             0.38976368
                            ]) )

    
if __name__=='__main__':
    main()


