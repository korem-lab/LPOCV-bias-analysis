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

def main():

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

    ## run the delong tests across all the pvals from other rocs
    import sys
    sys.path.append('../')
    import delong

    print( np.power(10, delong.delong_roc_test(x1.Severe_irAE, 
                                        x1.CompositeModelLOOCV,
                                        x2.CompositeModelLOOCV
                                        ) )
         )



    from scipy.stats import combine_pvalues

    print( combine_pvalues([0.01457697, 
                     0.539665, 
                     0.24930627, 
                     0.12259377
                     ]
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


    bootstrapped_preds = pd.read_csv('bootstrapped-with-correction-results.csv', 
                                     index_col=0)


    bootstrapped_preds['labels']=bootstrapped_preds['Severe_irAE']
    bootstrapped_preds['predictions']=bootstrapped_preds['CompositeModelLOOCV']


    rocs = format_rocs( pd.concat( [ make_roc_df(bootstrapped_preds.loc[bootstrapped_preds.run_num==rr], 
                  'RLOOCV (median auROC = {:.3f})'.format(
                      np.median( [ roc_auc_score(bootstrapped_preds.loc[bootstrapped_preds.run_num==rr].labels,
                             bootstrapped_preds.loc[bootstrapped_preds.run_num==rr].predictions
                             ) 
              for rr in bootstrapped_preds.run_num.unique() ] )
                       ), 
                  rr
                  )
                for rr in bootstrapped_preds.run_num.unique() 
               ]) ).drop('run', axis=1)


    rocs_all = pd.concat([
                            pd.DataFrame({'FPR':fpr, 
                                          'TPR':tpr, 
                    'Group':'LOOCV (original; auROC = {:.3f})'.format(auc(fpr, tpr))}), 
                            rocs ], axis=0
                    ).reset_index(drop=True)


    pal = sns.color_palette()

    plt.figure(figsize=(12, 12))
    ax = sns.lineplot(x='FPR', 
                      y='TPR', 
                      hue='Group',
                      linewidth=5, 
                      data=rocs_all.loc[rocs_all.Group.str[0]=='R'], 
                      ci=95, 
                      palette={rocs_all.Group.values[-1]:pal.as_hex()[1]}
                      )

    ax = sns.lineplot(x='FPR', 
                      y='TPR', 
                      hue='Group',
                      linewidth=5, 
                      data=rocs_all.loc[rocs_all.Group.str[0]=='L'], 
                      ci=0, 
                      ax=ax
                      )
    ax.legend_.set_title(None)
    
    plt.legend(loc='lower center')

    plt.title('Correcting for LOOCV distributional bias\n'+             'slightly improves prediction of ICI adverse outcomes\n'+             'from T Cell abundance and diversity')

    plt.savefig('../plots-latest/Lozano-roc-with-rloocv-bootstrap.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')
    
    
    
if __name__=='__main__':
    main()


