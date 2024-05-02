#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('PhageIPSeq_CFS-main/')
sys.path.append('..')
import delong

import sklearn
from sklearn.utils import (
    _approximate_mode,
    _safe_indexing,
    check_random_state,
    indexable,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples, check_array, column_or_1d

class RebalancedLeaveOneOut(BaseCrossValidator):
    
    def split(self, X, y, groups=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        X, y, groups = indexable(X, y, groups)
        
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            
            ## drop one sample with a `y` different from the test index
            susbample_inds = train_index != train_index[ 
                    np.random.choice( np.where( y[train_index] 
                                                         != y[test_index][0])[0] ) ]
            train_index=train_index[susbample_inds]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= 1:
            raise ValueError(
                "Cannot perform RebalancedLeaveOneOut with n_samples={}.".format(n_samples)
            )
        return range(n_samples)

    def get_n_splits(self, X, y, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : object
            Needed to maintain class balance consistency.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _num_samples(X)

import os
import numpy as np
np.random.seed(1)
import pandas as pd
from sklearn.model_selection import cross_val_predict, LeaveOneOut

## loading the prediction code and data from the original analysis
from PhageIPSeq_CFS.config import predictions_outcome_dir, predictors_info
from PhageIPSeq_CFS.helpers import get_data_with_outcome, split_xy_df_and_filter_by_threshold

def main():

    np.random.seed(1)
    res = {}
    for estimator_name, estimator_info in predictors_info.items():
        estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
        x, y = split_xy_df_and_filter_by_threshold(
            get_data_with_outcome(with_oligos=False, 
                                  with_bloodtests=True, 
                                  imputed=False))
        
        res[estimator_name] = pd.Series(index=y.index,
                                        data=cross_val_predict(estimator, 
                                                               x, 
                                                               y,
                                                               cv=LeaveOneOut(),## run LOOCV
                                                               method='predict_proba'
                                                               )[:, 1])
        
        
        
    res = pd.DataFrame(res)


    
    np.random.seed(1)
    res2 = {}
    for estimator_name, estimator_info in predictors_info.items():
        estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
        x, y = split_xy_df_and_filter_by_threshold(
            get_data_with_outcome(with_oligos=False, 
                                  with_bloodtests=True, 
                                  imputed=False))
        res2[estimator_name] = pd.Series(index=y.index,
                                         data=cross_val_predict(estimator, 
                                                                x, 
                                                                y,
                                                                cv=RebalancedLeaveOneOut(), ## run RLOOCV
                                                                method='predict_proba')[:, 1])
        
    res2 = pd.DataFrame(res2)


    from sklearn.metrics import roc_auc_score

    roc_auc_score( y, res.xgboost )

    roc_auc_score( y, res2.xgboost )

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y, res.xgboost, drop_intermediate=False)
    fpr2, tpr2, _ = roc_curve(y, res2.xgboost, drop_intermediate=False)
    fpr3, tpr3, _ = roc_curve(y, res.GBR, drop_intermediate=False)
    fpr4, tpr4, _ = roc_curve(y, res2.GBR, drop_intermediate=False)

    rocplotdf_xgboost = pd.concat([
                    pd.DataFrame({'FPR':fpr, 
                                  'TPR':tpr, 
            'Group':'LOOCV (original; auROC = {:.3f})'.format(auc(fpr, tpr))}), 
                    pd.DataFrame({'FPR':fpr2, 
                                  'TPR':tpr2, 
            'Group':'RLOOCV (auROC = {:.3f})'.format(auc(fpr2, tpr2))}),
                    ], axis=0
            ).reset_index(drop=True)


    rocplotdf_gbr = pd.concat([ pd.DataFrame({'FPR':fpr3, 
                                  'TPR':tpr3, 
            'Group':'LOOCV (original; auROC = {:.3f})'.format(auc(fpr3, tpr3))}), 
                    pd.DataFrame({'FPR':fpr4, 
                                  'TPR':tpr4, 
            'Group':'RLOOCV (auROC = {:.3f})'.format(auc(fpr4, tpr4))}),
                                ], axis=0
            ).reset_index(drop=True)



    import seaborn as sns
    import matplotlib.pyplot as plt


    sns.set(rc={'axes.facecolor':'white', 
                'figure.facecolor':'white', 
                'axes.edgecolor':'black', 
                'grid.color': 'black'
                }, 
           font_scale=3)

#     plt.figure(figsize=(12,12))
#     ax = sns.lineplot(x='FPR', 
#                  y='TPR', 
#                  hue='Group',
#                  linewidth=5, 
#                  data=rocplotdf_gbr, 
#                  ci=0
#                  )
#     ax.legend_.set_title(None)

#     sns.set(rc={'axes.facecolor':'white', 
#                 'figure.facecolor':'white', 
#                 'axes.edgecolor':'black', 
#                 'grid.color': 'black'
#                 }, 
#            font_scale=3)

#     plt.figure(figsize=(12, 12))
#     ax = sns.lineplot(x='FPR', 
#                  y='TPR', 
#                  hue='Group',
#                  linewidth=5, 
#                  data=rocplotdf_xgboost, 
#                  ci=0
#                  )
#     ax.legend_.set_title(None)

#     sns.set(rc={'axes.facecolor':'white', 
#                 'figure.facecolor':'white', 
#                 'axes.edgecolor':'black', 
#                 'grid.color': 'black'
#                 }, 
#            font_scale=3)

    plt.figure(figsize=(12, 12))
    ax = sns.lineplot(x='FPR', 
                 y='TPR', 
                 hue='Group',
                 linewidth=5, 
                 data=rocplotdf_xgboost, 
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
    plt.savefig('../plots-latest/CFS-xgboost-roc.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')
#     plt.show()


    plt.figure(figsize=(12, 12))
    ax = sns.lineplot(x='FPR', 
                 y='TPR', 
                 hue='Group',
                 linewidth=5, 
                 data=rocplotdf_gbr, 
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
    plt.savefig('../plots-latest/CFS-gbr-roc.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')
#     plt.show()

    multiple_res_base=[]
#     np.random.seed(1)
    for ss in range(10):
        res = {}
        np.random.seed(ss)
        for estimator_name, estimator_info in predictors_info.items():
            estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
            x, y = split_xy_df_and_filter_by_threshold(
                get_data_with_outcome(with_oligos=False, with_bloodtests=True, imputed=False))
            res[estimator_name] = pd.Series(index=y.index,
                                            data=cross_val_predict(estimator, 
                                                                   x, 
                                                                   y,
                                                                   cv=LeaveOneOut(),## run LOOCV
                                                                   method='predict_proba')[:, 1])


        res = pd.DataFrame(res)
        multiple_res_base.append(res)

        print(roc_auc_score( y, res.xgboost ))
        print(roc_auc_score( y, res.GBR ))

    multiple_res=[]
#     np.random.seed(1)
    for ss in range(10):
        res2 = {}
        np.random.seed(ss)
        for estimator_name, estimator_info in predictors_info.items():
            estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
            x, y = split_xy_df_and_filter_by_threshold(
                                    get_data_with_outcome(with_oligos=False, 
                                                          with_bloodtests=True, 
                                                          imputed=False) )
            res2[estimator_name] = pd.Series(index=y.index,
                                            data=cross_val_predict(estimator, 
                                                             x, 
                                                             y,
                                                             cv=RebalancedLeaveOneOut(), ## run RLOOCV
                                                             method='predict_proba')[:, 1])



        res2 = pd.DataFrame(res2)
        multiple_res.append(res2)

        print(roc_auc_score( y, res2.xgboost ))
        print(roc_auc_score( y, res2.GBR ))


    ## make xgboost ci roc plot

    for i in range(len(multiple_res_base)):
        multiple_res_base[i]['labels']=y
        multiple_res_base[i]['predictions']=multiple_res_base[i]['xgboost']

    for i in range(len(multiple_res)):
        multiple_res[i]['labels']=y
        multiple_res[i]['predictions']=multiple_res[i]['xgboost']

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


    def add_origin(rocs):
        return( pd.concat([pd.DataFrame( [0, 0, rocs.Group.values[0], rocs.run.values[0]], 
                               index=rocs.columns 
                             ).T, rocs] ).reset_index(drop=True) )



    rocs = format_rocs( pd.concat([ 
                        make_roc_df( a,
                     'RLOOCV (median auROC: {:.3f})'.format(
                                     np.median([roc_auc_score(a.labels, a.predictions) 
                                                for a in multiple_res ] )
                     ), 
                     i
                     )
          for i,a in enumerate(multiple_res) ]
                          )
               )


    rocs_base = format_rocs( pd.concat([ 
                        make_roc_df( a,
                     'LOOCV (median auROC: {:.3f})'.format(
                                     np.median([roc_auc_score(a.labels, a.predictions) 
                                                for a in multiple_res_base ] )
                     ), 
                     i
                     )
          for i,a in enumerate(multiple_res_base) ]
                          )
               )



    rocs_all = pd.concat([
                            add_origin( rocs_base ), 
                            add_origin( rocs )
                         ], axis=0
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
                      ci=95, 
                      ax=ax
                      )
    ax.legend_.set_title(None)
    plt.legend(loc='lower center')

    plt.savefig('../plots-latest/CFS-xgboost-roc-with-bootstrap.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')


    for i in range(len(multiple_res_base)):
        multiple_res_base[i]['label']=y
        multiple_res_base[i]['predictions']=multiple_res_base[i]['GBR']

    for i in range(len(multiple_res)):
        multiple_res[i]['label']=y
        multiple_res[i]['predictions']=multiple_res[i]['GBR']

    rocs = format_rocs( pd.concat([ 
                        make_roc_df( a,
                     'RLOOCV (median auROC: {:.3f})'.format(
                                     np.median([roc_auc_score(a.labels, a.predictions) 
                                                for a in multiple_res ] )
                     ), 
                     i
                     )
          for i,a in enumerate(multiple_res) ]
                          )
               )


    rocs_base = format_rocs( pd.concat([ 
                        make_roc_df( a,
                     'LOOCV (median auROC: {:.3f})'.format(
                                     np.median([roc_auc_score(a.labels, a.predictions) 
                                                for a in multiple_res_base ] )
                     ), 
                     i
                     )
          for i,a in enumerate(multiple_res_base) ]
                          )
               )



    rocs_all = pd.concat([
                            add_origin( rocs_base ), 
                            add_origin( rocs )
                         ], axis=0
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
                      ci=95, 
                      ax=ax
                      )
    ax.legend_.set_title(None)
    plt.legend(loc='lower center')

    plt.savefig('../plots-latest/CFS-gbr-roc-with-bootstrapping.pdf', 
               format='pdf', 
               dpi=900, 
               bbox_inches='tight')
    
    
    ## print delong pvals
    
    print( np.power(10, 
             delong.delong_roc_test(
                    multiple_res[0].labels,
                    np.vstack([a['xgboost'].values
                               for a in multiple_res_base]
                               ).mean(axis=0) ,
                    np.vstack([a['xgboost'].values
                               for a in multiple_res]
                               ).mean(axis=0) ,
                     )[0] 
            ) )

    print( np.power(10, 
             delong.delong_roc_test(
                    multiple_res[0].labels,
                    np.vstack([a['GBR'].values
                               for a in multiple_res_base]
                               ).mean(axis=0) ,
                    np.vstack([a['GBR'].values
                               for a in multiple_res]
                               ).mean(axis=0) ,
                     )[0] 
            ) )
if __name__=='__main__':
    main()

