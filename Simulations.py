#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, LeavePOut
from scipy.stats import ttest_1samp
import sys
from scipy.stats import zscore, wilcoxon
from debiasm.torch_functions import rescale
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


def main():

    y = np.random.permutation([1] * 50 + [0] * 50)
    y_pred = [-np.concatenate((y[:i], y[i+1:])).mean() for i in range(len(y))]
    out=roc_curve(y, y_pred)
    plt.figure(figsize=(8,8))
    plt.plot(
             out[0],
             out[1],
             linewidth=5, 
             color='red', 
                   )
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.title('ROC curve for LOOCV dummy negative-mean predictor\n' +           '$f(X_{test} | X_{train},y_{train}) = $'+          '$ - \overline{y_{train}}$')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('plots-latest/Fig1-dummy-anti-leakage-roc.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')


    y = np.random.permutation([1] * 50 + [0] * 50)
    y_pred = [-np.concatenate((y[:i], y[i+1:])).mean() for i in range(len(y))]
    out=precision_recall_curve(y, y_pred)
    plt.figure(figsize=(8,8))
    plt.plot(list(out[1]),
             list(out[0]),
             linewidth=5, 
             color='red', 
                   )
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('plots-latest/FigS1-dummy-anti-leakage-PR.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')


    # #### In practice, though, this will slighlty bias ML models in the opposite direction, since models tend to regress to the label mean; below we show how the direct 'label mean' prediction produces a perfectly bad ROC curve


    y = np.random.permutation([1] * 50 + [0] * 50)
    y_pred = [np.concatenate((y[:i], y[i+1:])).mean() for i in range(len(y))]
    out=roc_curve(y, y_pred)
    plt.figure(figsize=(8,8))
    plt.plot(
             out[0],
             out[1],
             linewidth=5, 
             color='blue', 
                   )
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.title('ROC curve for LOOCV dummy mean predictor\n' +           '$f(X_{test} | X_{train},y_{train}) = $'+          '$\overline{y_{train}}$')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('plots-latest/Fig1-dummy-leakage-roc.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')

    y = np.random.permutation([1] * 50 + [0] * 50)
    y_pred = [np.concatenate((y[:i], y[i+1:])).mean() for i in range(len(y))]
    out=precision_recall_curve(y, y_pred)
    plt.figure(figsize=(8,8))
    plt.plot(list(out[1]),
             list(out[0]),
             linewidth=5, 
             color='blue', 
                   )
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('plots-latest/FigS1-dummy-leakage-PR.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')


    # ## Distributional generalize to reassembly of dummy predictors across all validation Ns

    # #### while we can reproduce these perfect signals in leave-one-out cross validations, a wekaer yet similar leakage can be observed in any leave-P-out analysis in which the predictions across multiple folds are pooled together into a single ROC curve


    def flatten(l):
        return [item for sublist in l for item in sublist]

    def get_mean_anitleak_auroc(n=2, 
                                positive_frac=.5,
                                n_iters=100, 
                                smp_size=100
                                ):
        res = []
        n = int(n)
        n_rep_pos = int(smp_size*positive_frac)

        for iteration in range(n_iters):
            y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
            y_pred = flatten( [ [ -np.concatenate((y[:i*n], 
                                                   y[n*i+n:])
                                                   ).mean() ]*n for i in range(smp_size//n) ] )
            res.append(roc_auc_score(y, y_pred))
        return( np.mean(res) )

    def get_full_anitleak_aurocs(n=2, 
                                positive_frac=.5,
                                n_iters=100, 
                                smp_size=100
                                ):
        res = []
        n = int(n)
        n_rep_pos = int(smp_size*positive_frac)

        for iteration in range(n_iters):
            y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
            y_pred = flatten( [ [ -np.concatenate((y[:i*n], 
                                                   y[n*i+n:])
                                                   ).mean() ]*n for i in range(smp_size//n) ] )
            res.append(roc_auc_score(y, y_pred))
        return( res )


    n_its=100
    smp_s = 250
    class_ranges =  np.linspace(0,1,  11)[1:-1]
    n_left_outs = [ i for i in range(1, 250) if (n_its)%i==0 ]
    full_dummy_heatmap = [[ get_mean_anitleak_auroc(i, 
                                                     n_iters = n_its,
                                                     positive_frac = pos_frac,
                                                     smp_size = smp_s + smp_s%i 
                                                     ) for i in n_left_outs]
                                                for pos_frac in class_ranges]



    plt.figure(figsize=(12,9))
    sns.heatmap(full_dummy_heatmap[::-1], 
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                vmin=0,
                vmax=1,
                xticklabels=n_left_outs,
                yticklabels=[str(a)[:2] for a in class_ranges.round(2)[::-1]*100]
               )
    plt.title('auROCs for LPOCV dummy negative-mean predictors\n' +           '$f(X_{test} | X_{train},y_{train}) = $'+          '$- \overline{y_{train}}$')

    plt.xlabel('P left-out')
    plt.ylabel('Class balance (%)')
    plt.savefig('plots-latest/Fig1-dummy-leakage-heatmap.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')


    print( np.array(full_dummy_heatmap)[:, -1].mean(), 
           np.array(full_dummy_heatmap)[:, -1].min(), 
           np.array(full_dummy_heatmap)[:, -1].max() )


    print( ttest_1samp( np.array(full_dummy_heatmap)[:, -1], .5) )

    print( np.array(full_dummy_heatmap)[:, 2:4][np.array([0,-1])].mean(), 
           np.array(full_dummy_heatmap)[:, 2:4][np.array([0,-1])].min(), 
           np.array(full_dummy_heatmap)[:, 2:4][np.array([0,-1])].max() )


    print( np.array(full_dummy_heatmap)[:, 3][np.array([0])].mean(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([0])].min(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([0])].max() )


    print( np.array(full_dummy_heatmap)[:, 3][np.array([-1])].mean(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([-1])].min(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([-1])].max() )


    print( np.array(full_dummy_heatmap)[:, 3][np.array([4])].mean(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([4])].min(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([4])].max() )


    print( np.array(full_dummy_heatmap)[:, 3][np.array([0])].mean(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([0])].min(), 
           np.array(full_dummy_heatmap)[:, 3][np.array([0])].max() )


    print( np.array(full_dummy_heatmap)[:, 2:4][np.array([4])].mean(), 
           np.array(full_dummy_heatmap)[:, 2:4][np.array([4])].min(), 
           np.array(full_dummy_heatmap)[:, 2:4][np.array([4])].max() )


    p1 = get_full_anitleak_aurocs(n=4, positive_frac=0.5)
    p2 = get_full_anitleak_aurocs(n=4, positive_frac=0.1)
    p3 = get_full_anitleak_aurocs(n=4, positive_frac=0.9)


    nsmpls = len( p1 )
    plt.figure(figsize=(8,8))
    plt.yticks([0.7, 0.8, 0.9, 1])
    sns.boxplot( y = p2 + p1 +  p3, 
                 x = ['10%']*nsmpls + ['50%']*nsmpls + ['90%']*nsmpls,
                fliersize=0
               )

    sns.stripplot(y = p2 + p1 +  p3, 
                  x = ['10%']*nsmpls + ['50%']*nsmpls + ['90%']*nsmpls,
                  size=5, 
                  color='black'
                  )

    plt.xlabel('Class balance')
    plt.ylabel('auROC')
    plt.savefig('plots-latest/FigS1-dummy-leakage-4-left-out.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')

    print( np.mean(p3).round(2), np.max(p3).round(2), np.min(p3).round(2), np.std(p3).round(2) )

    print( np.mean(p2).round(2), np.max(p2).round(2), np.min(p2).round(2), np.std(p2).round(2) )

    print(  np.mean(p1).round(2), np.max(p1).round(2), np.min(p1).round(2), np.std(p1).round(2) )

    print( np.mean(p3) )

    print( np.mean( p2 ) )


    # ### Rebalanced-leave-one-out addresssed the issue highlighted in the dummy negative-mean predictor

    from sklearn.preprocessing import StandardScaler


    y = np.random.permutation([1] * 50 + [0] * 50)
    rloo=RebalancedLeaveOneOut()
    y_pred = [ - y[train_inds].mean() for train_inds, test_inds in rloo.split(y, y)]

    out=roc_curve(y, y_pred) 
    plt.figure(figsize=(8,8))
    plt.plot(
             out[0],
             out[1],
             linewidth=5, 
             color='red', 
                   )
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.title('Rebalancing addresses the\n'+            'LOOCV dummy reverse-mean predictor bias\n' +           '$f(X_{test} | X_{train},y_{train}) = $'+          '$- \overline{y_{train}}$')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.savefig('plots-latest/Fig1-dummy-rebalanced-roc.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf')


    # # Part 2 - LPOCV distribtuional bias impacts machine learning models on synthetic data of entirely random signals

    # ### Example on randomly generated data with no $y \sim X$ signals; where y are randomly generated numbers $\in \{0,1\}$ and $X$ is a dataset of 20 i.i.d features drawn from a uniform distribution $\in [0,1]$

    # #### The very short highlight: we can demonstrate even on these synthetic datasets that the underlying class balances shift the mean of the ML model's predictions, even though there is no real signal in the data, producing auROCs worse than 0.5

    pos_frac=.8
    smp_size=250
    n_features=20
    def plot_training_predictions(pos_frac=0.5, 
                                  model_format=LogisticRegression, 
                                  mod='Logistic regression'):
        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features)

        loo = LeaveOneOut()

        preds = [ model_format().fit(X[train_index], y[train_index])\
                 .predict_proba(X[test_index])[:, 1][0]
              for train_index, test_index in loo.split(X) ]

        sns.histplot(preds, 
                     binwidth=0.05, 
                    color='black')
        plt.title("{}'s training data predictions\n".format(mod) + \
                  'Class balance: {:.2f}'.format(pos_frac) )
        plt.xlim(0,1)
        plt.xlabel('Model predictions')
        return(preds)


    # #### Quick demonstration of regression to mean on Logistic regression


    all_ps = []
    class_ranges =  np.linspace(0,1,  11)[1:-1]
    for cr in class_ranges:
        all_ps.append( plot_training_predictions(cr) )


    plt.figure(figsize=(12,7))
    sns.violinplot(x='Class balance', 
                y='Prediction', 
                data =  pd.concat( [ pd.DataFrame({'Prediction':all_ps[i], 
                                                   'Class balance':round(class_ranges[i], 
                                                                         2)
                                                  })
                                           for i in range(len(all_ps)) ] )
               )
    plt.xticks(rotation=0.9)
    plt.savefig('plots-latest/FS2-logreg-boxplot-meanregress.pdf', 
                    format='pdf', 
                    bbox_inches='tight', 
                    dpi=900
                    )
    plt.ylim(-0.05, 1.05)


    # #### Similar trends are  also be observed in random forests

    class_ranges =  np.linspace(0,1,  11)[1:-1]
    all_ps = []
    for cr in class_ranges:
        all_ps.append( plot_training_predictions(cr, 
                                  model_format=RandomForestClassifier, 
                                  mod='Random Forest')
                     )


    print( pd.concat( [ pd.DataFrame({'Prediction':all_ps[i], 
                                                   'Class balance':round(class_ranges[i], 
                                                                         2)
                                                  })
                                           for i in range(len(all_ps)) ] ).Prediction.max() 
         )


    plt.figure(figsize=(12,7))
    sns.violinplot(x='Class balance', 
                   y='Prediction', 
                   data =  pd.concat( [ pd.DataFrame({'Prediction':all_ps[i], 
                                                   'Class balance':round(class_ranges[i], 
                                                                         2)
                                                  })
                                           for i in range(len(all_ps)) ] ),
                   cut=0
               )

    plt.savefig('plots-latest/FS2-RF-boxplot-meanregress.pdf', 
                    format='pdf', 
                    bbox_inches='tight', 
                    dpi=900
                    )


    class_ranges =  np.linspace(0,1,  11)[1:-1]
    all_ps = []
    for cr in class_ranges:
        all_ps.append( plot_training_predictions(cr, 
                                  model_format=KNeighborsClassifier, 
                                  mod='KNN')
                     )



    plt.figure(figsize=(12,7))
    sns.violinplot(x='Class balance', 
                   y='Prediction', 
                   data =  pd.concat( [ pd.DataFrame({'Prediction':all_ps[i], 
                                                      'Class balance':round(class_ranges[i],                                                                     2)  })
                                       for i in range(len(all_ps)) ] ), 
                   cut=0
                  )
    
    plt.savefig('plots-latest/FS2-KNN-boxplot-meanregress.pdf', 
                format='pdf', 
                bbox_inches='tight', 
                dpi=900
               )


    plt.figure(figsize=(12,7))
    sns.swarmplot( x='Class balance', 
                  y='Prediction', 
                  data =  pd.concat( [ pd.DataFrame({'Prediction':all_ps[i], 
                                                     'Class balance':round(class_ranges[i], 
                                                                           2)
                                                    })
                                      for i in range(len(all_ps)) ] ),
                  size=5,
                  color='k', 
                  #               alpha=0.05
                  #                cut=0
                 )
    
    plt.savefig('plots-latest/FS2-KNN-boxplot-meanregress.pdf', 
                format='pdf', 
                bbox_inches='tight',
                dpi=900
                )


    # ### Through the regression to the mean, we see the same anti-leakage class-balance impact ML performance on leave-one-out CV

    def flatten(l):
        return [item for sublist in l for item in sublist]
    def simulate_loo_aurocs(pos_frac=.8,
                            smp_size=250,
                            n_features=20,
                            reg_level=0.01
                            ):

        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features) 
        loo = LeaveOneOut()
        vals = [ LogisticRegression(C=reg_level)\
                        .fit(X[train_index], y[train_index])\
                        .predict_proba(X[test_index])[:, 1][0]
                  for train_index, test_index in loo.split(X) ]
        return(roc_auc_score(y,vals))

    roc_sims = [[ simulate_loo_aurocs(pos_frac = pf) for i in range(10) ]
                for pf in class_ranges ]



    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(roc_sims))] ), 
                y = flatten( roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(roc_sims))] ), 
                y = flatten( roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')
    plt.title('LOOCV predictors observing random data\n'  +          'Consistently Performs worse than a random guess\n' +           '1-sample T test p = {:.3f}'.format(ttest_1samp(flatten( roc_sims ), .5).pvalue))

    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(roc_sims))] ), 
                y = flatten( roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(roc_sims))] ), 
                y = flatten( roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')
    plt.savefig('plots-latest/Fig2-no-correction-boxplots.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf'
               )

    print( pd.Series(flatten(roc_sims)).describe() )


    # ### Next, we show the same analysis as above, but repeated aceross different regularization levels; we note that less regularization means we allow for more noise in the system, which does mask these signals. However, the overall auROCs are stil significantly below 0.5


    roc_sims = [ [[ simulate_loo_aurocs(pos_frac = pf,
                                        reg_level=reg_level_C, 
                                        smp_size=250
                                       ) for i in range(10) ]
                  for pf in class_ranges ] 
                for reg_level_C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e10] ]


    plt.figure(figsize=(12,9))
    sns.heatmap(np.array( [ [np.mean(b) for b in a] for a in roc_sims] ).T[::-1] , 
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                vmin=0,
                vmax=1,
                xticklabels=[ '$10^{' + str(int(a)) + '}$'
                                 for a in np.log10( [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e10] ) ],
                yticklabels=[str(a)[:2] for a in class_ranges.round(2)[::-1]*100]
               )
    plt.title('auROCs of logistic regression models on simulated random data. \n' +          'Showing the mean anti-leakge produced across 10 runs')

    plt.xlabel("Regularization (per scikit learn's `C` parameter)\nLower C indicates more regularization")
    plt.ylabel('Class balance (%)')


    print(np.mean(flatten(roc_sims[0])))
    print(np.mean(flatten(roc_sims[-1])))


    plt.figure(figsize=(12,9))
    sns.heatmap(np.array( [ [np.mean(b) for b in a] for a in roc_sims] )[::-1].T[::-1], 
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                vmin=0,
                vmax=1,
                xticklabels=[ '$10^{' + str(int(a)) + '}$'
                                 for a in np.log10( [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e10] )[::-1] ],
                yticklabels=[str(a)[:2] for a in class_ranges.round(2)[::-1]*100]
               )

    plt.xlabel("Regularization (per scikit learn's `C` parameter)\nLower C indicates more regularization")
    plt.ylabel('Class balance (%)')

    plt.savefig('plots-latest/Fig2-no-correction-heatmap.pdf', 
                format='pdf', 
                dpi=900, 
                bbox_inches='tight')

    def simulate_rf_loo_aurocs(pos_frac=.8,
                            smp_size=100,##smaller smp_size to bring down runtime
                            n_features=20,
                            ):

        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features) 
        loo = LeaveOneOut()
        vals = [ RandomForestClassifier(
                                        )\
                        .fit(X[train_index], y[train_index])\
                        .predict_proba(X[test_index])[:, 1][0]
                  for train_index, test_index in loo.split(X) ]
        return(roc_auc_score(y,vals))

    
    np.random.seed(0)
    rf_roc_sims = [[ simulate_rf_loo_aurocs(pos_frac = pf) for i in range(10) ]
                   for pf in class_ranges ]

    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(rf_roc_sims))] ), 
                y = flatten( rf_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(rf_roc_sims))]), 
                y = flatten( rf_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')
    plt.title('LOOCV Random forest predictors observing random data\n'  +          'Consistently Performs worse than a random guess\n' +           '1-sample T test p = {:.3f}'.format(ttest_1samp(flatten( rf_roc_sims ), .5).pvalue))
    plt.savefig('plots-latest/Fig-S1-RandomForest-simulations.pdf', 
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )

    print( pd.Series(flatten( rf_roc_sims )).describe() )


    def simulate_knn_loo_aurocs(pos_frac=.8,
                               smp_size=250,##smaller smp_size to bring down runtime
                               n_features=20,
                               ):

        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features) 
        loo = LeaveOneOut()
        vals = [ KNeighborsClassifier()                    .fit(X[train_index], y[train_index])                    .predict_proba(X[test_index])[:, 1][0]
                  for train_index, test_index in loo.split(X) ]
        return(roc_auc_score(y,vals))


    np.random.seed(0)
    knn_roc_sims = [[ simulate_knn_loo_aurocs(pos_frac = pf) for i in range(10) ]
                    for pf in class_ranges ]


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(knn_roc_sims))] ), 
                y = flatten( knn_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(knn_roc_sims))] ), 
                y = flatten( knn_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')
    plt.title('LOOCV KNN predictors observing random data\n'  +          'Consistently Performs worse than a random guess\n' +           '1-sample T test p = {:.3e}'.format(ttest_1samp(flatten( knn_roc_sims ), .5).pvalue))
    plt.savefig('plots-latest/Fig-S3-KNN-N5-simulations.pdf', 
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )
    
    print( pd.Series(flatten( knn_roc_sims )).describe() )


    def flatten(l):
        return [item for sublist in l for item in sublist]

    from scipy.stats import zscore

    def simulate_loo_zscore_aurocs(pos_frac=.8,
                                    smp_size=250,
                                    n_features=20,
                                    reg_level=0.01
                                   ):

        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features) 
        loo = LeaveOneOut()

        vals = [ zscore( 
                    LogisticRegression(C=reg_level,
                                       )\
                                .fit(X[train_index], y[train_index])\
                                .predict_proba(X)[:, 1]
                                )[test_index]
                          for train_index, test_index in loo.split(X) ]
        return(roc_auc_score(y,vals))

    def simulate_loo_zscore_aurocsNN(pos_frac=.8,
                                    smp_size=250,
                                    n_features=20,
                                    reg_level=0.01
                                   ):

        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features) 
        loo = LeaveOneOut()

        vals = [ zscore( 
                    KNeighborsClassifier(n_neighbors=1
                                       )\
                                .fit(X[train_index], y[train_index])\
                                .predict_proba(X)[:, 1]
                                )[test_index]
                          for train_index, test_index in loo.split(X) ]
        return(roc_auc_score(y,vals))


    np.random.seed(0)
    zscore_roc_sims = [[ simulate_loo_zscore_aurocs(pos_frac = pf) for i in range(10) ]
                       for pf in class_ranges ]


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims))] ), 
                y = flatten( zscore_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims))] ), 
                y = flatten( zscore_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')
    plt.savefig('plots-latest/Fig-Addendum-zscore-LogisticRegression.pdf', 
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims))] ), 
                y = flatten( zscore_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims))] ), 
                y = flatten( zscore_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')
    plt.title('LOOCV predictors with zscoring observing random data\n'  +          'Consistently performs similarly to a random guess\n' +           '1-sample T test vs "0.5" p = {:.3f}'.format(ttest_1samp(flatten( zscore_roc_sims ), 
                                                                       .5).pvalue))


    zscore_roc_sims_nn = [[ simulate_loo_zscore_aurocsNN(pos_frac = pf) for i in range(10) ]
                          for pf in class_ranges ]


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims_nn))] ), 
                y = flatten( zscore_roc_sims_nn ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims_nn))] ), 
                y = flatten( zscore_roc_sims_nn ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')
    plt.savefig('plots-latest/Fig-Addendum-zscore-NN-leakage.pdf', 
                format='pdf',
                dpi=900, 
                bbox_inches='tight'
                )



    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims_nn))] ), 
                y = flatten( zscore_roc_sims_nn ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(zscore_roc_sims_nn))] ), 
                y = flatten( zscore_roc_sims_nn ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LOOCV auROC')

    plt.title('LOOCV Nearest neighbor predictors with zscoring random data\n'  +          'Consistently outperforms a random guess\n' +           '1-sample T test vs "0.5" p = {:.3f}'.format(ttest_1samp(flatten( zscore_roc_sims_nn ), .5).pvalue))


    print(  np.quantile(np.array( [ [np.mean(b) for b in a] for a in zscore_roc_sims] ).T,  
                np.array([.25, .5, .75])
                ) )


    # ### Additionally, we demonstrate that stratifying the P left out can sufficiently address this issue, while increasing the N left out to capture the correct level of class-balance. For our simulation's construction, we note that a stratified leave-10-out can capture all our cases


    def simulate_rebalanced_loo_aurocs(pos_frac=.8,
                            smp_size=250,
                            n_features=20,
                            reg_level=0.01
                            ):

        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features) 
        rloo = RebalancedLeaveOneOut()
        vals = [ LogisticRegression(C=reg_level,
                                    )\
                        .fit(X[train_index], y[train_index])\
                        .predict_proba(X[test_index])[:, 1][0]
                  for train_index, test_index in rloo.split(X, y) ]
        return(roc_auc_score(y,vals))


    np.random.seed(0)
    rebalanced_roc_sims = [[ simulate_rebalanced_loo_aurocs(pos_frac = pf) for i in range(10) ]
                           for pf in class_ranges ]


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(rebalanced_roc_sims))] ), 
                y = flatten( rebalanced_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(rebalanced_roc_sims))] ), 
                y = flatten( rebalanced_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('RLOOCV auROC')
    plt.title('RLOOCV predictors with rebalancing observing random data\n'  +          'Consistently performs similarly to a random guess\n' +           '1-sample T test vs "0.5" p = {:.3f}'.format(ttest_1samp(flatten( rebalanced_roc_sims ), 
                                                                       .5).pvalue))


    print(  pd.Series(flatten( rebalanced_roc_sims )).describe() )
    print(ttest_1samp( flatten( rebalanced_roc_sims ), 0.5 ) )


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(rebalanced_roc_sims))] ), 
                y = flatten( rebalanced_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(rebalanced_roc_sims))] ), 
                y = flatten( rebalanced_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('RLOOCV auROC')

    plt.savefig('plots-latest/Fig2-rebalanced-boxplot.pdf', 
                dpi=900, 
                format='pdf', 
                bbox_inches='tight')


    rebalanced_roc_sims = [ [[ simulate_rebalanced_loo_aurocs(pos_frac = pf,
                                                              reg_level=reg_level_C, 
                                                              smp_size=250
                                                             ) for i in range(10) ]
                             for pf in class_ranges ] 
                           for reg_level_C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e10] ]


    plt.figure(figsize=(12,9))
    sns.heatmap(np.array( [ [np.mean(b) for b in a] for a in rebalanced_roc_sims] )[::-1].T[::-1], 
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                vmin=0,
                vmax=1,
                xticklabels=[ '$10^{' + str(int(a)) + '}$'
                                 for a in np.log10( [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e10] )[::-1] ],
                yticklabels=[str(a)[:2] for a in class_ranges.round(2)[::-1]*100]
               )

    plt.xlabel("Regularization (per scikit learn's `C` parameter)\nLower C indicates more regularization")
    plt.ylabel('Class balance (%)')

    plt.savefig('plots-latest/Fig2-rebalanced-correction-heatmap.pdf', 
                format='pdf', 
                dpi=900, 
                bbox_inches='tight')
    
    ##FIIND
    print('RLOOCV HEAT')
    from scipy.stats import combine_pvalues
    print(combine_pvalues( [ ttest_1samp(c, 0.5).pvalue
            for c in  np.array( [ [np.mean(b) for b in a] for a in rebalanced_roc_sims] 
                                  )[::-1].T[::-1].T ] ) )


    from sklearn.model_selection import StratifiedKFold

    def simulate_stratified_lno_aurocs(pos_frac=.8,
                                       smp_size=250,
                                       n_features=20,
                                       reg_level=0.01, 
                                       lno_n=10
                                       ):
        if lno_n==1:
            return( simulate_loo_aurocs(pos_frac = pos_frac, 
                                        smp_size = smp_size, 
                                        n_features = n_features, 
                                        reg_level = reg_level) 
                   )

        smp_size += smp_size%lno_n ### to make sure we can get the right split stratifications
        n_rep_pos = int( pos_frac*smp_size )
        y = np.random.permutation([1] * n_rep_pos + [0] * (smp_size-n_rep_pos) )
        X = np.random.rand(smp_size, n_features) 
        skf = StratifiedKFold(n_splits=smp_size//lno_n, 
                              shuffle=True)
        vals = [ ( LogisticRegression(C=reg_level
                                    )\
                        .fit(X[train_index], y[train_index])\
                        .predict_proba(X[test_index])[:, 1], 
                  y[test_index] )
                  for train_index, test_index in skf.split(X, y) ]

        return(roc_auc_score(np.hstack([a[1] for a in vals]),
                             np.hstack([a[0] for a in vals]) ) )


    # ### doing a stratified leave-10-out solves leakage in all our specific class-balances

    stratified_roc_sims = [[ simulate_stratified_lno_aurocs(pos_frac = pf, 
                                                            lno_n=10) for i in range(10) ]
                           for pf in class_ranges ]


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('L-Ten-OCV auROC')


    # ### Doing a stratified leave-2-out solves the 50% class-balance, although some leakage remains in other cases

    np.random.seed(0)
    stratified_roc_sims = [[ simulate_stratified_lno_aurocs(pos_frac = pf,                                                       lno_n=2) for i in range(10) ]
                           for pf in class_ranges ]

    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LPOCV auROC')


    print(  pd.Series( flatten( [stratified_roc_sims[i] 
                         for i in range(len(stratified_roc_sims)) if i==4] 
               ) ).describe() )


    # ### Doing a stratified leave-5-out solves leakage in all the specific class-balances of 0.2, 0.4. 0.6, and 0.8, which can be perfectly represented with 5 samples


    np.random.seed(123)
    stratified_roc_sims = [[ simulate_stratified_lno_aurocs(pos_frac = pf, 
                                                            lno_n=5, 
                                                            reg_level=1e-3)
                                                    for i in range(10) ]
                           for pf in class_ranges ]


    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LPOCV auROC (P=5)')

    plt.savefig('plots-latest/Fig2-stratified-leave-p-out-boxplot.pdf', 
                format='pdf', 
                dpi=900, 
                bbox_inches='tight')



    print(  np.quantile(np.array([ a.mean()
                            for a in np.array(stratified_roc_sims)\
                                  [np.array([1,3,5,7])] ]),  
                np.array([.25, .5, .75])
                ) )



    plt.figure(figsize=(12,7))
    sns.boxplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
               fliersize=0)
    sns.swarmplot(x = flatten( [[round(class_ranges[i], 1)]*10 for i in range(len(stratified_roc_sims))] ), 
                y = flatten( stratified_roc_sims ), 
                 color='black', 
                 s=10)
    plt.plot([-.5, 8.5], [.5, .5], '--',
             linewidth = 5, 
             color='black'
             )

    plt.xlabel('Class balance')
    plt.ylabel('LNO-CV auROC')



    print( pd.Series( flatten( [stratified_roc_sims[i] for i in range(len(stratified_roc_sims)) if i%2==1] 
               ) ).describe() )

    print('LPOCV EVEN')
    print(  ttest_1samp(flatten( [stratified_roc_sims[i] for i in range(len(stratified_roc_sims)) if i%2==1] 
                       ), 0.5) )
    
    print('LPOCV ODD')
    print(  ttest_1samp(flatten( [stratified_roc_sims[i] for i in range(len(stratified_roc_sims)) if i%2==0] 
                       ), 0.5) )



    # ### the general LPO anti-leakage heatmap trends demonstrate specific class-balance-dependent solutions

    stratified_roc_sims = [ [[ simulate_stratified_lno_aurocs(pos_frac = pf, 
                                                              lno_n=lno_n_, 
                                                              reg_level=1e-3
                                                             ) 
                              for i in range(10) ]
                             for pf in class_ranges ] 
                           for lno_n_ in range(1,11) ]


    plt.figure(figsize=(12,9))
    sns.heatmap(np.array( [ [np.mean(b) for b in a] for a in stratified_roc_sims] ).T[::-1], 
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                vmin=0,
                vmax=1,
                xticklabels=range(1,11),
                yticklabels=[str(a)[:2] for a in class_ranges.round(2)[::-1]*100]
               )
    plt.title('auROCs of stratified LPO on logistic regression models with simulated random data. \n' +          'This approach solves anti-leakage when all the right Ns and class balances align')

    plt.xlabel("P left-out")
    plt.ylabel('Class balance (%)')


    plt.figure(figsize=(12,9))
    tmp=np.array( [ [np.mean(b) for b in a] for a in stratified_roc_sims] ).T
    sns.heatmap(tmp[::-1], 
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                vmin=0,
                vmax=1,
                xticklabels=range(1,11),
                yticklabels=[str(a)[:2] for a in class_ranges.round(2)[::-1]*100]
               )

    plt.xlabel("P left-out")
    plt.ylabel('Class balance (%)')
    plt.savefig('plots-latest/Fig2-stratified-leave-p-out-heatmap.pdf', 
                format='pdf', 
                dpi=900, 
                bbox_inches='tight')
    
    return(None)



if __name__=='__main__':
    main()

