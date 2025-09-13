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
pd.set_option('display.float_format', '{:.3e}'.format)

def make_signal_var_plot(in_path='general_signal_variation.csv',
#                          in_path='general_signal_variation_test.csv',
                         pval_thresh = 0.01, 
                         plot_sig_ticks=False
                         ):
    
    default_setting_form = {'Signal noise':0.75,#4, 
                            'N samples':40, 
                            'N features':5,
                            }
    
    plot_df_main=pd.read_csv(in_path, index_col=0)
    

    for x_val in plot_df_main.columns[:3]:
        try:
            plot_df=plot_df_main.copy()
            for xvv in [a for a in default_setting_form if a != x_val]:
                plot_df=plot_df.loc[plot_df[xvv]==default_setting_form[xvv]]


            plt.figure(figsize=(8,8))
            ax=sns.boxplot(y='auROC', 
                           x=x_val,
                           hue='Group',
                        data=plot_df, 
                        fliersize=0
                        )
            handles, labels = ax.get_legend_handles_labels()
            plt.plot(ax.get_xlim(), [.5, .5], '--',
                     linewidth = 5, 
                     color='black'
                     )

            sns.swarmplot( y='auROC', 
                           x=x_val,
                           hue='Group',
                        data=plot_df, 
                        s=5, 
                          color='black', 
                          dodge=True,
                        ax=ax
                        )    

            sig_df = pd.DataFrame({a :\
        #                            ttest_rel(
                        wilcoxon(
                                 plot_df.loc[
                                     (plot_df[x_val]==a)&\
                                    (plot_df['Group']=='LOOCV')].auROC, 
                                 plot_df.loc[
                                     (plot_df[x_val]==a)&\
                                    (plot_df['Group']=='RLOOCV')].auROC
                                    ).pvalue
                            for a in plot_df[x_val].unique()
                              }, index=[0])

            sig_df=sig_df.T.reset_index()
            sig_df.columns=[x_val, 'pvalue']
            sig_df['is_sig'] = sig_df.pvalue < pval_thresh
            sig_df['y_val'] = 0.95

            print(sig_df)

            if (sum(sig_df.is_sig)>0) and plot_sig_ticks:
                sns.swarmplot(x=x_val,
                              y='y_val', 
                              data = sig_df.loc[sig_df.is_sig].reset_index(), 
                              marker='+',
                              size= 25/2, 
                              order=sig_df[x_val].unique(),
                              ax=ax,
                              edgecolor='red',
                              color='red',
                              linewidth=3.5/2
                              )

                sns.swarmplot(x=x_val, 
                              y='y_val', 
                              data = sig_df.loc[sig_df.is_sig].reset_index(), 
                              marker='x',
                              size= 18.5/2,
                              order=sig_df[x_val].unique(),
                              ax=ax,
                              edgecolor='red',
                              color='red',
                              linewidth=3.5/2,
                              )


            plt.legend().remove()
            plt.xticks(ticks=ax.get_xticks(), labels=[])
            plt.yticks(ticks=[0,.2,.4,.6,.8,1], labels=[])
            plt.ylim([0,1])
            plt.xlabel(None)
            plt.ylabel(None)


            plt.savefig('r2_plots/{}.pdf'.format(x_val), 
                        format='pdf', 
                        bbox_inches='tight', 
                        dpi=900)


            delta_plot_df = plot_df.loc[plot_df.Group=='RLOOCV']
            delta_plot_df['RLOOCV Gain'] = delta_plot_df.auROC.values - plot_df.loc[plot_df.Group!='RLOOCV'].auROC.values
            for hide_axes in [True, False]:

                plt.figure(figsize=(4,8))
                ax=sns.boxplot(y='RLOOCV Gain', 
                               x=x_val,
                               hue='Group',
                            data=delta_plot_df, 
                            fliersize=0
                            )

                sns.swarmplot( y='RLOOCV Gain', 
                               x=x_val,
                               hue='Group',
                            data=delta_plot_df, 
                            s=5, 
                              color='black', 
                              dodge=True,
                            ax=ax
                            )
                plt.ylim(delta_plot_df['RLOOCV Gain'].min()-0.01, 
                         delta_plot_df['RLOOCV Gain'].max()+0.01)

    #             plt.plot(ax.get_xlim(), [0,0], '--',
    #                      linewidth = 5, 
    #                      color='black'
    #                      )

                if hide_axes:
                    plt.legend().remove()
                    plt.xticks(ticks=ax.get_xticks(), labels=[])
                    plt.yticks(ticks=ax.get_yticks(), labels=[])
                    plt.xlabel(None)
                    plt.ylabel(None)
                else:
                    plt.legend().remove()
                    plt.xticks( ticks=ax.get_xticks() )
                    plt.yticks( ticks=ax.get_yticks() )

                plt.savefig('r2_plots/{}-delta-{}-axes.pdf'.format(x_val, 
                                                                   'without' if hide_axes else 'with' ), 
                            format='pdf', 
                            bbox_inches='tight', 
                            dpi=900
                            )

            print(plot_df.groupby([a for a in plot_df.columns if a!='auROC']).auROC.describe().round(2))
    
        except:
            pass

    return(None)


def make_n_classes_plot(in_path='n_classes_analysis.csv', 
                        out_path='r2_plots/n_classes.pdf',
                        pval_thresh = 0.01,
                        plot_sig_ticks=False
                        ):


    plot_df= pd.read_csv(in_path, index_col=0)
    x_val='N classes'

    plt.figure(figsize=(8,8))
    ax=sns.boxplot(y='auROC', 
                   x=x_val,
                   hue='Group',
                data=plot_df, 
                fliersize=0
                )
    handles, labels = ax.get_legend_handles_labels()
    plt.plot(ax.get_xlim(), [.5, .5], '--',
                 linewidth = 5, 
                 color='black'
                 )

    sns.swarmplot( y='auROC', 
                   x=x_val,
                   hue='Group',
                data=plot_df, 
                s=5, 
                  color='black', 
                  dodge=True,
                ax=ax
                )

    ax.legend(handles=handles[:2], labels=labels[:2])

    sig_df = pd.DataFrame({a :\
                wilcoxon(
                         plot_df.loc[
                             (plot_df[x_val]==a)&\
                            (plot_df['Group']=='LOOCV')].auROC, 
                         plot_df.loc[
                             (plot_df[x_val]==a)&\
                            (plot_df['Group']=='RLOOCV')].auROC
                            ).pvalue
                    for a in plot_df[x_val].unique()
                      }, index=[0])

    sig_df=sig_df.T.reset_index()
    sig_df.columns=[x_val, 'pvalue']
    sig_df['is_sig'] = sig_df.pvalue < pval_thresh
    sig_df['y_val'] = 0.95

    print(sig_df)

    if sum(sig_df.is_sig)>0 and plot_sig_ticks:
        sns.swarmplot(x=x_val,
                      y='y_val', 
                      data = sig_df.loc[sig_df.is_sig].reset_index(), 
                      marker='+',
                      size= 25/2, 
                      order=sig_df[x_val].unique(),
                      ax=ax,
                      edgecolor='red',
                      color='red',
                      linewidth=3.5/2
                      )

        sns.swarmplot(x=x_val, 
                      y='y_val', 
                      data = sig_df.loc[sig_df.is_sig].reset_index(), 
                      marker='x',
                      size= 18.5/2,
                      order=sig_df[x_val].unique(),
                      ax=ax,
                      color='red',
                      edgecolor='red',
                      linewidth=3.5/2,
                      )

    plt.legend().remove()
    plt.xticks(ticks=ax.get_xticks(), labels=[])
    plt.yticks(ticks=[0,.2,.4,.6,.8,1], labels=[])
    plt.ylim([0,1])
    plt.xlabel(None)
    plt.ylabel(None)


    plt.savefig('r2_plots/n_classes.pdf'.format(x_val), 
                format='pdf', 
                bbox_inches='tight', 
                dpi=900)
    
    delta_plot_df = plot_df.loc[plot_df.Group=='RLOOCV']
    delta_plot_df['RLOOCV Gain'] = delta_plot_df.auROC.values - plot_df.loc[plot_df.Group!='RLOOCV'].auROC.values

    for hide_axes in [True, False]:
        plt.figure(figsize=(4,8))
        ax=sns.boxplot(y='RLOOCV Gain', 
                       x=x_val,
                       hue='Group',
                    data=delta_plot_df, 
                    fliersize=0
                    )

        sns.swarmplot( y='RLOOCV Gain', 
                       x=x_val,
                       hue='Group',
                    data=delta_plot_df, 
                    s=5, 
                      color='black', 
                      dodge=True,
                    ax=ax
                    )  
        
        plt.ylim(delta_plot_df['RLOOCV Gain'].min()-0.01, 
                     delta_plot_df['RLOOCV Gain'].max()+0.01)


#         plt.plot(ax.get_xlim(), [0,0], '--',
#                  linewidth = 5, 
#                  color='black'
#                  )

        if hide_axes:
            plt.legend().remove()
            plt.xticks(ticks=ax.get_xticks(), labels=[])
            plt.yticks(ticks=ax.get_yticks(), labels=[])
        #         plt.yticks(ticks=[0,.2,.4,.6,.8,1], labels=[])
        #         plt.ylim([0,1])
            plt.xlabel(None)
            plt.ylabel(None)
        else:
            plt.legend().remove()
            plt.xticks( ticks=ax.get_xticks() )
            plt.yticks( ticks=ax.get_yticks() )
            

        plt.savefig('r2_plots/{}-delta-{}-axes.pdf'.format('N classes',  'without' if hide_axes else 'with' ), 
                    format='pdf', 
                    bbox_inches='tight', 
                    dpi=900
                    )
    
    print(plot_df.groupby([a for a in plot_df.columns if a!='auROC']).auROC.describe().round(2))
    
    return(None)


if __name__=='__main__':
    make_signal_var_plot(pval_thresh=0.05)
    make_n_classes_plot(pval_thresh=0.05)

