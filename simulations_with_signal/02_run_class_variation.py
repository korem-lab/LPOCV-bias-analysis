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

from sklearn.utils import indexable, check_random_state
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples, check_array, column_or_1d
import numpy as np

import numbers
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import type_of_target
from abc import ABCMeta, abstractmethod
from itertools import chain, combinations

def flatten(xss):
    return [x for xs in xss for x in xs]

class MulticlassRebalancedLeaveOneOut(BaseCrossValidator):
    """Rebalanced Leave-One-Out cross-validator

    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples are used to form the training set, 
    with subsampling to ensure consistent class balances across all splits.

    This class is designed to have the same functionality and 
    implementation structure as scikit-learn's ``LeaveOneOut()``
    
    At least two observations per class are needed for `RebalancedLeaveOneOut`

    Examples
    --------
    >>> import numpy as np
    >>> from rebalancedcv import RebalancedLeaveOneOut
    >>> X = np.array([[1, 2, 1, 2], [3, 4, 3, 4]]).T
    >>> y = np.array([1, 2, 1, 2])
    >>> rloo = RebalancedLeaveOneOut()
    >>> for i, (train_index, test_index) in enumerate(rloo.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
        Fold 0:
          Train: index=[2 3]
          Test:  index=[0]
        Fold 1:
          Train: index=[0 3]
          Test:  index=[1]
        Fold 2:
          Train: index=[0 1]
          Test:  index=[2]
        Fold 3:
          Train: index=[0 1]
          Test:  index=[3]
    """
    
    def split(self, X, y, groups=None, seed=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            
        seed : to enforce consistency in the subsampling

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        
        if seed is not None:
            np.random.seed(seed)
            
        X, y, groups = indexable(X, y, groups)
        self.unique_y = np.unique(y)
        
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            
            for to_drop_val in self.unique_y[self.unique_y!=y[test_index][0]]:
                ## drop one sample with a `y` different from the test index
                subsample_inds = train_index != train_index[ 
                        np.random.choice( np.where( y[train_index] 
                                                             == to_drop_val)[0] ) 
                                                ]
                train_index=train_index[subsample_inds]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= 1:
            raise ValueError(
                "Cannot perform LeaveOneOut with n_samples={}.".format(n_samples)
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
            Needed to maintin class balance consistency.

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



def run_analysis(
            n_samples=50,
            p=0.5,
            n_features=10,
            penalty='l2',
            noise_strength = 3,
            weight_std=1,
            noise_std=1, 
            seed=1, 
            max_depth=5,
            n_estimators=100, 
            n_classes=3
            ):
#     np.random.seed(seed)

#     model=LogisticRegressionCV(solver='newton-cg', max_iter=1000)
#     ## set up data
#     X=np.random.normal(size=(n_samples, n_features))
    
#     ## need minimum 6 samples in each class to run LogisticRegressionCV nested 5-fold
#     done=False
#     counter=0
#     while not done:

#         true_weights = np.random.normal(scale=weight_std,
#                                         size=(n_features, n_classes)
#                                         )

#         noise=np.random.normal(scale=noise_std, 
#                                size=(n_samples, n_classes))


#         y_float = ( X@true_weights )+noise
#         y=y_float.argmax(axis=1)
        
#         if pd.Series(y).value_counts().min()>=6:
#             done=True
            
#         counter+=1
#         if counter>100:
#             raise(ValueError('Could not set up y labels'))

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

        noise=np.random.normal(scale=noise_std,#*4, 
                               size=(n_samples, n_classes)
                               )


        y_float = ( X@true_weights )+noise
        y=y_float.argmax(axis=1)
        
        if pd.Series(y).value_counts().min()>=6:
            done=True
            
        counter+=1
        if counter>100:
            raise(ValueError('Could not set up y labels'))
    
    if n_classes > 2:
        loocv_auroc = roc_auc_score(y, 
                        cross_val_predict(model, 
                                          X=X, 
                                          y=y,
                                          cv=LeaveOneOut(), 
                                          method='predict_proba'), 
                                     multi_class='ovo'
                         )


        rloocv_auroc = \
            roc_auc_score(y, 
                        cross_val_predict(model, 
                                          X=X, 
                                          y=y,
                                          cv=MulticlassRebalancedLeaveOneOut(), 
                                          method='predict_proba'), 
                          multi_class='ovo'
                         )
    else:
        loocv_auroc = roc_auc_score(y, 
                        cross_val_predict(model, 
                                          X=X, 
                                          y=y,
                                          cv=LeaveOneOut(), 
                                          method='predict_proba')[:, 1], 
                         )


        rloocv_auroc = \
            roc_auc_score(y, 
                        cross_val_predict(model, 
                                          X=X, 
                                          y=y,
                                          cv=MulticlassRebalancedLeaveOneOut(), 
                                          method='predict_proba')[:, 1], 
                         )
    return(loocv_auroc, rloocv_auroc)




def main(out_csv_path = 'n_classes_analysis.csv'):
    loocv_aurocs=[]
    rloocv_aurocs=[]
    noise_stds=[]
    n_samples=[]
    all_nfeats=[]
    all_nclasses=[]

    for noise_std in [.75]:
        for n_sample in [40]:
            for nfeats in [5]:
                  for nclasses in [2,3,4,5]:#,6]:
                    for seed in range(25):

                        loocv_auroc, rloocv_auroc = \
                            run_analysis(noise_std=noise_std, 
                                         n_samples=n_sample, 
                                         n_features=nfeats,
                                         n_classes=nclasses, 
                                         seed=seed
                                         )

                        loocv_aurocs.append(loocv_auroc)
                        rloocv_aurocs.append(rloocv_auroc)
                        noise_stds.append(noise_std)
                        n_samples.append(n_sample)
                        all_nfeats.append(nfeats)
                        all_nclasses.append(nclasses)


                        plot_df= pd.DataFrame({'Signal noise':noise_stds*2, 
                                                'N samples':n_samples*2, 
                                                'N features':all_nfeats*2,
                                                'auROC':loocv_aurocs + rloocv_aurocs, 
                                                'Group':['LOOCV']*len(loocv_aurocs) + \
                                                        ['RLOOCV']*len(rloocv_aurocs), 
                                                'N classes':all_nclasses*2
                                               })

                        plot_df.to_csv(out_csv_path)
    
    
    
if __name__=='__main__':
    main()












