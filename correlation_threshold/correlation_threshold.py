import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import base, feature_selection


def _pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


class CorrelationThreshold(base.BaseEstimator, feature_selection.SelectorMixin):
    """
    Transformer that drops all features that have a high correlation to another feature.

    Inpired by this stackoverflow answer https://stackoverflow.com/a/49282823/4990485
    """

    def __init__(self, r_threshold=0.5, p_threshold=0.05):
        """
        Paramters:
        ----------
        `r_threshold`: Maximal correlation between features that is preserved. Has to be in [0, 1].
        `p_threshold`: Correlations are only considered to be significant if `p < p_threshold`
        """
        super().__init__()
        self.r_threshold = r_threshold
        self.p_threshold = p_threshold
        self.columns = None
        self.support_mask = None

    def fit(self, X, y=None):
        r_masked = self._get_masked_corr(X)
        p_masked = self._get_masked_corr(X, method=_pearsonr_pval)
        df_correlated = ((r_masked > self.r_threshold) &
                         (p_masked < self.p_threshold)).any()
        df_not_correlated = ~df_correlated
        self.columns = df_not_correlated.loc[df_not_correlated == True].index
        self.support_mask = np.array(
            [col in self.columns for col in df_not_correlated.index])
        return self

    def _get_masked_corr(self, X, method='pearson'):
        """
        Obtaines the correlation matrix using `method` 
        and masks it then to the upper triangle matrix
        """
        df_corr = pd.DataFrame(X).corr(method=method, min_periods=1)
        lower_triangle_matrix = np.tril(
            np.ones(shape=[len(df_corr)]*2, dtype=bool))
        corr_masked = df_corr.mask(lower_triangle_matrix).abs()
        return corr_masked

    def _get_support_mask(self):
        return self.support_mask

    def get_feature_names(self):
        return self.columns
