import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import base


def _pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


class CorrelationThreshold(base.BaseEstimator, base.TransformerMixin):
    """
    Transformer that drops all features that have a high correlation to another feature.

    Inpired by this stackoverflow answer https://stackoverflow.com/a/49282823/4990485
    """

    # TODO:
    # * Use feature_selection.SelectorMixin

    def __init__(self, r_threshold=0.5, p_threshold=0.05):
        """
        Paramters:
        ----------
        `r_threshold`: Maximal correlation between features that is preserved. Has to be in [0, 1].
        `p_threshold`: Corrleation are only considered to be significant if `p < p_threshold`
        """
        super().__init__()
        self.r_threshold = r_threshold
        self.p_threshold = p_threshold
        self.un_corr_idx = None

    def fit(self, X, y=None):
        r_masked = self._get_masked_corr(X)
        p_masked = self._get_masked_corr(X, method=_pearsonr_pval)
        df_correlated = ((r_masked > self.r_threshold) &
                         (p_masked < self.p_threshold)).any()
        df_not_correlated = ~df_correlated
        self.un_corr_idx = df_not_correlated.loc[df_not_correlated == True].index
        return self

    def _get_masked_corr(self, X, method='pearson'):
        df_corr = pd.DataFrame(X).corr(method=method, min_periods=1)
        lower_triangle_matrix = np.tril(
            np.ones(shape=[len(df_corr)]*2, dtype=bool))
        corr_masked = df_corr.mask(lower_triangle_matrix).abs()
        return corr_masked

    def transform(self, X):
        if hasattr(X, 'loc'):
            return X.loc[:, self.un_corr_idx]
        else:
            return X[:, self.un_corr_idx]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    a = np.random.normal(size=100)
    b = np.random.normal(size=100)
    c = b + np.random.normal(size=100)
    d = -a + np.random.normal(size=100)
    e = np.random.normal(size=100)
    X = pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e))

    X2 = CorrelationThreshold().fit_transform(X)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), dpi=100)
    fig.suptitle('Correlation Matrices', fontsize=16)
    ax[0].set_title('Original Data')
    ax[1].set_title('After eliminating correlated features')
    sns.heatmap(X.corr(), cmap='seismic', vmin=-1, vmax=1, ax=ax[0])
    sns.heatmap(X2.corr(), cmap='seismic', vmin=-1, vmax=1, ax=ax[1])
    plt.tight_layout()
    plt.show()
