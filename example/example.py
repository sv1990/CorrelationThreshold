import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from correlation_threshold import CorrelationThreshold

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
sns.heatmap(pd.DataFrame(X2).corr(), cmap='seismic', vmin=-1, vmax=1, ax=ax[1])
plt.tight_layout()
plt.show()
