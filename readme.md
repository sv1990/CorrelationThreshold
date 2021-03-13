# CorrelationThreshold

## Usage

```python
import pandas as pd
from correlation_threshold import CorrelationThreshold

X = pd.DataFrame(...)

ct = CorrelationThreshold(r_threshold=0.5, p_threshold=0.05)

X2 = ct.fit_transform(X)
```

