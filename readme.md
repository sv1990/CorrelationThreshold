# CorrelationThreshold

## Usage

```python
import pandas as pd
from correlation_threshold import CorrelationThreshold

X = pd.DataFrame(...)

ct = CorrelationThreshold(r_threshold=0.5, p_threshold=0.05)

X2 = ct.fit_transform(X)
```

## Example

![Example heatmaps](https://github.com/sv1990/CorrelationThreshold/blob/main/media/example.png?raw=true)
