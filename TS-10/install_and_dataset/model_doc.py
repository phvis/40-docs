import pandas as pd
import numpy as np
from paddlets import TSDataset

x = np.linspace(-np.pi, np.pi, 200)
sinx = np.sin(x) * 4 + np.random.randn(200)

df = pd.DataFrame(
    {
        'time_col': pd.date_range('2022-01-01', periods=200, freq='1h'),
        'value': sinx
    }
)
target_dataset = TSDataset.load_from_dataframe(
    df,  # Also can be path to the CSV file
    time_col='time_col',
    target_cols='value',
    freq='1h'
)
target_dataset.plot()