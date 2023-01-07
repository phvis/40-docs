import paddle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from paddlets.datasets.repository import dataset_list, get_dataset
from paddlets.metrics import MSE, MAE
from paddlets.models.forecasting import RNNBlockRegressor, LSTNetRegressor, NBEATSModel, NHiTSModel, MLPRegressor, \
    TCNRegressor, TransformerModel, InformerModel

import warnings

# warnings.filterwarnings("ignore", category=Warning)
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_rows', 120)

dataset = get_dataset('UNI_WTH')
train_dataset, val_test_dataset = dataset.split(0.8)
val_dataset, test_dataset = val_test_dataset.split(0.5)
train_dataset.plot(add_data=[val_dataset, test_dataset], labels=['Val', 'Test'])
plt.show()

rnn = RNNBlockRegressor(in_chunk_len=24 * 7,
                    out_chunk_len=24,
                    max_epochs=1000,
                    batch_size=32,
                    patience=150,
                    optimizer_params=dict(learning_rate=5e-4),

                    )
rnn.fit(train_dataset,val_dataset)
subset_test_pred_dataset = rnn.predict(val_dataset)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()

mae = MAE()
print(mae(subset_test_dataset, subset_test_pred_dataset))
mse = MSE()
print(mse(subset_test_dataset, subset_test_pred_dataset))

from paddlets.utils import backtest
score , pred_data= backtest(
    data=val_test_dataset,
    model=rnn,
    start=0.5, #start 可以控制回测的起始点如果设置 start 为0.5,那么回测将会在数据的中间位置开始。
    predict_window=24, # predict_window 是每次预测的窗口长度
    stride=24, # stride 是两次连续预测之间的移动步长
    return_predicts = True, #如果设置 return_predicts 为True，回测函数会同时返回指标结果和预测值 。
    metric=mae
)
print(f"mae: {score}")
val_test_dataset.plot(add_data=pred_data,labels="backtest")
plt.show()

rnn.save("./tcn2", network_model=True, dygraph_to_static=True)


