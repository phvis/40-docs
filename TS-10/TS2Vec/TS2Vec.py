import paddle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from paddlets.models import load

from paddlets.models.forecasting import DeepARModel
from paddlets.datasets.repository import dataset_list, get_dataset
from paddlets.metrics import MSE, MAE, QuantileLoss
from paddlets.models.forecasting import RNNBlockRegressor, LSTNetRegressor, NBEATSModel, NHiTSModel, MLPRegressor, \
    TCNRegressor, TransformerModel, InformerModel

import warnings

from paddlets.models.representation import TS2Vec, ReprForecasting

warnings.filterwarnings("ignore", category=Warning)
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_rows', 120)

dataset = get_dataset('UNI_WTH')
dataset, __ = dataset.split(0.1)
train_dataset, test_dataset = dataset.split(0.8)

ts2vec = TS2Vec(
 segment_size=200,
 repr_dims=320,
 batch_size=32,
 max_epochs=30,
)
ts2vec.fit(train_dataset)
# ts2vec.save("./ts2vec")
# ts2vec = load("./ts2vec")

sliding_len = 200 # Use past sliding_len length points to infer the representation of the current point in time
all_reprs = ts2vec.encode(dataset, sliding_len=sliding_len)
split_tag = len(train_dataset['WetBulbCelsius'])
train_reprs = all_reprs[:, :split_tag]
test_reprs = all_reprs[:, split_tag:]


# generate samples
def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
             labels.reshape(-1, labels.shape[2]*labels.shape[3])

pre_len = 24 # prediction lengths

# generate training samples
train_to_numpy = train_dataset.to_numpy()
train_to_numpy = np.expand_dims(train_to_numpy, 0) # keep the same dimensions as the encode output
train_features, train_labels = generate_pred_samples(train_reprs, train_to_numpy, pre_len, drop=sliding_len)

# generate test samples
test_to_numpy = test_dataset.to_numpy()
test_to_numpy = np.expand_dims(test_to_numpy, 0)
test_features, test_labels = generate_pred_samples(test_reprs, test_to_numpy, pre_len)

from sklearn.linear_model import Ridge
lr = Ridge(alpha=0.1)
lr.fit(train_features, train_labels)

# predict
subset_test_pred_dataset = lr.predict(test_features)


subset_test_pred_dataset = subset_test_pred_dataset[0]
subset_test_dataset = test_to_numpy.flatten()[:24]

mae = MAE()
print(mae.metric_fn(subset_test_dataset, subset_test_pred_dataset))
mse = MSE()
print(mse.metric_fn(subset_test_dataset, subset_test_pred_dataset))



subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
ts2vec_params = {"segment_size": 200,
                 "repr_dims": 320,
                 "batch_size": 128,
                 "sampling_stride": 200,
                 "max_epochs": 300}
model = ReprForecasting(in_chunk_len=24*7,
                        out_chunk_len=24,
                        sampling_stride=1,
                        repr_model=TS2Vec,
                        repr_model_params=ts2vec_params)
model.fit(train_dataset)
subset_test_pred_dataset = model.recursive_predict(val_dataset,24*4)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()

mae = MAE()
print(mae(subset_test_dataset, subset_test_pred_dataset))
mse = MSE()
print(mse(subset_test_dataset, subset_test_pred_dataset))

from paddlets.utils.backtest import backtest
score, pred_data = backtest(
    data=val_test_dataset,
    model=model,
    start=0.5, #start 可以控制回测的起始点如果设置 start 为0.5,那么回测将会在数据的中间位置开始。
    predict_window=24, # predict_window 是每次预测的窗口长度
    stride=24, # stride 是两次连续预测之间的移动步长
    return_predicts = True,
    metric=mae#如果设置 return_predicts 为True，回测函数会同时返回指标结果和预测值 。
    )

print(f"mae: {score}")

val_test_dataset.plot(add_data=pred_data,labels="backtest")
plt.show()



