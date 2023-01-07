import warnings
warnings.filterwarnings("ignore")

from paddlets.datasets.repository import get_dataset, dataset_list
from matplotlib import pyplot as plt

dataset = get_dataset('UNI_WTH')
train_dataset, val_test_dataset = dataset.split(0.7)
val_dataset, test_dataset = val_test_dataset.split(0.5)
# train_dataset.plot(add_data=[val_dataset,test_dataset], labels=['Val', 'Test'])
# plt.show()


from paddlets.models.forecasting import InformerModel
model = InformerModel(in_chunk_len=7*24,
                    out_chunk_len=24,
                    max_epochs=500,
                    sampling_stride=1,

                    optimizer_params={'learning_rate':5e-5},
                    batch_size=64,
                    patience=400,
                    nhead=4,
                    d_model=256,
                    ffn_channels=1024,
                    )
#模型训练
model.fit(train_dataset, val_dataset)

subset_test_pred_dataset = model.predict(val_dataset)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()

# subset_test_pred_dataset = model.recursive_predict(val_dataset, 24*4)
# subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
# subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
# plt.show()



mae = MAE()
print(mae(subset_test_dataset, subset_test_pred_dataset))
mse = MSE()
print(mse(subset_test_dataset, subset_test_pred_dataset))

from paddlets.utils import backtest

score , pred_data= backtest(
    data=val_test_dataset,
    model=model,
    start=0.5, #start 可以控制回测的起始点如果设置 start 为0.5,那么回测将会在数据的中间位置开始。
    predict_window=24, # predict_window 是每次预测的窗口长度
    stride=24, # stride 是两次连续预测之间的移动步长
    return_predicts = True, #如果设置 return_predicts 为True，回测函数会同时返回指标结果和预测值 。
    metric=mae
)
print(f"mae: {score}")
val_test_dataset.plot(add_data=pred_data,labels="backtest")
plt.show()

model.save("informer11")

# from paddlets.automl.autots import AutoTS
# autots_model = AutoTS(InformerModel, 168,24,sampling_stride=1)
# autots_model.fit(train_dataset, val_dataset)
#
#
# # Method 1
# best_estimator = autots_model.fit(train_dataset, val_dataset)
# best_estimator.save(path="./autots_best_estimator_m1")
#
# # Method 2
# best_estimator = autots_model.best_estimator()
# best_estimator.save(path="./autots_best_estimator_m2")
