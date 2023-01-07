from paddlets.datasets.repository import get_dataset, dataset_list
from matplotlib import pyplot as plt

dataset = get_dataset('UNI_WTH')
train_dataset, val_test_dataset = dataset.split(0.7)
val_dataset, test_dataset = val_test_dataset.split(0.5)
train_dataset.plot(add_data=[val_dataset,test_dataset], labels=['Val', 'Test'])
plt.show()

from paddlets.models.forecasting import TransformerModel

Trans =TransformerModel(in_chunk_len=24 * 7,
                    out_chunk_len=24,
                    max_epochs=2000,
                    batch_size=2056,
                    patience=100,
                    d_model =8,
                    num_encoder_layers = 1,
                    num_decoder_layers= 1,
                    dim_feedforward= 64,
                    optimizer_params=dict(learning_rate=1e-3),
                    )
Trans.fit(train_dataset,val_dataset)

# 训练结果可视化
from paddlets.models.model_loader import load
model = load("trans")
subset_test_pred_dataset = model.predict(val_dataset)
plt.show()
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()

subset_test_pred_dataset = model.recursive_predict(val_dataset,24*4)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()

from paddlets.metrics import MAE,MSE
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

Trans.save("trans")