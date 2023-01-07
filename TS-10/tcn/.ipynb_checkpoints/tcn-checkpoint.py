from paddlets.datasets.repository import get_dataset, dataset_list
from matplotlib import pyplot as plt

dataset = get_dataset('UNI_WTH')
train_dataset, val_test_dataset = dataset.split(0.8)
val_dataset, test_dataset = val_test_dataset.split(0.5)
train_dataset.plot(add_data=[val_dataset,test_dataset], labels=['Val', 'Test'])
plt.show()

from paddlets.models.forecasting import TCNRegressor
# 构建模型
model = TCNRegressor(
    in_chunk_len = 7 * 24,
    out_chunk_len = 24,
    max_epochs = 1000,
    patience = 30
                        )
#模型训练
model.fit(train_dataset, val_dataset)
# 训练结果可视化
subset_test_pred_dataset = model.predict(val_dataset)
plt.show()
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()