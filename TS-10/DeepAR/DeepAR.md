# DeepAR
概率预测，即根据过去估计时间序列的未来概率分布，是优化业务流程的关键因素。例如，在零售业务中，概率需求预测对于在正确的时间和正确的地点获得正确的库存至关重要。

DeepAR，是一种用于生成精确概率预测的方法，该方法基于在大量相关时间序列上训练自回归递归神经网络模型。

## PaddleTS内置的TCNRegressor

### 升级特性
均值迭代预测：在预测阶段，除实现原文中的采样路径外，我们提供了另一个可选的解码方式，通过分布均值进行迭代预测。

1. `in_chunk_len`必选参数

模型输入的时间序列长度。

Type int

2. `out_chunk_len`必选参数

模型输出的序列长度。

Type int

3. rnn_type

具体的RNN模型（”GRU” 或 “LSTM”）。

Type str

4. hidden_size

RNN模型隐藏状态h大小。

Type int

5. num_layers_recurrent

循环网络的层数。

Type int

6. dropout

dropout概率，除第一层外每层输入时的dropout概率。

Type float

7. skip_chunk_len

可选变量， 输入序列与输出序列之间跳过的序列长度，既不作为特征也不作为预测目标使用，默认值为0。

Type int

8. sampling_stride

相邻两个样本的采样间隔。

Type int, optional

9. likelihood_model

概率预测用到的分布似然函数。

Type Likelihood

10. num_samples

在评估和预测阶段的采样数量，用于计算分位数损失与生成预测结果。

Type int

11. loss_fn

对应于似然函数的概率预测损失函数。

Type Callable[..., paddle.Tensor]

12. regression_mode

用于迭代预测的回归方式，可选值有`mean`与`sampling`。

Type str

13. output_mode

模型输出的模式，分位数和预测是可选的。

Type str

14. optimizer_fn

优化器算法。

Type Callable, Optional

15. optimizer_params

优化器参数。

Type Dict, Optional

16. eval_metrics

模型评估指标。

Type List[str], Optional

17. callbacks

自定义的callback函数。

Type List[Callback], Optional

18. batch_size

每个batch中的样本数量。

Type int, Optional

19. max_epochs

训练过程中最大迭代轮数。

Type int, Optional

20. verbose

模型日志模式。

Type int, Optional

21. patience

训练停止所需的效果不再提升的轮数。

Type int, Optional

22. seed

全局随机种子。

Type int, Optional

### 使用DeepAR以及内置数据集进行预测

### 数据介绍与处理

使用内置数据集'UNI_WTH'作为训练数据

构建训练、验证以及测试数据集
````
from paddlets.datasets.repository import get_dataset, dataset_list
from matplotlib import pyplot as plt

dataset = get_dataset('UNI_WTH')
train_dataset, val_test_dataset = dataset.split(0.8)
val_dataset, test_dataset = val_test_dataset.split(0.5)
train_dataset.plot(add_data=[val_dataset, test_dataset], labels=['Val', 'Test'])
plt.show()
````
![img_6.png](img_6.png)


### 模型训练

初始化模型，模型输入的时间序列长度为24 * 7，模型输出的时间序列长度为24，最大迭代轮数300，不再减小（或增大）的累计次数设置为20，学习率设置为1e-4
````
# 构建模型
from paddlets.models.forecasting import DeepARModel

DeepAR =DeepARModel(in_chunk_len=24 * 7,
                    out_chunk_len=24,
                    max_epochs=250,
                    batch_size=512,
                    patience=40,
                    optimizer_params=dict(learning_rate=5e-4),
                    )

# 模型训练

DeepAR.fit(train_dataset,val_dataset)


````
### 模型预测

#### 单步预测

预测只能预测长度为长度为out_chunk_len的数据
将使用验证集进行预测，得到的结果如下，橙色为预测结果，蓝色为真实数据：

````
subset_test_pred_dataset = DeepAR.predict(val_dataset)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()
````

![img_35.png](img_35.png)

#### 递归多步预测

对模型进行递归多步预测,将predict改为recursive_predict，其支持指定想要输出的预测长度.
想要预测未来96个小时的 WetBulbCelsuis , 我们可以通过调用 recursive_predict 通过如下方法实现

````
from paddlets.models.forecasting import DeepARModel

DeepAR =DeepARModel(in_chunk_len=24 * 7,
                    out_chunk_len=24,
                    max_epochs=250,
                    batch_size=512,
                    patience=40,
                    optimizer_params=dict(learning_rate=5e-4),
                    )
DeepAR.fit(train_dataset,val_dataset)
subset_test_pred_dataset = DeepAR.recursive_predict(val_dataset, 24 * 4)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
plt.show()

````

结果如下：

![img_24.png](img_24.png)

### 模型评估

有了预测数据和真实数据后，可以计算相应的metrics指标
我们可以通过 backtest 实现对整个测试集的metrics指标计算。
以QuantileLoss为例：
回测用给定模型获得的历史上的模拟预测,是用来评测模型预测准确率的重要工具。

![img_28.png](img_28.png)

回测是一个迭代过程，回测用固定预测窗口在数据集上进行重复预测，然后通过固定步长向前移动到训练集的末尾。如上图所示，桔色部分是长度为3的预测窗口。在每次迭代中，预测窗口会向前移动3个长度，同样训练集也会向后扩张三个长度。这个过程会持续到窗口移动到数据末尾。
````
from paddlets.utils import backtest
from paddlets.metrics import QuantileLoss

qloss , pred_data= backtest(
    data=val_test_dataset,
    model=DeepAR,
    start=0.5, #start 可以控制回测的起始点如果设置 start 为0.5,那么回测将会在数据的中间位置开始。
    predict_window=24, # predict_window 是每次预测的窗口长度
    stride=24, # stride 是两次连续预测之间的移动步长
    return_predicts = True, #如果设置 return_predicts 为True，回测函数会同时返回指标结果和预测值 。
    metric=QuantileLoss([0.1, 0.5, 0.9])
    # metric=mae
)
print(f"QuantileLoss: {qloss}")

val_test_dataset.plot(add_data=pred_data,labels="backtest",low_quantile=0.05,
     high_quantile=0.95)
plt.show()
````
得到结果：
````
QuantileLoss: {'WetBulbCelsius': {0.1: 1.2298172970749692, 0.5: 1.4986214629289973, 0.9: 0.9213376155457629}}
````
![img_36.png](img_36.png)

### 模型持久化

模型训练完成后，我们需将训练完成的模型持久化，以便在未来使用该模型时无需对其重复训练。
同时，也可以加载一个已经被保存在硬盘上的PaddleBaseModel模型。
保存模型：
````
DeepAR.save("/G:/pycharm/pythonProject7")
````
加载模型：
````
loaded_DeepAR_reg = load("/G:/pycharm/pythonProject7")
````
#### 保存静态图模型
PaddleTS所有时序预测以及异常检测模型的save接口都新增了 network_model 以及 dygraph_to_static 的参数设置;其中, network_model默认是False, 表示仅导出只支持PaddleTS.predict推理的模型文件, 当network_model设置为True的时候, 在此基础上，会新增对paddle 原始network 的模型以及参数的导出, 可用于 Paddle Inference进行推理; dygraph_to_static参数仅当当network_model为True的时候起作用，表示将导出的模型从动态图转换成静态图, 参考 动转静.

````
model.save("./DeepAR", network_model=True, dygraph_to_static=True)

# 包含以下文件
# ./DeepAR.pdmodel
# ./DeepAR.pdiparams
# ./DeepAR_model_meta

````
其中DeepAR.pdmodel以及DeepAR.pdiparams作为paddle 原生模型以及模型参数, 可用于Paddle Inference的应用;同时PaddleTS生成了DeepAR_model_meta文件用于模型的描述, 里面包含了模型的输入数据类型以及shape的各种元信息, 便于用户对模型进行正确的部署应用.

静态图模型可以用于paddleinference进行快速推理

