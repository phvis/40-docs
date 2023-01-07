#Informer

Informer是2021年提出的一种编码器-解码器架构的深度学习模型. 核心特征是”ProbSparse注意力”机制, 在注意力计算上达到了O(LlogL)的时间复杂度和O(LlogL)的空间复杂度。

PaddleTS内置的Informer参数如下：

`in_chunk_len`

模型输入的时间序列长度.

Type
int

`out_chunk_len`

模型输出的时间序列长度.

Type
int

`start_token_len`

解码器输入时间序列的填充长度.

Type
int

`skip_chunk_len`

可选变量, 输入序列与输出序列之间跳过的序列长度, 既不作为特征也不作为序测目标使用, 默认值为0

Type
int

`sampling_stride`

相邻样本间的采样间隔.

Type
int

`loss_fn`

损失函数.

Type
Callable[…, paddle.Tensor]|None

`optimizer_fn`

优化算法.

Type
Callable[…, Optimizer]

`optimizer_params`

优化器参数.

Type
Dict[str, Any]

`eval_metrics`

模型训练过程中的需要观测的评估指标.

Type
List[str]

`callbacks`

自定义callback函数.

Type
List[Callback]

`batch_size`

训练数据或评估数据的批大小.

Type
int

`max_epochs`

训练的最大轮数.

Type
int

`verbose`

模型训练过程中打印日志信息的间隔.

Type
int

`patience`

模型训练过程中, 当评估指标超过一定轮数不再变优，模型提前停止训练.

Type
int

`seed`

全局随机数种子, 注: 保证每次模型参数初始化一致.

Type
int|None

`stop_training`

Type
bool

`d_model`

编码器/解码器的输入特征维度.

Type
int

`nhead`

多头注意力机制中的头数.

Type
int

`num_encoder_layers`

编码器中的编码层数.

Type
int

`num_decoder_layers`

解码器中的解码层数.

Type
int

`activation`

编码器/解码器中间层的激活函数, 可选[“relu”, “gelu”].

Type
str

`dropout_rate`

神经元丢弃概率.

Type
float

## PaddleTS内置的InformerModel模型

### 数据介绍与处理

使用内置数据集'UNI_WTH'作为训练数据，构建训练、验证以及测试数据集

    dataset = get_dataset('UNI_WTH')
    train_dataset, val_test_dataset = dataset.split(0.8)
    val_dataset, test_dataset = val_test_dataset.split(0.5)
    train_dataset.plot(add_data=[val_dataset,test_dataset], labels=['Val', 'Test'])
    plt.show()

![img_19.png](img_19.png)

### 模型训练

初始化模型，模型输入的时间序列长度为120，模型输出的时间序列长度为120，最大迭代轮数5000，不再减小（或增大）的累计次数设置为20，模型评估指标设置为["mae", "mse"]

    informer = InformerModel(in_chunk_len=120,
                        out_chunk_len=120,
                        max_epochs=5000,
                        eval_metrics=['mse', 'mae'],
                        patience=20
                        )


模型训练

    informer.fit(train_dataset)
    subset_test_pred_dataset = informer.predict(train_dataset)

### 模型预测

#### 预测

训练模型后预测结果并将预测结果可视化，图中标明预测结果和真实数据

    subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
    subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])
    plt.show()

![img_18.png](img_18.png)

#### 递归多步预测

对模型进行递归多步预测,将predict改为recursive_predict，其支持指定想要输出的预测长度.
想要预测未来240个小时的 WetBulbCelsuis , 我们可以通过调用 recursive_predict 通过如下方法实现

将模型输入的时间序列长度为20*24，其他不变

    nhits = NHiTSModel(
            in_chunk_len=24*20,
            out_chunk_len=24,
            max_epochs=5000,
            patience=20,
            eval_metrics=["mae", "mse"]
        )
    informer.fit(train_dataset)
    subset_test_pred_dataset = informer.recursive_predict(train_dataset,120*2)
    subset_test_dataset, _ = val_test_dataset.split(len(subset_test_pred_dataset.target))
    subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])

结果如下：

![img_20.png](img_20.png)

### 模型评估

有了预测数据和真实数据后，可以计算相应的metrics指标
使用PaddleTS中的MSE和MAE

    mae = MAE()
    print(mae(subset_test_dataset, subset_test_pred_dataset))
    mse = MSE()
    print(mse(subset_test_dataset, subset_test_pred_dataset))

结果如下：

    {'WetBulbCelsius': 3.909598967904846}#MAE
    {'WetBulbCelsius': 24.399359596988916}#MSE

### 模型持久化

模型训练完成后，我们需将训练完成的模型持久化，以便在未来使用该模型时无需对其重复训练。
同时，也可以加载一个已经被保存在硬盘上的PaddleBaseModel模型。
保存模型：

    informer.save("/G:/pycharm/pythonProject7")

加载模型：

    loaded_informer_reg = load("/G:/pycharm/pythonProject7")


