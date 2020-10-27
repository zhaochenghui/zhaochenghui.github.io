[TOC]

![](C:\Users\z00575241\Desktop\深度模型迁移\1.PNG)



# 1. 核心要点



| 分类                 | 解释              |
| -------------------- | ----------------- |
| 头文件               | 添加文件解释说明  |
| 日志输出             | 适配NPU输出信息   |
| 输入数据（数据处理） | 调整输入数据格式  |
| 机器信息（数据处理） | 指定训练机器和卡  |
| 算子替换（提升性能） | 替代使用TBE算子   |
| 提升精度（提升性能） | 更换不同数据类型  |
| 环境变量             | 告诉机器1-8包信息 |



# 2. 网络模型迁移



## 2.1 Estimator迁移

### 2.1.0 Estimator简介

​	Estimator API属于TensorFlow的高阶API，使用Estimator进行训练脚本开发的一般步骤为：

- 创建输入函数input_fn；

- 构建模型函数model_fn；

- 实例化Estimator，并传入Runconfig类对象作为运行参数；

- 在Estimator上调用训练方法Estimator.train()，利用指定输入对模型进行固定步数的训练；

  

### 2.1.1 输入函数

​	一般情况下，直接迁移，无需改造

##### train_input_fn()

```python
def train_input_fn(train_data,train_labels):  #训练输入，以numpy矩阵作为输入
return tf.estimator.inputs.numpy_input_fn(
  x={"x":train_data},
  y=train_labels,
  batch_size=FLAGS.batch_size,
  num_epochs=None,#训练多少个epochs
  shuffle=True)
```

##### drop_remainder

​	由于当前仅支持固定shape下的训练，也就是在进行图编译时shape的值必须是已知的。当原始网络脚本中使用dataset.batch(batch_size)返回动态形状时，由于数据流中剩余的样本数可能小于batch大小，因此，在昇腾AI处理器上进行训练时，请将drop_remainder设置为True：

```python
dataset = dataset.batch(batch_size, drop_remainder=True)
```

##### assert

​	有些脚本最后会加个断言，预测结果的数量要和预测数据的数量一致，此种情况会导致训练失败:

```python
assert num_written_lines == num_actual_predict_examples
```



### 2.1.2 模型函数

##### dropout()

​	对于原始网络中的dropout接口，修改以获得更优的性能:

```python
#TensorFlow原始代码：
layers = tf.nn.dropout()

#修改后代码：
from npu_bridge.estimator import npu_ops
layers = npu_ops.dropout()
```

##### gelu()

​	对于原始网络例如bert网络中的gelu接口，建议修改以获得更优的性能：

```python
#TensorFlow原始代码：
def gelu(x): 
  cdf = 0.5 * (1.0 + tf.tanh(
     (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))) 
  return x*cdf
layers = gelu()

#修改后代码：
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops
layers = npu_unary_ops.gelu(x)
```



### 2.1.3 运行配置

​	TensorFlow通过Runconfig配置运行参数，用户需要将Runconfig迁移为NPURunconfig：

```python
#TensorFlow原始代码：
config=tf.estimator.RunConfig(
  model_dir=FLAGS.model_dir, 
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
  
#修改后代码：
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator import npu_ops
npu_config=NPURunConfig(
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)#配置自动选择运行设备，不记录设备指派
  )
```



### 2.1.4 创建Estimator

​	用户需要将TensorFlow的Estimator迁移为NPUEstimator：

```python
#TensorFlow原始代码：
mnist_classifier=tf.estimator.Estimator(
  model_fn=cnn_model_fn,
  config=config,
  model_dir="/tmp/mnist_convnet_model")
  
#修改后代码：
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator

mnist_classifier=NPUEstimator(
  model_fn=cnn_model_fn,
  config=npu_config,
  model_dir="/tmp/mnist_convnet_model"
  )
```



### 2.1.5 执行训练

​	利用指定输入对模型进行固定步数的训练，此部分代码直接迁移，无需改造：

```python
mnist_classifier.train(
  input_fn=train_input_fn,
  steps=20000,
  hooks=[logging_hook])
```



## 2.2 sess.run迁移

### 2.2.1 sess.run简介

​	sess.run API属于TensorFlow的低阶API，使用sess.run API进行训练脚本开发的一般步骤为：

-  数据预处理；

- 模型搭建/计算Loss/梯度更新；

- 创建session并初始化资源；

- 执行训练；

  

### 2.2.2 数据预处理

​	一般情况下，直接迁移，无需改造

##### drop_remainder

​	由于当前仅支持固定shape下的训练，也就是在进行图编译时shape的值必须是已知的。当原始网络脚本中使用dataset.batch(batch_size)返回动态形状时，由于数据流中剩余的样本数可能小于batch大小，因此，在昇腾AI处理器上进行训练时，请将drop_remainder设置为True：

```python
dataset = dataset.batch(batch_size, drop_remainder=True)
```

##### assert

​	有些脚本最后会加个断言，预测结果的数量要和预测数据的数量一致，此种情况会导致训练失败:

```python
assert num_written_lines == num_actual_predict_examples
```



### 2.2.3 模型搭建/计算Loss/梯度更新

##### dropout()

​	对于原始网络中的dropout接口，修改以获得更优的性能:

```python
#TensorFlow原始代码：
layers = tf.nn.dropout()

#修改后代码：
from npu_bridge.estimator import npu_ops
layers = npu_ops.dropout()
```

##### gelu()

​	对于原始网络例如bert网络中的gelu接口，建议修改以获得更优的性能：

```python
#TensorFlow原始代码：
def gelu(x): 
  cdf = 0.5 * (1.0 + tf.tanh(
     (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))) 
  return x*cdf
layers = gelu()

#修改后代码：
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops
layers = npu_unary_ops.gelu(x)
```

##### 支持分布式计算

​	首先通过集合通信接口broadcast进行变量广播：

```python
rank_size = os.environ.get('RANK_SIZE', '').strip()
if int(rank_size) > 1:
	input = tf.trainable_variables()
	bcast_global_variables_op = hccl_ops.broadcast(input, 0)
```

​	各Device梯度数据计算后，通过集合通信接口allreduce或者直接使用NPUDistributedOptimizer分布式训练优化器对梯度数据进行聚合:

```python
rank_size = os.environ.get('RANK_SIZE', '').strip()
if int(rank_size) > 1:
	grads = [ hccl_ops.allreduce(grad, "sum") for grad in grads ]
```



### 2.2.4 创建session并初始化资源

​	在昇腾AI处理器上通过sess.run模式执行训练脚本时，相关配置说明：

```python
#以下配置默认关闭，请勿开启：
rewrite_options.disable_model_pruning
#以下配置默认开启，请勿关闭：
rewrite_options.function_optimization
rewrite_options.constant_folding
rewrite_options.shape_optimization
rewrite_options.arithmetic_optimization
rewrite_options.loop_optimization
rewrite_options.dependency_optimization
rewrite_options.layout_optimizer
rewrite_options.memory_optimization
#以下配置默认开启，必须显式关闭：
rewrite_options.remapping
#分布式场景下，需要手工添加GradFusionOptimizer优化器：
rewrite_options.optimizers.extend(["GradFusionOptimizer"])
#以下配置项默认关闭，必须显示开启，用于在昇腾AI处理器执行训练。
custom_op.parameter_map["use_off_line"].b = True
```

​	

​	实际训练代码：

```python
#构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

#取batch数据
next_batch=iterator.get_next()

#迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)

#变量初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

#构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

#取batch数据
next_batch=iterator.get_next()

#迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)
 
#变量初始化
init=tf.global_variables_initializer()

#创建session
config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True #在昇腾AI处理器执行训练
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关

sess = tf.Session(config=config)
sess.run(init)
 
#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```



### 2.2.5 执行训练

​	直接迁移，无需改造：

```python
开始循环迭代
for epoch in range(num_epochs):
  ##Initialize iterator with the training dataset
  sess.run(training_init_op)
  for step in range(train_batches_per_epoch):  
    #get next batch of data
    img_batch,label_batch=sess.run(next_batch)
    #run the training op
    _,train_loss = sess.run([train_op, loss],feed_dict={x:img_batch,y_:label_batch,is_training:True})
```



## 2.3 Keras迁移（待写）





# 3. 分布式训练



## 3.0 使用场景

##### 训练服务器单机

​	Server单机场景，即由1台训练服务器（Server）完成训练，每台Server包含8块芯片（即昇腾AI处理器）。其中参与集合通信的芯片数目只能为1/2/4/8，且0-3卡和4-7卡各为一个组网，使用2张卡或4张卡训练时，不支持跨组网创建设备集群。

##### 	训练服务器集群

​	Server集群场景，即由集群管理主节点+一组训练服务器（Server）组成训练服务器集群，Server当前支持的上限是128台。每台Server上包含8块芯片（即昇腾AI处理器），Server集群场景下，参与集合通信的的芯片数目为8*n（其中n为参与训练的Server个数），n为2的指数倍情况下，集群性能最好，建议用户优先采用此种方式进行集群组网。

##### 	训练卡场景

​	当前训练卡场景只支持单机单卡训练，不支持多机多卡分布式训练场景。



## 3.1 脚本迁移

### 3.1.1 数据并行模式加载数据集

​	分布式训练时，数据是以数据并行的方式导入的，用户可以通过集合通信接口get_rank_size获取芯片数量，通过get_rank_id获取芯片id，例如：

```python
dataset = dataset.shard(get_rank_size(),get_rank_id())
```



### 3.1.2 使用分布式训练优化器

​	TensorFlow通过Runconfig中的train_distribute来指定分布式训练策略，Ascend平台不支持配置train_distribute，用户可以通过NPUDistributedOptimizer类来封装单机训练优化器，构造成NPU分布式训练优化器，从而支持单机多卡、多机多卡等组网形式下，各个Device之间计算梯度后执行梯度聚合操作。

##### TensorFlow原始代码

```python
def cnn_model_fn(features,labels,mode):
  #搭建网络
  xxx
  #计算loss
  xxx
 
  #Configure the TrainingOp(训练模式)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)#使用SGD优化器
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())#最小化loss
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
```



##### 修改后代码

```python
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
#如果原始脚本使用Tensorflow接口计算梯度，例如grads = tf.gradients(loss, tvars)，在构造完NPUDistributedOptimizer后，需要替换成NPUDistributedOptimizer的compute_gradients和apply_gradients方法。
def cnn_model_fn(features,labels,mode):    
  #搭建网络   
  xxx    
  #计算loss
  xxx    

  #Configure the TrainingOp(for TRAIN mode)    
  if mode == tf.estimator.ModeKeys.TRAIN:      
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)#使用SGD优化器
    distributedOptimizer=NPUDistributedOptimizer(optimizer)#使用NPU分布式计算，更新梯度
    train_op=distributedOptimizer.minimize(loss=loss,global_step=tf.train.get_global_step())#最小化loss
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
```



# 4. 脚本介绍



## 4.1 资源文件配置

### 4.1.1 配置文件说明

​	进行训练之前，需要准备芯片资源配置文件（即ranktable文件），并上传到当前运行环境，该文件用于定义训练的芯片资源信息。最终通过环境变量RANK_TABLE_FILE指定ranktable文件文件路径。ranktable文件内容格式按照json格式要求，以2p场景为例，文件可以命名为rank_table_2p.json。

### 4.1.1 ranktable文件

​	当前支持两种配置模板，全新场景推荐使用模板一，模板二用于兼容部分已有场景。

- Server单机场景下，ranktable文件中只需要配置您需要使用的芯片，例如配置1/2/4/8个。#Server集群场景下，ranktable文件中必须配置满参与集合通信的芯片，即8*n（其中n为参与训练的Server个数）；
- Atlas 300 训练卡（型号 9000）场景下，ranktable文件中配置的参与训练的芯片数目不大于服务器上实际的芯片数目，并且必须使用模板一配置；
- Atlas 300T Lite 入门级训练卡（型号 6000）场景下，ranktable文件中配置的参与训练的芯片数目不大于服务器上实际的芯片数目，并且必须用模板一配置。

##### 模板一（推荐使用）

```yaml
{
"server_count":"1",  //server数目，此例中，只有一个server
"server_list":       //本次参与训练的Server实例列表
[
   {
        "device":[    // server中的device列表
                       {
                        "device_id":"0",  // 芯片物理ID，即Device在Server上的序列号，对应HDC通道号，取值范围：[0-7]
                        "device_ip":"192.168.0.2",  // 芯片真实网卡IP，全局唯一，以点分十进制表示的字符串
                        "rank_id":"0"   // Rank唯一标识，从0开始配置，且全局唯一，取值范围：[0, 总Device数量-1]
                        },
                        {
                         "device_id":"1",
                         "device_ip":"192.168.1.2",
                         "rank_id":"1"
                         }
                  ],
         "server_id":"10.0.0.10"  //server物理IP，以点分十进制表示的字符串
    }
],
"status":"completed",  // ranktable可用标识，completed为可用
"version":"1.0"        // ranktable模板版本信息,当前必须为"1.0"
}
```



##### 模板二（兼容部分已有场景）

```yaml
{
"status":"completed",   // Rank table可用标识，completed为可用
"group_count":"1",      // group数量，建议为1
"group_list":           // group列表
 [
   {
    "group_name":"hccl_world_group",//group名称，建议hccl_world_group
    "instance_count":"2",        // instance实例个数，容器场景下可理解为容器个数
    "device_count":"2",         // group中的所有device数目
    "instance_list":[
        {
           "pod_name":"tf-bae41",     //instance实例名称，一般为容器名称
           "server_id":"10.0.0.10",   //server标识，以IP格式填入
           "devices":[                //instance实例的device列表
            {
              "device_id":"0",           // 芯片HDC通道号
              "device_ip":"192.168.0.2"  // 芯片真实网卡IP
            }
           ]
        },
        {
            "pod_name":"tf-tbdf1",             
            "server_id":"10.0.0.10",
            "devices":[
                {
                    "device_id":"1",
                    "device_ip":"192.168.1.2"  
                }
             ]
          }
       ]
   }     
 ] 
}
```



## 4.2 环境变量配置

​	由于执行训练涉及到相关启动参数，建议构建bash启动脚本，并上传至运行环境。后续在进行训练时，可以直接执行bash run_npu.sh进行训练。启动脚本主要作用是，配置训练进程启动所依赖的环境变量、拉起训练脚本，脚本示例如下所示：

```shell
#训练进程启动所依赖的环境变量
export install_path=/usr/local/Ascend
export nnae_path=$install_path/nnae/latest/xxx-linux_gccx.x.x

export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:$nnae_path/fwkacllib/lib64/:$install_path/driver/lib64/common/:$install_path/driver/lib64/driver/:$install_path/add-ons/
export PYTHONPATH=$PYTHONPATH:$nnae_path/opp/op_impl/built-in/ai_core/tbe:$install_path/tfplugin/latest/xxx-linux_gccx.x.x/tfplugin/python/site-packages/:$install_path/fwkacllib/python/site-packages/hccl:$nnae_path/fwkacllib/python/site-packages/te:$nnae_path/fwkacllib/python/site-packages/topi
export PATH=$PATH:$nnae_path/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=$install_path/opp/

export SOC_VERSION=Ascend910
export JOB_ID=10087
export DEVICE_ID=0
export RANK_TABLE_FILE=/home/test/rank_table_2p.json
export RANK_ID=1
export RANK_SIZE=1
#拉起训练脚本
python3.7 /home/test/xxx.py \
#多P训练时，需要依次拉起所有训练进程，因此需要在每个训练进程启动前，需要分别设置DEVICE_ID和RANK_ID，例如：
export DEVICE_ID=1
export RANK_ID=1
```



# 5. ResNet50模型训练示例（Estimater）

## 5.1 原始代码目录

```
├── r1   // 原始模型目录
│   ├── resnet    // resnet主目录
│        ├── __init__.py     
│        ├── cifar10_download_and_extract.py     // 下载并解压数据集至指定文件夹
│        ├── cifar10_main.py       // 基于cifar10数据集训练网络模型
│        ├── cifar10_test.py       // 基于cifar10数据集训练好的模型进行测试和预测
│        ├── estimator_benchmark.py        //  执行基准测试和准确性测试
│        ├── imagenet_main.py      // 基于Imagenet数据集训练网络模型
│        ├── imagenet_preprocessing.py     // Imagenet数据集数据预处理模块
│        ├── imagenet_test.py     // 基于imagenet数据集训练好的模型进行测试和预测
│        ├── resnet_model.py     // resnet模型文件
│        ├── resnet_run_loop.py    // 数据输入处理与运行循环（训练、验证、测试）
│        ├── README.md   // 项目介绍文件
│   ├── utils
│   │   ├── export.py     // 数据接收函数，定义了导出的模型将会对何种格式的参数予以响应
│        ├── 省略部分文件。
├── utils
│   ├── flags
│   │   ├── core.py         // 包含了参数定义的公共接口
│        ├── 省略部分文件。
│   ├── logs
│   │   ├── hooks_helper.py  //自定义创建模型在测试/训练时的钩子，即工具，比如每秒钟计算步数的钩子、每N步或捕获CPU/GPU分析信息的钩子等
│   │   ├── logger.py      // 日志工具
│        ├── 省略部分文件。
│   ├── misc
│   │   ├── distribution_utils.py       // 进行分布式运行模型的辅助函数
│   │   ├── model_helpers.py      // 定义了一些能被模型调用的函数，比如控制模型是否停止
│        ├── 省略部分文件。
│   ├── testing  // 用于性能评估
│        ├── perfzero_benchmark.py
│        ├── 省略部分文件。
```



## 5.2 训练代码目录

```
├── r1
│   ├── resnet       // resnet主目录
│        ├── imagenet_main.py      // 基于Imagenet数据集训练网络模型,包含imagenet数据集数据预处理、模型构建定义、模型运行的相关函数接口
│        ├── imagenet_preprocessing.py     // Imagenet数据集数据预处理模块,训练过程中包括使用提供的边界框对训练图像进行采样、将图像裁剪到采样边界框、随机翻转图像，然后调整到目标输出大小（不保留纵横比）。评估过程中使用图像大小调整（保留纵横比）和中央剪裁。
│        ├── resnet_model.py    // ResNet模型的实现，包括辅助构建ResNet模型的函数以及ResNet block定义函数
│        ├── resnet_run_loop.py    // 数据输入处理与运行循环（训练、验证、测试）,模型运行文件，包括输入处理和运行循环两部分，输入处理包括对输入数据进行解码和格式转换，输出image和label，还根据是否是训练过程对数据的随机化、批次、预读取等细节做出了设定；运行循环部分包括构建Estimator，然后进行训练和验证过程。
├── utils
│   ├── flags
│   │   ├── _base.py     //定义模型的通用参数并设置默认值
```



## 5.3 训练流程

- **数据预处理**:创建输入函数input_fn；

- **模型构建**:构建模型函数model_fn；

- **运行配置**:实例化Estimator，并传入Runconfig类对象作为运行参数；

- **执行训练**:在Estimator上调用训练方法Estimator.train()，利用指定输入对模型进行固定步数的训练；

  

## 5.4 数据预处理

##### input_fn()  

​	输入函数，用于处理数据集用于Estimator训练，输出真实数据。  

##### resnet_main()  

​	包含数据输入、运行配置、训练以及验证的主接口。  

##### 需要修改的地方

- 增加或修改头文件；
- 添加日志输出路径；
- 用于训练和测试的输入函数接口中，drop_remainder必须设置为True；
- 适配昇腾910 AI处理器，用户自行构建网络时有如下情况建议修改；

  

## 5.5 模型构建

### 5.5.1 定义模型函数

##### imagenet_model_fn()  

​	基于imagenet构建的模型函数。 

##### process_record_dataset()  

​	用于构建模型训练所需要的可迭代的数据集。  

##### learning_rate_with_decay()  

​	建立学习率函数，当全局步数小于设定步数时，学习率线性增加，当超过设定步数时，学习率分阶段下降。

##### resnet_model_fn()  

​	用于构建EstimatorSpec，该类定义了由Estimator运行的模型。  

##### ImagenetModel() 

​	ImagenetModel继承自resnet_model模块下的Model，指定了适用于imagenet的ResNet模型的网络规模、版本、分类数、卷积参数和池化参数等。  

##### __call__()  

​	添加操作以对输入图片进行分类，包括：1、为了加速GPU运算，将输入由NHWC转换成NCHW；2、首次卷积运算；3、根据ResNet版本判断是否要做batch norm；4、首次pooling；5、堆叠block；6、计算输入图像的平均值；7、全连接层。  



### 5.5.2 性能提升

- 使用tf.compat.v1.logging.info()接口替换logging.info()接口；
- 不指定dataset.repeat()的次数，在模型正式开始训练的时候，根据实际训练轮数让数据重复多少遍；
- global_step使用int32数据类型，以获得更好的计算性能；
- 使用tf.test.is_built_with_cuda()接口替代tf.config.list_physical_devices('GPU')接口，用于获得更好的计算性能 使用max_pool_with_argmax算子替代max_pooling2d算子，用于获得更好的计算性能；
- 检查输入特征/图像数据类型；
- 给定loss_scale变量以512的数值，简化模型训练步骤，用于加速训练；
- 计算accuracy时labels使用float32数据类型以提升精度；
- 给resnet_main()函数多定义一个形参num_images，用于传入训练和测试图片的数量；
- 调用resnet_main()函数时，传入num_image这个实参；
- 重新计算max_train_steps最大训练步数的数值，输出schedule调度器里面的信息；
- 使用max_eval_steps变量替换flags_obj.max_train_steps变量；
- 修改input_context的值，添加日志输出；
- 适配昇腾910 AI处理器，用户自行构建网络时有如下情况建议修改；



### 5.5.3 分布式训练配置

- 在“official\r1\resnet\resnet_run_loop.py”文件中增加头文件；
- 添加分布式训练优化器NPUDistributedOptimizer，用于分布式训练；



### 5.5.4 设置模型通用参数

* 在“\official\utils\flags\_base.py”添加头文件;
* 修改get_num_gpus()函数，使用hccl包里面的接口直接获取Ascend910芯片的数量;
* 构建更加清晰、详细的日志输出



## 5.6 运行配置

* 实现模型的分布式训练；定义并修改模型本身在训练过程中日志的输出内容，使日志内容更加清晰和明确;

* 在“\official\r1\resnet\resnet_run_loop.py”修改头文件;

* 在导包之后，添加如下两个函数，用于日志打印;

* 添加init_npu()函数，用于初始化Ascend 910芯片;

* 通过NPURunconfig替代Runconfig来配置运行参数，设置混合精度模式precision_mode='allow_mix_precision'=;

* 创建NPUEstimator，使用NPUEstimator接口代替tf.estimator.Estimator;

* 使用tf.compat.v1.logging.info()接口替换logging.info()接口;

* 添加日志输出信息;

* 修改max_steps参数的值，使每一轮训练的步数逐渐增大，用于达到更好的训练效果;

* 单次训练/验证结束后释放设备资源;

  

## 5.7 执行训练

### 5.7.1 训练模块

##### main() 

 	主函数入口，包含训练之前集合通信初始化，执行模型训练。 

##### run_imagenet() 

 	模型训练入口，包含输入函数选择及返回训练结果。 

##### resnet_main() 

 	包含运行配置、训练以及验证的主要函数。 



### 5.7.2 修改日志输出信息

* 在“official\r1\resnet\imagenet_main.py”文件中修改头文件；
* 添加如下两个函数，用于日志打印；
* 添加上文中增加的env()函数，使得输出日志内容更加丰富；
* 修改batch_size的数值大小；
* 使用tf.compat.v1.logging.set_verbosity()接口替换logging.set_verbosity()接口，使tf.compat.v1.logging.INFO()接口替换logging.INFO()接口



### 5.7.3 定义程序入口

* 训练之前集合通信初始化；

* 在imagenet_main.py文件中添加两个函数为启动模型训练做准备；

  

## 5.8 脚本执行

### 5.8.1 准备数据集

​	准备数据集并上传到运行环境的目录下，例如：“/home/zch/Datasets/DatasetForModelTrain/TensporFlowDataSet/imagenet_TF”



### 5.8.2 重新修改目录结构

```shell
├── resnet
│   ├── tensorflow
│         ├── code      // 代码主体
│              ├── official   // 模型主体
│                    ├── benchmark
│                          ├── resnet_tf_r1_benchmark.py  // 模型启动py文件
│                    ├── r1
│                          ├── resnet    // resnet主目录     
│                                ├── imagenet_main.py      // 基于Imagenet数据集训练网络模型
│                                ├── imagenet_preprocessing.py     // Imagenet数据集数据预处理模块
│                                ├── resnet_model.py     // resnet模型文件
│                                ├── resnet_run_loop.py    // 数据输入处理与运行循环（训练、验证、测试）
│                    ├── utils
│                          ├── flags
│                                ├── performance.py
│                                ├── core.py         // 包含了参数定义的公共接口
│                                ├── 省略部分文件。
│                          ├── logs
│                                ├── hooks_helper.py  //自定义创建模型在测试/训练时的钩子
│                                ├── logger.py      // 日志工具
│                                ├── 省略部分文件。
│                          ├── misc
│                                ├── distribution_utils.py       // 进行分布式运行模型的辅助函数
│                                ├── model_helpers.py      // 定义了一些能被模型调用的函数
│                                ├── 省略部分文件。
│                          ├── testing             // 用于性能评估
│                                ├── perfzero_benchmark.py
│                                ├── 省略部分文件。
│         ├── result      // 用于保存模型日志以及输出的模型文件
│         ├── config      // 配置模型启动的参数
│              ├── 8p.json
│              ├── hccl_sample.json // 当前机器的训练环境
│              ├── npu_set_env.sh  // 环境变量
│              ├── resnet_config_1p_npu.py  // 基本参数配置
│         ├── scripts   // 启动脚本
│              ├── run_8p.sh
```



### 5.8.3 配置模型启动的参数

* **配置8p.json文件**:在“resnet/tensorflow/config”路径下新建8p.json文件，模型运行起来后会自动输出当前机器的训练信息到文件下，所以不用添加内容；
* **配置模型启动的基本参数**:在“resnet/tensorflow/config”路径下新建resnet_config_1p_npu.py文件；
* **配置hccl_sample.json文件**:在“resnet/tensorflow/config”路径下新建hccl_sample.json文件；
* **配置程序入口**:在“tensorflow/official/benchmark/”路径下新建resnet_tf_r1_benchmark.py文件，该文件用作最终拉起的训练代码；



### 5.8.4 配置启动脚本

* **配置环境变量**:在“resnet/tensorflow/config”路径下新建npu_set_env.sh文件；
* **配置run_8p.sh文件**:在“resnet/tensorflow/script”路径下新建run_8p.sh文件；
* **配置train_8p.sh文件**:在“resnet/tensorflow/script”路径下新建train_8p.sh文件；



### 5.8.5 运行启动脚本

```shell
cd resnet/tensorflow/script
chmod +x ./run_8p.sh
chmod +x ./train_8p.sh
./scripts/run_8p.sh
```



# 6. 源代码

## 6.1 imagenet_main.py

```python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

############## npu modify begin #############
# 在原代码中添加如下代码：
import sys
import time
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../../')
############## npu modify end #############


from absl import app as absl_app
from absl import flags
from six.moves import range
import tensorflow as tf

from official.r1.resnet import imagenet_preprocessing
from official.r1.resnet import resnet_model
from official.r1.resnet import resnet_run_loop
from official.utils.flags import core as flags_core

############## npu modify begin #############
# 修改logger的导入路径
from official.utils.logs import logger
# 原代码为以下注释部分。
# from official.r1.utils.logs import logger
############## npu modify end #############


############## npu modify begin #############
from npu_bridge.estimator import npu_ops
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id
from tensorflow.core.protobuf import rewriter_config_pb2

tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
############## npu modify end ###############


#import atlasboost module
import atlasboost.tensorflow.mpi_ops as atlasboost

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'

############## npu modify begin #############
# 修改log_file1路径。
file_path = os.path.abspath(os.path.dirname(__file__))
dir_path = file_path.split('tensorflow')[0]
log_file1 = os.path.join(dir_path, 'tensorflow/result/logger_resnet50.log')

# 原代码为以下注释部分：
# log_file1 = 'result/logger_resnet50.log'
############## npu modify end #############

############## npu modify begin #############
# 在原代码中添加如下代码：
def env(log_file):
    work_num = "work " + str(os.environ.get("DEVICE_INDEX"))
    root_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
    datatime = round(time.time(), 3) * 100
    logger1 = get_logger('rizhi', log_file)

    return work_num, root_dir, datatime, logger1


def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s', "%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)
############## npu modify end #############

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
      'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                 default_value=-1),
      'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(serialized=example_serialized,
                                        features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

  return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = _parse_example_proto(raw_record)

  ############## npu modify begin #############
  # 原代码中添加如下内容：
  work_num, root_dir, datatime, logger1 = env(log_file1)
  # 输出日志信息，输出开始时间，记录运行时长。
  logger1.info("namespace:%s,time_ts:%d, event_type:pre_process_event,root_dir:%s" % (work_num, datatime, root_dir))
  logger1.info("namespace:%s,time_ts:%d,event_type:init_start, root_dir:%s" % (work_num, datatime, root_dir))
  ############## npu modify begin #############

  image = imagenet_preprocessing.preprocess_image(
          image_buffer=image_buffer,
          bbox=bbox,
          output_height=DEFAULT_IMAGE_SIZE,
          output_width=DEFAULT_IMAGE_SIZE,
          num_channels=NUM_CHANNELS,
          is_training=is_training)

  ############## npu modify begin #############
  # 原代码中添加如下内容：
  # 输出日志信息，输出结束时间，记录运行时长。
  logger1.info("namespace:%s,time_ts:%d,event_type:init_end, root_dir:%s" % (work_num, datatime, root_dir))
  ############## npu modify end #############

  image = tf.cast(image, dtype)
  return image, label

def input_fn(is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.

  Returns:
    A dataset that can be used for iteration.
  """

  """提供训练和验证batches的函数。

  参数解释:
    is_training: 表示输入是否用于训练的布尔值。
    data_dir: 包含输入数据集的文件路径。
    batch_size: 每个batch的大小。
    num_epochs: 数据集的重复数。
    dtype: 图片/特征的数据类型。
    datasets_num_private_threads: tf.data的专用线程数。
    parse_record_fn: 解析tfrecords的入口函数。
    input_context: 由'tf.distribute.Strategy'传入的'tf.distribute.InputContext'对象。
    drop_remainder: 用于标示对于最后一个batch如果数据量达不到batch_size时保留还是抛弃。设置为True,则batch的维度固定。
    tf_data_experimental_slack: 是否启用tf.data的'experimental_slack'选项。

  Returns:
    返回一个可用于迭代的数据集。
  """

  # 获取文件路径
  filenames = get_filenames(is_training, data_dir)  
  # 按第一个维度切分文件
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if input_context:
    # 获取芯片数量以及芯片id，用于支持数据并行
    ############## npu modify begin #############
    dataset = dataset.shard(get_rank_size(),get_rank_id())
    ############## npu modify end ###############
    # 原代码数据并行代码为以下注释部分。
     # if input_context:
     #   tf.compat.v1.logging.info(
     #       'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
     #           input_context.input_pipeline_id, input_context.num_input_pipelines))
     #   dataset = dataset.shard(input_context.num_input_pipelines,
     #                           input_context.input_pipeline_id)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means that up to 10 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  # cycle_length = 10 并行读取并反序列化10个文件，CPU资源充足的场景下可适当增加该值。
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=10,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record_fn,
      num_epochs=num_epochs,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=tf_data_experimental_slack,
  )


def get_synth_input_fn(dtype):
  return resnet_run_loop.get_synth_input_fn(
      DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES,
      dtype=dtype)


###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, list(choices.keys())))
    raise ValueError(err)

def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  # Warmup and higher lr may not be valid for fine tuning with small batches
  # and smaller numbers of training images.
  if params['fine_tune']:
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  ############## npu modify begin #############
  # 修改 batch_size 大小
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
    batch_size=params['num_gpus'] * params['batch_size'],
    batch_denom=256, num_images=NUM_IMAGES['train'],
    boundary_epochs=[30, 60, 80, 90], decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
    warmup=warmup, base_lr=base_lr)

  # 原代码为以下注释部分：
  # learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
  #     batch_size=params['batch_size'] * params.get('num_workers', 1),
  #     batch_denom=256, num_images=NUM_IMAGES['train'],
  #     boundary_epochs=[30, 60, 80, 90], decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
  #     warmup=warmup, base_lr=base_lr)
  ############## npu modify end #############

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=ImagenetModel,
      resnet_size=params['resnet_size'],
      weight_decay=flags.FLAGS.weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      fine_tune=params['fine_tune'],
      label_smoothing=flags.FLAGS.label_smoothing
  )


def define_imagenet_flags():
  resnet_run_loop.define_resnet_flags(
      resnet_size_choices=['18', '34', '50', '101', '152', '200'],
      dynamic_loss_scale=True,
      fp16_implementation=True)
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(train_epochs=90)

  #Loss scale is defautt used because Davinci core supports mixed precision naturally
  flags_core.set_defaults(loss_scale='512')

def run_imagenet(flags_obj):
  """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.

  Returns:
    Dict of results of the run.  Contains the keys `eval_results` and
      `train_hooks`. `eval_results` contains accuracy (top_1) and
      accuracy_top_5. `train_hooks` is a list the instances of hooks used during
      training.
  """
  input_function = (flags_obj.use_synthetic_data and
                    get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                    input_fn)

  ############## npu modify begin #############
  result = resnet_run_loop.resnet_main(
      flags_obj, imagenet_model_fn, input_function, DATASET_NAME, NUM_IMAGES,
      shape=[DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS])

  # 原代码为以下注释部分：
  # result = resnet_run_loop.resnet_main(
  #     flags_obj, imagenet_model_fn, input_function, DATASET_NAME,
  #     shape=[DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS])
  ############## npu modify begin #############

  return result


############## npu modify begin #############
def main(flags_obj):
  # 初始化NPU，调用HCCL接口。
  init_sess, npu_init = resnet_run_loop.init_npu()
  init_sess.run(npu_init)

  with logger.benchmark_context(flags.FLAGS):
      run_imagenet(flags.FLAGS)
############## npu modify end ###############

# 原代码为以下注释部分。
# def main(_):
#     with logger.benchmark_context(flags.FLAGS):
#         run_imagenet(flags.FLAGS)


############## npu modify begin #############
# 添加以下两个函数。
def benchmark_main():
    work_num, root_dir, datatime, logger1 = env(log_file1)
    logger1.info("namespace:%s,time_ts:%d,event_type:benchmark_start, root_dir:%s" % (work_num, datatime, root_dir))
    absl_app.run(main)
    logger1.info("namespace:%s,time_ts:%d,event_type:benchmark_stop, root_dir:%s" % (work_num, datatime, root_dir))

def benchmark_pre():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    define_imagenet_flags()


if __name__ == '__main__':
  ############## npu modify begin #############
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  # 原代码为以下注释部分：
  # logging.set_verbosity(logging.INFO)
  ############## npu modify end #############

  atlasboost.init()
  device_id = atlasboost.local_rank()
  atlasboost.set_device_id(device_id)

  define_imagenet_flags()
  absl_app.run(main)
```



## 6.2 resnet_run_loop.py

```python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import os

from absl import flags

############## npu modify begin #############
import logging
# 原代码为以下注释部分。
# from absl import logging
############## npu modify end #############

import tensorflow as tf

############## npu modify begin #############
import time
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator  import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer  #分布式訓練配置
from npu_bridge.estimator import npu_ops
from npu_bridge.hccl import hccl_ops
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id
from tensorflow.core.protobuf import rewriter_config_pb2
############## npu modify end ###############

############## npu modify begin #############
from official.utils.logs import hooks_helper
from official.utils.logs import logger
# 原代码为以下注释部分：
# from official.r1.utils.logs import hooks_helper
# from official.r1.utils.logs import logger
############## npu modify end #############

from official.r1.resnet import imagenet_preprocessing
from official.r1.resnet import resnet_model
from official.r1.utils import export
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


############## npu modify begin #############
# 在原代码中添加如下代码：
# 日志文件输出路径
file_path = os.path.abspath(os.path.dirname(__file__))
dir_path = file_path.split('tensorflow')[0]
log_file1 = os.path.join(dir_path, 'tensorflow/result/logger_resnet50.log')

def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s', "%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)

def env(log_file):
    work_num = "work " + str(os.environ.get("DEVICE_INDEX"))
    logger1 = get_logger('rizhi', log_file1)

    return work_num, logger1
############## npu modify end #############


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           drop_remainder=False,
                           tf_data_experimental_slack=False):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private
      threadpool created for all datasets computation.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
        datasets_num_private_threads)
    dataset = dataset.with_options(options)
    tf.compat.v1.logging.info('datasets_num_private_threads: %s',
                              datasets_num_private_threads)

  # Disable intra-op parallelism to optimize for throughput instead of latency.
  options = tf.data.Options()
  options.experimental_threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.repeat()
  # Repeats the dataset for the number of epochs to train.
  #dataset = dataset.repeat(num_epochs)
  #dataset = dataset.repeat()
  # Parses the raw records into images and labels.
  dataset = dataset.map(
      lambda value: parse_record_fn(value, is_training, dtype),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  if tf_data_experimental_slack:
    options = tf.data.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tunning the full input pipeline.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  # pylint: disable=unused-argument
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    """Returns dataset filled with random data."""
    # Synthetic input should be within [0, 255].
    inputs = tf.random.truncated_normal(
        [batch_size] + [height, width, num_channels],
        dtype=dtype,
        mean=127,
        stddev=60,
        name='synthetic_inputs')

    labels = tf.random.uniform(
        [batch_size],
        minval=0,
        maxval=num_classes - 1,
        dtype=tf.int32,
        name='synthetic_labels')
    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data

  return input_fn


def image_bytes_serving_input_fn(image_shape, dtype=tf.float32):
  """Serving input fn for raw jpeg images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    # Bounding box around the whole image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=dtype, shape=[1, 1, 4])
    height, width, num_channels = image_shape
    image = imagenet_preprocessing.preprocess_image(
        image_bytes, bbox, height, width, num_channels, is_training=False)
    return image

  image_bytes_list = tf.compat.v1.placeholder(
      shape=[None], dtype=tf.string, name='input_tensor')
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=dtype)
  return tf.estimator.export.TensorServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def override_flags_and_set_envars_for_gpu_thread_pool(flags_obj):
  """Override flags and set env_vars for performance.

  These settings exist to test the difference between using stock settings
  and manual tuning. It also shows some of the ENV_VARS that can be tweaked to
  squeeze a few extra examples per second.  These settings are defaulted to the
  current platform of interest, which changes over time.

  On systems with small numbers of cpu cores, e.g. under 8 logical cores,
  setting up a gpu thread pool with `tf_gpu_thread_mode=gpu_private` may perform
  poorly.

  Args:
    flags_obj: Current flags, which will be adjusted possibly overriding
    what has been set by the user on the command-line.
  """
  cpu_count = multiprocessing.cpu_count()
  tf.compat.v1.logging.info('Logical CPU cores: %s', cpu_count)

  # Sets up thread pool for each GPU for op scheduling.
  per_gpu_thread_count = 1
  total_gpu_thread_count = per_gpu_thread_count * flags_obj.num_gpus
  os.environ['TF_GPU_THREAD_MODE'] = flags_obj.tf_gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  tf.compat.v1.logging.info('TF_GPU_THREAD_COUNT: %s',
                            os.environ['TF_GPU_THREAD_COUNT'])
  tf.compat.v1.logging.info('TF_GPU_THREAD_MODE: %s',
                            os.environ['TF_GPU_THREAD_MODE'])

  # Reduces general thread pool by number of threads used for GPU pool.
  main_thread_count = cpu_count - total_gpu_thread_count
  flags_obj.inter_op_parallelism_threads = main_thread_count

  # Sets thread count for tf.data. Logical cores minus threads assign to the
  # private GPU pool along with 2 thread per GPU for event monitoring and
  # sending / receiving tensors.
  num_monitoring_threads = 2 * flags_obj.num_gpus
  flags_obj.datasets_num_private_threads = (cpu_count - total_gpu_thread_count
                                            - num_monitoring_threads)


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    """建立学习率函数，当全局步数小于设定步数时，学习率线性增加，当超过设定步数时，学习率分阶段下降。"""

    ############## npu modify begin #############
    # 转换数据类型，将global_step的数据类型转换成int32类型以获得更好的计算性能。
    # 代码中添加如下内容：
    global_step = tf.cast(global_step, tf.int32)
    ############## npu modify end ###############

    # 建立分阶段下降的学习率
    lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      # 当全局步数小于warmup_steps，学习率采用warmup_lr，当学习率大于warmup_steps，学习率采取分阶段下降的策略
      return tf.cond(pred=global_step < warmup_steps,
                     true_fn=lambda: warmup_lr,
                     false_fn=lambda: lr)
    return lr

  def poly_rate_fn(global_step):
    """Handles linear scaling rule, gradual warmup, and LR decay.

    The learning rate starts at 0, then it increases linearly per step.  After
    FLAGS.poly_warmup_epochs, we reach the base learning rate (scaled to account
    for batch size). The learning rate is then decayed using a polynomial rate
    decay schedule with power 2.0.

    Args:
      global_step: the current global_step

    Returns:
      returns the current learning rate
    """

    # Learning rate schedule for LARS polynomial schedule
    if flags.FLAGS.batch_size < 8192:
      plr = 5.0
      w_epochs = 5
    elif flags.FLAGS.batch_size < 16384:
      plr = 10.0
      w_epochs = 5
    elif flags.FLAGS.batch_size < 32768:
      plr = 25.0
      w_epochs = 5
    else:
      plr = 32.0
      w_epochs = 14

    w_steps = int(w_epochs * batches_per_epoch)
    wrate = (plr * tf.cast(global_step, tf.float32) / tf.cast(
        w_steps, tf.float32))

    # TODO(pkanwar): use a flag to help calc num_epochs.
    num_epochs = 90
    train_steps = batches_per_epoch * num_epochs

    min_step = tf.constant(1, dtype=tf.int64)
    decay_steps = tf.maximum(min_step, tf.subtract(global_step, w_steps))
    poly_rate = tf.train.polynomial_decay(
        plr,
        decay_steps,
        train_steps - w_steps + 1,
        power=2.0)
    return tf.where(global_step <= w_steps, wrate, poly_rate)

  # For LARS we have a new learning rate schedule
  if flags.FLAGS.enable_lars:
    return poly_rate_fn

  return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
                    fine_tune=False, label_smoothing=0.0):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.
    fine_tune: If True only train the dense layers(final layers).
    label_smoothing: If greater than 0 then smooth the labels.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  tf.compat.v1.summary.image('images', features, max_outputs=6)

  ############# npu modify begin #############
  # 检查输入特征/图像是否与用于计算的数据类型一致
  if features.dtype != dtype:
    # 将特征的数据类型改成与dtype一致
    features = tf.cast(features, dtype)
  ############## npu modify end ###############
  # 原代码中数据类型修改如下：
  # assert features.dtype == dtype


  model = model_class(resnet_size, data_format, resnet_version=resnet_version,
                      dtype=dtype)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  if label_smoothing != 0.0:
    one_hot_labels = tf.one_hot(labels, 1001)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=one_hot_labels,
        label_smoothing=label_smoothing)
  else:
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [
          tf.nn.l2_loss(tf.cast(v, tf.float32))
          for v in tf.compat.v1.trainable_variables()
          if loss_filter_fn(v.name)
      ])
  tf.compat.v1.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)

    if flags.FLAGS.enable_lars:
      from tensorflow.contrib import opt as contrib_opt  # pylint: disable=g-import-not-at-top
      optimizer = contrib_opt.LARSOptimizer(
          learning_rate,
          momentum=momentum,
          weight_decay=weight_decay,
          skip_list=['batch_normalization', 'bias'])
    else:
      optimizer = tf.compat.v1.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=momentum
      )

    ############## npu modify begin #############
    # 使用分布式训练优化器封装单机优化器，用于支持分布式训练，在原代码中添加如下代码
    optimizer = NPUDistributedOptimizer(optimizer)
    ############## npu modify end ###############

    fp16_implementation = getattr(flags.FLAGS, 'fp16_implementation', None)
    if fp16_implementation == 'graph_rewrite':
      optimizer = (
          tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
              optimizer, loss_scale=loss_scale))

    def _dense_grad_filter(gvs):
      """Only apply gradient updates to the final layer.

      This function is used for fine tuning.

      Args:
        gvs: list of tuples with gradients and variable info
      Returns:
        filtered gradients so that only the dense layer remains
      """
      return [(g, v) for g, v in gvs if 'dense' in v.name]

    ############## npu modify begin #############
    loss_scale = 512
    scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

    if fine_tune:
      scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

    unscaled_grad_vars = [(grad / loss_scale, var)
                          for grad, var in scaled_grad_vars]
    minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None
      ############## npu modify end #############
      # 原代码为以下注释部分：
  """
    if loss_scale != 1 and fp16_implementation != 'graph_rewrite':
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)
      print(">>>>>>>>>>>>>>>>>>>")
      print(loss_scale)
      print("<<<<<<<<<<<<<<<<<<")
      if fine_tune:
        scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss)
      if fine_tune:
        grad_vars = _dense_grad_filter(grad_vars)
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

  """

  ############## npu modify begin #############
  # labels使用float32类型来提升精度。
  accuracy = tf.compat.v1.metrics.accuracy(tf.cast(labels, tf.float32), predictions['classes'])
  ############## npu modify end ###############
  # 原代码中计算accuracy如下：
  # accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])
  accuracy_top_5 = tf.compat.v1.metrics.mean(
      tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))

  ############## npu modify begin #############
  #Using for 8P
  rank_size = int(os.getenv('RANK_SIZE'))
  newaccuracy = (hccl_ops.allreduce(accuracy[0], "sum") / rank_size, accuracy[1])
  newaccuracy_top_5 = (hccl_ops.allreduce(accuracy_top_5[0], "sum") / rank_size, accuracy_top_5[1])
  ############## npu modify begin #############

  metrics = {'accuracy': newaccuracy,
             'accuracy_top_5': newaccuracy_top_5}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
  tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
  tf.compat.v1.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

############## npu modify begin #############
def init_npu():
  """Initialize npu manually.
  Returns:
    `init_sess` npu  init session config.
    `npu_init` npu  init ops.
  """
  npu_init = npu_ops.initialize_system()
  config = tf.ConfigProto()

  #npu mix precision attribute set to true when using mix precision
  config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  #custom_op.parameter_map["precision_mode"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  custom_op.parameter_map["use_off_line"].b = True

  init_sess = tf.Session(config=config)
  return init_sess,npu_init
############## npu modify end ###############

############## npu modify begin #############
# 给resnet_main()函数多定义一个形参num_images，用于传入训练和测试图片的数量
def resnet_main(
    flags_obj, model_function, input_function, dataset_name, num_images, shape=None):

# 原代码resnet_main()函数的定义如下：
# def resnet_main(
#         flags_obj, model_function, input_function, dataset_name, shape=None):
############## npu modify end #############

  """Shared main loop for ResNet Models.

  Args:
    flags_obj: An object containing parsed flags. See define_resnet_flags()
      for details.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    dataset_name: the name of the dataset for training and evaluation. This is
      used for logging purpose.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags_obj.export_dir is passed.

  Returns:
     Dict of results of the run.  Contains the keys `eval_results` and
    `train_hooks`. `eval_results` contains accuracy (top_1) and accuracy_top_5.
    `train_hooks` is a list the instances of hooks used during training.
  """

  model_helpers.apply_clean(flags.FLAGS)

  # Ensures flag override logic is only executed if explicitly triggered.
  if flags_obj.tf_gpu_thread_mode:
    override_flags_and_set_envars_for_gpu_thread_pool(flags_obj)

  # Configures cluster spec for distribution strategy.
  num_workers = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                                     flags_obj.task_index)

  # Creates session config. allow_soft_placement = True, is required for
  # multi-GPU and is not harmful for other modes.
  session_config = tf.compat.v1.ConfigProto(
      inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
      allow_soft_placement=True)

  distribution_strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_core.get_num_gpus(flags_obj),
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs)

  ############## npu modify begin #############
  # Creates a `NPURunConfig` that checkpoints every 115200 steps
  # 创建一个每115200步保存checkpoint的'NPURunConfig'
  run_config = NPURunConfig(
        model_dir=flags_obj.model_dir,
        session_config=session_config,
        keep_checkpoint_max=5,
        save_summary_steps=0,
        save_checkpoints_steps=115200,
        enable_data_pre_proc=True,
        iterations_per_loop=100,
        #enable_auto_mix_precision=True,
		    precision_mode='allow_mix_precision',  # 精度模式设置为混合精度模式
        hcom_parallel=True
      )
  # 原代码中运行参数配置如下：
  # run_config = tf.estimator.RunConfig(
  #     train_distribute=distribution_strategy,
  #     session_config=session_config,
  #     save_checkpoints_secs=60 * 60 * 24,
  #     save_checkpoints_steps=None)
  ############## npu modify end ###############

  # Initializes model with all but the dense layer from pretrained ResNet.
  if flags_obj.pretrained_model_checkpoint_path is not None:
    warm_start_settings = tf.estimator.WarmStartSettings(
        flags_obj.pretrained_model_checkpoint_path,
        vars_to_warm_start='^(?!.*dense)')
  else:
    warm_start_settings = None

  ############## npu modify begin #############
  # Creates a `NPUEstimator` instead of using tf.estimator.Estimator 
  # 使用`NPUEstimator`接口代替tf.estimator.Estimator
  classifier = NPUEstimator(
      model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
      params={
          'resnet_size': int(flags_obj.resnet_size),
          'data_format': flags_obj.data_format,
          'batch_size': flags_obj.batch_size,
          'resnet_version': int(flags_obj.resnet_version),
          'loss_scale': flags_core.get_loss_scale(flags_obj,
                                                  default_for_fp16=128),
          'dtype': flags_core.get_tf_dtype(flags_obj),
          'fine_tune': flags_obj.fine_tune,
          'num_workers': num_workers,
          'num_gpus' : flags_core.get_num_gpus(flags_obj),
      })
  # 原代码中创建Estimator如下：
  # classifier = tf.estimator.Estimator(
  #     model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
  #     warm_start_from=warm_start_settings, params={
  #         'resnet_size': int(flags_obj.resnet_size),
  #         'data_format': flags_obj.data_format,
  #         'batch_size': flags_obj.batch_size,
  #         'resnet_version': int(flags_obj.resnet_version),
  #         'loss_scale': flags_core.get_loss_scale(flags_obj,
  #                                                 default_for_fp16=128),
  #         'dtype': flags_core.get_tf_dtype(flags_obj),
  #         'fine_tune': flags_obj.fine_tune,
  #         'num_workers': num_workers,
  #     })
  ############## npu modify end ###############

  run_params = {
      'batch_size': flags_obj.batch_size,
      'dtype': flags_core.get_tf_dtype(flags_obj),
      'resnet_size': flags_obj.resnet_size,
      'resnet_version': flags_obj.resnet_version,
      'synthetic_data': flags_obj.use_synthetic_data,
      'train_epochs': flags_obj.train_epochs,
      'num_workers': num_workers,
  }
  if flags_obj.use_synthetic_data:
    dataset_name = dataset_name + '-synthetic'

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('resnet', dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=flags_obj.model_dir,
      batch_size=flags_obj.batch_size)

  def input_fn_train(num_epochs, input_context=None):
    ############## npu modify begin #############
    # Using dtype=tf.float16 for higher data transmission performance
    # drop_remainder currently only support true
    # batch_size means single card batch instead of global batch size
    # 使用dtype=tf.float16提高数据传输性能。
    # 当前版本的drop_remainder只支持为True。
    # 这里的batch_size指的是单卡的batch大小而不是全局batch大小。
    return input_function(
        is_training=True,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=num_epochs,
        dtype=tf.float16,
        input_context=input_context,
        drop_remainder=True)

  def input_fn_eval():
    # batch_size means single card batch instead of global batch size
    # Using dtype=tf.float16 for higher data transmission performance
    # drop_remainder currently only support true 
    # 使用dtype=tf.float16提高数据传输性能
    # 当前版本的drop_remainder只支持为True
    # 这里的batch_size指的是单卡的batch大小而不是全局batch大小
    return input_function(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=1,
        dtype=tf.float16,
        input_context=True,
        drop_remainder=True)
    ############## npu modify end ###############

# 原代码中用于训练的输入函数接口和用于验证的输入函数接口。
    # def input_fn_train(num_epochs, input_context=None):
    #     return input_function(
    #         is_training=True,
    #         data_dir=flags_obj.data_dir,
    #         batch_size=distribution_utils.per_replica_batch_size(
    #             flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
    #         num_epochs=num_epochs,
    #         dtype=flags_core.get_tf_dtype(flags_obj),
    #         datasets_num_private_threads=flags_obj.datasets_num_private_threads,
    #         input_context=input_context)
    #
    # def input_fn_eval():
    #     return input_function(
    #         is_training=False,
    #         data_dir=flags_obj.data_dir,
    #         batch_size=distribution_utils.per_replica_batch_size(
    #             flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
    #         num_epochs=1,
    #         dtype=flags_core.get_tf_dtype(flags_obj))


  train_epochs = (0 if flags_obj.eval_only or not flags_obj.train_epochs else
                  flags_obj.train_epochs)

  use_train_and_evaluate = flags_obj.use_train_and_evaluate or num_workers > 1
  if use_train_and_evaluate:
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda input_context=None: input_fn_train(
            train_epochs, input_context=input_context),
        hooks=train_hooks,
        max_steps=flags_obj.max_train_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval)
    ############## npu modify begin ###############
    # 使用tf.compat.v1.logging.info()接口替换logging.info()接口。
    tf.compat.v1.logging.info('Starting to train and evaluate.')
    # 原代码如下所示：
    # logging.info('Starting to train and evaluate.')
    ############## npu modify end ###############
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    # tf.estimator.train_and_evalute doesn't return anything in multi-worker
    # case.
    eval_results = {}
  else:
    if train_epochs == 0:
      # If --eval_only is set, perform a single loop with zero train epochs.
      schedule, n_loops = [0], 1
    else:
      # Compute the number of times to loop while training. All but the last
      # pass will train for `epochs_between_evals` epochs, while the last will
      # train for the number needed to reach `training_epochs`. For instance if
      #   train_epochs = 25 and epochs_between_evals = 10
      # schedule will be set to [10, 10, 5]. That is to say, the loop will:
      #   Train for 10 epochs and then evaluate.
      #   Train for another 10 epochs and then evaluate.
      #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
      n_loops = math.ceil(train_epochs / flags_obj.epochs_between_evals)
      schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
      schedule[-1] = train_epochs - sum(schedule[:-1])  # over counting.

    current_max_steps = 0

    ############## npu modify begin #############
    # 在原代码中添加如下内容：
    # 重新计算max_train_steps最大训练步数的数值
    if flags_obj.max_train_steps is None:
      flags_obj.max_train_steps = (num_images['train'] / flags_obj.batch_size) / flags_core.get_num_gpus(flags_obj)
      max_eval_steps = num_images['validation'] / flags_obj.batch_size
    else:
      max_eval_steps = flags_obj.max_train_steps
    # 输出schedule调度器里面的信息
    for cycle_index, num_train_epochs in enumerate(schedule):
      print(cycle_index)
      print(num_train_epochs)
    ############## npu modify end #############

    for cycle_index, num_train_epochs in enumerate(schedule):
      tf.compat.v1.logging.info('Starting cycle: %d/%d', cycle_index,int(n_loops))

      ############## npu modify begin #############
      # 记录开始时间，原代码中添如下代码：
      work_num, logger1 = env(log_file1)
      root_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
      datatime = round(time.time(), 3) * 100
      logger1.info("namespace:%s,time_ts:%d, key:global_batch_size,value:%d,num_train_epochs:%d,root_dir:%s" % (
                    work_num, datatime, flags_obj.batch_size, num_train_epochs, root_dir))
      logger1.info("nemespace:%s,time_ts:%d,event_type:epoch_start, root_dir:%s" % (work_num, datatime, root_dir))
      ############## npu modify end #############

      ############## npu modify begin #############
      if flags_obj.max_train_steps is None:
        current_max_steps += (num_images['train']/flags_obj.batch_size)*num_train_epochs/flags_core.get_num_gpus(flags_obj)
      else:
        current_max_steps += flags_obj.max_train_steps
      ############## npu modify end #############

      if num_train_epochs:
        # Since we are calling classifier.train immediately in each loop, the
        # value of num_train_epochs in the lambda function will not be changed
        # before it is used. So it is safe to ignore the pylint error here
        # pylint: disable=cell-var-from-loop

        '''
        classifier.train(
            input_fn=lambda input_context=True: input_fn_train(
                num_train_epochs, input_context=input_context),
            hooks=train_hooks,
            max_steps=current_max_steps)
        '''

        ############## npu modify begin #############
        classifier.train(
          input_fn=lambda input_context=True: input_fn_train(
                num_train_epochs, input_context=input_context),
          hooks=train_hooks,
          max_steps=flags_obj.max_train_steps * (cycle_index + 1) * num_train_epochs)

        # 记录结束时间，原代码中添如下代码：
        logger1.info("namespace:%s,time_ts:%d,event_type:epoch_stop, root_dir:%s" % (work_num, datatime, root_dir))


        # 原代码为以下注释部分：
        # classifier.train(
        #     input_fn=lambda input_context=None: input_fn_train(
        #         num_train_epochs, input_context=input_context),
        #     hooks=train_hooks,
        #     max_steps=max_steps=flags_obj.max_train_steps)
        ############## npu modify end #############

      ############## npu modify begin #############
      # npu resorce will be destoryed When the training is over
      # Reinitialize is needed if using hccl interface before next process
      # 单次训练结束时，npu资源将被释放，在下一次进程开始之前如果要用到hccl接口需要重新初始化，在原代码中添加如下内容：
      init_sess,npu_init=init_npu()
      npu_shutdown = npu_ops.shutdown_system()
      init_sess.run(npu_shutdown)
      init_sess.run(npu_init)
      ############## npu modify end ###############

      # flags_obj.max_train_steps is generally associated with testing and
      # profiling. As a result it is frequently called with synthetic data,
      # which will iterate forever. Passing steps=flags_obj.max_train_steps
      # allows the eval (which is generally unimportant in those circumstances)
      # to terminate.  Note that eval will run for max_train_steps each loop,
      # regardless of the global_step count.

      """
      tf.compat.v1.logging.info('Starting to evaluate.')
      eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                         steps=num_images['validation']/flags_obj.batch_size)
      """

      ############## npu modify begin #############
      # 使用tf.compat.v1.logging.info()接口替换logging.info()接口
      tf.compat.v1.logging.info('Starting to evaluate.')
      eval_results = classifier.evaluate(input_fn=input_fn_eval,
                  steps=max_eval_steps)
      # 原代码为以下注释部分：
      # eval_results = classifier.evaluate(input_fn=input_fn_eval,
      #                                    steps=flags_obj.max_train_steps)
      ############## npu modify end #############

      benchmark_logger.log_evaluation_result(eval_results)


      if model_helpers.past_stop_threshold(
          flags_obj.stop_threshold, eval_results['accuracy']):
        break

      ############## npu modify begin #############
      # 当评估结束时npu资源释放，在调用nccl接口之前需要重新初始化，在原代码中添加如下内容：
      init_sess, npu_init = init_npu()
      npu_shutdown = npu_ops.shutdown_system()
      init_sess.run(npu_shutdown)
      init_sess.run(npu_init)
      ############## npu modify end ###############      
      """
      
      ############## npu modify begin #############
      # npu resorce will be destoryed when evaluate finish
      # Reinitialize is needed before using hccl interface
      if cycle_index < n_loops-1:
          init_sess,npu_init=init_npu()
          npu_shutdown = npu_ops.shutdown_system()
          init_sess.run(npu_shutdown)
          init_sess.run(npu_init)
      ############## npu modify end ###############
      """

  if flags_obj.export_dir is not None:
    # Exports a saved model for the given classifier.
    export_dtype = flags_core.get_tf_dtype(flags_obj)
    if flags_obj.image_bytes_as_serving_input:
      input_receiver_fn = functools.partial(
          image_bytes_serving_input_fn, shape, dtype=export_dtype)
    else:
      input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
          shape, batch_size=flags_obj.batch_size, dtype=export_dtype)
    classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn,
                                 strip_default_attrs=True)

  ############## npu modify begin #############
  npu_shutdown = npu_ops.shutdown_system()
  init_sess.run(npu_shutdown)
  ############## npu modify end ###############

  stats = {}
  stats['eval_results'] = eval_results
  stats['train_hooks'] = train_hooks

  return stats


def define_resnet_flags(resnet_size_choices=None, dynamic_loss_scale=False,
                        fp16_implementation=False):
  """Add flags and validators for ResNet."""
  flags_core.define_base(clean=True, train_epochs=True,
                         epochs_between_evals=True, stop_threshold=True,
                         num_gpu=True, hooks=True, export_dir=True,
                         distribution_strategy=True)
  flags_core.define_performance(num_parallel_calls=False,
                                inter_op=True,
                                intra_op=True,
                                synthetic_data=True,
                                dtype=True,
                                all_reduce_alg=True,
                                num_packs=True,
                                tf_gpu_thread_mode=True,
                                datasets_num_private_threads=True,
                                dynamic_loss_scale=dynamic_loss_scale,
                                fp16_implementation=fp16_implementation,
                                loss_scale=True,
                                tf_data_experimental_slack=True,
                                max_train_steps=True)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags_core.define_distribution()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
      name='resnet_version', short_name='rv', default='1',
      enum_values=['1', '2'],
      help=flags_core.help_wrap(
          'Version of ResNet. (1 or 2) See README.md for details.'))
  flags.DEFINE_bool(
      name='fine_tune', short_name='ft', default=False,
      help=flags_core.help_wrap(
          'If True do not train any parameters except for the final layer.'))
  flags.DEFINE_string(
      name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
      help=flags_core.help_wrap(
          'If not None initialize all the network except the final layer with '
          'these values'))
  flags.DEFINE_boolean(
      name='eval_only', default=False,
      help=flags_core.help_wrap('Skip training and only perform evaluation on '
                                'the latest checkpoint.'))
  flags.DEFINE_boolean(
      name='image_bytes_as_serving_input', default=False,
      help=flags_core.help_wrap(
          'If True exports savedmodel with serving signature that accepts '
          'JPEG image bytes instead of a fixed size [HxWxC] tensor that '
          'represents the image. The former is easier to use for serving at '
          'the expense of image resize/cropping being done as part of model '
          'inference. Note, this flag only applies to ImageNet and cannot '
          'be used for CIFAR.'))
  flags.DEFINE_boolean(
      name='use_train_and_evaluate', default=False,
      help=flags_core.help_wrap(
          'If True, uses `tf.estimator.train_and_evaluate` for the training '
          'and evaluation loop, instead of separate calls to `classifier.train '
          'and `classifier.evaluate`, which is the default behavior.'))
  flags.DEFINE_bool(
      name='enable_lars', default=False,
      help=flags_core.help_wrap(
          'Enable LARS optimizer for large batch training.'))
  flags.DEFINE_float(
      name='label_smoothing', default=0.0,
      help=flags_core.help_wrap(
          'Label smoothing parameter used in the softmax_cross_entropy'))
  flags.DEFINE_float(
      name='weight_decay', default=1e-4,
      help=flags_core.help_wrap(
          'Weight decay coefficiant for l2 regularization.'))

  choice_kwargs = dict(
      name='resnet_size', short_name='rs', default='50',
      help=flags_core.help_wrap('The size of the ResNet model to use.'))

  if resnet_size_choices is None:
    flags.DEFINE_string(**choice_kwargs)
  else:
    flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)
```



# 7. 版本日志

#### 2020.10.27 -v1

* 第一版





