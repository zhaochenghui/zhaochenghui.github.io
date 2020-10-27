[TOC]

# 1. 目录结构

## 1.1 单算子验证

```
├── .idea
├── bulid                                  //编译生成的中间文件
├── framework                              //算子插件实现文件目录
│   ├── tensorflow_plugin
│   │   ├── tensorflow_add_plugin.cpp   
├── fusion_rules                           //融合规则文件目录（当前Add算子不做融合规则定义，可以不关注）
│   ├── ai_core
│   │   ├── graph_fusion_rules.json   
├── op_proto                               //算子IR定义文件目录
│   ├── add.cpp  
│   ├── add.h
├── out                                    //运行结果目录，创建工程时不生成，运行后生成。
│   ├── bin 
│   │   ├── kernel_meta                  //UT测试用例编译生成的算子.o文件和算子描述.json文件  
│   │   ├── llt
│   │   │   ├── ops
│   │   │   │   ├── common
│   │   │   │   │   ├── data          //ST、BBIT用到的测试数据，包括算子输入、输出对比数据
│   │   │   │   │   ├── kernel_bin    //ST、BBIT用例用到的算子.o文件和算子描述.json文件
│   ├── coverage_report                   //UT、ST覆盖率结果文件
│   │   ├── ut                           
│   │   ├── st                           
├── tbe                    
│   ├── impl                              //算子实现文件目录
│   │   ├── add.py              
│   ├── op_info_cfg                       //算子信息库文件目录
│   │   ├── aicore
│   │   │   ├── add.ini
│   ├── testcases                          //算子测试用例目录
│   │   ├── ut                            //UT测试用例目录
│   │   │   ├── add
│   │   │   │   ├── test_add.py   //UT测试用例，可编译生成算子.o文件和算子描述.json文件
│   │   │   │   ├── __init__.py
│   │   │   ├── __init__.py
│   │   ├── st                            //ST测试用例目录
│   │   │   ├── add
│   │   │   │   ├── __init__.py
│   │   │   │   ├── CMakieLists.txt
│   │   │   │   ├── add_datagen.py //用于生成测试数据，包括算子输入、输出对比数据
│   │   │   │   ├── add_st.cc      //ST测试的C++用例，可编译生成可执行程序，在仿真环境上运行
│   │   │   │   ├── test_add_st.py //ST测试的Python用例，可编译生成算子.o文件和算子描述.json文件
│   │   │   ├── __init__.py
│   │   │   ├── CMakieLists.txt
│   │   ├── bbit                           //BBIT测试用例目录
│   │   │   ├── add
│   │   │   │   ├── add_test.cc   //BBIT测试用例，可编译生成可执行程序，在真实硬件环境上运行
│   │   │   │   ├── add_test.hpp
│   │   │   ├── CMakieLists.txt
│   │   ├── tf_test                        //TenserFlow整网测试用例目录
│   │   │   ├── add
│   │   │   │   ├── tf_add.py
│   │   ├── me_test                        //GE整网测试用例目录，推荐使用TenserFlow整网进行测试，可暂不关注
│   │   │   ├── add
│   │   │   │   ├── main.cpp      
│   │   │   │   ├── CMakieLists.txt  
│   │   │   ├── CMakieLists.txt  
├── .project                                 //工程信息文件，包含工程类型、工程描述、运行目标设备类型以及DDK版本
├── CMakeLists.txt               
├── MyOperator.iml    
```



## 1.2 网络运行验证

```
├── .idea
├── bulid                                  //编译生成的中间文件
├── cmake                                  //编译相关公共文件存放目录
├── framework                              //算子插件实现文件目录
│   ├── tf_plugin                 //存放tensorflow框架的算子插件文件及编译规则文件
│   │   ├── tensorflow_add_plugin.cpp 
│   │   ├── CMakeLists.txt               //编译规则文件，会被上级目录中的CMakeLists.txt文件调用
│   ├── CMakeLists.txt                    //编译规则文件，会被算子工程根目录中的CMakeLists.txt文件调用
├── op_proto                               //算子IR定义文件目录
│   ├── add.cpp  
│   ├── add.h
│   ├── CMakeLists.txt                    //编译规则文件，会被算子工程根目录的CMakeLists.txt文件调用
├── scripts                                //工程相关脚本
├── tbe                    
│   ├── impl                              //算子实现文件目录
│   │   ├── add.py         
│   ├── op_info_cfg                       //算子信息库文件目录
│   │   ├── aicore
│   │   │   ├── soc_version             
│   │   │   │   ├── add.ini
│   ├── testcases                          //算子测试用例目录
│   │   ├── acl_op                     //通过ACL进行单算子验证目录
│   │   │   ├── inc
│   │   │   │   ├── operator_desc.h      //算子描述声明文件，包含算子输入/输出，算子类型以及输入描述与输出描述 
│   │   │   │   ├── op_runner.h    //算子描述声明文件，包含算子输入/输出，算子类型以及输入描述与输出描述 
│   │   │   │   ├── common.h   // 声明公共方法类，用于读取二进制文件 
│   │   │   ├── run             // 单算子执行需要的文件存放目录
│   │   │   │   ├── out          // 单算子执行需要的可执行文件存放目录
│   │   │   │   │   ├── test_data    // 测试数据存放目录
│   │   │   │   │   │   ├── config
│   │   │   │   │   │   │   ├── acl.json       //用于进行~acl初始化，请勿修改此文件
│   │   │   │   │   │   │   ├── op_list.txt    //算子描述文件，用于构造单算子模型文件
│   │   │   │   │   │   ├── data
│   │   │   │   │   │  │    ├── generate_data.py    // 生成测试数据的脚本，生成shape为（8,16），类型为int32的文件
│   │   │   ├── src
│   │   │   │   ├── main.cpp      // 将单算子Add编译为om文件并加载om文件执行，此文件中包含算子的相关配置，若验证其他单算子，基于此文件修改
│   │   │   │   ├── operator_desc.cpp     // 构造算子的输入与输出描述
│   │   │   │   ├── op_runner.cpp   // 单算子编译与运行函数实现文件
│   │   │   │   ├── CMakeLists.txt    // 编译规则文件
│   │   │   │   ├── common.cpp         // 公共函数，读取二进制文件函数的实现文件
│   │   ├── tf_test                        //TensorFlow整网测试用例目录
│   │   │   ├── add
│   │   │   │   ├── tf_add.py
├── .project                                 //工程信息文件，包含工程类型、工程描述、运行目标设备类型以及ADK版本
├── CMakeLists.txt               
├── MyOperator.iml    
```



## 1.3 opp组件目录结构

```
├── opp      //算子库目录
│   ├── op_impl
│       ├── built-in        
│       ├── custom
│           ├── ai_core
│               ├── tbe
│                   ├── config
│                       ├── soc_version     //昇腾AI处理器版本
│                           ├── aic-soc_version-ops-info.json     //自定义算子信息库文件
│                   ├── custom_impl               //自定义算子实现代码文件
│                       ├── add.py
│           ├── vector_core   //此目录预留，无需关注
│   ├── framework
│       ├── built-in
│       ├── custom
│           ├── caffe      
│           ├── tensorflow          //存放tensorflow框架的自定义算子插件库
│               ├── libcust_tf_parsers.so
│               ├── npu_supported_ops.json   //Ascend 910场景下使用的文件，Ascend 310场景下无需关注
│   ├── op_proto
│       ├── built-in
│       ├── custom
│           ├── libcust_op_proto.so    //自定义算子原型库文件
```



# 2. 全局理解



## 2.1 TBE 算子开发流程

![TBE算子开发流程](C:\Users\z00575241\Desktop\图片\TBE算子开发流程.png)



### 2.1.1 工程创建

​	通过Mind Studio工具创建Add算子工程，创建完之后工具会自动生成算子工程目录及相应的文件模板。



### 2.1.2 算子开发

​	开发者可以基于创建的Add算子工程模板进行算子开发。

* **算子分析**：使用TBE DSL方式开发Add算子前，我们需要确定算子功能、输入、输出，算子开发方式、算子类型以及算子实现函数名称等；
* **算子代码实现**：通过调用TBE DSL接口，在算子工程下的“tbe/impl/add.py”文件中进行Add算子的实现，包括算子函数定义、算子入参校验、compute过程实     现及调度与编译；
* **算子适配插件实现**：开发者需要进行算子适配插件的开发，实现将Tensorflow网络中的算子进行解析并映射成昇腾AI处理器中的算子；	
* **算子原型定义**：将算子注册到算子原型库中，网络运行时，GE会调用算子原型库的校验接口进行基本参数的校验，校验通过后，会根据原型库中的推导函数推导每个节点的输出shape与dtype，进行输出tensor的静态内存的分配；
* **算子信息定义**：需要通过配置算子信息文件，将算子的相关信息注册到算子信息库中。算子信息库主要体现算子在昇腾AI处理器上物理实现的限制，包括算子的输入输出dtype、format以及输入shape信息。网络运行时，FE会根据算子信息库中的算子信息做基本校验，判断是否需要为算子插入合适的转换节点，并根据算子信息库中信息找到对应的算子实现文件进行编译，生成算子二进制文件进行执行；

### 2.1.3 单算子验证

​	通过编写UT/ST/BBIT测试用例，并运行测试用例来验证算子程序正确性和逻辑正确性。

- **UT测试**：即单元测试（Uint Test），验证算子代码逻辑的正确性。UT测试侧重于保证算子程序能够跑通，场景覆盖全面，算子功能正确性由ST测试和BBIT测试保证；
- **ST测试**：即系统测试（System Test），在仿真环境下，验证算子功能的正确性；
- **BBIT测试**：即模块间接口测试（Building Block Integrated Test），在昇腾AI处理器所在硬件环境，验证算子功能的正确性；



### 2.1.4 算子编译

​	将算子插件实现文件、算子原型定义文件、算子信息定义文件分别编译成算子插件、算子原型库、算子信息库。



### 2.1.5 算子部署（推理）

​	将算子编译生成的自定义算子安装包custom_opp_Targert OS_Target Architecture.run部署到开发环境的opp目录下。



### 2.1.6 算子部署（训练）

​	将自定义算子安装包custom_opp_Targert OS_Target Architecture.run部署到昇腾AI处理器所在硬件环境的算子库中。



### 2.1.7 网络运行验证（推理）

​	通过ACL进行单算子验证的主要流程为：通过ACL将自定义算子转换为单算子模型文件，为此模型输入用户构造的测试数据二进制文件，进行单算子模型的推理操作，通过查看输出结果验证算子的执行结果是否正确。



### 2.1.8 网络运行验证（训练）

​	自定义算子部署到在昇腾AI处理器所在硬件环境后，需要编写测试用例，构造包含待验证算子的TensorFlow或MindSpore网络，从而验证该算子在网络中的运行结果是否正确。



## 2.2 TBE 中间文件输出



| **阶段**             | **目录**                                  | **文件**                            | **意义**               |
| -------------------- | ----------------------------------------- | ----------------------------------- | ---------------------- |
| 算子代码实现         | tbe/impl/kernel_meta                      | add.o                               | 算子二进制文件         |
|                      | tbe/impl/kernel_meta                      | add.json                            | 算子描述文件           |
| 算子编译             | AscendProjects/add/cmake-build            | custom_opp_Centos7.6_x86_64.run     |                        |
| 网络运行验证（推理） | tbe/testcases/acl_op/run/out/op_models    | 0_Add_3_2_8_16_3_2_8_16_3_2_8_16.om | 单算子模型文件         |
|                      | tbe/testcases/acl_op/run/out/result_files | output_0.bin                        | 输出数据的二进制文件   |
|                      | tbe/testcases/acl_op/run/out              | main                                | 单算子验证的可执行文件 |



## 2.3 TBE 工程目录解析



### 2.3.1 算子开发

| **阶段**     |     **步骤**     | **目录**                                       | **文件**                  |
| ------------ | :--------------: | ---------------------------------------------- | ------------------------- |
| 算子开发     |   算子代码实现   | tbe/impl/                                      | add.py                    |
|              | 算子适配插件实现 | framework/tf_plugin/                           | tensorflow_add_plugin.cpp |
|              |   算子原型定义   | op_proto/                                      | add.h  + add.cpp          |
|              |   算子信息定义   | tbe/op_info_cfg.ai_core.ascend310              | add.ini                   |
| 网络运行验证 |       推理       | tbe/testcases/acl_op/run/out/test_data/config/ | op_list.txt               |
|              |                  | tbe/testcases/acl_op/src/                      | main.cpp                  |
|              |                  | tbe/testcases/acl_op/run/out/test_data/data    | generate_date.py          |
|              |                  | tbe/testcases/acl_op/src                       | CMakeLists.txt            |
|              |       训练       | tbe/testcases/tf_test/                         | tf_add.py                 |



**add.py**：进行Add算子的实现，包括算子函数定义、算子入参校验、compute过程实现及调度与编译；

**tensorflow_add_plugin.cpp**：开发者需要进行算子适配插件的开发，实现将Tensorflow网络中的算子进行解析并映射成昇腾AI处理器中的算子；

**add.h**：IR实现文件，将算子注册到算子原型库中；

**add.cpp**：将算子注册到算子原型库中， 用于推导出算子的输出张量描述，这样在网络运行时就可以为所有的张量静态分配内存，避免动态内存分配带来的开销；

​                   Verify函数-IMPLEMT_VERIFIER(Add, AddVerify)函数-用于校验Add算子的两个输入的DataType是否一致

​                   InferShape函数-IMPLEMT_COMMON_INFERFUNC(AddInferShape)函数

**add.ini**：配置算子信息文件，将算子的相关信息注册到算子信息库中；



### 2.3.2 单算子验证

| 类型 | 目录                    | 文件           | 目的                                                         |
| ---- | ----------------------- | -------------- | ------------------------------------------------------------ |
| UT   | tbe/testcase/ut/add/    | test_add.py    | 编写UT测试用例                                               |
| ST   | tbe/testcases/st/add/   | test_add_st.py | 编写ST的Python用例，用于测试各种数据类型、shape等场景下生成的算子的正确性 |
|      |                         | add_datagen.py | 编写测试数据生成脚本，生成输入算子数据和算子期望输出数据     |
|      |                         | add_st.cc      | 编写ST的C++测试用例，用于计算出算子执行结果，并取回结果和预期结果进行比较，来测试算子逻辑的正确性 |
| BBIT | tbe/testcases/bbit/add/ | add_test.hpp   | 编写C++用例.hpp文件                                          |
|      |                         | add_test.cc    | 编写BBIT的C++测试用例，用于计算出在昇腾AI处理器所在硬件环境下算子的执行结果，并取回结果和预期结果进行比较，来测试算子逻辑的正确性 |

### 2.3.3 网络运行验证

| 目录                                           | 文件             | 目的                                                         |
| ---------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| tbe/testcases/acl_op/run/out/test_data/config/ | op_list.txt      | 配置算子描述信息，此文件用于ACL生成单算子模型文件时提供算子信息 |
| tbe/testcases/acl_op/src/                      | main.cpp         | 配置算子执行时所需信息+进行算子验证启动代码开发              |
| tbe/testcases/acl_op/run/out/test_data/data    | generate_date.py | 生成测试数据文件                                             |
| tbe/testcases/acl_op/src                       | CMakeLists.txt   | 编译文件                                                     |
| tbe/testcases/tf_test/                         | tf_add.py        | 进行测试用例开发                                             |





# 3. 算子开发



## 3.1 算子代码实现

### add.py（样例）

```python
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

@fusion_manager.register("add")
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)     # 将input_x.shape转换为list类型赋值给shape_x
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)     # 将input_y.shape转换诶list类型赋值给shape_y

    # 取shape_x、shape_y中每个维度的大值赋给shape_max
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_max)     # 对shape_max进行校验
    input_x = te.lang.cce.broadcast(input_x, shape_max)    # 将input_x的shape广播为shape_max
    input_y = te.lang.cce.broadcast(input_y, shape_max)    # 将input_y的shape广播为shape_max
    res = te.lang.cce.vadd(input_x, input_y)     # 执行input_x + input_y
    # 返回计算结果的tensor
    return res


@util.check_input_type(dict, dict, dict, str)
def add(input_x, input_y, output_z, kernel_name="add"):
    # 获取算子输入tensor的shape与dtype
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype_x = input_x.get("dtype").lower()
    dtype_y = input_y.get("dtype").lower()

    util.check_shape_rule(shape_x)    # 校验算子的shape，维度数需要大于等于1、小于等于8
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)  # 校验算子第一个输入shape大小
    util.check_tensor_shape_size(shape_y)  # 校验算子第二个输入shape大小
    util.check_kernel_name(kernel_name)  # 校验算子的kernel_name

    if dtype_x != dtype_y:      # 若两个输入tensor的shape不一致，则报错
        raise RuntimeError("dtype of inputs should be consistent")
    dtype = dtype_x
    check_tuple = ("float16", "float32", "int32")
    util.check_dtype_rule(dtype, check_tuple)

    # shpae_max取shape_x与shape_y的每个维度的最大值
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        # 如果shape的长度等于1，就直接赋值，如果shape的长度不等于1，做切片，将最后一个维度舍弃
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    util.check_tensor_shape_size(shape_max)
    # 使用TVM的placeholder接口对第一个输入tensor进行占位，返回一个tensor对象
    data_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype)
    # 使用TVM的placeholder接口对第二个输入tensor进行占位，返回一个tensor对象
    data_y = tvm.placeholder(shape_y, name="input_y", dtype=dtype)

    # 调用compute实现函数
    res = add_compute(data_x, data_y, output_z, kernel_name)
    # 自动调度
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)
    # 编译配置
    config = {"print_ir": False,      # 是否打印lower IR code
              "name": kernel_name,
              "tensor_list": (data_x, data_y, res)}
    te.lang.cce.cce_build_code(schedule, config)


# 设置DDK版本号
from te.platform import cce_conf
#cce_conf.cceProduct("x.x.x.x.x")
# 算子调用
if __name__ == '__main__':
    input_output_dict = {"shape": (5, 6, 7), "dtype": "float32"}
    add(input_output_dict, input_output_dict, input_output_dict, kernel_name="add")
```



### rsqrt.py（TBE初级课程模式）

```python
#题目一：请用DSL语言实现TBE算子,用于计算 x 元素的平方根的倒数，即(y = 1 / sqrt{x})：

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


@util.check_input_type(tuple, str, str)
def rsqrt(shape, dtype, kernel_name="rsqrt"):
    """
    请用DSL语言实现TBE算子,用于计算 x 元素的平方根的倒数，即(y = 1 / sqrt{x})
    calculating data rsqrt,y = 1/(x**0.5), because of the version,please don't use te.lang.cce.vrsqrt(data) or te.lang.cce.rsqrt(data)
    Parameters
    ----------
    shape: Tensor shape
    dtype: data type
    name: kernel name, default value is "rsqrt"
    Returns
    -------
    res: TVM tensor
    result of compute
    """

    util.check_shape_rule(shape)    # 校验算子的shape，维度数需要大于等于1、小于等于8
    util.check_tensor_shape_size(shape)  # 校验算子输入shape大小
    util.check_kernel_name(kernel_name)  # 校验算子的kernel_name

    data = tvm.placeholder(shape, name="data", dtype=dtype)

    # 計算y = 1 / sqrt{x},即 y = 1 / exp(0.5 * log(x))
    log_val = te.lang.cce.vlog(data)
    const_val = tvm.const(0.5, "float16")
    mul_val = te.lang.cce.vmuls(log_val, const_val)
    exp_val = te.lang.cce.vexp(mul_val)
    res = te.lang.cce.vrec(exp_val)

    """
    TODO:
    auto schedule
    """
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    """
    TODO:
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(schedule, config)
#文件名命名为rsqrt.py，其中测试用例的shape=(16, 16, 16, 16, 16)，dtype为float16，使用cloud形态。


# 设置DDK版本号
from te.platform import cce_conf 
#cce_conf.cceProduct("x.x.x.x.x")
# 算子调用
if __name__ == '__main__':
    shape = (16, 16, 16, 16, 16)
    dtype = "float16"
    rsqrt(shape, dtype, kernel_name="rsqrt")
```



### rsqrt.py（标准模式）

```python
#题目一：请用DSL语言实现TBE算子,用于计算 x 元素的平方根的倒数，即(y = 1 / sqrt{x})：

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


@fusion_manager.register("rsqrt")
def rsqrt_compute(input_x, output_y, kernel_name="rsqrt"):
    # 計算y = 1 / sqrt{x},即 y = 1 / exp(0.5 * log(x))
    log_val = te.lang.cce.vlog(input_x)
    const_val = tvm.const(0.5, "float16")
    mul_val = te.lang.cce.vmuls(log_val, const_val)
    exp_val = te.lang.cce.vexp(mul_val)
    res = te.lang.cce.vrec(exp_val)

    # 返回计算结果的tensor
    return res


@util.check_input_type(dict, dict, str)
def rsqrt(input_x, output_y, kernel_name="rsqrt"):
    """
    请用DSL语言实现TBE算子,用于计算 x 元素的平方根的倒数，即(y = 1 / sqrt{x})
    calculating input_x rsqrt,y = 1/(x**0.5), because of the version,please don't use te.lang.cce.vrsqrt(input_x) or te.lang.cce.rsqrt(input_x)
    Parameters
    ----------
    shape: Tensor shape
    dtype: input_x type
    name: kernel name, default value is "rsqrt"
    Returns
    -------
    res: TVM tensor
    result of compute
    """
    # 获取算子输入tensor的shape与dtype
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()

    util.check_shape_rule(shape_x)    # 校验算子的shape，维度数需要大于等于1、小于等于8
    util.check_tensor_shape_size(shape_x)  # 校验算子输入shape大小
    util.check_kernel_name(kernel_name)  # 校验算子的kernel_name

    data_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    res = rsqrt_compute(data_x, output_y, kernel_name)
    """
    TODO:
    auto schedule
    """
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    """
    TODO:
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}

    te.lang.cce.cce_build_code(schedule, config)
#文件名命名为rsqrt.py，其中测试用例的shape=(16, 16, 16, 16, 16)，dtype为float16，使用cloud形态。


# 设置DDK版本号
from te.platform import cce_conf
#cce_conf.cceProduct("x.x.x.x.x")
# 算子调用
if __name__ == '__main__':
    #input_output_dict = {"shape": (5, 6, 7), "dtype": "float32"}
    input_output_dict = {"shape": (16, 16, 16, 16, 16), "dtype": "float16"}
    rsqrt(input_output_dict, input_output_dict, kernel_name="rsqrt")
```



### sinh.py （TBE初级课程模式）

```python
#题目二：请用DSL语言实现TBE算子,用于计算双曲正弦函数，即(y = (exp(x) - exp(-x)) / 2)：

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

@fusion_manager.register("sinh")
def sinh_compute(input_x, output_y, kernel_name="sinh"):

    """
    calculating input_x's sinh = (exp(x) - exp(-x)) / 2
    Parameters
    ----------
    shape: Tensor shape
    dtype: input_x type
    name: kernel name, default value is "sinh"
    Returns
    -------
    res: TVM tensor
    result of compute
    """
    exp_val_1 = te.lang.cce.vexp(input_x)
    const_val_1 = tvm.const(-1.0, "float16")
    mul_val = te.lang.cce.vmuls(input_x, const_val_1)
    exp_val_2 = te.lang.cce.vexp(mul_val)
    sub_val = te.lang.cce.vsub(exp_val_1, exp_val_2)
    const_val_2= tvm.const(0.5, "float16")
    res = te.lang.cce.vmuls(sub_val, const_val_2)

    return res


@util.check_input_type(dict, dict, str)
def sinh(input_x, output_y, kernel_name="sinh"):
    # 获取算子输入tensor的shape与dtype
    shape_x = input_x.get("shape")      
    dtype_x = input_x.get("dtype").lower()

    util.check_shape_rule(shape_x)    # 校验算子的shape，维度数需要大于等于1、小于等于8
    util.check_tensor_shape_size(shape_x)  # 校验算子输入shape大小
    util.check_kernel_name(kernel_name)  # 校验算子的kernel_name

    data_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    res = sinh_compute(data_x, output_y, kernel_name)
    """
    TODO:
    auto schedule
    """
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    """
    TODO:
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}

    te.lang.cce.cce_build_code(schedule, config)
#文件名命名为sinh.py，其中测试用例的shape=(16, 16, 16, 16, 16)，dtype为float16，使用cloud形态。


# 设置DDK版本号
from te.platform import cce_conf 
#cce_conf.cceProduct("x.x.x.x.x")
# 算子调用
if __name__ == '__main__':
    input_output_dict = {"shape": (64, 64), "dtype": "float16"}
    sinh(input_output_dict, input_output_dict, kernel_name="sinh")


#文件名命名为sinh.py，其中测试用例的shape=(64, 64)，dtype为float16，使用cloud形态。
#提示：该算子实现时，fp16数据类型丢失精度较多，需要先转成floa32数据类型计算，计算结束后再转回float16数据类型
```



### sinh.py （标准模式）

```python
#题目二：请用DSL语言实现TBE算子,用于计算双曲正弦函数，即(y = (exp(x) - exp(-x)) / 2)：

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


@util.check_input_type(tuple, str, str)
def sinh(shape, dtype, kernel_name="sinh"):
    """
    calculating data's sinh = (exp(x) - exp(-x)) / 2
    Parameters
    ----------
    shape: Tensor shape
    dtype: data type
    name: kernel name, default value is "sinh"
    Returns
    -------
    res: TVM tensor
    result of compute
    """

    util.check_shape_rule(shape)    # 校验算子的shape，维度数需要大于等于1、小于等于8
    util.check_tensor_shape_size(shape)  # 校验算子输入shape大小
    util.check_kernel_name(kernel_name)  # 校验算子的kernel_name

    data = tvm.placeholder(shape, name="data", dtype=dtype)

    if dtype == "float16":
        data_middle = te.lang.cce.cast_to(data, "float32")

    exp_val_1 = te.lang.cce.vexp(data_middle)
    const_val_1 = tvm.const(-1.0, "float16")
    mul_val = te.lang.cce.vmuls(data_middle, const_val_1)
    exp_val_2 = te.lang.cce.vexp(mul_val)
    sub_val = te.lang.cce.vsub(exp_val_1, exp_val_2)
    const_val_2= tvm.const(0.5, "float16")
    res_middle = te.lang.cce.vmuls(sub_val, const_val_2)
    res = te.lang.cce.cast_to(res_middle, "float16")

    """
    TODO:
    auto schedule
    """
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    """
    TODO:
    operator build
    """
    config = {"name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(schedule, config)
#文件名命名为rsqrt.py，其中测试用例的shape=(16, 16, 16, 16, 16)，dtype为float16，使用cloud形态。


# 设置DDK版本号
from te.platform import cce_conf 
#cce_conf.cceProduct("x.x.x.x.x")
# 算子调用
if __name__ == '__main__':
    shape = (16, 16, 16, 16, 16)
    dtype = "float16"
    sinh(shape, dtype,  kernel_name="sinh")


#文件名命名为sinh.py，其中测试用例的shape=(64, 64)，dtype为float16，使用cloud形态。
#提示：该算子实现时，fp16数据类型丢失精度较多，需要先转成floa32数据类型计算，计算结束后再转回float16数据类型
```



## 3.2 算子适配插件实现

### tensorflow_add_plugin.cpp

```python
#include "register/register.h"

namespace domi {

// register add op info to GE
REGISTER_CUSTOM_OP("add")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("add")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
```



## 3.3 算子原型定义

### add.h

```C++
#ifndef GE_OP_OPERATORTYPE_H       //条件编译
#define GE_OP_OPERATORTYPE_H       //进行宏定义

#include "graph/operator_reg.h"

namespace ge {
    REG_OP(Add)
    .INPUT(x1,
    TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
    DT_COMPLEX64, DT_STRING}))
    .INPUT(x2,
    TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
    DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y,
    TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
    DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add)
}

#endif //GE_OPS_OP_PROTO_ADD_H_
```



### add.cpp

```c++
#include "add.h"
#include <string>
#include <vector>

namespace ge {
    bool InferShapeAndTypeAdd(Operator &op, const string &input_name1, const string &input_name2, const string &output_name)
    {
        // vOutputDesc.push_back(op.GetInputDesc(0));
        TensorDesc vOutputDesc = op.GetOutputDesc(output_name);

        DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
        Format input_format = op.GetInputDesc(input_name1).GetFormat();
        // 针对shape维度大小进行交换
        ge::Shape shapeX = op.GetInputDesc(input_name1).GetShape();
        ge::Shape shapeY = op.GetInputDesc(input_name2).GetShape();
        std::vector<int64_t> dimsX = shapeX.GetDims();
        std::vector<int64_t> dimsY = shapeY.GetDims();
        if (dimsX.size() < dimsY.size()) {
            std::vector<int64_t> dimsTmp = dimsX;
            dimsX = dimsY;
            dimsY = dimsTmp;
        }

        // 对小的shape进行1补齐
        if (dimsX.size() != dimsY.size()) {
            int dec = dimsX.size() - dimsY.size();
            for (int i = 0; i < dec; i++) {
                dimsY.insert(dimsY.begin(), (int64_t)1);
            }
        }

        // 设置输出的shape维度
        std::vector<int64_t> dimVec;
        for (size_t i = 0; i < dimsX.size(); i++) {
            if ((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1)) {
                return false;
            }

            int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
            dimVec.push_back(dims);
        }
        ge::Shape outputShape = ge::Shape(dimVec);

        vOutputDesc.SetShape(outputShape);
        vOutputDesc.SetDataType(input_dtype);
        vOutputDesc.SetFormat(input_format);
        op.UpdateOutputDesc(output_name, vOutputDesc);

        return true;
    }

    // ----------------Add-------------------
    IMPLEMT_VERIFIER(Add, AddVerify)
    {
        if (op.GetInputDesc("x1").GetDataType() != op.GetInputDesc("x2").GetDataType()) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(AddInferShape)
    {
        if (InferShapeAndTypeAdd(op, "x1", "x2", "y")) {
            return GRAPH_SUCCESS;
        }
        return GRAPH_FAILED;
    }

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(Add, AddInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(Add, AddVerify);
    // ----------------Add-------------------
} // namespace ge
```



## 3.4 算子信息定义

### add.ini

```
[Add]
input0.name=input_x
input0.shape=all
input0.dtype=float16,float16,float16,float16,float,float,float,float,int32,int32,int32,int32
input0.format=NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND
input0.paramType=required
input1.name=input_y
input1.shape=all
input1.dtype=float16,float16,float16,float16,float,float,float,float,int32,int32,int32,int32
input1.format=NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND
input1.paramType=required
output0.name=output_z
output0.shape=all
output0.dtype=float16,float16,float16,float16,float,float,float,float,int32,int32,int32,int32
output0.format=NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND
output0.paramType=required
opFile.value=add
opInterface.value=add
```



# 4. 单算子验证



## 4.1 UT测试

### test_add.py

```python
import unittest
from impl.add import add

def add_cce(shape_x, shape_y, dtype, kernel_name="add"):
    add({"shape": shape_x, "dtype": dtype}, {"shape": shape_y, "dtype": dtype},
        {"shape": shape_x, "dtype": dtype}, kernel_name=kernel_name)


class Test_add_cce(unittest.TestCase):
    def setUp(self):
        # 每个测试用例执行之前做操作
        pass

    def tearDown(self):
        # 每个测试用例执行之后做操作
        pass

    @classmethod
    def tearDownClass(self):
        # 必须使用 @ classmethod装饰器, 所有test运行完后运行一次
        print("---------------------------------------------------")

    @classmethod
    def setUpClass(self):
        # 必须使用@classmethod 装饰器,所有test运行前运行一次
        print("---------------------------------------------------")

    def test_function_case1(self):
        try:
            add_cce((1,), (1,), "float32",
                    "cce_add_1_float32")
        except RuntimeError:
            pass

    def test_function_case2(self):
        try:
            add_cce((1, 1), (1, 1), "float32",
                    "cce_add_1_1_float32"),
        except RuntimeError:
            pass

    def test_function_case3(self):
        try:
            add_cce((16, 32), (16, 32), "float32",
                    "cce_add_16_32_float32"),
        except RuntimeError:
            pass

    def test_function_case4(self):
        try:
            add_cce((16, 2, 32), (16, 2, 32), "float32",
                    "cce_add_16_2_32_float32"),
        except RuntimeError:
            pass

    def test_function_case5(self):
        try:
            add_cce((16, 2, 4, 32), (16, 2, 4, 32), "float32",
                    "cce_add_16_2_4_32_float32"),
        except RuntimeError:
            pass

    def test_function_case6(self):
        try:
            add_cce((512, 1024), (512, 1024), "float32",
                    "cce_add_512_1024_float32"),
        except RuntimeError:
            pass

    def test_function_case7(self):
        try:
            add_cce((2, 1024), (2, 1024), "float32",
                    "cce_add_2_1024_float32"),
        except RuntimeError:
            pass

    def test_function_case8(self):
        try:
            add_cce((4096, 1024), (4096, 1024), "float32",
                    "cce_add_4096_1024_float32"),
        except RuntimeError:
            pass

    def test_function_case9(self):
        try:
            add_cce((32, 128, 1024), (32, 128, 1024), "float32",
                    "cce_add_32_128_1024_float32"),
        except RuntimeError:
            pass

    def test_function_case10(self):
        try:
            add_cce((100, 100), (100, 100), "float32",
                    "cce_add_100_100_float32"),
        except RuntimeError:
            pass

    def test_function_case11(self):
        try:
            add_cce((9973, 1), (9973, 1), "float32",
                    "cce_add_9973_1_float32_9973_1_float32")
        except RuntimeError:
            pass

    def test_function_case12(self):
        try:
            add_cce((1024, 1024, 256), (1024, 1024, 256), "float32",
                    "cce_add_1024_1024_256_float32_1024_1024_256_float32")
        except RuntimeError:
            pass

    def test_function_case13(self):
        try:
            add_cce((11, 33), (11, 33), "float32",
                    "cce_add_11_33_float32_11_33_float32")
        except RuntimeError:
            pass

    def test_function_failed_case1(self):
        try:
            add_cce((10, 12), (10, 11), "float32",
                    "cce_add_10_12_float32_10_11_float32")
        except RuntimeError:
            pass

    def test_function_failed_case2(self):
        try:
            add_cce((10, 13), (10, 11, 12), "float32",
                    "cce_add_10_13_float32_10_11_13_float32")
        except RuntimeError:
            pass


if __name__ == "__main__":
    unittest.main()
```



## 4.2 ST测试



### test_add_st.py

```python
import unittest
import os
import shutil
from impl.add import add
from run_testcase import run_testcase, get_path_val, print_func_name

testcases_add_aicore = {
    "op_name": "add",
    "all": {
        "test_add_cce_100_100_float32": ((100, 100), (100, 100), "float32", "cce_add_100_100_float32"),
        "test_add_cce_9973_1_float32": ((9973, 1), (9973, 1), "float32", "cce_add_9973_1_float32")
    },
    "mini": {},
    "cloud": {},
}


bin_path_val = get_path_val(testcases)
def test_cce_add(shape_x_val, shape_y_val, dtype_val, kernel_name_val):
    add({"shape": shape_x_val, "dtype": dtype_val}, {"shape": shape_y_val, "dtype": dtype_val},
        {"shape": shape_x_val, "dtype": dtype_val}, kernel_name=kernel_name_val)
    kernel_meta_path = "./kernel_meta/"
    lib_kernel_name = "lib" + kernel_name_val + ".so"

    if (os.path.isfile(kernel_meta_path + lib_kernel_name)):
        shutil.move(kernel_meta_path + lib_kernel_name, bin_path_val + "/" + lib_kernel_name)
    else:
        shutil.move(kernel_meta_path + kernel_name_val + ".o", bin_path_val + "/" + kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json", bin_path_val + "/" + kernel_name_val + ".json")


class Test_add_cce(unittest.TestCase):
    def tearDown(self):
        pass

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @classmethod
    def setUpClass(self):
        pass

    @print_func_name
    def test_cce_add(self):
        run_testcase(testcases, test_add)


if __name__ == "__main__":
    unittest.main()
```



### add_datagen.py

```python
import numpy as np
import sys
from dataFormat import *

def _add_data(name, shape, src_type):
    sys.stdout.write("Info: writing input for %s...\n" % name)
    input_shape = shape

    shape_str = ""
    for dim in shape:
        shape_str += str(dim) + "_"
    feature_name = shape_str + src_type
    #调用np.random（）生成输入数据。
    data_a = np.random.uniform(1, 3, input_shape).astype(src_type)
    #调用dumpData（）将输入数据写入文件。
    dumpData(data_a,
             name + "_input_" + feature_name + ".data",
             fmt="binary", data_type="float32",
             path="../data/" + name + "/" + feature_name) # input array a path: out/bin/llt/ops/common/data

    data_b = np.random.uniform(1, 3, input_shape).astype(src_type)
    dumpData(data_b,
             name + "_input_B_" + feature_name + ".data",
             fmt="binary", data_type="float32",
             path="../data/" + name + "/" + feature_name)# input array b path: out/bin/llt/ops/common/data
   #定义输出数据的变量
    in_tensor0 = data_a + data_b
    dumpData(in_tensor0,
             name + "_output_" + feature_name + ".data",
             fmt="binary", data_type="float32",
             path="../data/" + name + "/" + feature_name)# output path: out/bin/llt/ops/common/data
    sys.stdout.write("Info: writing output for %s done!!!\n" % name)




def add(name, shape, src_type):
    sys.stdout.write("Info: writing input for %s...\n" % name)
    """
    TODO:
    write codes for generating data.
    """
    sys.stdout.write("Info: writing output for %s done!!!\n" % name)


def gen_add_data():
    _add_data("add", (100, 100), "float32")
    _add_data("add", (9973, 1), "float32")


if __name__ == "__main__":
    gen_add_data()
```



### add_st.cc

```c++
#include "gtest/gtest.h"
#include "two_in_one_out_layer.hpp"//调用two_in_one_out_layer.hpp

class ADD_ST : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ADD_ST ST SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ADD_ST ST TearDown" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

//该测试用例中入参"test_add_cce_100_100_float32"需要和"test_add_st.py"中的"testcases_add_aicore"
//定义保持一致。
TEST_F(ADD_ST, test_add_cce_100_100_float32) {
    std::string op_name = "add";
	std::string input_size_str = "100_100_float32";
	uint32_t input_size = 100*100;
	uint32_t input_b_size = 100*100;
    uint32_t output_size = 100*100;
//stub_Func、tiling_name的value需要保持一致。加粗部分需要和kernel_name的取值保持一致。
	std::string stub_Func =  "cce_add_100_100_float32__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/add/cce_add_100_100_float32.o";
	
	std::string tiling_name = "cce_add_100_100_float32__kernel0";
	
	std::string input_array_a_path = "./llt/ops/common/data/add/100_100_float32/add_input_100_100_float32.data";
    std::string input_array_b_path = "./llt/ops/common/data/add/100_100_float32/add_input_B_100_100_float32.data";

	std::string expect_output_data_path = "./llt/ops/common/data/add/100_100_float32/add_output_100_100_float32.data";
   //算子结果对比误差在双千分之一以内，则代表算子的逻辑正确。
    float ratios[2] = {0.001 ,0.001};

	TwoInOneOutLayer<float,float> layer{
		op_name,
		input_size_str,
		input_size,
		input_b_size,
		output_size,
		bin_path,
		tiling_name,
		input_array_a_path,
		input_array_b_path,
		expect_output_data_path,
		ratios,
		(void*)stub_Func.c_str()
	};

	bool ret = layer.test();
    //当数据对比没有通过时，会将实际得到的数据存到"actual_add_output_100_100_float32.data"文件。
    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/add/100_100_float32/actual_add_output_100_100_float32.data",
        output_size * sizeof(float));
    }

	assert(true == ret);
}
```



## 4.3 BBIT测试

### add_test.hpp

```c++
//调用BBIT用到的头文件。
#include "tvm_bbit.hpp"
#include "register.hpp"

class AddTest : public BaseBbitTest{
public:
    AddTest(){
        testcases.push_back("add_test_100_100_float32");
        testcases.push_back("add_test_9973_1_float32");
    };

    virtual ~AddTest(){};
    bool test(string name);
    bool add_test_100_100_float32();
    bool add_test_9973_1_float32();
};

REGISTER_CLASS(AddTest)
```



### add_test.cc

```C++
//调用add_test.cc需要的头文件。
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include "log.hpp"
#include "./add_test.hpp"
#include "two_in_one_out_layer.hpp"
using namespace std;


/*
* op: add
* input_shape: (100,100)
* output_shape: (100,100)
* stype: float32
* dtype: float32
*/
bool AddTest::add_test_100_100_float32()
{
    std::string op_name = "add";
    std::string input_size_str = "100_100_float32";
    uint32_t input_size = 100*100;
    uint32_t input_b_size = 100*100;
    uint32_t output_size = 100*100;

    const char* stub_func =  "cce_add_100_100_float32__kernel0";

    std::string bin_path = "./llt/ops/common/kernel_bin/add/cce_add_100_100_float32";

    std::string tiling_name = "cce_add_100_100_float32__kernel0";

    std::string input_array_a_path = "./llt/ops/common/data/add/100_100_float32/add_input_100_100_float32.data";
    std::string input_array_b_path = "./llt/ops/common/data/add/100_100_float32/add_input_B_100_100_float32.data";

    std::string expect_output_data_path = "./llt/ops/common/data/add/100_100_float32/add_output_100_100_float32.data";
    float ratios[2] = {0.001 ,0.001};

    TwoInOneOutLayer<float,float,float> layer{
        op_name,
        input_size_str,
        input_size,
        input_b_size,
        output_size,
        bin_path,
        tiling_name,
        input_array_a_path,
        input_array_b_path,
        expect_output_data_path,
        ratios,
        (void*)stub_func,
        false
    };

    return layer.test();
}

/*
* op: add
* input_shape: (9973, 1)
* output_shape: (9973, 1)
* stype: float32
* dtype: float32
*/
bool AddTest::add_test_9973_1_float32()
{
    std::string op_name = "add";
    std::string input_size_str = "9973_1_float32";
    uint32_t input_size = 9973*1;
    uint32_t input_b_size = 9973*1;
    uint32_t output_size = 9973*1;

    const char* stub_func =  "cce_add_9973_1_float32__kernel0";

    std::string bin_path = "./llt/ops/common/kernel_bin/add/cce_add_9973_1_float32";

    std::string tiling_name = "cce_add_9973_1_float32__kernel0";

    std::string input_array_a_path = "./llt/ops/common/data/add/9973_1_float32/add_input_9973_1_float32.data";
    std::string input_array_b_path = "./llt/ops/common/data/add/9973_1_float32/add_input_B_9973_1_float32.data";

    std::string expect_output_data_path = "./llt/ops/common/data/add/9973_1_float32/add_output_9973_1_float32.data";
    float ratios[2] = {0.001 ,0.001};

    TwoInOneOutLayer<float,float,float> layer{
        op_name,
        input_size_str,
        input_size,
        input_b_size,
        output_size,
        bin_path,
        tiling_name,
        input_array_a_path,
        input_array_b_path,
        expect_output_data_path,
        ratios,
        (void*)stub_func,
        false
    };

    return layer.test();
}

bool AddTest::test(string name)
{
    TVM_LOG(CC_LOG_INFO, "TVM add BBIT begin.");

    bool ret = false;

    if("all" == name)
    {
        ret = add_test_100_100_float32();
        if (!ret) {
            TVM_LOG(CC_LOG_ERROR, "add_test_100_100_float32 falied");
            return false;
        }
        ret = add_test_9973_1_float32();
        if (!ret) {
            TVM_LOG(CC_LOG_ERROR, "add_test_9973_1_float32 falied");
            return false;
        }
    }
    else if("add_test_100_100_float32" == name)
    {
        ret = add_test_100_100_float32();
        if (!ret) {
            TVM_LOG(CC_LOG_ERROR, "add_test_100_100_float32 falied");
            return false;
        }
    }
    else if("add_test_9973_1_float32" == name)
    {
        ret = add_test_9973_1_float32();
        if (!ret) {
            TVM_LOG(CC_LOG_ERROR, "add_test_9973_1_float32 falied");
            return false;
        }
    }
    else
    {
        TVM_LOG(CC_LOG_INFO, "The case[%s] not exist", name.c_str());
        return false;
    }
    TVM_LOG(CC_LOG_INFO, "TVM add BBIT end.");
    return true;
```



# 5. 网络运行验证



## 5.1 推理

###  op_list.txt

```
[
  {
    "op": "Add",
    "attr": [],
    "input_desc": [
      {
        "format": "ND",
        "shape": [8, 16],
        "type": "int32"
      },
      {
        "format": "ND",
        "shape": [8, 16],
        "type": "int32"
      }
    ],
    "output_desc": [
      {
        "format": "ND",
        "shape": [8, 16],
        "type": "int32"
      }
    ]
  }
]
```



### main.cpp

```c++
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;


OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> shape{8, 16};    //构造的算子输入的shape
    std::string opType = "Add";           //算子类型
    aclDataType dataType = ACL_INT32;     //构造的测试数据的数据类型
    aclFormat format = ACL_FORMAT_ND;     //测试数据的排布格式
    OperatorDesc opDesc(opType);
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);  //算子输入
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);  //算子输入
    opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format); //算子输出
    return opDesc;
}


bool SetInputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumInputs(); ++i) {
        size_t fileSize;
        std::string filePath = "test_data/data/input_" + std::to_string(i) + ".bin";
        char *fileData = ReadFile(filePath, fileSize,
        runner.GetInputBuffer<void>(i), runner.GetInputSize(i));
        if (fileData == nullptr) {
            ERROR_LOG("Read input[%zu] failed", i);
            return false;
        }

        INFO_LOG("Set input[%zu] from %s success.", i, filePath.c_str());
        INFO_LOG("Input[%zu]:", i);
        runner.PrintInput(i);
    }

    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumOutputs(); ++i) {
        INFO_LOG("Output[%zu]:", i);
        runner.PrintOutput(i);

        std::string filePath = "result_files/output_" + std::to_string(i) + ".bin";
        if (!WriteFile(filePath, runner.GetOutputBuffer<void>(i), runner.GetOutputSize(i))) {
            ERROR_LOG("Write output[%zu] failed.", i);
            return false;
        }

        INFO_LOG("Write output[%zu] success. output file = %s", i, filePath.c_str());
    }
    return true;
}

bool RunAddOp()
{
    // [TODO] create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // [TODO] create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // [TODO] Load inputs
    if (!SetInputData(opRunner)) {
        return false;
    }

    // [TODO] Run op
    if (!opRunner.RunOp()) {
        return false;
    }

    // [TODO] process output data
    if (!ProcessOutputData(opRunner)) {
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}


int main()
{
    //设置输出文件路径
    std::string output = "./result_files";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("make output directory successfully");
        }
        else {
            ERROR_LOG("make output directory fail");
            return FAILED;
        }
    }
    //初始化ACL
    if (aclInit("test_data/config/acl.json") != ACL_ERROR_NONE) {
        ERROR_LOG("Init acl failed");
        return FAILED;
    }

    int deviceId = 0;      //设置需要运行的设备ID
    //设置模型文件的相对路径，op_models表示在当前工程的out/op_models目录下生成模型文件
    if (aclopSetModelDir("op_models") != ACL_ERROR_NONE) {
        std::cerr << "Load single op model failed" << std::endl;
        return FAILED;
    }
    //指定用于运算的Device
    if (aclrtSetDevice(deviceId) != ACL_ERROR_NONE) {
        std::cerr << "Open device failed. device id = " << deviceId << std::endl;
        return FAILED;
    }
    INFO_LOG("Open device[%d] success", deviceId);
    //获取昇腾AI处理器运行模式
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    //执行算子
    if (!RunAddOp()) {
        (void) aclrtResetDevice(deviceId);
        return FAILED;
    }
    //复位当前运算的Device
    (void) aclrtResetDevice(deviceId);
    return SUCCESS;
}
```



### generate_date.py

```python
import numpy as np

# [TODO] generate input data by user.
a = np.random.randint(100, size=(8, 16)).astype(np.int32)
b = np.random.randint(100, size=(8, 16)).astype(np.int32)

a.tofile('input_0.bin')
b.tofile('input_1.bin')
```



### CMakeLists.txt

```
# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# target
set(CUSTOM_OBJECT_NAME "main")

# project information
project(acl_op)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../../outputs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../outputs")
set(CMAKE_INSTALL_PREFIX "../../../run")
set(CMAKE_OUTPUT_DIR "out")

# Header path
if ("x${ACLLIB_ROOT_PATH}" STREQUAL "x")
    set(ACLLIB_ROOT_PATH $ENV{DDK_PATH})
endif()

message(STATUS "ACLLIB_ROOT_PATH=${ACLLIB_ROOT_PATH}")
include_directories(
        ${ACLLIB_ROOT_PATH}/acllib/include
        ../inc
)

# add host lib path
link_directories(
        ${ACLLIB_ROOT_PATH}/acllib/lib64/stub
)

add_executable(${CUSTOM_OBJECT_NAME}
        operator_desc.cpp
        op_runner.cpp
        main.cpp
        common.cpp)

target_link_libraries(${CUSTOM_OBJECT_NAME}
        ascendcl
        stdc++)

install(TARGETS ${CUSTOM_OBJECT_NAME} DESTINATION ${CMAKE_OUTPUT_DIR})

# custom command process om conversion
# [TODO] Please uncomment the following part when your project is ready to convert single-op om.

message(STATUS "SOC_VERSION=${SOC_VERSION}")
set(ASCEND_HOME $ENV{ADK_PATH})
add_custom_command(TARGET ${CUSTOM_OBJECT_NAME}
    POST_BUILD
    COMMAND export LD_LIBRARY_PATH="${ASCEND_HOME}/atc/python/site-packages/te.egg/lib:${ASCEND_HOME}/atc/lib64" &&
        export PATH="${PYTHON3_BIN}:${ASCEND_HOME}/atc/ccec_compiler/bin:${ASCEND_HOME}/atc/bin:$ENV{PATH}" &&
        export ASCEND_OPP_PATH="${ASCEND_HOME}/opp" &&
        export PYTHONPATH="$ENV{PYTHONPATH}:${ASCEND_HOME}/atc/python/site-packages/te.egg:${ASCEND_HOME}/atc/python/site-packages/topi.egg" &&
        ${ASCEND_HOME}/atc/bin/atc --singleop=$ENV{PROJECT_PATH}/tbe/testcases/acl_op/run/out/test_data/config/op_list.txt --soc_version=${SOC_VERSION} --output=$ENV{PROJECT_PATH}/tbe/testcases/acl_op/run/out/op_models> atc.log
    USES_TERMINAL
    COMMENT "single-op om conversion, if got failing log below, please check build/intermediates/acl_op/atc.log for detail."
)
```



## 5.2 训练

### tf_add.py

```python
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""

import tensorflow as tf
import numpy as np
import os

import logging            # Python标准库日志模块
import tensorflow as tf   # 导入Tensorflow开源库
from npu_bridge.estimator import npu_ops   # 导入Tensorflow开源库中的npu_ops模块
import numpy as np    # 导入Python的数学基础库
from npu_bridge.estimator import npu_ops

tf.flags.DEFINE_string("local_log_dir", "output/train_logs.txt", "Log file path")
FLAGS = tf.flags.FLAGS

#np.allclose比较函数的相对公差参数
atol = 0.001
#np.allclose比较函数的绝对公差参数
rtol = 0.001


def config(excute_type):
    if excute_type == 'ai_core':
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True   # 开启数据预处理下沉到Device侧执行
        custom_op.parameter_map["mix_compile_mode"].b = True    
        custom_op.parameter_map["use_off_line"].b = True     # True表示在昇腾AI处理器上执行训练
        
    elif excute_type == 'cpu':
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

    return session_config

def main(unused_argv):
    shape_params = (2, 2, 2)
    dtype_params = "float16"

    # 构造Add算子的两个输入数据,shape为shape_params，范围在[-2,2]之间的随机数
    x_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)
    y_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)
    # 分别对Add算子的两个输入数据进行占位
    x = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    y = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    # 计算算子输出
    out = tf.math.add(x, y)
    # 在Host侧CPU上运行单算子，得到期望运行结果
    with tf.compat.v1.Session(config=config('cpu')) as session:
        result_cpu = session.run(out, feed_dict={x: x_data, y: y_data})
    # 在昇腾AI处理器上运行单算子，得到实际运行结果
    with tf.compat.v1.Session(config=config('ai_core')) as session:
        result_ai_core = session.run(out, feed_dict={x: x_data, y: y_data})

    np.array(result_ai_core).astype(dtype_params)
    np.array(result_cpu).astype(dtype_params)
    print('====================================')
   # 通过np.allclose比较昇腾AI处理器上运行的实际结果和cpu上运行的期望结果，其中atol和rtol为np.allclose比较函数的相对公差参数和绝对公差参数，请见步骤3设置。
    cmp_result = np.allclose(result_ai_core, result_cpu, atol, rtol)
    print(cmp_result)
    print('====================================')

if __name__ == "__main__":
    tf.app.run()
```



# 6. 参考链接

[关于Mind Studio](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0004.html)

[自定义算子开发](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0026.html)



# 7. 版本日志

#### 2020.10.26 -v1

* 第一版





























