# 农机租赁及农产品交易推荐系统
## 项目简介
该仓库为农机租赁及农产品交易推荐系统的算法部分，包括：
- 农产品内容理解系统
- 农机租赁推荐系统
- 农产品交易推荐系统

农产品内容理解系统使用的模型为：
- Ghost-EfficientNetV2-S

内容理解系统包含：
- 数据预处理模块
- 模型训练模块
- 模型测试及筛选模块
- 模型线上部署模块

农机租赁推荐系统使用的算法为：
- 特征工程算法
- 召回部分：二部图召回算法
- 排序部分：LambdaMART算法

农产品交易推荐系统使用的算法为：
- 特征工程算法
- 召回部分：itemcf、二部图、word2vec算法、多路召回合并算法
- 排序部分：GOSS算法、MART算法、bagging融合模型、stacking融合模型

## 代码部分介绍
本项目于2023年2月22日在colab平台测试成功，各个子目录的功能如下：  
.  
├─README.md  
├─crop_identification/：农产品内容理解算法工程目录  
│&emsp; ├─models/：模型定义目录  
│&emsp; ├─log/：模型训练日志目录  
│&emsp; ├─class_dict/：类别真实名称与编号的对照表  
│&emsp; ├─checkpoints/：训练参数目录  
│&emsp; ├─best_checkpoints/：最优训练参数目录  
│&emsp; ├─utils.py：一些数据流工具  
│&emsp; ├─train.ipynb：模型训练  
│&emsp; ├─test.ipynb：模型测试  
│&emsp; ├─test_result.csv：模型测试结果汇总  
│&emsp; ├─tensorboardlauncher.ipynb：tensorboard启动器  
│&emsp; ├─plot.ipynb：结果可视化  
│&emsp; ├─model_builder.py：建立模型所需代码  
│&emsp; ├─debugtools.py：数据流调试工具  
│&emsp; ├─data.rar：数据压缩包  
│&emsp; └─config.py：超参数设定文件  
└─reco/：特征工程、召回与排序算法工程目录  
&emsp; ├─veg-fru-reco/：农产品交易推荐系统工程目录  
&emsp; │&emsp; ├─01data_preprocess.ipynb：数据预处理  
&emsp; │&emsp; ├─02recall_feature：召回特征工程  
&emsp; │&emsp; ├─03recall.ipynb：召回算法  
&emsp; │&emsp; ├─04ranking_feature.ipynb：排序特征工程  
&emsp; │&emsp; ├─05ranking.ipynb：排序算法  
&emsp; │&emsp; ├─cache/：代码运行过程中产出的中间文件目录  
&emsp; │&emsp; └─data/：原始数据目录  
&emsp; └─agri-machine-reco/：农机租赁推荐系统工程目录  
&emsp; &emsp; ├─01data_preprocess.ipynb：数据预处理  
&emsp; &emsp; ├─02recall_feature：召回特征工程  
&emsp; &emsp; ├─03recall.ipynb：召回算法  
&emsp; &emsp; ├─04ranking_feature.ipynb：排序特征工程  
&emsp; &emsp; ├─05ranking.ipynb：排序算法  
&emsp; &emsp; ├─cache/：代码运行过程中产出的中间文件目录  
&emsp; &emsp; └─data/：原始数据目录  

其中各个ipynb为colab笔记本文件，将项目目录上传至Google drive后使用colab即可按照编号顺序运行。

> **备注1：如果你发现代码复现的指标与论文中的存在差异，可能的原因是**：
>  1. 训练、测试集的划分具有随机性，可能会对模型性能造成影响
>  2. 模型的参数初始化及训练过程具有随机性，可能会对指标造成影响  

> **备注2：如果在colab运行代码报错，可能的原因是：**
>  1. colab平台的代码执行环境是实时更新的，可能当前版本过新导致api改变，无法与项目创建时的环境兼容
>  2. colab是付费平台，免费用户拥有的资源十分有限，可能不足以运行这段代码。
>  3. 项目中使用到的deepctr是基于tensorflow2.1版本开发的，对更高版本的tensorflow可能不兼容
>  4. 您的目录可能没有配置正确
>  5. 农产品交易项目的数据量较大，单机运行速度十分缓慢，而colab对免费用户有使用时常限制，可能不足以运行这段代码。
>  6. 农产品内容理解算法需要GPU运行，如果您的colab账号没有GPU资源，可能无法运行或运行极慢。

> **备注3：目前已知colab平台的一些坑：**
>  1. 如果代码时间运行太久，colab的相对目录可能会失效，请仿照reco/veg-fru-reco/目录下04、05笔记本的做法配置为绝对目录。
>  2. 目前已知tensorflow2.2.0版本存在内存泄漏会OOM，请避开该版本，可以尝试2.3.0版本。

