# 项目简介：
使用YOLOX+Deepsort实现车辆行人追踪和计数，代码封装成一个Detector类，更容易嵌入到自己的项目中。

代码地址（欢迎star）：

[https://github.com/Sharpiless/yolox-deepsort/](https://github.com/Sharpiless/yolox-deepsort/)

最终效果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/7768e8e4cf0a4bbf97bb10ab56ea028c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

# 运行demo：

```bash
python demo.py
```

# 训练自己的模型：


训练好后放到 weights 文件夹下

# 调用接口：

## 创建检测器：

```python
from AIDetector_pytorch import Detector

det = Detector()
```

## 调用检测接口：

```python
result = det.feedCap(im)
```

其中 im 为 BGR 图像

返回的 result 是字典，result['frame'] 返回可视化后的图像

# 联系作者：

> B站：[https://space.bilibili.com/470550823](https://space.bilibili.com/470550823)

> CSDN：[https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)

> AI Studio：[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)

> Github：[https://github.com/Sharpiless](https://github.com/Sharpiless)

