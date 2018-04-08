# darknet
darknet是一个较为轻型的完全基于C与CUDA的开源深度学习框架，其主要特点就是容易安装，没有任何依赖项（OpenCV都可以不用），移植性非常好，支持CPU与GPU两种计算方式。

更多信息（包括安装、使用）可以参考：[Darknet: Open Source Neural Networks in C](https://pjreddie.com/darknet/)

# 为什么选择darknet？

相比于TensorFlow来说，darknet并没有那么强大，但这也成了darknet的优势：

1. darknet完全由C语言实现，没有任何依赖项，当然可以使用OpenCV，但只是用其来显示图片、为了更好的可视化；

2. darknet支持CPU（所以没有GPU也不用紧的）与GPU（CUDA/cuDNN，使用GPU当然更块更好了）；

3. 正是因为其较为轻型，没有像TensorFlow那般强大的API，所以给我的感觉就是有另一种味道的灵活性，适合用来研究底层，可以更为方便的从底层对其进行改进与扩展；

4. darknet的实现与caffe的实现存在相似的地方，熟悉了darknet，相信对上手caffe有帮助；

# 本项目目的与状态

目的很简单，研究darknet底层，窥探深度学习框架原理与具体实现，同时巩固C语言编程（所以注释中不单有很多的框架原理/逻辑分析，也有很多语法分析）。目前只完成部分代码（主要是卷积神经网络）的分析，其注释非常详细（可能很多人会觉得罗嗦了：)，那就强忍着吧～），未来会不定期的更新!（忙ing，什么时候会再更呢？）

很希望有相同兴趣的人加入我，一起研究（若有兴趣，欢迎给我发邮件～～）！

# 小小声明

注释中有些地方提及了参考什么什么的，这些多半是指我所作的图表+文字用来帮助理解代码的笔记，原谅我这些笔记还躺在我的电脑里，并没有上传，但不用紧，因为注释真的真的很详细，基本上不用图表也说明清楚了～～

大部分的代码都是本人个人完成的（lonely...），所以难免会有理解错误的地方（可能还不少，害怕ing...），还请多多包涵，若发现与您理解相左的地方，欢迎发邮件给我～～

# 两三点小说明

1. src文件夹中凡是.cu文件，都被我改为.c结尾了（为了一点点方便～），替换之前的文件全被我放在了src/cu文件夹中（没啥用，可以随便删掉～）。如果你没用gpu的话，这没有任何影响，因为没有用gpu就不会用到nvcc编译；但是如果你用gpu的话，还得麻烦你将.c改回.cu，不然编译会出问题的（你可以看一下Makefile文件，.cu文件要用nvcc编译的，要改为.c那就通通用gcc编译了～～）

2. 如果你也愿意解析代码，为其写注释，也可以pull requests给我，我来merge（注释风格如果能够保持一致就最好了～～）

3. Contributors:
    * [Goffic](https://github.com/Goffic)： 为rnn_layer.c添加了注释
    * [LamHoCN](https://github.com/LamHoCN)： 修改了一些注释，并提供了一些yolo的应用代码（已push到extension分支中）

# 疑惑求解

1. 始终不明白softmax_layer层反向传播函数backward_softmax_layer()中为什么不用对softmax函数求导？

2. region_layer层前向函数forward_region_layer()在求l.output过程中（也就是第一次使用activate_array函数），为什么只对x,y进行了logistic激活函数处理，而没有对w,h处理（也就是activate_array()函数的第二个参数为什么是2*l.w*l.h，而不是4*l.w*l.h）？还有，region_layer这一层没有训练参数吗？

3. 另外，就是region_layer中cost和delta的计算了，感觉并没有弄懂？
