#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/*
** 在darknet中，softmax_layer一般作为网络的倒数第二层（因为darknet中把cost也算作一层，
** 一般cost_layer作为最后一层），可以参见vgg以及alexnet的网络配置文件（vgg-16.cfg,alexnet.cfg）。
** softmax_layer本身也没有训练参数，所以比较简单，只是darknet中的实现似乎引入了一些不太常见的东西，导致有些地方理解上比较费劲。
** softmax_layer构建函数
** 输入： batch    
**       intputs
**       groups    
** 注意：此处的softmax_layer层，单张图片的输入元素个数l.inputs等于输出元素个数l.outputs（总输入元素与总输出元素个数也将相同），
*/
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    // 断言：inputs与groups之间必须是整除关系，不然就会出错（有些元素访问不到或者访问索引的偏移将会出问题）
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;      // softmax_layer的输入输出元素相同
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

/*
** softmax层前向传播函数
** 输入： l   当前softmax层
**       net 整个网络
** 说明：softmax层的前向比较简单，只需要对输入中的每个元素做softmax处理就可以，但是darknet的实现引入了softmax_tree，
**      这个参数的用法尚需要去推敲。
*/
void forward_softmax_layer(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        // 调用softmax_cpu()对输入的每一个元素进行softmax处理
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }
}

/*
** softmax层反向传播函数
** 输入： l   当前softmax层
**       net 整个网络
** 说明：
//      下面注释理解应该有错，另外，对此处softmax反向的实现存在很大的疑问？？？？？？？
**      softmax层的反向很简单，由于自身没有训练参数，虽然有激活函数（即softmax函数），但是又因其所处位置特殊，一般处于网络导数第二层，
**      下一层就是cost层，
**      其自身的敏感度图l.delta已经计算完成（如果此处不太明白再说什么，可以参看卷积层的注释，或者全连接层的注释，以及最大池化层的注释），
//
**      剩下要做的仅剩下利用自身的l.delta计算其上一层的敏感度图（完成大部分计算，还差乘以上一层激活函数关于其加权输入的导数值），
**      即将l.delta中的元素乘以对应的当前层与上一次层之间的权重，而softmax层与上一层输出之间没有权重或者说权重都为1
**      （因为是将输入直接送进softmax函数处理的，并没有加权什么的），且softmax的输出与上一层的输出存在一一对应的关系，
**      所以求取上一层的敏感度图也是很简单，很直接，详见下面的注释。
*/
void backward_softmax_layer(const softmax_layer l, network net)
{
    // 由当前softmax层的敏感度图l.delta计算上一层的敏感度图net.delta，调用的函数为axpy_cpu()，
    // 为什么调用axpy_cpu()函数，因为softmax层的输出元素与上一层的输出元素存在一一对应的关系（由此可以看出softmax的stride取值为1是必然的，
    // 再次照应blas.c中softmax_cpu()的注释，如果不为1,肯定不能是一一对应关系），所以由softmax层的敏感度图计算上一层的敏感度图，
    // 可以逐个逐个元素计算，不需要类似全连接层中的矩阵乘法，更没有卷积层中的那般复杂。
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
            count += group_size;
        }
    } else {
        softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
    }
}

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
