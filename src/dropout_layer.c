#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

/*
** 构建dropout层
** 输入： batch         一个batch中含有的图片张数（等于net.batch）
**       inputs        一张输入图片中的元素个数（等net.inputs）
**       probability   dropout概率，即某个输入神经元被丢弃的概率，由配置文件指定，如果配置文件中未指定，则默认值为0.5（间parse_dropout_layer()函数）
** 返回： dropout_layer
** 碎碎念： 相比很多网络层的构建，dropout层的构建函数需要的输入参数比较少，网络输入数据尺寸h,w,c也不需要；
**         注意dropout层有l.inputs = l.outputs；另外此处实现的是inverted dropout，不是标准的dropout
*/
dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;            // 舍弃概率（1-probability为保留概率）
    l.inputs = inputs;                      // dropout层不会改变输入输出的个数，因此有l.inputs==l.outputs
    l.outputs = inputs;                     // 虽然dropout会丢弃一些输入神经元，但这丢弃只是置该输入元素值为0,并没有删除
    l.batch = batch;                   
    l.rand = calloc(inputs*batch, sizeof(float));   // 动态分配内存，详细见layer.h中的注释
    l.scale = 1./(1.-probability);          // 使用inverted dropout，比例因子取为保留概率的倒数，关于这个顺便在网上找个关于dropout的博客都会讲到
    
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

/*
** 重新配置dropout的相关参数，主要修改的是l.rand的尺寸，在cpu中，可以直接用realloc重新动态分配内存，
** 而在GPU中，得先用cuda_free释放原来的内存，而后重新用cuda_make_arry分配
** 输入： l     dropout网络层指针
**       inputs    dropout层新输入元素个数值（单张输入图片）
*/
void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}

/*
** dropout层前向传播函数
** 输入： l    当前dropout层网络
**       net  整个网络
** 说明：dropout层同样没有训练参数，因此前向传播比较简单，只完成一个事：按指定概率l.probability，
**      丢弃输入元素，并将保留下来的输入元素乘以比例因子（采用的是inverted dropout，这种方式实现更为方便，
**      且代码接口比较统一，想想看，如果采用标准的droput，则测试阶段还需要进入forward_dropout_layer()，
**      使每个输入乘以保留概率，而使用inverted dropout，测试阶段根本就不需要进入到forward_dropout_layer）。
** 说明2：dropout层输入与输出元素个数相同（即l.intputs=l.outputs）
** 说明3：关于inverted dropout，在网上随便搜索关于dropout的博客，都会讲到，这里给一个博客链接：https://yq.aliyun.com/articles/68901
*/
void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    // 如果当前网络不是处于训练阶段而处于测试阶段，则直接返回（使用inverted dropout带来的方便）
    if (!net.train) return;

    // 遍历dropout层的每一个输入元素（包含整个batch的），按照指定的概率l.probability置为0或者按l.scale缩放
    for(i = 0; i < l.batch * l.inputs; ++i){
        // 产生一个0~1之间均匀分布的随机数
        float r = rand_uniform(0, 1);

        // 每个输入元素都对应一个随机数，保存在l.rand中
        l.rand[i] = r;

        // 如果r小于l.probability（l.probability是舍弃概率），则舍弃该输入元素，注意，舍弃并不是删除，
        // 而是将其值置为0,所以输入元素个数总数没变（因故输出元素个数l.outputs等于l.inputs）
        if(r < l.probability) net.input[i] = 0;
        // 否则保留该输入元素，并乘以比例因子
        else net.input[i] *= l.scale;
    }
}

/*
** dropout层反向传播函数
** 输入： l    当前dropout层网络
**       net  整个网络
** 说明：dropout层的反向传播相对简单，因为其本身没有训练参数，也没有激活函数，或者说激活函数就为f(x) = x，也
**      也就是激活函数关于加权输入的导数值为1,因此其自身的敏感度值已经由其下一层网络反向传播时计算完了，
**      没有必要再乘以激活函数关于加权输入的导数了。剩下要做的就是计算上一层的敏感度图net.delta，这个计算也很简单，详见下面注释。
*/
void backward_dropout_layer(dropout_layer l, network net)
{
    // 如果进入backward_dropout_layer()函数，那没得说，一定是训练阶段，因为测试阶段压根就没有反向过程，只有前向过程，
    // 所以就不再需要像forward_dropout_layer()函数中一样判断net.train是不是处于训练阶段了
    int i;

    // 如果net.delta为空，则返回（net.delta为空则说明已经反向到第一层了，此处所指第一层，是net.layers[0]，
    // 也是与输入层直接相连的第一层隐含层，详细参见：network.c中的forward_network()函数）
    if(!net.delta) return;

    // 因为dropout层的输入输出元素个数相等，所以dropout层的敏感度图的维度就为l.batch*l.inputs（每一层的敏感度值与该层的输出维度一致），
    // 以下循环遍历当前层的敏感度图，并根据l.rand的指示反向计算上一层的敏感度值，由于当前dropout层与上一层之间的连接没有权重，
    // 或者说连接权重为0（对于舍弃的输入）或固定的l.scale（保留的输入，这个比例因子是固定的，不需要训练），所以计算过程比较简单，
    // 只需让保留输入对应输出的敏感度值乘以l.scale，其他输入（输入是针对当前dropout层而言，实际为上一层的输出）的敏感度值直接置为0即可
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        // 与前向过程forward_dropout_layer照应，根据l.rand指示，如果r小于l.probability，说明是舍弃的输入，其敏感度值为0；
        // 反之是保留下来的输入元素，其敏感度值为当前层对应输出的敏感度值乘以l.scale
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}


// 显然，dropout层没有训练参数，故而，也没有update_dropout_layer()函数
