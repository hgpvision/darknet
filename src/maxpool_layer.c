#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

/*
** 以image格式获取最大池化层输出l.output：将l.output简单打包以image格式返回（并没有改变什么值）
** 输入： l     最大池化层
** 返回： image数据类型
** 说明：本函数其实没有做什么实质性操作，就是将数据简单的打包了一下，float_to_image()函数中，创建了一个image类型数据out，指定了每张图片的
**      宽、高、通道数为最大池化层输出图的宽、高、通道数，然后简单的将out.data置为l.output，这样简单的打包也许是为了更方便的访问吧～
*/
image get_maxpool_image(maxpool_layer l)
{
    // 获取最大池化层输出图片的高度，宽度，通道数
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;        // 对于最大池化层，l.c == l.out_c，也即输入通道数等于输出通道数（因为没有卷积核。。。）
    return float_to_image(w,h,c,l.output);
}

/*
** 以image格式获取最大池化层输出敏感度图l.delta，与上面get_maxpool_image差不多
*/
image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

/*
** 构建最大池化层
** 输入： batch     该层输入中一个batch所含有的图片张数，等于net.batch
**       h,w,c     该层输入图片的高度（行），宽度（列）与通道数
**       size      池化核尺寸
**       stride    跨度
**       padding   四周补0长度
** 返回： 最大池化层l
** 说明：最大池化层与卷积层比较类似，所以有较多的变量可以类比卷积层参数，比如池化核，池化核尺寸，跨度，补0长度等等
*/
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;         // 网络类型为最大池化层
    l.batch = batch;          // 一个batch中含有的图片张数（等于net.batch）
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;

    // 由最大池化层的输入图像尺寸以及跨度计算输出图像尺寸，目前对以下公式存在疑问，公式中居然没有包含池化核尺寸l.size的信息，而且这计算方式与
    // 卷积层中convolutional_out_width()的计算方式不同，也与http://blog.csdn.net/u013989576/article/details/70154421提到的caffe的计算方式不同，
    // 总之，目前对此公式持强烈怀疑态度！！！
    l.out_w = (w + 2*padding)/stride;
    l.out_h = (h + 2*padding)/stride;

    l.out_c = c;              // 最大池化层输出图像的通道数等于输入图像的通道数（因为连卷积核都没有～）
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;            // 池化核尺寸
    l.stride = stride;        // 跨度
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;          // 前向函数
    l.backward = backward_maxpool_layer;        // 反向函数
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

/*
** 重新设置最大池化层的各种尺寸，包括输入输出图的尺寸（当然输出尺寸根据新设置的输入尺寸以及跨度计算得到）以及相关变量的尺寸
*/
void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + 2*l->pad)/l->stride;
    l->out_h = (h + 2*l->pad)/l->stride;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

/*
** 最大池化层前向传播函数：计算l层的输出
** 输入： l    当前层（最大池化层）
**       net  整个网络结构
** 说明：最大池化层处理图像的方式与卷积层类似，也是将最大池化核在图像平面上按照指定的跨度移动，并取对应池化核区域中最大元素值为对应输出元素。
**      最大池化层没有训练参数（没有权重以及偏置），因此，相对与卷积来说，其前向（以及下面的反向）过程比较简单，实现上也是非常直接，不需要什么技巧。
*/
void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    // 初始偏移设定为四周补0长度的负值
    int w_offset = -l.pad;
    int h_offset = -l.pad;

    // 获取当前层的输出尺寸
    int h = l.out_h;
    int w = l.out_w;

    // 获取当前层输入图像的通道数，为什么是输入通道数？不应该为输出通道数吗？实际二者没有区别，对于最大池化层来说，输入有多少通道，输出就有多少通道！
    int c = l.c;

    // 遍历batch中每一张输入图片，计算得到与每一张输入图片具有相同通道数的输出图
    for(b = 0; b < l.batch; ++b){
        // 对于每张输入图片，将得到通道数一样的输出图，以输出图为基准，按输出图通道，行，列依次遍历
        // （这对应图像在l.output的存储方式，每张图片按行铺排成一大行，然后图片与图片之间再并成一行）。
        // 以输出图为基准进行遍历，最终循环的总次数刚好覆盖池化核在输入图片不同位置进行池化操作。
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){

                    // out_index为输出图中的索引：out_index = b * c * w * h + k * w * h + h * w + w，展开写可能更为清晰些
                    int out_index = j + w*(i + h*(k + c*b));

                    float max = -FLT_MAX;   // FLT_MAX为c语言中float.h定义的对大浮点数，此处初始化最大元素值为最小浮点数
                    int max_i = -1;         // 最大元素值的索引初始化为-1

                    // 下面两个循环回到了输入图片，计算得到的cur_h以及cur_w都是在当前层所有输入元素的索引，内外循环的目的是找寻输入图像中，
                    // 以(h_offset + i*l.stride, w_offset + j*l.stride)为左上起点，尺寸为l.size池化区域中的最大元素值max及其在所有输入元素中的索引max_i
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            // cur_h，cur_w是在所有输入图像中第k通道中的cur_h行与cur_w列，index是在所有输入图像元素中的总索引。
                            // 为什么这里少一层对输入通道数的遍历循环呢？因为对于最大池化层来说输入与输出通道数是一样的，并在上面的通道数循环了！
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));

                            // 边界检查：正常情况下，是不会越界的，但是如果有补0操作，就会越界了，这里的处理方式是直接让这些元素值为-FLT_MAX
                            // （注意虽然称之为补0操作，但实际不是补0），总之，这些补的元素永远不会充当最大元素值。
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;

                            // 记录这个池化区域中的最大的元素值及其在所有输入元素中的总索引
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    // 由此得到最大池化层每一个输出元素值及其在所有输入元素中的总索引。
                    // 为什么需要记录每个输出元素值对应在输入元素中的总索引呢？因为在下面的反向过程中需要用到，在计算当前最大池化层上一层网络的敏感度时，
                    // 需要该索引明确当前层的每个元素究竟是取上一层输出（也即上前层输入）的哪一个元素的值，具体见下面backward_maxpool_layer()函数的注释。
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

/*
** 最大池化层反向传播传播函数
** 输入： l     当前最大池化层
**       net   整个网络
** 说明：这个函数看上去很简单，比起backward_convolutional_layer()少了很多，这都是有原因的。实际上，在darknet中，不管是什么层，
**      其反向传播函数都会先后做两件事：1）计算当前层的敏感度图l.delta、权重更新值以及偏置更新值；2）计算上一层的敏感度图net.delta（部分计算，
**      要完成计算得等到真正到了这一层再说）。而这里，显然没有第一步，只有第二步，而且很简单，这是为什么呢？首先回答为什么没有第一步。注意当前层l是最大池化层，
**      最大池化层没有训练参数，说的再直白一点就是没有激活函数，或者认为激活函数就是f(x)=x，所以激活函数对于加权输入的导数其实就是1,
**      正如在backward_convolutional_layer()注释的那样，每一层的反向传播函数的第一步是将之前（就是下一层计算得到的，注意过程是反向的）
**      未计算完得到的l.delta乘以激活函数对加权输入的导数，以最终得到当前层的敏感度图，而对于最大池化层来说，每一个输出对于加权输入的导数值都是1,
**      同时并没有权重及偏置这些需要训练的参数，自然不再需要第一步；对于第二步为什么会如此简单，可以参考：
**      https://www.zybuluo.com/hanbingtao/note/485480，最大池化层它就是这么简单，剩下的参考下面的注释。
*/
void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    // 获取当前最大池化层l的输出尺寸h,w
    int h = l.out_h;
    int w = l.out_w;

    // 获取当前层输入的通道数，为什么是输入通道数？不应该为输出通道数吗？实际二者没有区别，对于最大池化层来说，输入有多少通道，输出就有多少通道！
    int c = l.c;

    // 计算上一层的敏感度图（未计算完全，还差一个环节，这个环节等真正反向到了那层再执行）
    // 这个循环很有意思，循环总次数为当前层输出总元素个数（包含所有输入图片的输出，即维度为l.out_h * l.out_w * l.c * l.batch，注意此处l.c==l.out_c）,
    // 而不是上一层输出总元素个数，为什么呢？是因为对于最大池化层而言，其每个输出元素对仅受上一层输出对应池化核区域中最大值元素的影响，所以当前池化层每个输出元素
    // 对于上一层输出中的很多元素的导数值为0,而对最大值元素，其导数值为1；再乘以当前层的敏感度图，导数值为0的还是为0,导数值为1则就等于当前层的敏感度值。
    // 以输出图总元素个数进行遍历，刚好可以找出上一层输出中所有真正起作用（在某个池化区域中充当了最大元素值）也即敏感度值不为0的元素，而那些没有起作用的元素，
    // 可以不用理会，保持其初始值0就可以了。
    // 详细原理推导可以参见：https://www.zybuluo.com/hanbingtao/note/485480
    for(i = 0; i < h*w*c*l.batch; ++i){
        // 遍历的基准是以当前层的输出元素为基准的，l.indexes记录了当前层每一个输出元素与上一层哪一个输出元素有真正联系（也即上一层对应池化核区域中最大值元素的索引），
        // 所以index是上一层中所有输出元素的索引，且该元素在当前层某个池化域中充当了最大值元素，这个元素的敏感度值将直接传承当前层对应元素的敏感度值。
        // 而net.delta中，剩下没有被index按索引访问到的元素，就是那些没有真正起到作用的元素，这些元素的敏感度值为0（net.delta已经在前向时将所有元素值初始化为0）
        // 至于为什么要用+=运算符，原因有两个，和卷积类似：一是池化核由于跨度较小，导致有重叠区域；二是batch中有多张图片，需要将所有图片的影响加起来。
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

// 最大池化层没有训练参数，所以，没有update_maxpool_layer()函数～

