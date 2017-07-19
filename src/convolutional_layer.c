#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

/*
**  根据输入图像的高度(h)，两边补0的个数(pad)，卷积核尺寸(size)以及跨度(stride)计算输出的特征图的高度
**  输入：l    卷积层，包含该卷积层的所有参数，实际这里没有必要输入整个l，因为只需要到其中的四个参数而已
**  输出：int类型，输出图像的高度
**  说明：这个函数的实现应该可以进一步改善一下，虽然这个函数只是在最初构建网络时调用一次，之后就不调用了，不怎么影响性能，
**       但输入整个l实在不妥（l比较大，按值传递复制过程比较冗长），要么就只输入用到的四个参数，要么传入l的指针，
**       并且不需要返回值了，直接在函数内部为l.out_h赋值
*/
int convolutional_out_height(convolutional_layer l)
{
    // pad是每边补0的个数，因此乘以2
    // 当stride=1，pad=size/2（整数除法，会往下取整）时，输出高度就等于输入高度（same策略）；
    // 当stride=1,pad=0时，为valid策略
    // 当stride不等于1时，输出高度恒小于输入高度（尺寸一定会缩小）
    // 计算公式推导：设输出高度为x，总图像高度为h+2*pad个像素，输出高度为x，则共有x-1次卷积核移位，
    // 共占有(x-1)*stride+size个像素，可能还剩余res个像素，且res一定小于stride（否则还可以再移位一次），
    // 因此有(x-1)*stride+size+res=h+2*pad，->x=(h+2*pad-size)/stride+1-res/stride，因为res<stride，
    // 对于整数除法来说，值为0,于是得到最终的输出高度为x=(h+2*pad-size)/stride+1
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

/*
**  根据输入图像的宽度(w)，两边补0的个数(pad)，卷积核尺寸(size)以及跨度(stride)计算输出的特征图的宽度
**  与上一个函数convolutional_out_height()类似，不再赘述
*/
int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);
}
#endif
#endif

/*
**
**  
**  输入：batch    每个batch含有的图片数
**      h               图片高度（行数）
**      w               图片宽度（列数）
        c               输入图片通道数
        n               卷积核个数
        size            卷积核尺寸
        stride          跨度
        padding         四周补0长度
        activation      激活函数类别
        batch_normalize 是否进行BN(规范化)
        binary          是否对权重进行二值化
        xnor            是否对权重以及输入进行二值化
        adam            使用
*/
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    // convolutional_layer是使用typedef定义的layer的别名
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL; // 层属性：卷积层

    l.h = h;                // 输入图像高度
    l.w = w;                // 输入图像宽度
    l.c = c;                // 输入图像通道
    l.n = n;                // 卷积核个数（即滤波器个数）
    l.binary = binary;      // 是否对权重进行二值化
    l.xnor = xnor;          // 是否对权重以及输入进行二值化
    l.batch = batch;        // 每个batch含有的图片数
    l.stride = stride;      // 跨度
    l.size = size;          // 卷积核尺寸
    l.pad = padding;        // 四周补0长度
    l.batch_normalize = batch_normalize;    // 是否进行BN(规范化)

    // 该卷积层总的权重元素（卷积核元素）个数=输入图像通道数*卷积核个数*卷积核尺寸
    // (实际真正的卷积层元素个数不需要乘以c，但为了计算方便，需要复制c份，使可以同时对所有通道的输入进行卷积)
    l.weights = calloc(c*n*size*size, sizeof(float));
    // 
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    // bias就是Wx+b中的b（上面的weights就是W），有多少个卷积核，就有多少个b（与W的个数一一对应，每个W的元素个数为c*size*size）
    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    // 该卷积层总的权重元素个数（实际只有n*size*size个，但复制了c份）
    l.nweights = c*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    // 初始化权重：缩放因子*标准正态分布随机数
    // TODO：个人感觉，这里应该加一个if条件语句：if(weightfile)，因为如果导入了预训练权重文件，就没有必要这样初始化了
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    
    // 根据该层输入图像的尺寸、卷积核尺寸以及跨度计算输出特征图的宽度和高度
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;        // 输出图像高度
    l.out_w = out_w;        // 输出图像宽度
    l.out_c = n;            // 输出图像通道（等于卷积核个数，有多少个卷积核，最终就得到多少张特征图，每张图是一个通道）

    l.outputs = l.out_h * l.out_w * l.out_c;    // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n张特征图）
    l.inputs = l.w * l.h * l.c;                 // mini-batch中每张输入图片的像素元素个数
    // 关于上面两个参数的说明：
    // 一个mini-batch中有多张图片，每张图片可能有多个通道（彩色图有三通道），l.inputs是每张输入图片所有通道的总元素个数，
    // 而每张输入图片会有n个卷积核对其进行卷积操作，因此一张输入图片会输出n张特征图，这n张特征图的总元素个数就为l.outputs

    // l.output为该层所有的输出（包括mini-batch所有输入图片的输出）
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    // 卷积层三种指针函数，对应三种计算：前向，反向，更新
    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.adam = 1;
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;

    // 初始化输出l.output全为0.0；输入l.outputs*l.batch为输出的总元素个数，其中l.outputs为batch
    // 中一个输入对应的输出的所有元素的个数，l.batch为batch中总的输入个数；0表示初始化所有输出为0；
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    // 是否进行二值化操作（这个操作应该只有第一个卷积层使用吧？因为下面直接对net.input操作）
    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n;                // 该层卷积核个数
    int k = l.size*l.size*l.c;  // 该层每个卷积核的参数元素个数
    int n = out_h*out_w;        // 该层每个特征图的尺寸（元素个数）


    float *a = l.weights;       // 所有卷积核（也即权重），元素个数为l.n*l.c*l.size*l.size，按行存储，共有l*n行，l.c*l.size*l.size列
    float *b = net.workspace;   // 对输入图像进行重排之后的图像数据
    float *c = l.output;        // 存储一张输入图片（多通道）所有的输出特征图（输入图片是多通道的，输出图片也是多通道的，有多少个卷积核就有多少个通道，每个卷积核得到一张特征图即为一个通道）

    // 该循环即为卷积层核心代码：对batch中每张图片进行卷积运算！！
    // 可以参考：https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    // 进行辅助理解（主要是辅助理解，实际执行并不一样）
    for(i = 0; i < l.batch; ++i){
        // 将多通道二维图像net.input变成列数组b，以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂）
        // 注意net.input包含batch中所有图片的数据，但是每次循环只处理一张，因此在im2col_cpu仅会对其中一张图片
        // 进行重排，l.c为每张图片的通道数，l.h为每张图片的高度，l.w为每张图片的宽度，l.size为卷积核尺寸，l.stride为跨度
        // b为一张图片重排后的结果
        im2col_cpu(net.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);

        // GEneral Matrix to Matrix Multiplication
        // 此处在im2col_cpu操作基础上，利用矩阵乘法c=alpha*a*b+beta*c完成对图像卷积的操作
        // 0,0表示不对输入a,b进行转置，
        // m是输入a,c的行数，具体含义为卷积核的个数，
        // n是输入b,c的列数，具体含义为每个特征图的元素个数(out_h*out_w)，
        // k是输入a的列数也是b的行数，具体含义为卷积核元素个数乘以输入图像的通道数（l.size*l.size*l.c），
        // a,b,c即为三个参与运算的矩阵（用一维数组存储）,alpha=beta=1为常系数，
        // a为所有卷积核集合,元素个数为l.n*l.c*l.size*l.size，按行存储，共有l*n行，l.c*l.size*l.size列，
        // 即a中每行代表一个可以作用在3通道上的卷积核，
        // b为输入图像经过im2col_cpu重排后的图像数据，
        // c为输出，相当于输出，包含所有特征图（每个卷积核得到一张特征图），c中一行代表一张特征图。
        // 所有特征图铺排开成一行后，再并成一行，存储在c中
        // 详细查看该函数注释（比较复杂）
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);

        // 对c进行指针偏移：移到batch中下一张图片对应输出的起始位置（每循环一次，将完成对一张图片的卷积操作，
        // 产生的所有特征图的元素个数总和为n*m）
        c += n*m;
        // 同样，输入也进行指针偏移，移动到下一张图片元素的起始位置，以便下一次循环处理
        // （batch中每张图片的元素个数为通道数*高度*宽度，即l.c*l.h*l.w）
        net.input += l.c*l.h*l.w;
    }

    // 如需需要规范化（BN在非线性激活函数处理之前完成）
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        float *im = net.input+i*l.c*l.h*l.w;

        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta){
            a = l.weights;
            b = l.delta + i*m*k;
            c = net.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta+i*l.c*l.h*l.w);
        }
    }
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

