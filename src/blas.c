#include "blas.h"
#include "math.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

/*
** axpy是线性代数中一种基本操作，完成y= alpha*x + y操作，其中x,y为矢量，alpha为实数系数
** 可以参考：https://www.youtube.com/watch?v=PQ1Q85JGgZg
** 输入：  N       X中包含的有效元素个数
**        ALPHA   系数alpha
**        X       参与运算的矢量X
**        INCX    步长（倍数步长），即X中凡是INCX的倍数编号参与运算
**        Y       参与运算的矢量，也相当于是输出
*/
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

/*
**  初始化X数组所有元素的值为ALPHA
**  输入： N       X中包含的有效元素个数
**        ALPHA   初始化的值
**        X       待初始化的float数组指针
**        INCX    步长（倍数步长），即X中凡是INCX的倍数编号进行初始化赋值操作
*/
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

/*
**  将输入X中的数据复制到Y中（值复制，并不是指针复制，即之后X与Y之间再无关联）
**  输入： N       X中包含的有效元素个数
**        X       待初始化的float数组指针
**        INCX    步长（倍数步长），即X中凡是INCX的倍数编号进行初始化赋值操作
*/
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

/*
** 
*/
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

/*
** 输入： input   一组输入图片数据（含义见下面softmax_cpu()注释，下同）
**       n       一组输入数据中含有的元素个数n=l.inputs/l.groups
**       temp    温度参数，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
**       stride  跨度
**       output  这一组输入图片数据对应的输出（也即l.output中与这一组输入对应的某一部分）
** 说明：本函数实现的就是标准的softmax函数处理，唯一有点变化的就是在做指数运算之前，将每个输入元素减去了该组输入元素中的最大值，以增加数值稳定性，
**      关于此，可以参加博客：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/，
**      这篇博客写的不错，博客中还提到了softmax-loss，此处没有实现（此处实现的也即博客中提到的softmax函数，将softmax-loss分开实现了）。
*/
void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    // 赋初始最大值为float中的最小值-FLT_MAX（定义在float.h中）
    float largest = -FLT_MAX;
    // 寻找输入中的最大值，至于为什么要找出最大值，是为了数值计算上的稳定，详细请戳：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    // 这篇博客写的不错，博客在接近尾声的时候，提到了为什么要减去输入中的最大值。
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        // 在进行指数运算之间，如上面博客所说，首先减去最大值（当然温度参数也要除）
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;                       // 求和
        output[i*stride] = e;           // 并将每一个输入的结果保存在相应的输出中
    }
    // 最后一步：归一化转换为概率（就是softmax函数的原型～），最后的输出结果保存在output中
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

/*
** 输入： input    softmax层所有输入数据（包含整个batch的），即net.input（上一层的输出）
**       n        一组输入数据中含有的元素个数n=l.inputs/l.groups
**       batch    一个batch中所含有的图片张数（等于net.batch）
**       batch_offset    一张输入图片含有的元素个数，即值等于l.inputs（所以叫做batch_offset，目的是要借助该参数在input中整张整张照片移位）
**       groups   一张输入图片的元素被分成了几组，值为l.groups（这个参数由配置文件指定，如果未指定，则默认为1）,这个参数暂时还没遇到怎么用，大部分的网络值都为1,也即相当于没有这个参数
**       group_offset    值等于n，组偏移（在每张输入图片元素中整组整组偏移）
**       stride   跨度，这个参数类似于axpy_cpu()函数中的INCX参数，一定注意不同于卷积层中的l.stride，这个参数是指按照stride间隔从每组输入数据中抽取元素，即会抽取所有索引为stride倍数的输入元素，
**                而其他的输入元素，实际没有用到；stride=1时，显然，相当于没有这个参数，所有输入数据都用到了（这个参数在softmax_layer层中，相当于没用，因为在forward_softmax_layer()中，
**                调用该函数时，stride已经被写死为1,并不能改，不知道还有没有其他地方使用了这个参数）
**       temp     softmax的温度参数l.temperature，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
**       output   经softmax处理之后得到的输出l.output（即概率），与input具有相同的元素个数（见make_softmax_layer()），其实由此也可知，
**                stride的值必然为1,不然output的元素个数肯定少于input的元素个数（所以对于softmax来说，感觉设置stride是没有必要的，有点自相矛盾的意思）
** 说明：上面的注释出现了新的量词单位，这里厘清一下关系：输入input中包括batch中所有图片的输入数据，其中一张图片具有inputs个元素，一张图片的元素又分成了groups组，
**      每组元素个数为n=l.inputs/l.groups
*/
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    // 遍历batch中的每张图片
    for(b = 0; b < batch; ++b){
        // 每张图片又按组遍历：一组一组遍历
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

