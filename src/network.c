#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"

load_args get_base_args(network net)
{
    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.size = net.w;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.center = net.center;
    args.saturation = net.saturation;
    args.hue = net.hue;
    return args;
}

network load_network(char *cfg, char *weights, int clear)
{
    network net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(&net, weights);
    }
    if(clear) *net.seen = 0;
    return net;
}

/*
** 计算当前已经读入多少个batch（提醒一下：网络配置.cfg文件中的batch不是指总共有多少个batch，而是指每个batch中有多少张图片，
** tensorflow实战中一般用batch_size来表示一个batch中函数的图片张数，用num_batches来表示共有多少个batch，这样更加清晰）
** 输入： net    构建的整个神经网络
** 返回： 已经读入的batch个数
*/
int get_current_batch(network net)
{
    // net.seen为截至目前已经读入的图片张数，batch*subdivisons为一个batch含有的图片张数，二者一除即可得截至目前已经读入的batch个数
    // net.subdivisions这个参数目前还不知道有什么用，总之net.batch*net.subdivisions等于.cfg中指定的batch值（参看：parser.c中的parse_net_options()函数）
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

void reset_momentum(network net)
{
    if (net.momentum == 0) return;
    net.learning_rate = 0;
    net.momentum = 0;
    net.decay = 0;
    #ifdef GPU
        //if(net.gpu_index >= 0) update_network_gpu(net);
    #endif
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                //if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

/*
** 新建一个空网络，并为网络中部分指针参数动态分配内存
** 输入： n    神经网络层数
** 说明： 该函数只为网络的三个指针参数动态分配了内存，并没有为所有指针参数分配内存
*/
network make_network(int n)
{
    network net = {0};
    net.n = n;
    // 为每一层分配内存
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(int));
    net.cost = calloc(1, sizeof(float));
    return net;
}

/** 前向计算网络net每一层的输出.
 * @param net 构建好的整个网络模型
 * @details 遍历net的每一层网络，从第0层到最后一层，逐层计算每层的输出
*/
void forward_network(network net)
{
    int i;
    /// 遍历所有层，从第一层到最后一层，逐层进行前向传播（网络总共有net.n层）
    for(i = 0; i < net.n; ++i){
        /// 置网络当前活跃层为当前层，即第i层
        net.index = i;
        /// 获取当前层
        layer l = net.layers[i];
        /// 如果当前层的l.delta已经动态分配了内存，则调用fill_cpu()函数，将其所有元素的值初始化为0
        if(l.delta){
            /// 第一个参数为l.delta的元素个数，第二个参数为初始化值，为0
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }

        /// 前向传播（前向推理）：完成l层的前向推理
        l.forward(l, net);

        /// 完成某一层的推理时，置网络的输入为当前层的输出（这将成为下一层网络的输入），要注意的是，此处是直接更改指针变量net.input本身的值，
        /// 也就是此处是通过改变指针net.input所指的地址来改变其中所存内容的值，并不是直接改变其所指的内容而指针所指的地址没变，
        /// 所以在退出forward_network()函数后，其对net.input的改变都将失效，net.input将回到进入forward_network()之前时的值。
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(net);
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
        }
    }
}

void calc_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network net)
{
    return max_index(net.output, net.outputs);
}

/*
** 反向计算网络net每一层的敏感度图，并进而计算每一层的权重、偏置更新值，最后完成每一层权重与偏置更新
** 流程：遍历net的每一层网络，从最后一层到第一层（此处所指的第一层不是指输入层，而是与输入层直接相连的第一层隐含层），反向传播
*/
void backward_network(network net)
{
    int i;
    // 在进行反向之前，先保存一下原先的net，下面会用到orig的input
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        // i = 0时，也即已经到了网络的第1层（或者说第0层，看个人习惯了～）了，就是直接与输入层相连的第一层隐含层（注意不是输入层，我理解的输入层就是指输入的图像数据，
        // 严格来说，输入层不算一层网络，因为输入层没有训练参数，也没有激活函数），这个时候，不需要else中的赋值，1）对于第1层来说，其前面已经没有网络层了（输入层不算），
        // 因此没有必要再计算前一层的参数，故没有必要在获取上一层；2）第一层的输入就是图像输入，也即整个net最原始的输入，在开始进行反向传播之前，已经用orig变量保存了
        // 最为原始的net，所以net.input就是第一层的输入，不需要通过net.input=prev.output获取上一层的输出作为当前层的输入；3）同1），第一层之前已经没有层了，
        // 也就不需要计算上一层的delta，即不需要再将net.delta链接到prev.delta，此时进入到l.backward()中后，net.delta就是NULL（可以参看network.h中关于delta
        // 的注释），也就不会再计算上一层的敏感度了（比如卷积神经网络中的backward_convolutional_layer()函数）
        if(i == 0){
            net = orig;
        }else{
            // 获取上一层
            layer prev = net.layers[i-1];
            // 上一层的输出作为当前层的输入（下面l.backward()会用到，具体是在计算当前层权重更新值时要用到）
            net.input = prev.output;
            // 上一层的敏感度图（l.backward()会同时计算上一层的敏感度图）
            net.delta = prev.delta;
        }
        // 置网络当前活跃层为当前层，即第i层
        net.index = i;
        // 反向计算第i层的敏感度图、权重及偏置更新值，并更新权重、偏置（同时会计算上一层的敏感度图，
        // 存储在net.delta中，但是还差一个环节：乘上上一层输出对加权输入的导数，也即上一层激活函数对加权输入的导数）
        l.backward(l, net);
    }
}

float train_network_datum(network net)
{
    // 如果使用GPU则调用gpu版的train_network_datum
#ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net);
#endif
    // 更新目前已经处理的图片数量：每次处理一个batch，故直接添加l.batch
    *net.seen += net.batch;
    // 标记处于训练阶段
    net.train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net.cost;
    if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net.input, net.truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

/*
** 训练一个batch（此处所指一个batch含有的图片是配置文件中真实指定的一个batch中含有的图片数量，也即图片张数为：net.batch*net.subdivision）
** 输入： net     已经构建好待训练的整个网络
**       d       此番训练所用到的所有图片数据（包含net.batch*net.subdivision张图片）
** 说明：train_network()函数对应处理ngpu=1的情况，如果有多个gpu，那么会首先调用train_networks()函数，在其之中会调用get_data_part()
**      将在detector.c train_detector()函数中读入的net.batch*net.subdivision*ngpu张图片平分至每个gpu上，
**      而后再调用train_network()，总之最后train_work()的输入d中只包含net.batch*net.subdivision张图片
*/
float train_network(network net, data d)
{
    // 事实上对于图像检测而言，d.X.rows/net.batch=net.subdivision，因此恒有d.X.rows % net.batch == 0，且下面的n就等于net.subdivision
    // （可以参看detector.c中的train_detector()），因此对于图像检测而言，下面三句略有冗余，但对于其他种情况（比如其他应用，非图像检测甚至非视觉情况），
    // 不知道是不是这样
    assert(d.X.rows % net.batch == 0);
    int batch = net.batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        // 从d中读取batch张图片到net.input中，进行训练
        // 第一个参数d包含了net.batch*net.subdivision张图片的数据，第二个参数batch即为每次循环读入到net.input也即参与train_network_datum()
        // 训练的图片张数，第三个参数为在d中的偏移量，第四个参数为网络的输入数据，第五个参数为输入数据net.input对应的标签数据（真实数据）
        get_next_batch(d, batch, i*batch, net.input, net.truth);

        // 训练网络：本次训练的数据共有net.batch张图片。
        // 训练包括一次前向过程：计算每一层网络的输出并计算cost；一次反向过程：计算敏感度、权重更新值、偏置更新值；适时更新过程：更新权重与偏置
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    cuda_free(net->workspace);
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(*net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        cuda_free(net->input_gpu);
        cuda_free(net->truth_gpu);
        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
        net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

detection_layer get_network_detection_layer(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].type == DETECTION){
            return net.layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    detection_layer l = {0};
    return l;
}

image get_network_image_layer(network net, int i)
{
    layer l = net.layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network net, int k, int *index)
{
    top_k(net.output, net.outputs, k, index);
}

/** 利用构建好的网络net对输入数据input进行处理.
 * @param net   构建好且已经训练好的网络
 * @param input 输入数据（如果是一张图片，则其尺寸已经调整至网络需要的尺寸）
 * @return net.output 网络层输出
 * @details 本函数为整个网络的前向推理函数，调用该函数可完成对某一输入数据进行前向推理的过程，
 *          可以参考detector.c中test_detector()函数调用该函数完成对一张输入图片检测的用法。
*/
float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif
    net.input = input;
    net.truth = 0;
    net.train = 0;
    net.delta = 0;
    forward_network(net);
    return net.output;
}

matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = net.outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net.batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = net.outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network net)
{
    int i,j;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network net)
{
    int i;
    for(i = net.n - 1; i >= 0; --i){
        if(net.layers[i].type != COST) break;
    }
    return net.layers[i];
}

float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        free_layer(net.layers[i]);
    }
    free(net.layers);
    if(net.input) free(net.input);
    if(net.truth) free(net.truth);
#ifdef GPU
    if(net.input_gpu) cuda_free(net.input_gpu);
    if(net.truth_gpu) cuda_free(net.truth_gpu);
#endif
}

// Some day...


layer network_output_layer(network net)
{
    int i;
    for(i = net.n - 1; i >= 0; --i){
        if(net.layers[i].type != COST) break;
    }
    return net.layers[i];
}

int network_inputs(network net)
{
    return net.layers[0].inputs;
}

int network_outputs(network net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network net)
{
    return network_output_layer(net).output;
}

