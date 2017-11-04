// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;              // 网络总层数（make_network()时赋值）
    int batch;          // parse_net_options()中赋值：一个batch含有的图片张数，下面还有个subdivisions参数，暂时没搞清楚有什么用处，
                        // 总之此处的batch*subdivision才等于网络配置文件中指定的batch值
    int *seen;          // 目前已经读入的图片张数（网络已经处理的图片张数）（在make_network()中动态分配内存）
    float epoch;        
    int subdivisions;   // parse_net_options()中赋值，如上batch注释
    float momentum;     // parse_net_options()中赋值
    float decay;        // parse_net_options()中赋值
    layer *layers;      // 存储网络所有的层，在make_network()中动态分配内存
    float *output;
    learning_rate_policy policy;

    float learning_rate; // parse_net_options()中赋值
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;         // 一张输入图片的元素个数，如果网络配置文件中未指定，则默认等于net->h * net->w * net->c，在parse_net_options()中赋值
    int outputs;        // 一张输入图片对应的输出元素个数，对于一些网络，可由输入图片的尺寸及相关参数计算出，比如卷积层，可以通过输入尺寸以及跨度、核大小计算出；
                        // 对于另一些尺寸，则需要通过网络配置文件指定，如未指定，取默认值1，比如全连接层
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;

    int gpu_index;
    tree *hierarchy;



    float *input;       // 中间变量，用来暂存某层网络的输入（包含一个batch的输入，比如某层网络完成前向，将其输出赋给该变量，作为下一层的输入，可以参看network.c中的forward_network()与backward_network()两个函数），
                        // 当然，也是网络接受最原始输入数据（即第一层网络接收的输入）的变量（比如在图像检测训练中，最早在train_detector()->train_network()->get_next_batch()函数中赋值）

    float *truth;       // 中间变量，与上面的input对应，用来暂存input数据对应的标签数据（真实数据）
    
    float *delta;       // 中间变量，用来暂存某层网络的敏感度图（反向传播处理当前层时，用来存储上一层的敏感度图，因为当前层会计算部分上一层的敏感度图，可以参看network.c中的backward_network()函数），
                        // net.delta并没有在创建网络之初就为其动态分配了内存，而是等到反向传播时，直接将其等于某一层的l.delta（l.delta是在创建每一层网络之初就动态为其分配了内存），这才为net.delta分配了内存，
                        // 如果没有令net.delta=l.delta，则net.delta是未定义的（没有动态分配内存的）
    float *workspace;   // 整个网络的工作空间，其元素个数为所有层中最大的l.workspace_size = l.out_h*l.out_w*l.size*l.size*l.c
                        // （在make_convolutional_layer()计算得到workspace_size的大小，在parse_network_cfg()中动态分配内存，
                        // 此值对应未使用gpu时的情况），该变量貌似不轻易被释放内存，目前只发现在network.c的resize_network()函数对其进行了释放。
                        // net.workspace充当一个临时工作空间的作用，存储临时所需要的计算参数，比如每层单张图片重排后的结果
                        // （这些参数马上就会参与卷积运算），一旦用完，就会被马上更新（因此该变量的值的更新频率比较大）
    int train;          // 标志参数，网络是否处于训练阶段，如果是，则值为1（这个参数一般用于训练与测试有不同操作的情况，比如dropout层，对于训练，才需要进行forward_dropout_layer()函数，对于测试，不需要进入到该函数）
    int index;          // 标志参数，当前网络的活跃层（活跃包括前向和反向，可参考network.c中forward_network()与backward_network()函数）
    float *cost;        

    #ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
    #endif

} network;


#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net);
float *network_predict_gpu(network net, float *input);
void pull_network_output(network net);
void forward_network_gpu(network net);
void backward_network_gpu(network net);
void update_network_gpu(network net);
void harmless_update_network_gpu(network net);
#endif

float get_current_rate(network net);
int get_current_batch(network net);
void free_network(network net);
void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net);
void backward_network(network net);
void update_network(network net);

float train_network(network net, data d);
float train_network_sgd(network net, data d, int n);
float train_network_datum(network net);

matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
layer get_network_output_layer(network net);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
network load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network net);
void calc_network_cost(network net);

#endif

