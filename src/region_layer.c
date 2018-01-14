#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};
    l.type = REGION;

    /// 以下众多参数含义参考layer.h中的注释
    l.n = n;                                                ///< 一个cell（网格）中预测多少个矩形框（box）
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);                         ///< region_layer输出的通道数
    l.out_w = l.w;                                          ///< region_layer层的输入和输出尺寸一致，通道数也一样，也就是这一层并不改变输入数据的维度
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;                                    ///< 物体类别种数（训练数据集中所拥有的物体类别总数）
    l.coords = coords;                                      ///< 定位一个物体所需的参数个数（一般值为4,包括矩形中心点坐标x,y以及长宽w,h）
    l.cost = calloc(1, sizeof(float));                      ///< 目标函数值，为单精度浮点型指针
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);               ///< 一张训练图片经过region_layer层后得到的输出元素个数（等于网格数*每个网格预测的矩形框数*每个矩形框的参数个数）
    l.inputs = l.outputs;                                   ///< 一张训练图片输入到reigon_layer层的元素个数（注意是一张图片，对于region_layer，输入和输出的元素个数相等）
    /**
     * 每张图片含有的真实矩形框参数的个数（30表示一张图片中最多有30个ground truth矩形框，每个真实矩形框有
     * 5个参数，包括x,y,w,h四个定位参数，以及物体类别）,注意30是darknet程序内写死的，实际上每张图片可能
     * 并没有30个真实矩形框，也能没有这么多参数，但为了保持一致性，还是会留着这么大的存储空间，只是其中的
     * 值未空而已.
    */
    l.truths = 30*(5);
    l.delta = calloc(batch*l.outputs, sizeof(float));       ///< l.delta,l.input,l.output三个参数的大小是一样的
    /**
     * region_layer的输出维度为l.out_w*l.out_h，等于输入的维度，输出通道数为l.out_c，等于输入通道数，
     * 且通道数等于n*(classes+coords+1)。那region_layer的输出l.output中到底存储了什么呢？存储了
     * 所有网格（grid cell）中预测矩形框（box）的所有信息。看Yolo论文就知道，Yolo检测模型最终将图片
     * 划分成了S*S（论文中为7*7）个网格，每个网格中预测B个（论文中B=2）矩形框，最后一层输出的就是这些
     * 网格中所包含的所有预测矩形框信息。目标检测模型中，作者用矩形框来表示并定位检测到的物体，每个矩形框中
     * 包含了矩形框定位信息x,y,w,h，含有物体的自信度信息c，以及属于各类的概率（如果有20类，那么就有矩形框
     * 中所包含物体属于这20类的概率）。注意了，这里的实现与论文中的描述有不同，首先参数固然可能不同（比如
     * 并不像论文中那样每个网格预测2个box，也有可能更多），更为关键的是，输出维度的计算方式不同，论文中提到
     * 最后一层输出的维度为一个S_w*S_c*(B*5+C)的tensor（作者在论文中是S*S，这里我写成S_w，S_c是考虑到
     * 网格划分维度不一定S_w=S_c=S，不过貌似作者用的都是S_w=S_c的，比如7*7,13*13，总之明白就可以了），
     * 实际上，这里有点不同，输出的维度应该为S_w*S_c*B*(5+C),C为类别数目，比如共有20类；5是因为有4个定位
     * 信息，外加一个自信度信息c，共有5个参数。也即每个矩形框都包含一个属于各类的概率，并不是所有矩形框共有
     * 一组属于各类的概率，这点可以从l.outputs的计算方式中看出（可以对应上，l.out_w = S_w, l.out_c = S_c, 
     * l.out_c = B*(5+C)）。知道输出到底存储什么之后，接下来要搞清是怎么存储的，毕竟输出的是一个三维张量，
     * 但实现中是用一个一维数组来存储的，详细的注释可以参考下面forward_region_layer()以及entry_index()
     * 函数的注释，这个东西仅用文字还是比较难叙述的，应该借助图来说明～
    */
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

/** 获取某个矩形框的4个定位信息（根据输入的矩形框索引从l.output中获取该矩形框的定位信息x,y,w,h）.
 * @param x region_layer的输出，即l.output，包含所有batch预测得到的矩形框信息
 * @param biases 
 * @param n 
 * @param index 矩形框的首地址（索引，矩形框中存储的首个参数x在l.output中的索引）
 * @param i 第几行（region_layer维度为l.out_w*l.out_c，通道数为）
 * @param j 
 * @param w
 * @param h
 * @param stride个数
*/
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

/** 
 * @brief 计算某个矩形框中某个参数在l.output中的索引。一个矩形框包含了x,y,w,h,c,C1,C2...,Cn信息，
 *        前四个用于定位，第五个为矩形框含有物体的自信度信息c，即矩形框中存在物体的概率为多大，而C1到Cn
 *        为矩形框中所包含的物体分别属于这n类物体的概率。本函数负责获取该矩形框首个定位信息也即x值在
 *        l.output中索引、获取该矩形框自信度信息c在l.output中的索引、获取该矩形框分类所属概率的首个
 *        概率也即C1值的索引，具体是获取矩形框哪个参数的索引，取决于输入参数entry的值，这些在
 *        forward_region_layer()函数中都有用到，由于l.output的存储方式，当entry=0时，就是获取矩形框
 *        x参数在l.output中的索引；当entry=4时，就是获取矩形框自信度信息c在l.output中的索引；当
 *        entry=5时，就是获取矩形框首个所属概率C1在l.output中的索引，具体可以参考forward_region_layer()
 *        中调用本函数时的注释.
 * @param l 当前region_layer
 * @param batch 当前照片是整个batch中的第几张，因为l.output中包含整个batch的输出，所以要定位某张训练图片
 *              输出的众多网格中的某个矩形框，当然需要该参数.
 * @param location 这个参数，说实话，感觉像个鸡肋参数，函数中用这个参数获取n和loc的值，这个n就是表示网格中
 *                 的第几个预测矩形框（比如每个网格预测5个矩形框，那么n取值范围就是从0~4），loc就是某个
 *                 通道上的元素偏移（region_layer输出的通道数为l.out_c = (classes + coords + 1)），
 *                 这样说可能没有说明白，这都与l.output的存储结构相关，见下面详细注释以及其他说明。总之，
 *                 查看一下调用本函数的父函数orward_region_layer()就知道了，可以直接输入n和j*l.w+i的，
 *                 没有必要输入location，这样还得重新计算一次n和loc.               
 * @param entry 切入点偏移系数，关于这个参数，就又要扯到l.output的存储结构了，见下面详细注释以及其他说明.
 * @details l.output这个参数的存储内容以及存储方式已经在多个地方说明了，再多的文字都不及图文说明，此处再
 *          简要罗嗦几句，更为具体的参考图文说明。l.output中存储了整个batch的训练输出，每张训练图片都会输出
 *          l.out_w*l.out_h个网格，每个网格会预测l.n个矩形框，每个矩形框含有l.classes+l.coords+1个参数，
 *          而最后一层的输出通道数为l.n*(l.classes+l.coords+1)，可以想象下最终输出的三维张量是个什么样子的。
 *          展成一维数组存储时，l.output可以首先分成batch个大段，每个大段存储了一张训练图片的所有输出；进一步细分，
 *          取其中第一大段分析，该大段中存储了第一张训练图片所有输出网格预测的矩形框信息，每个网格预测了l.n个矩形框，
 *          存储时，l.n个矩形框是分开存储的，也就是先存储所有网格中的第一个矩形框，而后存储所有网格中的第二个矩形框，
 *          依次类推，如果每个网格中预测5个矩形框，则可以继续把这一大段分成5个中段。继续细分，5个中段中取第
 *          一个中段来分析，这个中段中按行（有l.out_w*l.out_h个网格，按行存储）依次存储了这张训练图片所有输出网格中
 *          的第一个矩形框信息，要注意的是，这个中段存储的顺序并不是挨个挨个存储每个矩形框的所有信息，
 *          而是先存储所有矩形框的x，而后是所有的y,然后是所有的w,再是h，c，最后的的概率数组也是拆分进行存储，
 *          并不是一下子存储完一个矩形框所有类的概率，而是先存储所有网格所属第一类的概率，再存储所属第二类的概率，
 *          具体来说这一中段首先存储了l.out_w*l.out_h个x，然后是l.out_w*l.out_c个y，依次下去，
 *          最后是l.out_w*l.out_h个C1（属于第一类的概率，用C1表示，下面类似），l.out_w*l.outh个C2,...,
 *          l.out_w*l.out_c*Cn（假设共有n类），所以可以继续将中段分成几个小段，依次为x,y,w,h,c,C1,C2,...Cn
 *          小段，每小段的长度都为l.out_w*l.out_c.
 *          现在回过来看本函数的输入参数，batch就是大段的偏移数（从第几个大段开始，对应是第几张训练图片），
 *          由location计算得到的n就是中段的偏移数（从第几个中段开始，对应是第几个矩形框），
 *          entry就是小段的偏移数（从几个小段开始，对应具体是那种参数，x,c还是C1），而loc则是最后的定位，
 *          前面确定好第几大段中的第几中段中的第几小段的首地址，loc就是从该首地址往后数loc个元素，得到最终定位
 *          某个具体参数（x或c或C1）的索引值，比如l.output中存储的数据如下所示（这里假设只存了一张训练图片的输出，
 *          因此batch只能为0；并假设l.out_w=l.out_h=2,l.classes=2）：
 *          xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2-#-xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2，
 *          n=0则定位到-#-左边的首地址（表示每个网格预测的第一个矩形框），n=1则定位到-#-右边的首地址（表示每个网格预测的第二个矩形框）
 *          entry=0,loc=0获取的是x的索引，且获取的是第一个x也即l.out_w*l.out_h个网格中第一个网格中第一个矩形框x参数的索引；
 *          entry=4,loc=1获取的是c的索引，且获取的是第二个c也即l.out_w*l.out_h个网格中第二个网格中第一个矩形框c参数的索引；
 *          entry=5,loc=2获取的是C1的索引，且获取的是第三个C1也即l.out_w*l.out_h个网格中第三个网格中第一个矩形框C1参数的索引；
 *          如果要获取第一个网格中第一个矩形框w参数的索引呢？如果已经获取了其x值的索引，显然用x的索引加上3*l.out_w*l.out_h即可获取到，
 *          这正是delta_region_box()函数的做法；
 *          如果要获取第三个网格中第一个矩形框C2参数的索引呢？如果已经获取了其C1值的索引，显然用C1的索引加上l.out_w*l.out_h即可获取到，
 *          这正是delta_region_class()函数中的做法；
 *          由上可知，entry=0时,即偏移0个小段，是获取x的索引；entry=4,是获取自信度信息c的索引；entry=5，是获取C1的索引.
 *          l.output的存储方式大致就是这样，个人觉得说的已经很清楚了，但可视化效果终究不如图文说明～
*/
int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);


/**
 * @param l
 * @param net
 * @details 本函数多次调用了entry_index()函数，且使用的参数不尽相同，尤其是最后一个参数，通过最后一个参数，
 *          可以确定出region_layer输出l.output的数据存储方式。为方便叙述，假设本层输出参数l.w = 2, l.h= 3,
 *          l.n = 2, l.classes = 2, l.coords = 4, l.c = l.n * (l.coords + l.classes + 1) = 21,
 *          l.output中存储了所有矩形框的信息参数，每个矩形框包括4条定位信息参数x,y,w,h，一条自信度（confidience）
 *          参数c，以及所有类别的概率C1,C2（本例中，假设就只有两个类别，l.classes=2），那么一张样本图片最终会有
 *          l.w*l.h*l.n个矩形框（l.w*l.h即为最终图像划分层网格的个数，每个网格预测l.n个矩形框），那么
 *          l.output中存储的元素个数共有l.w*l.h*l.n*(l.coords + 1 + l.classes)，这些元素全部拉伸成一维数组
 *          的形式存储在l.output中，存储的顺序为：
 *          xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C1C1C1C1C1C2C2C2C2C2C2-##-xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C2C1C2C1C2C1C2C1C2C1C2
 *          文字说明如下：-##-隔开分成两段，左右分别是代表所有网格的第1个box和第2个box（因为l.n=2，表示每个网格预测两个box），
 *          总共有l.w*l.h个网格，且存储时，把所有网格的x,y,w,h,c信息聚到一起再拼接起来，因此xxxxxx及其他信息都有l.w*l.h=6个，
 *          因为每个有l.classes个物体类别，而且也是和xywh一样，每一类都集中存储，先存储l.w*l.h=6个C1类，而后存储6个C2类，
 *         更为具体的注释可以函数中的语句注释（注意不是C1C2C1C2C1C2C1C2C1C2C1C2的模式，而是将所有的类别拆开分别集中存储）。
 * @details 自信度参数c表示的是该矩形框内存在物体的概率，而C1，C2分别表示矩形框内存在物体时属于物体1和物体2的概率，
 *          因此c*C1即得矩形框内存在物体1的概率，c*C2即得矩形框内存在物体2的概率
 */
void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    /// 将net.input中的元素全部拷贝至l.output中
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

    /// 这个#ifndef预编译指令没有必要用的，因为forward_region_layer()函数本身就对应没有定义gpu版的，所以肯定会执行其中的语句,
    /// 其中的语句的作用是为了计算region_layer层的输出l.output
#ifndef GPU
    /// 遍历batch中的每张图片（l.output含有整个batch训练图片对应的输出）
    for (b = 0; b < l.batch; ++b){
        /// 注意region_layer层中的l.n含义是每个cell grid（网格）中预测的矩形框个数（不是卷积层中卷积核的个数）
        for(n = 0; n < l.n; ++n){
            /// 获取 某一中段首个x的地址（中段的含义参考entry_idnex()函数的注释），此处仅用两层循环处理所有的输入，直观上应该需要四层的，
            /// 即还需要两层遍历l.w和l.h（遍历每一个网格），但实际上并不需要，因为每次循环，其都会处理一个中段内某一小段的数据，这一小段数据
            /// 就包含所有网格的数据。比如处理第1个中段内所有x和y（分别有l.w*l.h个x和y）.
            int index = entry_index(l, b, n*l.w*l.h, 0);
            /// 注意第二个参数是2*l.w*l.h，也就是从index+l.output处开始，对之后2*l.w*l.h个元素进行logistic激活函数处理，也就是对
            /// 一个中段内所有的x,y进行logistic函数处理，这里没有搞明白的是，为什么对x,y进行激活函数处理？后面的w,h呢？还有，region_layer
            /// 怎么只有激活函数处理，没有训练参数吗？
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            /// 和上面一样，此处是获取一个中段内首个自信度信息c值的地址，而后对该中段内所有的c值（该中段内共有l.w*l.h个c值）进行logistic激活函数处理
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index,   l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else if (l.softmax){
        /// 与上面一样，此处是获取l.output中首个类别概率C1的地址，而后对l.output中所有的类别概率（共有l.batch*l.n*l.w*l.h*l.classes个）进行softmax函数处理,
        /// 注意这里的措辞，是整个l.output中首个类别概率C1的地址，因为entry_index()的第二个和第三个参数都是0，
        int index = entry_index(l, 0, 0, 5);

        /// net.input+index为region_layer的输入（加上索引偏移量index）
        /// l.classes-->n，物体种类数，对应softmax_cpu()中输入参数n；
        /// l.batch*l.n-->batch，一个batch的图片张数乘以每个网格预测的矩形框数，得到值可以这么理解：所有batch数据（net.input）可以分成的中段的总段数，
        /// 该参数对应softmax_cpu()中输入参数batch；
        /// l.inputs/l.n-->batch_offset，注意l.inputs仅是一张训练图片输入到region_layer的元素个数，l.inputs/l.n得到的值实际是一个小段的元素个数
        /// （即所有网格中某个矩形框的所有参数个数）,对应softmax_cpu()中输入参数batch_offset参数；
        /// l.w*l.h-->groups，对应softmax_cpu()中输入参数groups;
        /// softmax_cpu()中输入参数group_offset值恒为1；
        /// l.w*l.h-->stride，对应softmax_cpu()中输入参数stride;
        /// softmax_cpu()中输入参数temp的值恒为1；
        /// l.output+index为region_layer的输出（同样加上索引偏移量index，region_layer的输入与输出元素一一对应）；
        /// 详细说一下这里的过程（对比着softmax_cpu()及其调用的softmax()函数来说明）：softmax_cpu()中的第一层for循环遍历了batch次，即遍历了所有中段；
        /// 第二层循环遍历了groups次，也即l.w*l.h次，实际上遍历了所有网格；而后每次调用softmax()实际上会处理一个网格某个矩形框的所有类别概率，因此可以在
        /// softmax()函数中看到，遍历的次数为n，也即l.classes的值；在softmax()函数中，用上了跨度stride，其值为l.w*l.h，之所以用到跨度，是因为net.input
        /// 和l.output的存储方式，详见entry_index()函数的注释，由于每次调用softmax()，需要处理一个矩形框所有类别的概率，这些概率值都是分开存储的，间隔
        /// 就是stride=l.w*l.h。这样，softmax_cpu()的两层循环以及softmax()中遍历的n次合起来就会处理得到l.output中所有l.batch*l.n*l.w*l.h*l.classes个
        /// 概率类别值。（此处提到的中段，小段等名词都需参考entry_index()的注释，尤其是l.output数据的存储方式，只有熟悉了此处才能理解好，另外再次强调一下，
        /// region_layer的输入和输出元素个数是一样的，一一对应，因此其存储方式也是一样的）
        softmax_cpu(net.input + index, l.classes, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
#endif

    /// 数组初始化：将l.delta中所有元素（共有l.outputs*l.batch个元素，每个元素sizeof(float)个字节）清零
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

    /// 如果不是训练过程，则返回不再执行下面的语句（前向推理即检测过程也会调用这个函数，这时就不需要执行下面训练时才会用到的语句了）
    if(!net.train) return;

    float avg_iou = 0;        ///< 平均IoU（Intersection over Union）
    float recall = 0;         ///< 召回率
    float avg_cat = 0;        ///< 
    float avg_obj = 0;        ///< 
    float avg_anyobj = 0;     ///< 一张训练图片所有预测矩形框的平均自信度（矩形框中含有物体的概率），该参数没有实际用处，仅用于输出打印
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass = 0;
            /// 循环30次，每张图片固定处理30个矩形框
            for(t = 0; t < 30; ++t){
                /// 通过移位来获取每一个真实矩形框的信息，net.truth存储了网络吞入的所有图片的真实矩形框信息（一次吞入一个batch的训练图片），
                /// net.truth作为这一个大数组的首地址，l.truths参数是每一张图片含有的真实值参数个数（可参考layer.h中的truths参数中的注释），
                /// b是batch中已经处理完图片的图片的张数，5是每个真实矩形框需要5个参数值（也即每条矩形框真值有5个参数），t是本张图片已经处理
                /// 过的矩形框的个数（每张图片最多处理30张图片），明白了上面的参数之后对于下面的移位获取对应矩形框真实值的代码就不难了。
                box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);

                /// 这个if语句是用来判断一下是否有读到真实矩形框值（每个矩形框有5个参数,float_to_box只读取其中的4个定位参数，
                /// 只要验证x的值不为0,那肯定是4个参数值都读取到了，要么全部读取到了，要么一个也没有），另外，因为程序中写死了每张图片处理30个矩形框，
                /// 那么有些图片没有这么多矩形框，就会出现没有读到的情况。
                if(!truth.x) break;

                /// float_to_box()中没有读取矩形框中包含的物体类别编号的信息，就在此处获取。（darknet中，物体类别标签值为编号，
                /// 每一个类别都有一个编号值，这些物体具体的字符名称存储在一个文件中，如data/*.names文件，其所在行数就是其编号值）
                int class = net.truth[t*5 + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, 5);
                        int obj_index = entry_index(l, b, n, 4);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, 5);
                    int obj_index = entry_index(l, b, maxi, 4);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        /**
         * 下面三层主循环的顺序，与最后一层输出的存储方式有关。外层循环遍历所有行，中层循环遍历所有列，这两层循环合在一起就是按行遍历
         * region_layer输出中的每个网格，内层循环l.n表示的是每个grid cell中预测的box数目。首先要明白这一点，region_layer层的l.w,l.h
         * 就是指最后图片划分的网格（grid cell）数，也就是说最后一层输出的图片被划分成了l.w*l.h个网格，每个网格中预测l.n个矩形框
         * （box），每个矩形框就是一个潜在的物体，每个矩形框中包含了矩形框定位信息x,y,w,h，含有物体的自信度信息c，以及属于各类的概率，
         * 最终该网格挑出概率最大的作为该网格中含有的物体，当然最终还需要检测，如果其概率没有超过一定阈值，那么判定该网格中不含物体。
         * 搞清楚这点之后，理解下面三层循环就容易一些了，外面两层循环是在遍历每一个网格，内层循环则遍历每个网格中预测的所有box。
         * 除了这三层主循环之后，里面还有一个循环，循环次数固定为30次，这个循环是遍历一张训练图片中30个真实的矩形框（30是指定的一张训练
         * 图片中最多能够拥有的真实矩形框个数）。知道了每一层在遍历什么，那么这整个4层循环合起来是用来干什么的呢？与紧跟着这4层循环之后
         * 还有一个固定30次的循环在功能上有什么区别呢？此处这四层循环目的在于
         */
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    /// 根据i,j,n计算该矩形框的索引，实际是矩形框中存储的x参数在l.output中的索引，矩形框中包含多个参数，x是其存储的首个参数，
                    /// 所以也可以说是获取该矩形框的首地址。更为详细的注释，参考entry_index()的注释。
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    /// 根据矩形框的索引，获取矩形框的定位信息
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);

                    /// 最高IoU，赋初值0
                    float best_iou = 0;

                    /// 为什么这里总是看到30？查看layer.h中关于max_boxes变量的注释就知道了，每张图片最多能够有30真实个框;
                    /// 还要说明的一点是，此处所说最多处理30个矩形框，是指真实值中，一张图片含有的最大真实物体标签数，
                    /// 也即真实的物体矩形框个数最大为30,并不是模型预测限制最多就30个，如上注释，一个图片如果分成7*7个网格，
                    /// 每个网格预测两个矩形框，那就有98个了，所以不是指模型只能预测30个。模型你可以尽管预测多的矩形框，
                    /// 只是我默认一张图片中最多就打了30个物体的标签，所以之后会有滤除过程。滤除过程有两层，首先第一层就是
                    /// 下面的for循环了，
                    for(t = 0; t < 30; ++t){
                        /// 通过移位来获取每一个真实矩形框的信息，net.truth存储了网络吞入的所有图片的真实矩形框信息（一次吞入一个batch的训练图片），
                        /// net.truth作为这一个大数组的首地址，l.truths参数是每一张图片含有的真实值参数个数（可参考layer.h中的truths参数中的注释），
                        /// b是batch中已经处理完图片的图片的张数，5是每个真实矩形框需要5个参数值（也即每条矩形框真值有5个参数），t是本张图片已经处理
                        /// 过的矩形框的个数（每张图片最多处理30张图片），明白了上面的参数之后对于下面的移位获取对应矩形框真实值的代码就不难了。
                        box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);

                        /// 这个if语句是用来判断一下是否有读到真实矩形框值（每个矩形框有5个参数,float_to_box只读取其中的4个定位参数，
                        /// 只要验证x的值不为0,那肯定是4个参数值都读取到了，要么全部读取到了，要么一个也没有），另外，因为程序中写死了每张图片处理30个矩形框，
                        /// 那么有些图片没有这么多矩形框，就会出现没有读到的情况。
                        if(!truth.x) break;

                        /// 获取完真实标签矩形定位坐标后，与模型检测出的矩形框求IoU，具体参考box_iou()函数注释
                        float iou = box_iou(pred, truth);

                        /// 找出最大的IoU值
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    /// 获取当前遍历矩形框含有物体的自信度信息c（该矩形框中的确存在物体的概率）在l.output中的索引值
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    /// 叠加每个预测矩形框的自信度c（也即每个矩形框中含有物体的概率）
                    avg_anyobj += l.output[obj_index]; 
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    /// 上面30次循环使得本矩形框已经与训练图片中所有30个（30个只是最大值，可能没有这么多）真实矩形标签进行了对比，只要在这30个中
                    /// 找到一个真实矩形标签与该预测矩形框的iou大于指定的阈值，则判定该
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }

                    if(*(net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            if(iou > .5) recall += 1;
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
            avg_obj += l.output[obj_index];
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore) {
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }

            int class = net.truth[t*5 + b*l.truths + 4];

            /// 参考layer.h中关于map的注释：将coco数据集的物体类别编号，变换至在联合9k数据集中的物体类别编号，
            /// 如果l.map不为NULL，说明使用了yolo9000检测模型，其他模型不用这个参数（没有联合多个数据集训练），
            /// 目前只有yolo9000.cfg中设置了map文件所在路径。
            if (l.map) class = l.map[class];
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 5);
            delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

void get_region_boxes(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh, int relative)
{
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + 5; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                probs[index][j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            float scale = predictions[obj_index];
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);

            int class_index = entry_index(l, 0, n*l.w*l.h + i, 5);
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, 5 + map[j]);
                        float prob = scale*predictions[class_index];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    probs[index][j] = (scale > thresh) ? scale : 0;
                    probs[index][l.classes] = scale;
                }
            } else {
                float max = 0;
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 5 + j);
                    float prob = scale*predictions[class_index];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if(prob > max) max = prob;
                    // TODO REMOVE
                    // if (j == 56 ) probs[index][j] = 0; 
                    /*
                       if (j != 0) probs[index][j] = 0; 
                       int blacklist[] = {121, 497, 482, 504, 122, 518,481, 418, 542, 491, 914, 478, 120, 510,500};
                       int bb;
                       for (bb = 0; bb < sizeof(blacklist)/sizeof(int); ++bb){
                       if(index == blacklist[bb]) probs[index][j] = 0;
                       }
                     */
                }
                probs[index][l.classes] = max;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
    correct_region_boxes(boxes, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU

/**
 * 
 */
void forward_region_layer_gpu(const layer l, network net)
{
    copy_ongpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            int index = entry_index(l, 0, 0, count);
            softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
            count += group_size;
        }
    } else if (l.softmax) {
        int index = entry_index(l, 0, 0, 5);
        //printf("%d\n", index);
        softmax_gpu(net.input_gpu + index, l.classes, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    float *truth_cpu = 0;
    if(net.truth_gpu){
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(net.truth_gpu, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_region_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            index = entry_index(l, b, n*l.w*l.h, 4);
            gradient_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            l.output[obj_index] = 0;
        }
    }
}

