#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/*
**  从文件filename中读取数据信息（不是具体的图像数据，只是关于数据的相关信息），存至链表返回：依次调用fgetl()->list_insert()函数
**  输入： filename    文件名称
**  输出： list指针，包含从文件中读取的信息
**  调用： 比如在data.c->get_labels()调用，目的是为了从data/**.names文件中，读取所有物体类别的名称/标签信息；
**        在train_detector()中调用，目的是从train.txt（该文件的生成参考Yolo官网）中读入所有训练图片的路径（文件中每一行就是一张图片的全路径）
*/
list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    // fget1读入一整行到path,并将其插入到列表lines中
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        indexes[i] = index;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
*/

/*
** 从paths中读取n条路径信息：paths包含所有训练图片的路径，二维数组，每行对应一张图片的路径，m为paths的行数，即为训练图片总数
** 返回一个二维数组（矩阵），每行代表一张图片的路径，共n行
*/
char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    // paths这个变量可能会被不同线程访问（读取数据本来就是多线程的），所以访问之前，先锁住，结束后解锁
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        // 随机产生索引：随机读入图片路径
        
        //随意读取图片的目的：举个例子：一般的训练集都是猫的图片在一起，狗的图片在一起，如果不随机读取，就是一个或者几个batch都是猫或者狗，容易过拟合同时泛化能力也差
        int index = rand()%m;
        random_paths[i] = paths[index];
        //if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop;
        if(center){
            crop = center_crop_image(im, size, size);
        } else {
            crop = random_augment_image(im, angle, aspect, min, max, size);
        }
        int flip = rand()%2;
        if (flip) flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        /*
        show_image(im, "orig");
        show_image(crop, "crop");
        cvWaitKey(0);
        */
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}

/*
** 读入一张图片的所有box：一张图片可能有多个物体，每个物体都有一个矩形框框起来（物体检测不单识别类别，更包括定位），
** 本函数就是读入一张图片的所有box信息。每个box包括5条信息，依次为：物体类别id，矩形中心点x坐标，矩形中心点y坐标，
** 矩形框宽度w,矩形框高度h。
** 输入： filename    标签数据所在路径（标签数据需要下载，然后调用voc_label.py生成指定的格式，具体路径视情况而定，相见darknet/yolo网页）
**       n           该图片中的物体个数，也就是读到的矩形框个数（也是一个返回值）
** 返回： box_label*，包含这张图片中所有的box标签信息
*/
box_label *read_boxes(char *filename, int *n)
{
    // 新建一个标签数据box，并动态分配内存（之后，如果检测到多个矩形框标签数据，则利用realloc函数重新分配内存）
    box_label *boxes = calloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;

    // 读入一行数据：图片检测数据文件中一行包含了一个box的信息，依次为id,x,y,w,h
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        // 根据box个数重新分配内存：分配count+1个box_label的内存
        boxes = realloc(boxes, (count+1)*sizeof(box_label));
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        // 通过x,y,w,h计算矩形框四个角点的最小最大x,y坐标
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

/*
** 随机打乱一张照片中所有box的索引编号
*/
void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        // 通过随机交换值来打扰box在box集合中的索引编号
        box_label swap = b[i];
        // 生成0~n-1之间的索引号
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

/*
** 矫正矩形框标签数据在标准化尺寸图片中的值：输入的图片，经过place_image()函数对图片尺寸进行规范化以及数据增强后,
** 尺寸发生了变化，由于矩形框的x,y,w,h（分别为矩形中心点坐标，矩形宽高）都是相对于原始图片宽高的比例值，所以，
** 如果仅对原始图片进行缩放（等比例也好，不等比例也好），是不会改变x,y,w,h等值的，也就是中间图与原始图矩形框的x,y,w,h值是一样的，但关键在于，在函数place_image()中，
** 不单涉及缩放操作，还涉及平移操作，place_image()函数的最后一步是将中间图随机的嵌入到最终输出图上，因为这一步，
** 就彻底改变了x,y,w,h的值，为了计算新的x,y,w,h值，很简单，先按原来的值乘以中间图的宽高，得到矩形框四个角点的真实坐标，
** 而后进行平移，统一除以最终输出图的宽高，得到新的x,y,w,h值。
** 除此之外，左右翻转也会导致x,y,w,h值的变化。
** 输出： boxes     一张图片中包含的所有矩形框标签数据
**       n         一张图片中包含的矩形框个素
**       dx        place_image()函数中，中间图相对于最终输出图canvas的起点的x坐标，用占比表示（或者说x方向上的偏移坐标），正值表示中间图嵌入到最终输出图中，负值表示输出图是中间图的一个mask
**       dy        place_image()函数中，中间图相对于最终输出图canvas的起点的y坐标，用占比表示（或者说y方向上的偏移坐标），正值表示中间图嵌入到最终输出图中，负值表示输出图是中间图的一个mask
**       sx        nw/w，place_image()函数中中间图宽度与最终输出图宽度的比值
**       sy        nw/w，place_image()函数中中间图高度与最终输出图高度的比值
**       flip      是否进行了翻转操作，在load_data_detection()中，为了进行数据增强，还可能进行了翻转操作
*/
void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    // 遍历并依次矫正每个矩形框标签数据
    for(i = 0; i < n; ++i){
        // x,y是矩形框中心点的坐标，因此，二者不可能为0（为0的话，说明矩形框宽高只能为0,相当于不存在这个矩形框或者物体），
        // 要搞清一个概念，最初的矩形框的x,y,w,h是相对于输入的训练图片的宽高比例值，因此矩形框必须在图片内，就算有一个物体，图片没有照全，
        // 那么给的矩形框也必须是图片内的矩形框，这时矩形框只覆盖物体的部分内容，总之不可能出现矩形框中心坐标为(0,0)（或者为负）的情况。
        // 个人觉得这里的与条件应该改为或，因为x,y二者都不能为0
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        // sx = nw / w, dx = -dx / w, (boxes[i].left  * nw + dx) / w括号内为在原矩形框在中间图中真实的长度，除以w即可得新的x值，
        // 其他几个参数也一样，思路都是先获取在中间图中的绝对坐标，然后除以最终输出图的尺寸即可得到矩形框四个角点相对输出图的坐标（占比）
        // 此函数需要与load_data_detection()函数中调用的place_image()函数一起看。
        // 要注意的是，这里首先获取的是在中间图中的绝对坐标，不是原始输入图的，因为place_image()函数最后一步，是将
        // 中间图嵌入到最终输出图中，因此，已经与原始输入图没有关系了。
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        // 如果load_data_detection()函数中还对最终输出图进行了左右翻转，那么相应的矩形框的位置也有改动
        if(flip){
            // 左右翻转，就是交换一下值就可以了（因为这里都使用占比来表示坐标值，所以用1相减）
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        // 将矩形框的四个角点坐标严格限制在0~1之间（超出边界值的直接置为边界值）
        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        // 计算矩形框新的中心点坐标以及宽高
        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        // 严格限制新的矩形框宽高在0~1之间（感觉没有必要，安全起见吧）
        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 30; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);

    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .005 || h < .005) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

/*
** 用来获取一张图的真实标签信息，对于图像检测，标签信息包括物体的类别（用类别id表示）以及定位信息，定位用矩形框来表示，包含矩形中心点坐标x,y以及宽高w,h，
** 本函数读入一张图片中所有标签信息（一张图片可能存在多个物体，每个物体都含有一条类别信息以及一个矩形框信息）
** 输入： path     一张图片所在路径，字符数组
**       num_boxes 每张图片允许处理的最大的矩形框数（如果图片中包含的矩形框大于num_boxes，那么不管，随机取其中num_boxes个参与训练）
**       truth    存储一张图片包含的所有真实信息（标签信息），相当于返回值，对于检测而言，主要包括物体类别以及定位（矩形框）信息，
**                truth是一个一维数组，每张矩形框有5条信息，因此truth中每5个数对应一个矩形框数据
**       classes  本函数并未使用该参数
**       flip     图片在之前读入时（比如在load_data_detection函数中）是否进行过了左右翻转
**       dx       此参数需要参考load_data_detection函数中的注释，dx是中间图相对最终图的起点位置的x坐标除以最终图的宽度（并取负值）
**       dy       此参数需要参考load_data_detection函数中的注释，dy是中间图相对最终图的起点位置的x坐标除以最终图的高度（并取负值）
**       sx       此参数需要参考load_data_detection函数中的注释，sx是中间图宽度与最终图宽度的比值
**       sy       此参数需要参考load_data_detection函数中的注释，sy是中间图高度与最终图高度的比值
** 说明： 后面五个参数，用来矫正矩形框的信息，因为在此函数之前，对输入图片进行了缩放、平移、左右翻转一系列的数据增强操作，这些操作不会改变物体的类别信息，
**       但会改变物体的位置信息，也即矩形框信息，需要进行矫正，这些参数的具体含义上面可能未说清，具体可参看本函数内部调用的correct_boxes()函数的用法
*/
void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    // 定义一个数组，分配4096个字符（字节）内存，用以存放本图片标签信息文件所在路径
    char labelpath[4096];

    // 下面一连串地调用find_replace()函数，是为了得到每张训练图片的标签数据（.txt文件）所在路径
    // 通过调用find_replace()函数，对每张图片的绝对路径进行修改，得到对应的标签数据所在路径。
    // 比如，图片的路径为：/home/happy/Downloads/darknet_dataset/VOCdevkit/VOC2007/JPEGImages/000001.jpg，
    // 通过连续调用find_place()函数，最终可以得到对应的标签数据路径labelpath为：
    // /home/happy/Downloads/darknet_dataset/VOCdevkit/VOC2007/labels/000001.txt
    // 注意，下面共调用了7次find_replace函数，可以分为两部分，第一部分是将图片的文件夹名字替换为labels，
    // 图片的路径可能为JPEGImages,images或raw中的一种，所以调用了三次以应对多种情况，实际只有一次调用真正有效;
    // 第二部分是将修改后缀，图片的格式有可能为jpg,png,JPG,JPEG四种中的一种，不管是哪种，
    // 最终替换成标签数据格式，即.txt格式，因此，此处也是为了应对可能的四种情况，才四次调用了find_replace，实际起作用的只有一次调用。
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, "raw", "labels", labelpath);

    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);


    int count = 0;
    // 读入一张图片的所有box标签信息，count为读到的box个数
    box_label *boxes = read_boxes(labelpath, &count);
    // 随机打乱一张照片中所有box的索引编号
    randomize_boxes(boxes, count);
    // 从输入的原始图片，到真正给神经网络用的图片，可能经过了平移，随机截取，左右翻转等数据增强操作，这些操作，都会改变矩形框的值，需要进行矫正
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);

    // 如果图片中含有的矩形框数多于num_boxes，则保持count = num_boxes，因为num_boxes是指定的每张图片最多参与训练的矩形框数，如果超过这个数，
    // 就在其中随机选择num_boxes个（box的顺序已经随机打乱了）
    if(count > num_boxes) count = num_boxes;
    float x,y,w,h;
    int id;
    int i;

    // 提取count个矩形框信息
    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;   // 物体的类别并不是用字符串来表示，而是用物体类别对应的id来表示，如对于VOC数据集，共有20类物体，那么对应的id编号为从0~19号

        // 矩形框大小下限：如果长宽小于0.001（矩形框的长宽不到图片长宽的0.001）认为没有包含物体
        if ((w < .001 || h < .001)) continue;

        // 最后将矩形框信息赋给truth
        truth[i*5+0] = x;
        truth[i*5+1] = y;
        truth[i*5+2] = w;
        truth[i*5+3] = h;
        truth[i*5+4] = id;
    }
    // 所有矩形框的信息已经提取，及时释放堆内存
    free(boxes);
}

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
        }
    }
    if(count != 1 && (k != 1 || count != 0)) printf("Too many or too few labels: %d, %s\n", count, path);
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

matrix load_regression_labels_paths(char **paths, int n)
{
    matrix y = make_matrix(n, 1);
    int i;
    for(i = 0; i < n; ++i){
        char labelpath[4096];
        find_replace(paths[i], "images", "targets", labelpath);
        find_replace(labelpath, "JPEGImages", "targets", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".png", ".txt", labelpath);
        
        FILE *file = fopen(labelpath, "r");
        fscanf(file, "%f", &(y.vals[i][0]));
        fclose(file);
    }
    return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy){
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "imgs", "labels", label);
        find_replace(label, "_iconl.jpeg", ".txt", label);
        FILE *file = fopen(label, "r");
        if(!file){
            find_replace(label, "labels", "labels2", label);
            file = fopen(label, "r");
            if(!file) continue;
        }
        ++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    printf("%d/%d\n", count, n);
    return y;
}

/*
**  首先调用get_paths()从filename中读取数据到list变量中，而后调用list_to_array()转存至二维字符数组中返回
**  输入： filename    文件路径
**  返回： char**类型，包含从文件中读取的所有信息（该函数只用于读入名称/标签信息，即读取data/**.names文件的信息）
**  调用：该函数在detector.c->test_detector()函数中调用，目的正如函数名，从文件中读取数据集中所有类别的名称/标签信息
*/
char **get_labels(char *filename)
{
    // 从包含数据集中所有物体类别信息的文件（data/**.names文件）中读取所有物体类别信息，存入链表plist中
    list *plist = get_paths(filename);

    // 将所用数据集的所有类别信息（即包含所有类别的名称/标签信息）从plist中提取出来，存至二维字符数组labels中
    // labels将包含数据集中所有类别信息，比如使用coco.data，里面包含80类物体，有person, bicycle等等
    char **labels = (char **)list_to_array(plist);

    // 将指针复制给labels之后，就可以释放plist了，不会影响labels的
    // TODO:这里应该可以额外写个函数直接将数据读取至labels，没有必要调用list_to_array先读到plist，而后转存至labels，
    //      当然，这里绕一下弯可能是出于代码统一规范的考虑
    free_list(plist);
    return labels;
}

/*
** 释放数据d的堆内存
** 说明： 该函数输入虽然是data类型数据，但实际上是释放data结构体中matrix类型元素X,y的vals的堆内存（有两层深度），
**       分两种情况，浅层释放与深层释放，决定于d的标志位shallow的取值，shallow=1表示浅层释放，
**       shallow=0表示深层释放，关于浅层与深层释放，见data结构体定义处的注释，此不再赘述。
**       什么时候需浅层释放呢？当不想删除二维数组的数据，只想清除第一层指针存储的第二层指针变量时，
**       采用浅层释放，在释放前，二维数组的数据应当转移，使得另有指针可以访问这些保留的数据；
**       而当不想要保留任何数据时，采用深层释放。
*/
void free_data(data d)
{
    if(!d.shallow){
        // 深层释放堆内存（注意虽然输入是d.X,d.y，但不是直接释放d.X，d.y，这二者在data结构体中根本连指针都不算。
        // 在free_matrix()中是逐行释放d.X.vals和d.y.vals的内存，再直接释放d.X.vals和d.y.vals）
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        // 浅层释放堆内存
        free(d.X.vals);
        free(d.y.vals);
    }
}

data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    int k = size*size*(5+classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = rand_uniform(-dw, dw);
        int pright = rand_uniform(-dw, dw);
        int ptop   = rand_uniform(-dh, dh);
        int pbot   = rand_uniform(-dh, dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = rand()%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = calloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char imlabel1[4096];
        char imlabel2[4096];
        find_replace(paths[i*2],   "imgs", "labels", imlabel1);
        find_replace(imlabel1, "jpg", "txt", imlabel1);
        FILE *fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        find_replace(paths[i*2+1], "imgs", "labels", imlabel2);
        find_replace(imlabel2, "jpg", "txt", imlabel2);
        FILE *fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }

        for (j = 0; j < classes; ++j){
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5){
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            } else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5){
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            } else {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = rand()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*30;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = rand()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

/*
** 可以参考，看一下对图像进行jitter处理的各种效果:
** https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3
** 从所有训练图片中，随机读取n张，并对这n张图片进行数据增强，同时矫正增强后的数据标签信息。最终得到的图片的宽高为w,h（原始训练集中的图片尺寸不定），也就是网络能够处理的图片尺寸，
** 数据增强包括：对原始图片进行宽高方向上的插值缩放（两方向上缩放系数不一定相同），下面称之为缩放抖动；随机抠取或者平移图片（位置抖动）；
** 在hsv颜色空间增加噪声（颜色抖动）；左右水平翻转，不含旋转抖动。
** 输入： n         一个线程读入的图片张数（详见函数内部注释）
**       paths     所有训练图片所在路径集合，是一个二维数组，每一行对应一张图片的路径（将在其中随机取n个）
**       m         paths的行数，也即训练图片总数
**       w         网络能够处理的图的宽度（也就是输入图片经过一系列数据增强、变换之后最终输入到网络的图的宽度）
**       h         网络能够处理的图的高度（也就是输入图片经过一系列数据增强、变换之后最终输入到网络的图的高度）
**       boxes     每张训练图片最大处理的矩形框数（图片内可能含有更多的物体，即更多的矩形框，那么就在其中随机选择boxes个参与训练，具体执行在fill_truth_detection()函数中）
**       classes   类别总数，本函数并未用到（fill_truth_detection函数其实并没有用这个参数）
**       jitter    这个参数为缩放抖动系数，就是图片缩放抖动的剧烈程度，越大，允许的抖动范围越大（所谓缩放抖动，就是在宽高上插值缩放图片，宽高两方向上缩放的系数不一定相同）
**       hue       颜色（hsv颜色空间）数据增强参数：色调（取值0度到360度）偏差最大值，实际色调偏差为-hue~hue之间的随机值
**       saturation 颜色（hsv颜色空间）数据增强参数：色彩饱和度（取值范围0~1）缩放最大值，实际为范围内的随机值
**       exposure  颜色（hsv颜色空间）数据增强参数：明度（色彩明亮程度，0~1）缩放最大值，实际为范围内的随机值
** 返回： data类型数据，包含一个线程读入的所有图片数据（含有n张图片）
** 说明： 最后四个参数用于数据增强，主要对原图进行缩放抖动，位置抖动（平移）以及颜色抖动（颜色值增加一定噪声），抖动一定程度上可以理解成对图像增加噪声。
**       通过对原始图像进行抖动，实现数据增强。最后三个参数具体用法参考本函数内调用的random_distort_image()函数
** 说明2：从此函数可以看出，darknet对训练集中图片的尺寸没有要求，可以是任意尺寸的图片，因为经该函数处理（缩放/裁剪）之后，
**       不管是什么尺寸的照片，都会统一为网络训练使用的尺寸
*/
data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
{
    // paths包含所有训练图片的路径，get_random_paths函数从中随机提出n条，即为此次读入的n张图片的路径
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    // 初始化为0,清楚内存中之前的旧值（类似data d ={};，参考：https://www.zhihu.com/question/46405621/answer/101218929）
    data d = {0};
    d.shallow = 0;

    // 一次读入的图片张数：d.X中每行就是一张图片的数据，因此d.X.cols等于h*w*3
    // n = net.batch * net.subdivisions * ngpus，net中的subdivisions这个参数暂时还没搞懂有什么用，
    // 从parse_net_option()函数可知，net.batch = net.batch / net.subdivision，等号右边的那个batch就是
    // 网络配置文件.cfg中设置的每个batch的图片数量，但是不知道为什么多了subdivision这个参数？总之，
    // net.batch * net.subdivisions又得到了在网络配置文件中设定的batch值，然后乘以ngpus，是考虑多个GPU实现数据并行，
    // 一次读入多个batch的数据，分配到不同GPU上进行训练。在load_threads()函数中，又将整个的n仅可能均匀的划分到每个线程上，
    // 也就是总的读入图片张数为n = net.batch * net.subdivisions * ngpus，但这些图片不是一个线程读完的，而是分配到多个线程并行读入，
    // 因此本函数中的n实际不是总的n，而是分配到该线程上的n，比如总共要读入128张图片，共开启8个线程读数据，那么本函数中的n为16,而不是总数128
    d.X.rows = n;
    // d.X为一个matrix类型数据，其中d.X.vals是其具体数据，是指针的指针（即为二维数组），此处先为第一维动态分配内存
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    // d.y存储了所有读入照片的标签信息，每条标签包含5条信息：类别，以及矩形框的x,y,w,h
    // boxes为一张图片最多能够处理（参与训练）的矩形框的数（如果图片中的矩形框数多于这个数，那么随机挑选boxes个，这个参数仅在parse_region以及parse_detection中出现，好奇怪？    
    // 在其他网络解析函数中并没有出现）。同样，d.y是一个matrix，make_matrix会指定y的行数和列数，同时会为其第一维动态分配内存
    d.y = make_matrix(n, 5*boxes);

    // 依次读入每一张图片到d.X.vals的适当位置，同时读入对应的标签信息到d.y.vals的适当位置
    for(i = 0; i < n; ++i){
        // 读入的原始图片
        image orig = load_image_color(random_paths[i], 0, 0);

        // 原始图片经过一系列处理（重排及变换）之后的最终得到的图片，并初始化像素值全为0.5（下面会称之为输出图或者最终图之类的）
        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);
        
        // 缩放抖动大小：缩放抖动系数乘以原始图宽高即得像素单位意义上的缩放抖动
        float dw = jitter * orig.w;
        float dh = jitter * orig.h;

        // 中间图的长宽比（aspect ratio，此处其实是宽高比）：新的长宽比是一个随机数
        float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
        
        // 为了方便，引入了一个虚拟的中间图（之所以称为虚拟，是因为这个中间图并不是一个真实存在的变量），
        // 下面两个变量nh,nw其实是中间图的高宽，而scale就是中间图相对于输出图sized的缩放尺寸（比sized大或者小）
        // 中间图与sized 并不是保持长宽比等比例缩放，中间图的长宽比为new_ar，而sized的长宽比为w/h，
        // 二者之间的唯一的关系就是有一条边（宽或高）的长度比例为scale
        float scale = rand_uniform(.25, 2);

        // nw,nh为中间图的宽高, new_ar为中间图的宽高比
        float nw, nh;
        
        if(new_ar < 1){
            // new_ar<1，说明宽度小于高度，则以高度为主，宽度按高度的比例计算
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            // 否则说明高度小于等于宽度，则以宽度为主，高度按宽度比例计算 
            nw = scale * w;
            nh = nw / new_ar;
        }

        // 得到0~w-nw之间的均匀随机数（w-nw可能大于0,可能小于0，因为scale可能大于1,也可能小于1）
        float dx = rand_uniform(0, w - nw);
        // 得到0~h-nh之间的均匀随机数（h-nh可能大于0,可能小于0）
        float dy = rand_uniform(0, h - nh);

        // place_image先将orig根据中间图的尺寸nw,nh进行重排（双线性插值，不是等比例缩放，长宽比可能会变），而后，将中间图放入到sized，
        // dx,dy是将中间图放入到sized的起始坐标位置（dx,dy若大于0,说明sized的尺寸大于中间图的尺寸，这时
        // 可以说是将中间图随机嵌入到sized中的某个位置；dx,dy若小于0,说明sized的尺寸小于中间图的尺寸，这时
        // sized相当于是中间图的一个mask，在中间图上随机抠图）
        place_image(orig, nw, nh, dx, dy, sized);

        // 随机对图像jitter（在hsv三个通道上添加扰动），实现数据增强
        random_distort_image(sized, hue, saturation, exposure);
        // 随机的决定是否进行左右翻转操作来实现数据增强（注意是直接对sized，不是对原始图，也不是中间图）
        int flip = rand()%2;
        if(flip) flip_image(sized);

        // d.X为图像数据，是一个矩阵（二维数组），每一行为一张图片的数据
        d.X.vals[i] = sized.data;

        // d.y包含所有图像的标签信息（包括真实类别与位置），d.y.vals是一个矩阵（二维数组），每一行含一张图片的标签信息
        // 因为对原始图片进行了数据增强，其中的平移抖动势必会改动每个物体的矩形框标签信息（主要是矩形框的像素坐标信息），需要根据具体的数据增强方式进行相应矫正
        // 后面4个参数就是用于数据增强后的矩形框信息矫正（nw,nh是中间图宽高，w,h是最终图宽高）
        fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h);

        free_image(orig);
    }
    free(random_paths);
    return d;
}

/*
** 
*/
void *load_thread(void *ptr)
{
    //printf("Loading data: %d\n", rand());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == REGRESSION_DATA){
        *a.d = load_data_regression(a.paths, a.n, a.m, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center);
    } else if (a.type == SUPER_DATA){
        *a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
    } else if (a.type == WRITING_DATA){
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    } else if (a.type == REGION_DATA){
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == DETECTION_DATA){
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == SWAG_DATA){
        *a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    } else if (a.type == COMPARE_DATA){
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    } else if (a.type == LETTERBOX_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    } else if (a.type == TAG_DATA){
        *a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}

/*
** 创建一个线程，读入相应图片数据（此时args.n不再是一次迭代读入的所有图片的张数，而是经过load_threads()均匀分配给每个线程的图片张数）
** 输入： ptr    包含该线程要读入图片数据的信息（读入多少张，读入图片最终的宽高，图片路径等等）
** 返回： phtread_t   线程id
** 说明： 本函数实际没有做什么，就是深拷贝了args给ptr,然后创建了一个调用load_thread()函数的线程并返回线程id
*/
pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    // 同样第一件事深拷贝了args给ptr（为什么每次都要做这一步呢？求指点啊～）
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    // 创建一个线程，读入相应数据，绑定load_thread()函数到该线程上，第四个参数是load_thread()的输入参数，第二个参数表示线程属性，设置为0（即NULL）
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

/*
** 开辟多个线程读入图片数据，读入数据存储至ptr.d中（主要调用load_in_thread()函数完成）
** 输入： ptr    包含所有线程要读入图片数据的信息（读入多少张，开几个线程读入，读入图片最终的宽高，图片路径等等）
** 返回： void*  万能指针（实际上不需要返回什么）
** 说明： 1) load_threads()是一个指针函数，只是一个返回变量为void*的普通函数，不是函数指针
**       2) 输入ptr是一个void*指针（万能指针），使用时需要强转为具体类型的指针
**       3) 函数中涉及四个用来存储读入数据的变量：ptr, args, out, buffers，除args外都是data*类型，所有这些变量的
**          指针变量其实都指向同一块内存（当然函数中间有些动态变化），因此读入的数据都是互通的。
** 流程： 本函数首先会获取要读入图片的张数、要开启线程的个数，而后计算每个线程应该读入的图片张数（尽可能的均匀分配），
**       并创建所有的线程，并行读入数据，最后合并每个线程读入的数据至一个大data中，这个data的指针变量与ptr的指针变量
**       指向的是统一块内存，因此也就最终将数据读入到ptr.d中（所以其实没有返回值）
*/
void *load_threads(void *ptr)
{
    int i;
    // 先使用(load_args*)强转void*指针，而后取ptr所指内容赋值给args
    // 虽然args不是指针，args是深拷贝了ptr中的内容，但是要知道prt（也就是load_args数据类型），有很多的
    // 指针变量，args深拷贝将拷贝这些指针变量到args中（这些指针变量本身对ptr来说就是内容，
    // 而args所指的值是args的内容，不是ptr的，不要混为一谈），因此，args与ptr将会共享所有指针变量所指的内容
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    // 另指针变量out=args.d，使得out与args.d指向统一块内存，之后，args.d所指的内存块会变（反正也没什么用了，变就变吧），
    // 但out不会变，这样可以保证out与最原始的ptr指向同一块存储读入图片数据的内存块，因此最终将图片读到out中，
    // 实际就是读到了最原始的ptr中，比如train_detector()函数中定义的args.d中
    data *out = args.d;
    // 读入图片的总张数= batch * subdivision * ngpus，可参见train_detector()函数中的赋值
    int total = args.n;
    // 释放ptr：ptr是传入的指针变量，传入的指针变量本身也是按值传递的，即传入函数之后，指针变量得到复制，函数内的形参ptr
    // 获取外部实参的值之后，二者本身没有关系，但是由于是指针变量，二者之间又存在一丝关系，那就是函数内形参与函数外实参指向
    // 同一块内存。又由于函数外实参内存是动态分配的，因此函数内的形参可以使用free()函数进行内存释放，但一般不推荐这么做，因为函数内释放内存，
    // 会影响函数外实参的使用，可能使之成为野指针，那为什么这里可以用free()释放ptr呢，不会出现问题吗？
    // 其一，因为ptr是一个结构体，是一个包含众多的指针变量的结构体，如data* d等（当然还有其他非指针变量如int h等），
    // 直接free(ptr)将会导致函数外实参无法再访问非指针变量int h等（实际经过测试，在gcc编译器下，能访问但是值被重新初始化为0），
    // 因为函数内形参和函数外实参共享一块堆内存，而这些非指针变量都是存在这块堆内存上的，内存一释放，就无法访问了；
    // 但是对于指针变量，free(ptr)将无作为（这个结论也是经过测试的，也是用的gcc编译器），不会释放或者擦写掉ptr指针变量本身的值，
    // 当然也不会影响函数外实参，更不会牵扯到这些指针变量所指的内存块，总的来说，
    // free(ptr)将使得ptr不能再访问指针变量（如int h等，实际经过测试，在gcc编译器下，能访问但是值被重新初始化为0），
    // 但其指针变量本身没有受影响，依旧可以访问；对于函数外实参，同样不能访问非指针变量，而指针变量不受影响，依旧可以访问。
    // 其二，darknet数据读取的实现一层套一层（似乎有点罗嗦，总感觉代码可以不用这么写的:)），具体调用过程如下：
    // load_data(load_args args)->load_threads(load_args* ptr)->load_data_in_thread(load_args args)->load_thread(load_args* ptr)，
    // 就在load_data()中，重新定义了ptr，并为之动态分配了内存，且深拷贝了传给load_data()函数的值args，也就是说在此之后load_data()函数中的args除了其中的指针变量指着同一块堆内存之外，
    // 二者的非指针变量再无瓜葛，不管之后经过多少个函数，对ptr的非指针变量做了什么改动，比如这里直接free(ptr)，使得非指针变量值为0,都不会影响load_data()中的args的非指针变量，也就不会影响更为顶层函数中定义的args的非指针变量的值，
    // 比如train_detector()函数中的args，train_detector()对args非指针变量赋的值都不会受影响，保持不变。综其两点，此处直接free(ptr)是安全的。
    // 说明：free(ptr)函数，确定会做的事是使得内存块可以重新分配，且不会影响指针变量ptr本身的值，也就是ptr还是指向那块地址， 虽然可以使用，但很危险，因为这块内存实际是无效的，
    //      系统已经认为这块内存是可分配的，会毫不考虑的将这块内存分给其他变量，这样，其值随时都可能会被其他变量改变，这种情况下的ptr指针就是所谓的野指针（所以经常可以看到free之后，置原指针为NULL）。
    //      而至于free(ptr)还不会做其他事情，比如会不会重新初始化这块内存为0（擦写掉），以及怎么擦写，这些操作，是不确定的，可能跟具体的编译器有关（个人猜测），
    //      经过测试，对于gcc编译器，free(ptr)之后，ptr中的非指针变量的地址不变，但其值全部擦写为0；ptr中的指针变量，丝毫不受影响，指针变量本身没有被擦写，
    //      存储的地址还是指向先前分配的内存块，所以ptr能够正常访问其指针变量所指的值。测试代码为darknet_test_struct_memory_free.c。
    //      不知道这段测试代码在VS中执行会怎样，还没经过测试，也不知道换用其他编译器（darknet的Makefile文件中，指定了编译器为gcc），darknet的编译会不会有什么问题？？
    //      关于free()，可以看看：http://blog.sina.com.cn/s/blog_615ec1630102uwle.html，文章最后有一个很有意思的比喻，但意思好像就和我这里说的有点不一样了（到底是不是编译器搞得鬼呢？？）。
    free(ptr);

    // 每一个线程都会读入一个data，定义并分配args.thread个data的内存
    data *buffers = calloc(args.threads, sizeof(data));

    // 此处定义了多个线程，并为每个线程动态分配内存
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i){
        // 此处就承应了上面的注释，args.d指针变量本身发生了改动，使得本函数的args.d与out不再指向同一块内存，
        // 改为指向buffers指向的某一段内存，因为下面的load_data_in_thread()函数统一了结口，需要输入一个load_args类型参数，
        // 实际是想把图片数据读入到buffers[i]中，只能令args.d与buffers[i]指向同一块内存
        args.d = buffers + i;
        // 下面这句很有意思，因为有多个线程，所有线程读入的总图片张数为total，需要将total均匀的分到各个线程上，
        // 但很可能会遇到total不能整除的args.threads的情况，比如total = 61, args.threads =8,显然不能做到
        // 完全均匀的分配，但又要保证读入图片的总张数一定等于total，用下面的语句刚好在尽量均匀的情况下，
        // 保证总和为total，比如61,那么8个线程各自读入的照片张数分别为：7, 8, 7, 8, 8, 7, 8, 8
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        // 开启线程，读入数据到args.d中（也就读入到buffers[i]中）
        // load_data_in_thread()函数返回所开启的线程，并存储之前已经动态分配内存用来存储所有线程的threads中，
        // 方便下面使用pthread_join()函数控制相应线程
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i){
        // 以阻塞的方式等待线程threads[i]结束：阻塞是指阻塞启动该子线程的母线程（此处应为主线程），
        // 是母线程处于阻塞状态，一直等待所有子线程执行完（读完所有数据）才会继续执行下面的语句
        // 关于多线程的使用，进行过代码测试，测试代码对应：darknet_test_pthread_join.c
        pthread_join(threads[i], 0);
    }
    // 多个线程读入所有数据之后，分别存储到buffers[0],buffers[1]...中，接着使用concat_datas()函数将buffers中的数据全部合并成一个大数组得到out
    *out = concat_datas(buffers, args.threads);
    // 也就只有out的shallow敢置为0了，为什么呢？因为out是此次迭代读入的最终数据，该数据参与训练（用完）之后，当然可以深层释放了，而此前的都是中间变量，
    // 还处于读入数据阶段，万不可设置shallow=0
    out->shallow = 0;

    // 释放buffers，buffers也是个中间变量，切记shallow设置为1,如果设置为0,那就连out中的数据也没了
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    // 最终直接释放buffers,threads，注意buffers是一个存储data的一维数组，上面循环中的内存释放，实际是释放每一个data的部分内存
    // （这部分内存对data而言是非主要内存，不是存储读入数据的内存块，而是存储指向这些内存块的指针变量，可以释放的）
    free(buffers);
    free(threads);
    return 0;
}

/*
** 开辟线程，读入一次迭代所需的所有图片数据：读入图片的张数为args.n = net.batch * net.subdivisions * ngpus，读入数据将存入args.d中（虽然args是按值传递的，但是args.d是指针变量，函数内改变args.d所指的内容在函数外也是有效的）
** 输入： args    包含要读入图片数据的信息（读入多少张，开几个线程读入，读入图片最终的宽高，图片路径等等）
** 返回： 创建的读取数据的线程id，以便于外界控制线程的进行
** 说明： darknet作者在实现读入图片数据的时候，感觉有点绕来绕去的（也许此中有深意，只是我还未明白～），
**       总的流程是：load_data(load_args args)->load_threads(load_args* ptr)->load_data_in_thread(load_args args)->load_thread(load_args* ptr),
**       load_thread()函数中会选择具体的读入函数，比如load_data_detection()，要强调的是，所有这些函数，都会输入load_args类型参数，或者是其指针，
**       而且很多函数开头就会新建一个args，并且深拷贝传入的args，而后返回新建的args，同时可能会随意改动甚至释放传入的args，这一整串流程只要记住一点：不管args在之后的函数中怎么变，
**       其中的指针变量所指的内存块其实都是一样的（所以不管在哪个函数中，不要随意深层释放args.d的数据，除非是真正的用完了，而这里还处于数据读入阶段，所以决不能深层释放args.d）；
**       而非指针变量，每个函数都不尽相同，但是没有关系，因为load_data()函数中传入的args是按值传递（不是指针）的，不管之后的函数怎么改动args的非指针变量，都不会影响load_data()函数外的值。
** 碎碎念（TODO）：又是按值传递的哎，虽然上面说道因为按值传递使得更改args非指针变量不会影响外界的args，但是传入指针也可以做到啊，反正一进load_data()就深拷贝了args给ptr，而load_data()也没改动args的非指针变量。
*/
pthread_t load_data(load_args args)
{
    // 定义一个线程id
    pthread_t thread;
    // 深拷贝args到ptr（虽说是深拷贝，但要注意args还有很多的指针变量，所有这些指针变量，ptr与args都指向同一内存块）
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;

    // 创建相应线程，并绑定load_threads()函数到该线程上，第二参数是线程的属性，这里设置为0（即NULL）,第四个参数ptr就是load_threads()的输入参数
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    // 返回创建的线程id
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0);
    if(m) free(paths);
    return d;
}

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;

    int i;
    d.X.rows = n;
    d.X.vals = calloc(n, sizeof(float*));
    d.X.cols = w*h*3;

    d.y.rows = n;
    d.y.vals = calloc(n, sizeof(float*));
    d.y.cols = w*scale * h*scale * 3;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_crop_image(im, w*scale, h*scale);
        int flip = rand()%2;
        if (flip) flip_image(crop);
        image resize = resize_image(crop, w, h);
        d.X.vals[i] = resize.data;
        d.y.vals[i] = crop.data;
        free_image(im);
    }

    if(m) free(paths);
    return d;
}

data load_data_regression(char **paths, int n, int m, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_regression_labels_paths(paths, n);
    if(m) free(paths);
    return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy);
    if(m) free(paths);
    return d;
}

data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.w = size;
    d.h = size;
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_tags_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

/*
** 合并矩阵m1,m2至一个大矩阵并返回
** 输入： m1    待合并的矩阵1
**       m2    待合并的矩阵2
** 返回： matrix类型，合并后的大矩阵
*/
matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    // 新建返回的大矩阵
    matrix m;
    // 大矩阵列数不变，行数是m1,m2之后（一行对应一张图片数据）
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    // 为vals的第一维动态分配内存（下面并没有为第二维动态分配内存，从下面的for循环可以看出，第二维采用浅拷贝方式）
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));

    // 将m1中的图片数据逐行拷贝给大矩阵m（浅拷贝）
    for(i = 0; i < m1.rows; ++i){
        // 为vals的每一行赋值
        // 注意m1.vals[i]是每一行的首地址，因此m1.vals[i]是个指针变量，也就是此处直接将每行的地址赋值给了待输出矩阵的每一行的指针变量（前拷贝），
        // 所以m实际与m1共享了数据（指向了同一块内存）
        m.vals[count++] = m1.vals[i];
    }

    // 紧接着将m2中的图片数据逐行拷贝给大矩阵m（浅拷贝）
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

/*
** 合并某一块图片数据到大data中并返回
** 输入： d1    待合并的data块
**       d2    待合并的data块
** 返回： data类型数据
** 碎碎念（TODO）：为什么总是动不动返回或者传入data等其他结构体实例呢，传指针不是更快么:(
*/
data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    // 合并图片数据至d.X中
    d.X = concat_matrix(d1.X, d2.X);
    // 合并图片的标签数据至d.y中
    d.y = concat_matrix(d1.y, d2.y);
    return d;
}

/*
** 合并读入的所有图片数据至一个data中：图片数据不是一起读入的，分了多个线程读入，全部存储在d中（d是一个元素类型为data的一维数组）
** 输入： d    读入的所有图片数据，第i个线程读入的部分图片数据存储在d[i]中
**       n    d的维数，也即数据分了多少块（使用多少个线程读入）
** 返回： data类型，包含所有图片
** 说明： 虽然返回的out是在函数内新建的，但实际上，out都只是浅拷贝了这些碎块的图片数据，也即与这些碎块共享数据，
**       因此，无论是哪个变量，其数据都不能释放（即data.shallow只能设置为1,不能深层释放数据的内存）
** 碎碎念（TODO）：为什么总是动不动返回或者传入data等其他结构体实例呢，传指针不是更快么:(
*/
data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        // i = 0时，因为out = {0},表明默认初始化其中所有的元素为0值，对于指针变量而言，那就是空指针，
        // free()函数可以接受一个空指针，此时free()函数将无作为，但不会出错；当i != 1时，out等于上一次循环得到的new，
        // 在concat_data()函数中，会置其shallow元素值为1,也即进行浅层释放，为什么进行浅层释放呢？
        // 整个的concat_datas函数在合并矩阵，在合并每一个小矩阵时，小矩阵的数据并没有进行深拷贝（深拷贝是指内容拷贝），
        // 而是直接将小矩阵每一行的指针变量赋值给了大矩阵的对应行的指针变量（浅拷贝）（可以回看data.c文件中的concat_matrix()函数for循环中的赋值操作，
        // m1.vals[i]是指向二维数组每一行开头的指针变量，也即复制的是指针变量本身，而不是指针变量所指的值，这就是浅拷贝），
        // 所以小矩阵out与得到的新的大矩阵实际会共享数据，因此，下面只能释放out.X.vals及out.y.vals用于存储每一行指针变量的堆内存，
        // 而不能内释放用于存储真实数据的堆内存，也即只能进行浅层释放，不能进行深层释放。
        free_data(out);
        // 置out等于新得到的大矩阵new，此时out.X.vals以及out.y.vals存储的指向每一行数据的指针变量包括
        // 目前为止得到的整个大矩阵的所有行
        out = new;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d = {0};
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

/*
** 从输入d中深拷贝n张图片的数据与标签信息至X与y中：将d.X.vals以及d.y.vals（如有必要）逐行深拷贝至X,y中
** 输入： d    读入的图片数据，按行存入其中（一张图片对应一行，每一行又按行存储每张图片）
**       n    从d中深拷贝n张图片的数据与标签信息到X与y中
**       offset    相对d首地址的偏移，表示从d中哪张图片开始拷贝（从此往后拷n张图片）
**       X    d.X.vals目标存储变量
**       y    d.y.vals目标存储变量
** 注意：举例说明，在network.c中的train_network()中，调用了本函数，其中输入n=net.batch（关于net.batch这个参数，可以参考network.h中的注释），
**      而d中实际包含了的图片张数为：net.batch*net.subdivision（可以参考detector.c中的train_detector(),你可能注意到这里没有乘以ngpu，
**      是因为train_network()函数对应处理ngpu=1的情况，如果有多个gpu，那么会首先调用train_networks()函数，在其之中会调用get_data_part()将数据平分至每个gpu上，
**      而后再调用train_network()，总之最后train_work()的输入d中只包含net.batch*net.subdivision张图片），可知本函数只是获取其中一个小batch图片，事实上，
**      从train_network()函数中也可以看出，每次训练一个真实的batch的图片（所谓真实的batch，是指网络配置文件中指定的一个batch中所含的图片张数，详细参考network.h中的注释），
**      又分了d.X.rows/batch次完成，d.X.rows/batch实际就是net.subdivision的值（可以参考data.c中load_data_detection()）。
*/
void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    // 下面逐行将输入d中的数据深拷贝至X中
    for(j = 0; j < n; ++j){
        int index = offset + j;
        // memcpy(void* destination,const void *source,size_t num);函数用来复制块内存，常用于数组间的复制赋值（指针所指内容复制，不是指针复制）
        // 此处将d.X中的数据d.X.vals中的某一行深拷贝至输入X中
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));

        // 如果y也分配了内存（也有这个需求），那么也将d.y中的某一行拷贝至y中
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class = bytes[0];
            y.vals[i+b*10000][class] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}

data load_go(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp))){
        int i;
        if(count == X.rows){
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, "%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i){
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = rand()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

data copy_data(data d)
{
    data c = {0};
    c.w = d.w;
    c.h = d.h;
    c.shallow = 0;
    c.num_boxes = d.num_boxes;
    c.boxes = d.boxes;
    c.X = copy_matrix(d.X);
    c.y = copy_matrix(d.y);
    return c;
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = calloc(num, sizeof(float *));
    r.y.vals = calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i){
        int index = rand()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

