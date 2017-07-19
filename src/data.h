#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#include "matrix.h"
#include "list.h"
#include "image.h"
#include "tree.h"

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}

/*
** 一次
*/
typedef struct{
    int w, h;
    matrix X;
    matrix y;
    // shallow是指深层释放X,y中vals的内存还是浅层释放X,y中vals（注意是X,y的vals元素，不是X,y本身，X,y本身是万不能用free释放的，因为其连指针都不是）的内存
    // X,y的vals是一个二维数组，如果浅层释放，即直接释放free(vals)，这种释放并不会真正释放二维数组的内容，因为二维数组实际是指针的指针，
    // 这种直接释放只是释放了用于存放第二层指针变量的内存，也就导致二维数组的每一行将不再能通过二维数组名来访问了（因此这种释放，
    // 需要先将数据转移，使得有其他指针能够指向这些数据块，不能就会造成内存溢出了）；
    // 深层释放，是循环逐行释放为每一行动态分配的内存，然后再释放vals，释放完后，整个二维数组包括第一层存储的第二维指针变量以及实际数据将都不再存在，所有数据都被清空。
    // 详细可查看free_data()以及free_matrix()函数。
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

/*
** 图片检测标签数据：图片检测包括识别与定位，定位通过一个矩形框来实现，
** 因此，对于图片检测，标签数据依次包括：物体类别id，矩形框中心点x,y坐标，
** 矩形框宽高，以及矩形框四个角点的最小最大x,y坐标
*/
typedef struct{
    int id;                             // 矩形框类别（即矩形框框起来的物体的标签/类别id）
    float x,y,w,h;                      // 矩形中心点的x,y坐标，以及矩形宽高w,h（值不是真实的像素坐标，而是相对输入图片的宽高比例）
    float left, right, top, bottom;     // 矩形四个角点的最大最小x,y坐标（值不是真实的像素坐标，而是相对输入图片的宽高比例）
} box_label;

void free_data(data d);

pthread_t load_data(load_args args);

pthread_t load_data_in_thread(load_args args);

void print_letters(float *pred, int n);
data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
data load_data_super(char **paths, int n, int m, int w, int h, int scale);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
data load_data_regression(char **paths, int n, int m, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
data load_go(char *filename);

box_label *read_boxes(char *filename, int *n);
data load_cifar10_data(char *filename);
data load_all_cifar10();

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

list *get_paths(char *filename);
char **get_labels(char *filename);
void get_random_batch(data d, int n, float *X, float *y);
data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
void get_next_batch(data d, int n, int offset, float *X, float *y);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_data(data d1, data d2);
data concat_datas(data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);
data copy_data(data d);

#endif
