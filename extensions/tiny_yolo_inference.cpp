#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <arm_neon.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

/*
浅谈deep learning framework

所有的DL framework 其实都是一个数据流处理系统；在事先会定义整个神经网络的计算图，最初的静态计算图，以及比较fashion的动态计算图；
这里解释一下是什么静态计算图，以及实现过程；
我们都知道一个神经网络有很多的层组成（卷积，池化，全接连，激活，BN，反卷积，损失层，解析层，等等）；可以理解为每一个层都是一个特殊的处理函数f(x)；
下面在一个简短的例子中，将这些抽象的概念通俗化；

这里假设输入为x，输出为y；
卷积层为conv(),池化层为pool()，batchnorm层为bn(),激活层为relu() (这里选择relu作为激活函数，就以此命名) 先定义这么多，开始举例：
假设我们现在有一个三层的卷积神经网络，定义可如下所示:（这里使用伪代码来表示）
												其中一层如下：
												y = conv(x)
												y = bn(y)
												y = relu(y)
												y = pool(y)
												
										当然可以这么写 y = pool(relu(bn(conv(x))))
										这里神经网络有三层，完整如下：
											y = pool(relu(bn(conv(x))))
											y = pool(relu(bn(conv(y))))
											y = pool(relu(bn(conv(y))))
如何这个神经网络用作目标检测，那我们还需要加一个精心设计的detection层，用来解析神经网络的输出y，得到我们最终的结果，也就是在图像上画框框，也就是yolo里面的regionlayer层,这里我们定义为detector()
											y = detector(y)


上述就是一个静态计算图，在caffe中定义在prototxt当中，在darknet中也就是.cfg文件
那么在嵌入式移植的过程中，如何尽量小的减少开销，不使用任何的框架呢，那就直接手写不需要计算图，因为我们每一步的运算就是计算图；
下面一整个cpp时本人在实际项目中的tiny_yolo的整体inference算法移植；
实际读取图片并为使用opencv
但是为了方便给大家做展示这里只依赖于opencv，当然会一些针对neon的写法，这里可忽略不计，用正常的写法即可；
*／

/*
						简介：这是一个对目标识别算法 yolov2 的inference移植；
						本文件分为三部分：
						第一部分：所有中间层处理函数的定义
						第二部分：申请，释放内存操作函数以及结构体的定义
						第三部分：main() 函数，进行tiny_yolo的inference 也就是上述得到y的过程
*/





typedef struct _img_inf
{
	int rows;
	int cols;
	int channels;
}img_inf;

inline float get_pixel(float *img, int x, int y, int c,img_inf &picture)
{
	return *(img + c*picture.rows*picture.cols+y*picture.cols + x);
}

inline void set_pixel(float *img, int x, int y, int c, float val, img_inf &picture)
{
	if( static_cast<unsigned int>(x) < picture.cols || static_cast<unsigned int>(y) < picture.rows )
		img[c*picture.rows*picture.cols + y*picture.cols + x] = val;
	else
		return;

	return;
}

inline void add_pixel(float *img, const int x, const int y, const int c, const float val, img_inf &picture)
{
	img[c*picture.rows*picture.cols + y*picture.cols + x] += val;
	return;
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

//最终画图框的颜色定义
cv::Scalar colors[] = { { 0, 0, 0 }, { 47, 79, 79 }, { 255, 218, 185 }, { 100, 149, 237 }, { 139, 58, 98 }, { 255, 255, 0 }, { 178, 34, 34 }, { 148, 0, 211 }, { 0, 255, 255 }, 
                      { 0, 255, 255 }, { 255, 140, 0 }, { 255, 236, 139 }, { 255, 105, 180 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 178, 238 }, { 191, 239, 255 }, { 0, 0, 139 },
					  { 255, 127, 36 }, { 255, 0, 255 }}; // CvScalar

//检测目标名称
const string labels[20] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                            "table", "dog", "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "monitor" };

//img2col
void im2col_cpu( const float* data_im, const int channels,
                 const int height, const int width, const int kernel_size, const int pad, const int stride, float* data_col) 
{
  const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
  const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
        int input_row = -pad + kernel_row;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad + kernel_col;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride;
            }
          }
          input_row += stride;
        }
      }
    }
  }

  return;
}

//gemm with bn
void sparse_gemm_naive(float* bottom, float*weights_nozero_value, int*rowPtr, unsigned short* colIndex, int weights_h, int bottom_w, float* top, float* biases, float* scale, float* mean, float* val)
{
	int nozero_index = 0;
	for(int w = 0; w < weights_h; ++w)
	{
		int nozero_per_row = rowPtr[w + 1] - rowPtr[w];
		//printf("nozero_per_row = %d\n", nozero_per_row);
		for(int j = 0; j < bottom_w; ++j)
		{
			int output_index = w * bottom_w + j;
			//printf("output_index = %d\n", output_index);
			for(int k = 0; k < nozero_per_row; ++k)
			{
				int weight_index = colIndex[nozero_index + k];
				int input_index  = weight_index * bottom_w + j;
				top[output_index] += weights_nozero_value[nozero_index + k] * bottom[input_index];
				// printf("weight = %f, bottom = %f\n", weights_nozero_value[nozero_index + k], bottom[input_index]);
			}
			//

			top[output_index] = scale[w] * (top[output_index] - mean[w]) / sqrt(val[w] + 0.0000001);
			//

			top[output_index] += biases[w];
			top[output_index] = (top[output_index] > 0) ? top[output_index] : 0.1 * top[output_index];
			// if(top[output_index] != 0)
			// 	printf("index = %d,  %f\n", output_index, top[output_index]);
		}
		nozero_index += nozero_per_row;
	}

	return;
}

//gemm  without bn
void sparse_gemm_naive_no_BN(float* bottom, float*weights_nozero_value, int*rowPtr, unsigned short* colIndex, int weights_h, int bottom_w, float* top, float* biases)
{
	int nozero_index = 0;
	for(int w = 0; w < weights_h; ++w)
	{
		int nozero_per_row = rowPtr[w + 1] - rowPtr[w];
		//printf("nozero_per_row = %d\n", nozero_per_row);
		for(int j = 0; j < bottom_w; ++j)
		{
			int output_index = w * bottom_w + j;
			//printf("output_index = %d\n", output_index);
			for(int k = 0; k < nozero_per_row; ++k)
			{
				int weight_index = colIndex[nozero_index + k];
				int input_index  = weight_index * bottom_w + j;
				top[output_index] += weights_nozero_value[nozero_index + k] * bottom[input_index];
				// printf("weight = %f, bottom = %f\n", weights_nozero_value[nozero_index + k], bottom[input_index]);
			}
			//

			top[output_index] += biases[w];
			// top[output_index] = (top[output_index] > 0) ? top[output_index] : 0.1 * top[output_index];
			// if(top[output_index] != 0)
			// 	printf("index = %d,  %f\n", output_index, top[output_index]);
		}
		nozero_index += nozero_per_row;
	}

	return;
}

// 使用neno进行的 maxpooling 2*2
void max_pooling2x2(float* bottom, int width_in, int width_out, int channels, float* top)
{
	float* ptr[2];
	int width_temp = (width_in / 8) * 8;
	float32x4_t input[4], output[2], out_temp;
	int out_index = 0;
	for(int c = 0; c < channels; ++c)
	{
		for(int i = 0; i < width_in; i += 2)
		{
			ptr[0] = bottom + (c * width_in + i) * width_in;
			ptr[1] = bottom + (c * width_in + i + 1) * width_in;
			for(int j = 0; j < width_temp; j += 8)
			{
				input[0] = vld1q_f32(ptr[0] + j);
				input[1] = vld1q_f32(ptr[1] + j);
				input[2] = vld1q_f32(ptr[0] + j + 4);
				input[3] = vld1q_f32(ptr[1] + j + 4);

				output[0] = vmaxq_f32(input[0], input[1]);
				output[1] = vmaxq_f32(input[2], input[3]);

				out_temp  = vpmaxq_f32(output[0], output[1]);
				out_index = (c * width_out + i / 2) * width_out + j / 2;
				vst1q_f32(top + out_index, out_temp);
			}
			for(int k = width_temp; k < width_in; k += 2)
			{
				float max0 = std::max(*(ptr[0] + k), *(ptr[0] + k + 1));
				float max1 = std::max(*(ptr[1] + k), *(ptr[1] + k + 1));
				out_index = (c * width_out + i / 2) * width_out + k / 2;
				top[out_index] = std::max(max0, max1);
			}

		}
	}

	return;
}

//maxpooling 1*1 without neno
void maxpool_layer_1x1_const(float *input,float *output)
{

    //float *indexes = (float*)calloc(output_size, sizeof(int));
    int stride = 1;
    int kernel_size = 2;
    int w_offset = 0; // -padding
    int h_offset = 0;
    int b,i,j,k,m,n;

    int batch = 1;
    int h = 13;
    int w = 13;
    int c = 512; //output channel

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < kernel_size; ++n){
                        for(m = 0; m < kernel_size; ++m){
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                         cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    output[out_index] = max;
                    //ndexes[out_index] = max_i;
                }
            }
        }
    }
}


typedef struct _box 
{
	float x;
	float y;
	float w;
	float h;
	float p;
}box;

typedef struct _sortable_bbox
{
	int index;
	int Class;
	float **probs;
}sortable_bbox;

inline int nms_comparator(const void *pa, const void *pb)
{
	const sortable_bbox a = *(sortable_bbox* )pa;
	const sortable_bbox b = *(sortable_bbox* )pb;
	float diff = a.probs[a.index][b.Class] - b.probs[b.index][b.Class];
	
	if (diff < 0.f) return 1;
	else if (diff > 0.f) return -1;
	
	return 0;
}

static inline float logistic_activate(const float x){ return 1.f/(1.f + exp(-x)); }

void flatten(float *x, int size, int layers, int batch = 1, int forward = 1)
{
    float swap[125 * 169] = {0.f};
	int i,c;
        for(c = 0; c < layers; ++c)
		{
            for(i = 0; i < size; ++i){
                int i1 = c*size + i; 
                int i2 = i*layers + c; 
                swap[i2] = x[i1];
            }
        }
    memcpy(x, &swap[0], size*layers*batch*sizeof(float));
    free(swap); swap = NULL;

    return;
}

void softmax(const float *input, const int n, const float temp, float *output)
{
    float sum = 0;
    float largest = -FLT_MAX;
    for( int i = 0; i < n; ++i)
	{
        if(input[i] > largest) largest = input[i];
    }
	float tmp = 1.f / temp;
    for( int i = 0; i < n; ++i)
    {
		//TODO: Taylor expasion to optimize the exp
        const float e = exp( (input[i] - largest) * tmp );
        sum += e;
        output[i] = e;
    }
	tmp = 1.f / sum;
    for( int i = 0; i < n; ++i )
	{
        output[i] *= tmp;
    }

    return;
}

//linhao
box get_region_box( const float *x, const vector<float>& anchor_boxes, const int n, const int index, 
                     const int i, const int j, const int w, const int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * anchor_boxes[2*n]   / w;
    b.h = exp(x[index + 3]) * anchor_boxes[2*n+1] / h;

    return b;
}

void get_region_boxes(float *region_data, int w, int h, float thresh, float **probs, box *boxes,vector<float> anchor_boxes)
{	
	int width = 13;
	int height = 13;
	int classes = 20;
	int number_n = 5;

    int i,j,n;
    float *predictions = region_data; //region_data
    for (i = 0; i < width*height; ++i)
    {
        int row = i / width; //l.w = 13
        int col = i % width;
        for(n = 0; n < number_n; ++n){ //l.n = 5
            int index = i*number_n + n;
            int p_index = index * (classes + 5) + 4; //l.classes = 20
            float scale = predictions[p_index];
            int box_index = index * (classes + 5);
            boxes[index] = get_region_box(predictions, anchor_boxes, n, box_index, col, row, width, height); //l.biases 

            // printf("%f, %f, %f, %f\n", boxes[index].x, boxes[index].y, boxes[index].w, boxes[index].h);

            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;



            int class_index = index * (classes + 5) + 5;

            for(j = 0; j < classes; ++j)
            {
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
        }
    }

    return;
}

inline float overlap(const float x1, const float w1, const float x2, const float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;

    return right - left;
}

float box_intersection(const box& a, const box& b)
{
    const float w = overlap(a.x, a.w, b.x, b.w);
    const float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0){
    	return 0.f;
    }else{
    	return w*h;
    }
}

inline float box_union(const box& a, const box& b)
{
    const float i = box_intersection(a, b);
    const float u = a.w*a.h + b.w*b.h - i;
    return u;
}

const float inline box_iou(box& a, box& b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = (sortable_bbox*)calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].Class = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k)
    {
        for(i = 0; i < total; ++i){
            s[i].Class = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i)
		{
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s); s = NULL;

    return;
}

inline int max_index(float *a, const int n)
{
	int i, max_i = 0;
	float max = a[0];
	for (i = 0; i < n; ++i)
	{
		if (a[i] > max)
		{
			max = a[i];
			max_i = i;
		}
	}

	return max_i;
}

//定义每层所需内存变量
typedef struct _data_package
{
//conv 0
    float *biases_0; 
    float *scale_0;
    float *mean_0;
    float *variance_0;
    float *weights_conv_0_NoneZeroValue;
    int *row_0_ptr;
    unsigned short *col_0_ptr;
    float* feature_matrix_0;
    float *conv_top_data_0;
    float *max_pooling_1;
//conv 2
    float *biases_2; 
    float *scale_2;
    float *mean_2;
    float *variance_2;
    float *weights_conv_2_NoneZeroValue;
    int *row_2_ptr;
    unsigned short *col_2_ptr;
    float* feature_matrix_2;
    float *conv_top_data_2;
    float *max_pooling_3;
//conv 4
    float *biases_4; 
    float *scale_4;
    float *mean_4;
    float *variance_4;
    float *weights_conv_4_NoneZeroValue;
    int *row_4_ptr;
    unsigned short *col_4_ptr;
    float* feature_matrix_4;
    float *conv_top_data_4;
    float *max_pooling_5;
//conv 6
    float *biases_6; 
    float *scale_6;
    float *mean_6;
    float *variance_6;
    float *weights_conv_6_NoneZeroValue;
    int *row_6_ptr;
    unsigned short *col_6_ptr;
    float* feature_matrix_6;
    float *conv_top_data_6;
    float *max_pooling_7;
//conv 8
    float *biases_8; 
    float *scale_8;
    float *mean_8;
    float *variance_8;
    float *weights_conv_8_NoneZeroValue;
    int *row_8_ptr;
    unsigned short *col_8_ptr;
    float* feature_matrix_8;
    float *conv_top_data_8;
    float *max_pooling_9;
//conv 10
    float *biases_10; 
    float *scale_10;
    float *mean_10;
    float *variance_10;
    float *weights_conv_10_NoneZeroValue;
    int *row_10_ptr;
    unsigned short *col_10_ptr;
    float* feature_matrix_10;
    float *conv_top_data_10;
    float *max_pooling_11;
//conv 12
    float *biases_12; 
    float *scale_12;
    float *mean_12;
    float *variance_12;
    float *weights_conv_12_NoneZeroValue;
    int *row_12_ptr;
    unsigned short *col_12_ptr;
    float* feature_matrix_12;
    float *conv_top_data_12;
//conv 13
    float *biases_13; 
    float *scale_13;
    float *mean_13;
    float *variance_13;
    float *weights_conv_13_NoneZeroValue;
    int *row_13_ptr;
    unsigned short *col_13_ptr;
    float* feature_matrix_13;
    float *conv_top_data_13;
//conv 14
    float *biases_14; 
    // float *scale_14;
    // float *mean_14;
    // float *variance_14;
    float *weights_conv_14_NoneZeroValue;
    int *row_14_ptr;
    unsigned short *col_14_ptr;
    float *conv_top_data_14;
}data_package;

//定义内存变量开启所需常量
typedef struct _parameter_package
{
    int kernel_single_size_1 = 3;
    int kernel_single_size_2 = 1;
    int conv_stride = 1;
    int max_stride = 2;
    int padding = 1;
//conv_0
    int conv_height_0_in = 416;
    int conv_width_0_in = 416;
    int conv_channel_0_in = 3;

    int conv_height_0_out = 416;
    int conv_width_0_out = 416;
    int conv_channel_0_out = 16;
    int weights_num_0 = 3*3*3*16;
//conv2
    int conv_height_2_in = 208;
    int conv_width_2_in = 208;
    int conv_channel_2_in = 16;

    int conv_height_2_out = 208;
    int conv_width_2_out = 208;
    int conv_channel_2_out = 32;

    int weights_num_2 = 16*3*3*32; 
//conv4
    int conv_height_4_in = 104;
    int conv_width_4_in = 104;
    int conv_channel_4_in = 32;

    int conv_height_4_out = 104;
    int conv_width_4_out = 104;
    int conv_channel_4_out = 64;

    int weights_num_4 = 32*3*3*64;
//conv6
    int conv_height_6_in = 52;
    int conv_width_6_in = 52;
    int conv_channel_6_in = 64;

    int conv_height_6_out = 52;
    int conv_width_6_out = 52;
    int conv_channel_6_out = 128;

    int weights_num_6 = 64*3*3*128;
//conv8
    int conv_height_8_in = 26;
    int conv_width_8_in = 26;
    int conv_channel_8_in =128;

    int conv_height_8_out = 26;
    int conv_width_8_out = 26;
    int conv_channel_8_out =256;

    int weights_num_8 = 128*3*3*256;
//conv10
    int conv_height_10_in = 13;
    int conv_width_10_in = 13;
    int conv_channel_10_in = 256;

    int conv_height_10_out = 13;
    int conv_width_10_out = 13;
    int conv_channel_10_out = 512;

    int weights_num_10 = 256*3*3*512;
//conv12
    int conv_height_12_in = 13;
    int conv_width_12_in = 13;
    int conv_channel_12_in = 512;

    int conv_height_12_out = 13;
    int conv_width_12_out = 13;
    int conv_channel_12_out = 1024;

    int weights_num_12 = 512*3*3*1024;
//conv13
    int conv_height_13_in = 13;
    int conv_width_13_in = 13;
    int conv_channel_13_in = 1024;

    int conv_height_13_out = 13;
    int conv_width_13_out = 13;
    int conv_channel_13_out = 1024;

    int weights_num_13 = 1024*3*3*1024;
//conv14
    int conv_height_14_in = 13;
    int conv_width_14_in = 13;
    int conv_channel_14_in = 1024;

    int conv_height_14_out = 13;
    int conv_width_14_out = 13;
    int conv_channel_14_out = 125;
    int weights_num_14 = 1024*1*1*125;
}parameter_package;

//申请内存 + 从二进制权重文件中读取  并初始化
void initial_network_space_parameters( data_package& d_pkg, parameter_package& p_pkg )
{
	//printf("");
    const char *path_weights = "tiny-yolo-voc_41000_csr.weights"; //读取权重
    FILE *fptr = fopen(path_weights, "rb");
    /*
	int major;
    int minor;
    int revision;
    int net_seen;
    fread(&major, sizeof(int), 1, fptr);
    fread(&minor, sizeof(int), 1, fptr);
    fread(&revision, sizeof(int), 1, fptr);
    fread(&net_seen, sizeof(int), 1, fptr);
	*/
	// skip the four int vars, linhao
	fseek(fptr, sizeof(int)*4, 0);
//conv 0
    d_pkg.biases_0         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_0_out);
    memset(d_pkg.biases_0, 0x0, sizeof(float) * p_pkg.conv_channel_0_out);
    fread(d_pkg.biases_0, sizeof(float), p_pkg.conv_channel_0_out, fptr);

    d_pkg.scale_0          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_0_out);
	memset(d_pkg.scale_0, 0x0, sizeof(float) * p_pkg.conv_channel_0_out);
	fread(d_pkg.scale_0, sizeof(float), p_pkg.conv_channel_0_out, fptr);

    d_pkg.mean_0           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_0_out);
    memset(d_pkg.mean_0, 0x0, sizeof(float) * p_pkg.conv_channel_0_out);
	fread(d_pkg.mean_0, sizeof(float), p_pkg.conv_channel_0_out, fptr);

    d_pkg.variance_0       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_0_out);
    memset(d_pkg.variance_0, 0x0, sizeof(float) * p_pkg.conv_channel_0_out);
	fread(d_pkg.variance_0, sizeof(float), p_pkg.conv_channel_0_out, fptr);

    int NoneZeroCount_conv_0;
    fread(&NoneZeroCount_conv_0, sizeof(int), 1, fptr);
   // printf("%d\n", NoneZeroCount_conv_0);
    d_pkg.weights_conv_0_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_0);
	memset(d_pkg.weights_conv_0_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_0);
	fread(d_pkg.weights_conv_0_NoneZeroValue, sizeof(float), NoneZeroCount_conv_0, fptr);

    d_pkg.row_0_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_0_out+1));
    memset(d_pkg.row_0_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_0_out+1));
	fread(d_pkg.row_0_ptr, sizeof(int), (p_pkg.conv_channel_0_out+1), fptr);

    d_pkg.col_0_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_0);
    memset(d_pkg.col_0_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_0);
	fread(d_pkg.col_0_ptr, sizeof(unsigned short), NoneZeroCount_conv_0, fptr);

    d_pkg.feature_matrix_0 = (float*)malloc( 9 * p_pkg.conv_channel_0_in * p_pkg.conv_height_0_out * p_pkg.conv_width_0_out * sizeof(float) );
    d_pkg.conv_top_data_0 = (float*)malloc(sizeof(float)*p_pkg.conv_width_0_out*p_pkg.conv_height_0_out*p_pkg.conv_channel_0_out);
    d_pkg.max_pooling_1 = (float*)malloc(sizeof(float)*p_pkg.conv_width_2_in*p_pkg.conv_height_2_in*p_pkg.conv_channel_2_in);
//conv 2    
    d_pkg.biases_2         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_2_out);
    memset(d_pkg.biases_2, 0x0, sizeof(float) * p_pkg.conv_channel_2_out);
    fread(d_pkg.biases_2, sizeof(float), p_pkg.conv_channel_2_out, fptr);

    d_pkg.scale_2          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_2_out);
    memset(d_pkg.scale_2, 0x0, sizeof(float) * p_pkg.conv_channel_2_out);
	fread(d_pkg.scale_2, sizeof(float), p_pkg.conv_channel_2_out, fptr);

    d_pkg.mean_2           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_2_out);
    memset(d_pkg.mean_2, 0x0, sizeof(float) * p_pkg.conv_channel_2_out);
	fread(d_pkg.mean_2, sizeof(float), p_pkg.conv_channel_2_out, fptr);

    d_pkg.variance_2       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_2_out);
    memset(d_pkg.variance_2, 0x0, sizeof(float) * p_pkg.conv_channel_2_out);
	fread(d_pkg.variance_2, sizeof(float), p_pkg.conv_channel_2_out, fptr);

    int NoneZeroCount_conv_2;
    fread(&NoneZeroCount_conv_2, sizeof(int), 1, fptr);
   // printf("%d\n", NoneZeroCount_conv_2);
    d_pkg.weights_conv_2_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_2);
    memset(d_pkg.weights_conv_2_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_2);
	fread(d_pkg.weights_conv_2_NoneZeroValue, sizeof(float), NoneZeroCount_conv_2, fptr);

    d_pkg.row_2_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_2_out+1));
    memset(d_pkg.row_2_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_2_out+1));
	fread(d_pkg.row_2_ptr, sizeof(int), (p_pkg.conv_channel_2_out+1), fptr);

    d_pkg.col_2_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_2);
    memset(d_pkg.col_2_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_2);
	fread(d_pkg.col_2_ptr, sizeof(unsigned short), NoneZeroCount_conv_2, fptr);

    d_pkg.feature_matrix_2 = (float*)malloc( 9 * p_pkg.conv_channel_2_in * p_pkg.conv_height_2_out * p_pkg.conv_width_2_out * sizeof(float) );
    d_pkg.conv_top_data_2 = (float*)malloc(sizeof(float)*p_pkg.conv_width_2_out*p_pkg.conv_height_2_out*p_pkg.conv_channel_2_out);
    d_pkg.max_pooling_3 = (float*)malloc(sizeof(float)*p_pkg.conv_width_4_in*p_pkg.conv_height_4_in*p_pkg.conv_channel_4_in);
//conv 4
    d_pkg.biases_4         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_4_out);
    memset(d_pkg.biases_4, 0x0, sizeof(float) * p_pkg.conv_channel_4_out);
    fread(d_pkg.biases_4, sizeof(float), p_pkg.conv_channel_4_out, fptr);
    d_pkg.scale_4          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_4_out);
    memset(d_pkg.scale_4, 0x0, sizeof(float) * p_pkg.conv_channel_4_out);
	fread(d_pkg.scale_4, sizeof(float), p_pkg.conv_channel_4_out, fptr);

    d_pkg.mean_4           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_4_out);
    memset(d_pkg.mean_4, 0x0, sizeof(float) * p_pkg.conv_channel_4_out);
	fread(d_pkg.mean_4, sizeof(float), p_pkg.conv_channel_4_out, fptr);

    d_pkg.variance_4       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_4_out);
    memset(d_pkg.variance_4, 0x0, sizeof(float) * p_pkg.conv_channel_4_out);
	fread(d_pkg.variance_4, sizeof(float), p_pkg.conv_channel_4_out, fptr);

    int NoneZeroCount_conv_4;
    fread(&NoneZeroCount_conv_4, sizeof(int), 1, fptr);
   // printf("%d\n", NoneZeroCount_conv_4);
    d_pkg.weights_conv_4_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_4);
    memset(d_pkg.weights_conv_4_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_4);
	fread(d_pkg.weights_conv_4_NoneZeroValue, sizeof(float), NoneZeroCount_conv_4, fptr);

    d_pkg.row_4_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_4_out+1));
    memset(d_pkg.row_4_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_4_out+1));
	fread(d_pkg.row_4_ptr, sizeof(int), (p_pkg.conv_channel_4_out+1), fptr);

    d_pkg.col_4_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_4);
    memset(d_pkg.col_4_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_4);
	fread(d_pkg.col_4_ptr, sizeof(unsigned short), NoneZeroCount_conv_4, fptr);

    d_pkg.feature_matrix_4 = (float*)malloc( 9 * p_pkg.conv_channel_4_in * p_pkg.conv_height_4_out * p_pkg.conv_width_4_out * sizeof(float) );
    d_pkg.conv_top_data_4 = (float*)malloc(sizeof(float)*p_pkg.conv_width_4_out*p_pkg.conv_height_4_out*p_pkg.conv_channel_4_out);
    d_pkg.max_pooling_5 = (float*)malloc(sizeof(float)*p_pkg.conv_width_6_in*p_pkg.conv_height_6_in*p_pkg.conv_channel_6_in);
//conv 6    
    d_pkg.biases_6         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_6_out);
    memset(d_pkg.biases_6, 0x0, sizeof(float) * p_pkg.conv_channel_6_out);
    fread(d_pkg.biases_6, sizeof(float), p_pkg.conv_channel_6_out, fptr);
    d_pkg.scale_6          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_6_out);
    memset(d_pkg.scale_6, 0x0, sizeof(float) * p_pkg.conv_channel_6_out);
	fread(d_pkg.scale_6, sizeof(float), p_pkg.conv_channel_6_out, fptr);

    d_pkg.mean_6           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_6_out);
    memset(d_pkg.mean_6, 0x0, sizeof(float) * p_pkg.conv_channel_6_out);
	fread(d_pkg.mean_6, sizeof(float), p_pkg.conv_channel_6_out, fptr);

    d_pkg.variance_6       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_6_out);
    memset(d_pkg.variance_6, 0x0, sizeof(float) * p_pkg.conv_channel_6_out);
	fread(d_pkg.variance_6, sizeof(float), p_pkg.conv_channel_6_out, fptr);

    int NoneZeroCount_conv_6;
    fread(&NoneZeroCount_conv_6, sizeof(int), 1, fptr);
   // printf("%d\n", NoneZeroCount_conv_6);
    d_pkg.weights_conv_6_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_6);
    memset(d_pkg.weights_conv_6_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_6);
	fread(d_pkg.weights_conv_6_NoneZeroValue, sizeof(float), NoneZeroCount_conv_6, fptr);

    d_pkg.row_6_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_6_out+1));
    memset(d_pkg.row_6_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_6_out+1));
	fread(d_pkg.row_6_ptr, sizeof(int), (p_pkg.conv_channel_6_out+1), fptr);

    d_pkg.col_6_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_6);
    memset(d_pkg.col_6_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_6);
	fread(d_pkg.col_6_ptr, sizeof(unsigned short), NoneZeroCount_conv_6, fptr);

    d_pkg.feature_matrix_6 = (float*)malloc( 9 * p_pkg.conv_channel_6_in * p_pkg.conv_height_6_out * p_pkg.conv_width_6_out * sizeof(float) );
    d_pkg.conv_top_data_6 = (float*)malloc(sizeof(float)*p_pkg.conv_width_6_out*p_pkg.conv_height_6_out*p_pkg.conv_channel_6_out);
    d_pkg.max_pooling_7 = (float*)malloc(sizeof(float)*p_pkg.conv_width_8_in*p_pkg.conv_height_8_in*p_pkg.conv_channel_8_in);
//conv 8    
    d_pkg.biases_8         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_8_out);
    memset(d_pkg.biases_8, 0x0, sizeof(float) * p_pkg.conv_channel_8_out);
    fread(d_pkg.biases_8, sizeof(float), p_pkg.conv_channel_8_out, fptr);
    d_pkg.scale_8          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_8_out);
    memset(d_pkg.scale_8, 0x0, sizeof(float) * p_pkg.conv_channel_8_out);
	fread(d_pkg.scale_8, sizeof(float), p_pkg.conv_channel_8_out, fptr);

    d_pkg.mean_8           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_8_out);
    memset(d_pkg.mean_8, 0x0, sizeof(float) * p_pkg.conv_channel_8_out);
	fread(d_pkg.mean_8, sizeof(float), p_pkg.conv_channel_8_out, fptr);

    d_pkg.variance_8       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_8_out);
    memset(d_pkg.variance_8, 0x0, sizeof(float) * p_pkg.conv_channel_8_out);
	fread(d_pkg.variance_8, sizeof(float), p_pkg.conv_channel_8_out, fptr);

    int NoneZeroCount_conv_8;
    fread(&NoneZeroCount_conv_8, sizeof(int), 1, fptr);
  //  printf("%d\n", NoneZeroCount_conv_8);
    d_pkg.weights_conv_8_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_8);
    memset(d_pkg.weights_conv_8_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_8);
	fread(d_pkg.weights_conv_8_NoneZeroValue, sizeof(float), NoneZeroCount_conv_8, fptr);

    d_pkg.row_8_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_8_out+1));
    memset(d_pkg.row_8_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_8_out+1));
	fread(d_pkg.row_8_ptr, sizeof(int), (p_pkg.conv_channel_8_out+1), fptr);

    d_pkg.col_8_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_8);
    memset(d_pkg.col_8_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_8);
	fread(d_pkg.col_8_ptr, sizeof(unsigned short), NoneZeroCount_conv_8, fptr);

    d_pkg.feature_matrix_8 = (float*)malloc( 9 * p_pkg.conv_channel_8_in * p_pkg.conv_height_8_out * p_pkg.conv_width_8_out * sizeof(float) );
    d_pkg.conv_top_data_8 = (float*)malloc(sizeof(float)*p_pkg.conv_width_8_out*p_pkg.conv_height_8_out*p_pkg.conv_channel_8_out);
    d_pkg.max_pooling_9 = (float*)malloc(sizeof(float)*p_pkg.conv_width_10_in*p_pkg.conv_height_10_in*p_pkg.conv_channel_10_in);
//conv 10
    d_pkg.biases_10         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_10_out);
    memset(d_pkg.biases_10, 0x0, sizeof(float) * p_pkg.conv_channel_10_out);
    fread(d_pkg.biases_10, sizeof(float), p_pkg.conv_channel_10_out, fptr);
    d_pkg.scale_10          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_10_out);
    memset(d_pkg.scale_10, 0x0, sizeof(float) * p_pkg.conv_channel_10_out);
	fread(d_pkg.scale_10, sizeof(float), p_pkg.conv_channel_10_out, fptr);

    d_pkg.mean_10           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_10_out);
    memset(d_pkg.mean_10, 0x0, sizeof(float) * p_pkg.conv_channel_10_out);
	fread(d_pkg.mean_10, sizeof(float), p_pkg.conv_channel_10_out, fptr);

    d_pkg.variance_10       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_10_out);
    memset(d_pkg.variance_10, 0x0, sizeof(float) * p_pkg.conv_channel_10_out);
	fread(d_pkg.variance_10, sizeof(float), p_pkg.conv_channel_10_out, fptr);

    int NoneZeroCount_conv_10;
    fread(&NoneZeroCount_conv_10, sizeof(int), 1, fptr);
  //  printf("%d\n", NoneZeroCount_conv_10);
    d_pkg.weights_conv_10_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_10);
    memset(d_pkg.weights_conv_10_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_10);
	fread(d_pkg.weights_conv_10_NoneZeroValue, sizeof(float), NoneZeroCount_conv_10, fptr);

    d_pkg.row_10_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_10_out+1));
    memset(d_pkg.row_10_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_10_out+1));
	fread(d_pkg.row_10_ptr, sizeof(int), (p_pkg.conv_channel_10_out+1), fptr);

    d_pkg.col_10_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_10);
    memset(d_pkg.col_10_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_10);
	fread(d_pkg.col_10_ptr, sizeof(unsigned short), NoneZeroCount_conv_10, fptr);

    d_pkg.feature_matrix_10 = (float*)malloc( 9 * p_pkg.conv_channel_10_in * p_pkg.conv_height_10_out * p_pkg.conv_width_10_out * sizeof(float) );
    d_pkg.conv_top_data_10 = (float*)malloc(sizeof(float)*p_pkg.conv_width_10_out*p_pkg.conv_height_10_out*p_pkg.conv_channel_10_out);
    d_pkg.max_pooling_11 = (float*)malloc(sizeof(float)*p_pkg.conv_width_12_in*p_pkg.conv_height_12_in*p_pkg.conv_channel_12_in);
//conv 12
    d_pkg.biases_12         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_12_out);
    memset(d_pkg.biases_12, 0x0, sizeof(float) * p_pkg.conv_channel_12_out);
    fread(d_pkg.biases_12, sizeof(float), p_pkg.conv_channel_12_out, fptr);
    d_pkg.scale_12          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_12_out);
    memset(d_pkg.scale_12, 0x0, sizeof(float) * p_pkg.conv_channel_12_out);
	fread(d_pkg.scale_12, sizeof(float), p_pkg.conv_channel_12_out, fptr);

    d_pkg.mean_12           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_12_out);
    memset(d_pkg.mean_12, 0x0, sizeof(float) * p_pkg.conv_channel_12_out);
	fread(d_pkg.mean_12, sizeof(float), p_pkg.conv_channel_12_out, fptr);

    d_pkg.variance_12       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_12_out);
    memset(d_pkg.variance_12, 0x0, sizeof(float) * p_pkg.conv_channel_12_out);
	fread(d_pkg.variance_12, sizeof(float), p_pkg.conv_channel_12_out, fptr);

    int NoneZeroCount_conv_12;
    fread(&NoneZeroCount_conv_12, sizeof(int), 1, fptr);
   // printf("conv12: %d\n", NoneZeroCount_conv_12);
    d_pkg.weights_conv_12_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_12);
    memset(d_pkg.weights_conv_12_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_12);
	fread(d_pkg.weights_conv_12_NoneZeroValue, sizeof(float), NoneZeroCount_conv_12, fptr);

    d_pkg.row_12_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_12_out+1));
    memset(d_pkg.row_12_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_12_out+1));
	fread(d_pkg.row_12_ptr, sizeof(int), (p_pkg.conv_channel_12_out+1), fptr);

    d_pkg.col_12_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_12);
    memset(d_pkg.col_12_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_12);
	fread(d_pkg.col_12_ptr, sizeof(unsigned short), NoneZeroCount_conv_12, fptr);

    d_pkg.feature_matrix_12 = (float*)malloc( 9 * p_pkg.conv_channel_12_in * p_pkg.conv_height_12_out * p_pkg.conv_width_12_out * sizeof(float) );
    d_pkg.conv_top_data_12 = (float*)malloc(sizeof(float)*p_pkg.conv_width_12_out*p_pkg.conv_height_12_out*p_pkg.conv_channel_12_out);
//conv 13   
    d_pkg.biases_13         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_13_out);
    memset(d_pkg.biases_13, 0x0, sizeof(float) * p_pkg.conv_channel_13_out);
    fread(d_pkg.biases_13, sizeof(float), p_pkg.conv_channel_13_out, fptr);
    d_pkg.scale_13          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_13_out);
    memset(d_pkg.scale_13, 0x0, sizeof(float) * p_pkg.conv_channel_13_out);
	fread(d_pkg.scale_13, sizeof(float), p_pkg.conv_channel_13_out, fptr);

    d_pkg.mean_13           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_13_out);
    memset(d_pkg.mean_13, 0x0, sizeof(float) * p_pkg.conv_channel_13_out);
	fread(d_pkg.mean_13, sizeof(float), p_pkg.conv_channel_13_out, fptr);

    d_pkg.variance_13       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_13_out);
    memset(d_pkg.variance_13, 0x0, sizeof(float) * p_pkg.conv_channel_13_out);
	fread(d_pkg.variance_13, sizeof(float), p_pkg.conv_channel_13_out, fptr);

    int NoneZeroCount_conv_13;
    fread(&NoneZeroCount_conv_13, sizeof(int), 1, fptr);
  //  printf("conv13: %d\n", NoneZeroCount_conv_13);
    d_pkg.weights_conv_13_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_13);
    memset(d_pkg.weights_conv_13_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_13);
	fread(d_pkg.weights_conv_13_NoneZeroValue, sizeof(float), NoneZeroCount_conv_13, fptr);

    d_pkg.row_13_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_13_out+1));
    memset(d_pkg.row_13_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_13_out+1));
	fread(d_pkg.row_13_ptr, sizeof(int), (p_pkg.conv_channel_13_out+1), fptr);

    d_pkg.col_13_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_13);
    memset(d_pkg.col_13_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_13);
	fread(d_pkg.col_13_ptr, sizeof(unsigned short), NoneZeroCount_conv_13, fptr);

    d_pkg.feature_matrix_13 = (float*)malloc( 9 * p_pkg.conv_channel_13_in * p_pkg.conv_height_13_out * p_pkg.conv_width_13_out * sizeof(float) );
    d_pkg.conv_top_data_13 = (float*)malloc(sizeof(float)*p_pkg.conv_width_13_out*p_pkg.conv_height_13_out*p_pkg.conv_channel_13_out);
//conv 14    
    d_pkg.biases_14         = (float*)malloc(sizeof(float)*p_pkg.conv_channel_14_out);
    memset(d_pkg.biases_14, 0x0, sizeof(float) * p_pkg.conv_channel_14_out);
    fread(d_pkg.biases_14, sizeof(float), p_pkg.conv_channel_14_out, fptr);
    // d_pkg.scale_14          = (float*)malloc(sizeof(float)*p_pkg.conv_channel_14_out);
    // d_pkg.mean_14           = (float*)malloc(sizeof(float)*p_pkg.conv_channel_14_out);
    // d_pkg.variance_14       = (float*)malloc(sizeof(float)*p_pkg.conv_channel_14_out);
    int NoneZeroCount_conv_14;
    fread(&NoneZeroCount_conv_14, sizeof(int), 1, fptr);
  //  printf("conv14: %d\n", NoneZeroCount_conv_14);
    d_pkg.weights_conv_14_NoneZeroValue = (float *)malloc(sizeof(float)*NoneZeroCount_conv_14);
    memset(d_pkg.weights_conv_14_NoneZeroValue, 0x0, sizeof(float) * NoneZeroCount_conv_14);
	fread(d_pkg.weights_conv_14_NoneZeroValue, sizeof(float), NoneZeroCount_conv_14, fptr);

    d_pkg.row_14_ptr        = (int*)malloc(sizeof(int)*(p_pkg.conv_channel_14_out+1));
    memset(d_pkg.row_14_ptr, 0x0, sizeof(int) * (p_pkg.conv_channel_14_out+1));
	fread(d_pkg.row_14_ptr, sizeof(int), (p_pkg.conv_channel_14_out+1), fptr);

    d_pkg.col_14_ptr        = (unsigned short*)malloc(sizeof(unsigned short)*NoneZeroCount_conv_14);
    memset(d_pkg.col_14_ptr, 0x0, sizeof(unsigned short) * NoneZeroCount_conv_14);
	fread(d_pkg.col_14_ptr, sizeof(unsigned short), NoneZeroCount_conv_14, fptr);

    d_pkg.conv_top_data_14  = (float*)malloc(sizeof(float)*p_pkg.conv_width_14_out*p_pkg.conv_height_14_out*p_pkg.conv_channel_14_out);

	return;
} 

//释放内存空间
void destroy_network_parameters(data_package& d_pkg )
{
//conv 0
    if(d_pkg.biases_0)
    {
        free(d_pkg.biases_0);
        d_pkg.biases_0 = NULL;
    }   
    if(d_pkg.scale_0)
    {
        free(d_pkg.scale_0);
        d_pkg.scale_0 = NULL;
    }
    if(d_pkg.mean_0)
    {
        free(d_pkg.mean_0);
        d_pkg.mean_0 = NULL;
    }    
    if(d_pkg.variance_0)
    {
        free(d_pkg.variance_0);
        d_pkg.variance_0 = NULL;
    }    
    if(d_pkg.weights_conv_0_NoneZeroValue)
    {
        free(d_pkg.weights_conv_0_NoneZeroValue);
        d_pkg.weights_conv_0_NoneZeroValue = NULL;
    }    
    if(d_pkg.row_0_ptr)
    {
        free(d_pkg.row_0_ptr);
        d_pkg.row_0_ptr = NULL;
    }   
    if(d_pkg.col_0_ptr)
    {
        free(d_pkg.col_0_ptr);
        d_pkg.col_0_ptr = NULL;
    }   
    if(d_pkg.feature_matrix_0)
    {
        free(d_pkg.feature_matrix_0);
        d_pkg.feature_matrix_0 = NULL;
    }
//conv 2 
    if(d_pkg.biases_2)
    {
        free(d_pkg.biases_2);
        d_pkg.biases_2 = NULL;
    }

    
    if(d_pkg.scale_2)
    {
        free(d_pkg.scale_2);
        d_pkg.scale_2 = NULL;
    }

    
    if(d_pkg.mean_2)
    {
        free(d_pkg.mean_2);
        d_pkg.mean_2 = NULL;
    }

    
    if(d_pkg.variance_2)
    {
        free(d_pkg.variance_2);
        d_pkg.variance_2 = NULL;
    }

    
    if(d_pkg.weights_conv_2_NoneZeroValue)
    {
        free(d_pkg.weights_conv_2_NoneZeroValue);
        d_pkg.weights_conv_2_NoneZeroValue = NULL;
    }

    
    if(d_pkg.row_2_ptr)
    {
        free(d_pkg.row_2_ptr);
        d_pkg.row_2_ptr = NULL;
    }

    
    if(d_pkg.col_2_ptr)
    {
        free(d_pkg.col_2_ptr);
        d_pkg.col_2_ptr = NULL;
    }

    
    if(d_pkg.feature_matrix_2)
    {
        free(d_pkg.feature_matrix_2);
        d_pkg.feature_matrix_2 = NULL;
    }

//conv 4    
    if(d_pkg.biases_4)
    {
        free(d_pkg.biases_4);
        d_pkg.biases_4 = NULL;
    }

    
    if(d_pkg.scale_4)
    {
        free(d_pkg.scale_4);
        d_pkg.scale_4 = NULL;
    }

    
    if(d_pkg.mean_4)
    {
        free(d_pkg.mean_4);
        d_pkg.mean_4 = NULL;
    }

    
    if(d_pkg.variance_4)
    {
        free(d_pkg.variance_4);
        d_pkg.variance_4 = NULL;
    }

    
    if(d_pkg.weights_conv_4_NoneZeroValue)
    {
        free(d_pkg.weights_conv_4_NoneZeroValue);
        d_pkg.weights_conv_4_NoneZeroValue = NULL;
    }

    
    if(d_pkg.row_4_ptr)
    {
        free(d_pkg.row_4_ptr);
        d_pkg.row_4_ptr = NULL;
    }

    
    if(d_pkg.col_4_ptr)
    {
        free(d_pkg.col_4_ptr);
        d_pkg.col_4_ptr = NULL;
    }

    
    if(d_pkg.feature_matrix_4)
    {
        free(d_pkg.feature_matrix_4);
        d_pkg.feature_matrix_4 = NULL;
    }

//conv 6   
    if(d_pkg.biases_6)
    {
        free(d_pkg.biases_6);
        d_pkg.biases_6 = NULL;
    }

    
    if(d_pkg.scale_6)
    {
        free(d_pkg.scale_6);
        d_pkg.scale_6 = NULL;
    }

    
    if(d_pkg.mean_6)
    {
        free(d_pkg.mean_6);
        d_pkg.mean_6 = NULL;
    }

    
    if(d_pkg.variance_6)
    {
        free(d_pkg.variance_6);
        d_pkg.variance_6 = NULL;
    }

    
    if(d_pkg.weights_conv_6_NoneZeroValue)
    {
        free(d_pkg.weights_conv_6_NoneZeroValue);
        d_pkg.weights_conv_6_NoneZeroValue = NULL;
    }

    
    if(d_pkg.row_6_ptr)
    {
        free(d_pkg.row_6_ptr);
        d_pkg.row_6_ptr = NULL;
    }

    
    if(d_pkg.col_6_ptr)
    {
        free(d_pkg.col_6_ptr);
        d_pkg.col_6_ptr = NULL;
    }

    
    if(d_pkg.feature_matrix_6)
    {
        free(d_pkg.feature_matrix_6);
        d_pkg.feature_matrix_6 = NULL;
    }

//conv 8    
    if(d_pkg.biases_8)
    {
        free(d_pkg.biases_8);
        d_pkg.biases_8 = NULL;
    }

    
    if(d_pkg.scale_8)
    {
        free(d_pkg.scale_8);
        d_pkg.scale_8 = NULL;
    }

    
    if(d_pkg.mean_8)
    {
        free(d_pkg.mean_8);
        d_pkg.mean_8 = NULL;
    }

    
    if(d_pkg.variance_8)
    {
        free(d_pkg.variance_8);
        d_pkg.variance_8 = NULL;
    }

    
    if(d_pkg.weights_conv_8_NoneZeroValue)
    {
        free(d_pkg.weights_conv_8_NoneZeroValue);
        d_pkg.weights_conv_8_NoneZeroValue = NULL;
    }

    
    if(d_pkg.row_8_ptr)
    {
        free(d_pkg.row_8_ptr);
        d_pkg.row_8_ptr = NULL;
    }

    
    if(d_pkg.col_8_ptr)
    {
        free(d_pkg.col_8_ptr);
        d_pkg.col_8_ptr = NULL;
    }

    
    if(d_pkg.feature_matrix_8)
    {
        free(d_pkg.feature_matrix_8);
        d_pkg.feature_matrix_8 = NULL;
    }

//conv 10    
    if(d_pkg.biases_10)
    {
        free(d_pkg.biases_10);
        d_pkg.biases_10 = NULL;
    }

    
    if(d_pkg.scale_10)
    {
        free(d_pkg.scale_10);
        d_pkg.scale_10 = NULL;
    }

    
    if(d_pkg.mean_10)
    {
        free(d_pkg.mean_10);
        d_pkg.mean_10 = NULL;
    }

    
    if(d_pkg.variance_10)
    {
        free(d_pkg.variance_10);
        d_pkg.variance_10 = NULL;
    }

    
    if(d_pkg.weights_conv_10_NoneZeroValue)
    {
        free(d_pkg.weights_conv_10_NoneZeroValue);
        d_pkg.weights_conv_10_NoneZeroValue = NULL;
    }

    
    if(d_pkg.row_10_ptr)
    {
        free(d_pkg.row_10_ptr);
        d_pkg.row_10_ptr = NULL;
    }

    
    if(d_pkg.col_10_ptr)
    {
        free(d_pkg.col_10_ptr);
        d_pkg.col_10_ptr = NULL;
    }

    
    if(d_pkg.feature_matrix_10)
    {
        free(d_pkg.feature_matrix_10);
        d_pkg.feature_matrix_10 = NULL;
    }

//conv 12    
    if(d_pkg.biases_12)
    {
        free(d_pkg.biases_12);
        d_pkg.biases_12 = NULL;
    }

    
    if(d_pkg.scale_12)
    {
        free(d_pkg.scale_12);
        d_pkg.scale_12 = NULL;
    }

    
    if(d_pkg.mean_12)
    {
        free(d_pkg.mean_12);
        d_pkg.mean_12 = NULL;
    }

    
    if(d_pkg.variance_12)
    {
        free(d_pkg.variance_12);
        d_pkg.variance_12 = NULL;
    }

    
    if(d_pkg.weights_conv_12_NoneZeroValue)
    {
        free(d_pkg.weights_conv_12_NoneZeroValue);
        d_pkg.weights_conv_12_NoneZeroValue = NULL;
    }

    
    if(d_pkg.row_12_ptr)
    {
        free(d_pkg.row_12_ptr);
        d_pkg.row_12_ptr = NULL;
    }

    
    if(d_pkg.col_12_ptr)
    {
        free(d_pkg.col_12_ptr);
        d_pkg.col_12_ptr = NULL;
    }

    
    if(d_pkg.feature_matrix_12)
    {
        free(d_pkg.feature_matrix_12);
        d_pkg.feature_matrix_12 = NULL;
    }


//conv 13
    if(d_pkg.biases_13)
    {
        free(d_pkg.biases_13);
        d_pkg.biases_13 = NULL;
    }

    if(d_pkg.scale_13)
    {
        free(d_pkg.scale_13);
        d_pkg.scale_13 = NULL;
    }

    if(d_pkg.mean_13)
    {
        free(d_pkg.mean_13);
        d_pkg.mean_13 = NULL;
    }

    if(d_pkg.variance_13)
    {
        free(d_pkg.variance_13);
        d_pkg.variance_13 = NULL;
    }

    if(d_pkg.weights_conv_13_NoneZeroValue)
    {
        free(d_pkg.weights_conv_13_NoneZeroValue);
        d_pkg.weights_conv_13_NoneZeroValue = NULL;
    }

    if(d_pkg.row_13_ptr)
    {
        free(d_pkg.row_13_ptr);
        d_pkg.row_13_ptr = NULL;
    }

    if(d_pkg.col_13_ptr)
    {
        free(d_pkg.col_13_ptr);
        d_pkg.col_13_ptr = NULL;
    }

    if(d_pkg.feature_matrix_13)
    {
        free(d_pkg.feature_matrix_13);
        d_pkg.feature_matrix_13 = NULL;
    }
//conv 14

    if(d_pkg.biases_14)
    {
        free(d_pkg.biases_14);
        d_pkg.biases_14 = NULL;
    }

    // if(d_pkg.scale_14)
    // {
    //     free(d_pkg.scale_14);
    //     d_pkg.scale_14 = NULL;
    // }

    // if(d_pkg.mean_14)
    // {
    //     free(d_pkg.mean_14);
    //     d_pkg.mean_14 = NULL;
    // }

    // if(d_pkg.variance_14)
    // {
    //     free(d_pkg.variance_14);
    //     d_pkg.variance_14 = NULL;
    // }

    if(d_pkg.weights_conv_14_NoneZeroValue)
    {
        free(d_pkg.weights_conv_14_NoneZeroValue);
        d_pkg.weights_conv_14_NoneZeroValue = NULL;
    }

    if(d_pkg.row_14_ptr)
    {
        free(d_pkg.row_14_ptr);
        d_pkg.row_14_ptr = NULL;
    }

    if(d_pkg.col_14_ptr)
    {
        free(d_pkg.col_14_ptr);
        d_pkg.col_14_ptr = NULL;
    }

    return;
}


//数据预处理+整体前向+解析
int main()
{
	//初始化
	data_package d_pkg;
	parameter_package p_pkg;
	initial_network_space_parameters(d_pkg, p_pkg);

	//anchor boxes yolo当初用来修正bounding box的anchor boxes 这里直接定义 不从外部读取
	vector<float> anchor_boxes;
	anchor_boxes.resize(10);
	anchor_boxes[0] = 1.08f;
	anchor_boxes[1] = 1.19f;
	anchor_boxes[2] = 3.42f;
	anchor_boxes[3] = 4.41f;
	anchor_boxes[4] = 6.63f;
	anchor_boxes[5] = 11.38f;
	anchor_boxes[6] = 9.42f;
	anchor_boxes[7] = 5.11f;
	anchor_boxes[8] = 16.62f;
	anchor_boxes[9] = 10.52f;

	//for region
	const int number_box = 13 * 13 * 5; //l.w*l.h*l.n
	box *boxes = (box*)calloc(number_box, sizeof(box));
    float **probs = (float**)calloc(number_box, sizeof(float *));
    for(int j = 0; j < number_box; ++j) probs[j] = (float*)calloc(20 + 1, sizeof(float)); //20:classes
	
	const int w = 416;
	const int h = 416;
	int offset, r, c;
	bool flag = false;
	float *buffer = NULL, *img_part = NULL, *img_resized = NULL;
	cv::Mat frame;
	img_inf img_information[3];
	vector<cv::Mat> channels;
	channels.resize(3);

	//opencv读取的数据在排列顺序上需要进行分离，再拼接。
	frame = cv::imread("1.jpg");
	cv::split(frame, channels);
	img_information[0].rows = frame.rows;
	img_information[0].cols = frame.cols;
	img_information[0].channels = 3;

	img_information[1].rows = frame.rows;
	img_information[1].cols = w;
	img_information[1].channels = 3;

	img_information[2].rows = h;
	img_information[2].cols = w;
	img_information[2].channels = 3;

	offset = frame.rows*frame.cols;
	buffer = new float[offset*frame.channels()];
	img_part = new float[w*frame.rows*frame.channels()];
	img_resized = new float[w*h*frame.channels()];
	flag = true;

	memcpy(frame.data, channels[2].data, sizeof(unsigned char)* offset);
	memcpy(frame.data + offset, channels[1].data, sizeof(unsigned char)* offset);
	memcpy(frame.data + (offset << 1), channels[0].data, sizeof(unsigned char)* offset);
	//归一化操作
	for (int i = 0; i < offset*frame.channels(); ++i) buffer[i] = static_cast<float>(frame.data[i]) / 255.0f;
		
	const float w_scale = (float)(frame.cols - 1) / (w - 1);
	const float h_scale = (float)(frame.rows - 1) / (h - 1);
	
	//okey  这里你也可以使用openmp
	//进行 crop 以及 resize的操作
	//is openmp support?? yes!!!
	for(int pp = 0; pp < 3; ++pp)
	{
		for (r = 0; r < frame.rows; ++r)
		{
			for (c = 0; c < w; ++c)
			{
				float val = 0;
				if (c == w - 1 || frame.cols == 1)
				{
					val = get_pixel(buffer, frame.cols - 1, r, pp, img_information[0]);
					
				}
				else
				{
					float sx = c*w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx)*get_pixel(buffer, ix, r, pp, img_information[0]) + dx * get_pixel(buffer, ix + 1, r, pp, img_information[0]);
					// printf("val:%f\n",val);
				}
				set_pixel(img_part, c, r, pp, val, img_information[1]);
				// printf("%f\n", img_part[pp * w * frame.rows + r * w + c]);
			}
		}
		for (r = 0; r < h; r++)
		{
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c)
			{
				float val = (1 - dy)*get_pixel(img_part, c, iy, pp, img_information[1]);
				set_pixel(img_resized, c, r, pp, val, img_information[2]);
			}
			if (r == h - 1 || frame.rows == 1) continue;
			for (c = 0; c < w; ++c)
			{
				float val = dy*get_pixel(img_part, c, iy + 1, pp, img_information[1]);
				add_pixel(img_resized, c, r, pp, val, img_information[2]);
			}
		}
	}

	// for(int i = 0 ;i<100;i++) printf("%d,%f\n",i, img_resized[i]);
	// 	getchar();

	//resize 完毕  图片预处理操作结束 下面进入前向
	//run Forward
    //float *conv_bottom_data_0 = img_resized;//////////////// input
	im2col_cpu(img_resized, p_pkg.conv_channel_0_in, p_pkg.conv_height_0_in, p_pkg.conv_width_0_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_0);
    sparse_gemm_naive(d_pkg.feature_matrix_0, d_pkg.weights_conv_0_NoneZeroValue, d_pkg.row_0_ptr, d_pkg.col_0_ptr, p_pkg.conv_channel_0_out, p_pkg.conv_width_0_out * p_pkg.conv_height_0_out,
     d_pkg.conv_top_data_0, d_pkg.biases_0, d_pkg.scale_0, d_pkg.mean_0, d_pkg.variance_0);
	
    max_pooling2x2(d_pkg.conv_top_data_0, p_pkg.conv_width_0_out, p_pkg.conv_width_2_in, p_pkg.conv_channel_0_out, d_pkg.max_pooling_1);

    printf("conv0 over\n");
    // for(int i = 0; i < 100; ++i)
    // {
    // 	printf("%d, %f\n", i, d_pkg.max_pooling_1[i]);
    // }
	// getchar();

#if 1
    
    im2col_cpu(d_pkg.max_pooling_1, p_pkg.conv_channel_2_in, p_pkg.conv_height_2_in, p_pkg.conv_width_2_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_2);
    sparse_gemm_naive(d_pkg.feature_matrix_2, d_pkg.weights_conv_2_NoneZeroValue, d_pkg.row_2_ptr, d_pkg.col_2_ptr, p_pkg.conv_channel_2_out, p_pkg.conv_width_2_out * p_pkg.conv_height_2_out, 
    	d_pkg.conv_top_data_2, d_pkg.biases_2, d_pkg.scale_2, d_pkg.mean_2, d_pkg.variance_2);
    
    max_pooling2x2(d_pkg.conv_top_data_2, p_pkg.conv_width_2_out, p_pkg.conv_width_4_in, p_pkg.conv_channel_2_out, d_pkg.max_pooling_3);
    printf("conv2 over\n");
    //     for(int i = 0; i < 100; ++i)
    // {
    // 	printf("%d, %f\n", i, d_pkg.max_pooling_3[i]);
    // }
    // getchar();

    
    im2col_cpu(d_pkg.max_pooling_3, p_pkg.conv_channel_4_in, p_pkg.conv_height_4_in, p_pkg.conv_width_4_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_4);
    sparse_gemm_naive(d_pkg.feature_matrix_4, d_pkg.weights_conv_4_NoneZeroValue, d_pkg.row_4_ptr, d_pkg.col_4_ptr, p_pkg.conv_channel_4_out, p_pkg.conv_width_4_out * p_pkg.conv_height_4_out, 
    	d_pkg.conv_top_data_4, d_pkg.biases_4, d_pkg.scale_4, d_pkg.mean_4, d_pkg.variance_4);

    max_pooling2x2(d_pkg.conv_top_data_4, p_pkg.conv_width_4_out, p_pkg.conv_width_6_in, p_pkg.conv_channel_4_out, d_pkg.max_pooling_5);
    printf("conv4 over\n");
    // for(int i = 0; i < 100; ++i)
    // {
    // 	printf("%d, %f\n", i, d_pkg.max_pooling_5[i]);
    // }
    // getchar();
    im2col_cpu(d_pkg.max_pooling_5, p_pkg.conv_channel_6_in, p_pkg.conv_height_6_in, p_pkg.conv_width_6_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_6);
    sparse_gemm_naive(d_pkg.feature_matrix_6, d_pkg.weights_conv_6_NoneZeroValue, d_pkg.row_6_ptr, d_pkg.col_6_ptr, p_pkg.conv_channel_6_out, p_pkg.conv_width_6_out * p_pkg.conv_height_6_out, 
    	d_pkg.conv_top_data_6, d_pkg.biases_6, d_pkg.scale_6, d_pkg.mean_6, d_pkg.variance_6);
    
    max_pooling2x2(d_pkg.conv_top_data_6, p_pkg.conv_width_6_out, p_pkg.conv_width_8_in, p_pkg.conv_channel_6_out, d_pkg.max_pooling_7);
    printf("conv6 over\n");
    // for(int i = 0; i < 100; ++i)
    // {
    // 	printf("%d, %f\n", i, d_pkg.max_pooling_7[i]);
    // }
    // getchar();

    
    im2col_cpu(d_pkg.max_pooling_7, p_pkg.conv_channel_8_in, p_pkg.conv_height_8_in, p_pkg.conv_width_8_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_8);
    sparse_gemm_naive(d_pkg.feature_matrix_8, d_pkg.weights_conv_8_NoneZeroValue, d_pkg.row_8_ptr, d_pkg.col_8_ptr, p_pkg.conv_channel_8_out, p_pkg.conv_width_8_out * p_pkg.conv_height_8_out, 
    	d_pkg.conv_top_data_8, d_pkg.biases_8, d_pkg.scale_8, d_pkg.mean_8, d_pkg.variance_8);
    
    max_pooling2x2(d_pkg.conv_top_data_8, p_pkg.conv_width_8_out, p_pkg.conv_width_10_in, p_pkg.conv_channel_8_out, d_pkg.max_pooling_9);
    printf("conv8 over\n");
    // for(int i = 0; i < 100; ++i)
    // {
    // 	printf("%d, %f\n", i, d_pkg.max_pooling_9[i]);
    // }
    // getchar();

    im2col_cpu(d_pkg.max_pooling_9, p_pkg.conv_channel_10_in, p_pkg.conv_height_10_in, p_pkg.conv_width_10_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_10);
    sparse_gemm_naive(d_pkg.feature_matrix_10, d_pkg.weights_conv_10_NoneZeroValue, d_pkg.row_10_ptr, d_pkg.col_10_ptr, p_pkg.conv_channel_10_out, p_pkg.conv_width_10_out * p_pkg.conv_height_10_out, 
    	d_pkg.conv_top_data_10, d_pkg.biases_10, d_pkg.scale_10, d_pkg.mean_10, d_pkg.variance_10);
    
   
    maxpool_layer_1x1_const(d_pkg.conv_top_data_10, d_pkg.max_pooling_11);
    printf("conv10 over\n");
    // for(int i = 0; i < 100; ++i)
    // {
    // 	printf("%d, %f\n", i, d_pkg.max_pooling_11[i]);
    // }
    // getchar();


    im2col_cpu(d_pkg.max_pooling_11, p_pkg.conv_channel_12_in, p_pkg.conv_height_12_in, p_pkg.conv_width_12_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_12);
    sparse_gemm_naive(d_pkg.feature_matrix_12, d_pkg.weights_conv_12_NoneZeroValue, d_pkg.row_12_ptr, d_pkg.col_12_ptr, p_pkg.conv_channel_12_out, p_pkg.conv_width_12_out * p_pkg.conv_height_12_out, 
    	d_pkg.conv_top_data_12, d_pkg.biases_12, d_pkg.scale_12, d_pkg.mean_12, d_pkg.variance_12);
    
    printf("conv12 over\n");


    im2col_cpu(d_pkg.conv_top_data_12, p_pkg.conv_channel_13_in, p_pkg.conv_height_13_in, p_pkg.conv_width_13_in, p_pkg.kernel_single_size_1, p_pkg.padding, p_pkg.conv_stride, d_pkg.feature_matrix_13);
    sparse_gemm_naive(d_pkg.feature_matrix_13, d_pkg.weights_conv_13_NoneZeroValue, d_pkg.row_13_ptr, d_pkg.col_13_ptr, p_pkg.conv_channel_13_out, p_pkg.conv_width_13_out * p_pkg.conv_height_13_out, 
    	d_pkg.conv_top_data_13, d_pkg.biases_13, d_pkg.scale_13, d_pkg.mean_13, d_pkg.variance_13);
    
    printf("conv13 over\n");


    sparse_gemm_naive_no_BN(d_pkg.conv_top_data_13, d_pkg.weights_conv_14_NoneZeroValue, d_pkg.row_14_ptr, d_pkg.col_14_ptr, p_pkg.conv_channel_14_out, p_pkg.conv_width_14_out * p_pkg.conv_height_14_out, d_pkg.conv_top_data_14, d_pkg.biases_14);
	printf("conv14 over\n");
	// for(int i = 0; i < 100; ++i)
	// {
	// 	printf("%d, %f\n",i, d_pkg.conv_top_data_14[i]);
	// }
	// getchar();
#endif
	
	float *region_data = d_pkg.conv_top_data_14;
	int region_height = 13;//l.h
	int region_width = 13; //l.w
	int out_put_n = 5;     //l.n
	int size = 25;
	int output_size = 13*13*25*5;//20+4+1 l.coords + l.classes + 1;
	int classes = 20;
	float nms = 0.4f;

	flatten(region_data, 169, 125);

    	for(int i = 0; i < region_width*region_width*out_put_n; ++i){
        	int index = size*i;// + b*output_size;
        	region_data[index + 4] = logistic_activate(region_data[index + 4]);
        	        	        	
    	}


        for(int i = 0; i < region_width*region_width*out_put_n; ++i){
            int index = size*i;// + b*output_size;
            softmax(region_data + index + 5, classes, 1, region_data + index + 5);
        }

    
    //predict
	get_region_boxes(region_data, 1, 1, 0.24, probs, boxes, anchor_boxes);//0.24:thresh
	// getchar();
	do_nms_sort(boxes, probs, region_width*region_height*out_put_n, classes, nms);
	float *cpu_data = (float*)calloc(500,sizeof(float));
	int n = 0; 
	for (int i = 0; i < 845; i++) //TODO
	{
		int class_final = max_index(probs[i], classes);
		// printf("class_final = %d\n", class_final);
		float prob = probs[i][class_final];
		// printf("%f\n", prob);
		if (prob > 0.24) //thresh_final
		{
			cpu_data[n * 5]     = (boxes[i].x - boxes[i].w *0.5f);
			cpu_data[n * 5 + 1] = (boxes[i].x + boxes[i].w *0.5f);
			cpu_data[n * 5 + 2] = (boxes[i].y - boxes[i].h *0.5f);
			cpu_data[n * 5 + 3] = (boxes[i].y + boxes[i].h *0.5f);
			cpu_data[n * 5 + 4] = class_final;
			n++;
			// printf("%f, %f, %f, %f\n", cpu_data[n * 5], cpu_data[n * 5 + 1], cpu_data[n * 5 + 2], cpu_data[n * 5 + 3]);
		}
		if (n > 100) {
			cout << "too much bounding box" << endl;
			exit(1);
		}	
	}
	// printf("n = %d\n", n);
	//check the box
	cv::merge(channels, frame);
	//auto cpu_data = joints_vec[0]->mutable_cpu_data();
	for (int i = 0; i < n; i++){
		if (cpu_data[i * 5] != 0.f){ 
			cpu_data[i * 5] *= frame.cols;
			cpu_data[i * 5 + 1] *= frame.cols;
			cpu_data[i * 5 + 2] *= frame.rows;
			cpu_data[i * 5 + 3] *= frame.rows;

			printf("%f, %f, %f, %f\n", cpu_data[i * 5], cpu_data[i * 5 + 2], cpu_data[i * 5 + 1], cpu_data[i * 5 + 3]);
			
			rectangle( frame, cvPoint(cpu_data[i * 5], cpu_data[i * 5 + 2]), cvPoint(cpu_data[i * 5 + 1], cpu_data[i * 5 + 3]), colors[(int)cpu_data[i * 5 + 4]], 3);
			putText( frame, labels[(int)cpu_data[i * 5 + 4]], cvPoint(cpu_data[i * 5], cpu_data[i * 5 + 2] - 10 > 0 ? cpu_data[i * 5 + 2] - 10 : 0), 
				     FONT_HERSHEY_SIMPLEX, 1.9, colors[(int)cpu_data[i * 5 + 4]], 3);
		}
	}
	imwrite("yolo.jpg", frame);
	//imshow("demo", frame);
	//if (waitKey(400) >= 0) break;
	
	//capture.release();
	delete[] buffer; buffer = nullptr;
	delete[] img_resized; img_resized = nullptr;
	delete[] img_part; img_part = nullptr;
	//cvDestroyWindow("demo");

	destroy_network_parameters(d_pkg);

	return 0;

}


//看到这里你相比对伸进网络的架构有了一定的理解，其实就是这样，为了方便而已，这里是一种很直接的方式，让所谓的神经网络完的前向完全全暴露在你的眼前
//此文件只供学习使用，不具备商业价值。
//林皞    2017  11 5 日



