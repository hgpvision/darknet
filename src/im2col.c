#include "im2col.h"
#include <stdio.h>
/*
**  从输入的多通道数组im（存储图像数据）中获取指定行、列、、通道数处的元素值
**  输入： im      输入，所有数据存成一个一维数组，例如对于3通道的二维图像而言，
**                每一通道按行存储（每一通道所有行并成一行），三通道依次再并成一行
**        height  每一通道的高度（即输入图像的真正的高度，补0之前）
**        width   每一通道的宽度（即输入图像的宽度，补0之前）
**        channels 输入im的通道数，比如彩色图为3通道，之后每一卷积层的输入的通道数等于上一卷积层卷积核的个数
**        row     要提取的元素所在的行（二维图像补0之后的行数）
**        col     要提取的元素所在的列（二维图像补0之后的列数）
**        channel 要提取的元素所在的通道
**        pad     图像左右上下各补0的长度（四边补0的长度一样）
**  返回： float类型数据，为im中channel通道，row-pad行，col-pad列处的元素值
**  注意：在im中并没有存储补0的元素值，因此height，width都是没有补0时输入图像真正的
**       高、宽；而row与col则是补0之后，元素所在的行列，因此，要准确获取在im中的元素值，
**       首先需要减去pad以获取在im中真实的行列数
*/
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    // 减去补0长度，获取元素真实的行列数
    row -= pad;
    col -= pad;

    // 如果行列数小于0,则返回0（刚好是补0的效果）
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    
    // im存储多通道二维图像的数据的格式为：各通道所有行并成一行，再多通道依次并成一行，
    // 因此width*height*channel首先移位到所在通道的起点位置，加上width*row移位到
    // 所在指定通道所在行，再加上col移位到所在列
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*
** 将输入图片转为便于计算的数组格式，可以参考https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
** 进行辅助理解（但执行方式并不同，只是用于概念上的辅助理解），由作者的注释可知，这是直接从caffe移植过来的
** 输入： data_im    输入图像
**       channels   输入图像的通道数（对于第一层，一般是颜色图，3通道，中间层通道数为上一层卷积核个数）
**       height     输入图像的高度（行）
**       width      输入图像的宽度（列）
**       ksize      卷积核尺寸
**       stride     卷积核跨度
**       pad        四周补0长度
**       data_col   相当于输出，为进行格式重排后的输入图像数据
** 说明：1）此函数个人感觉在实现上存在不足，传入参数没有必要这么多，只需传入当前卷积层的指针即可，这样函数中的一些代码就会变的多余
**      2）输出data_col的元素个数与data_im元素个数不相等，一般比data_im的元素个数多，因为stride较小，各个卷积核之间有很多重叠，
**         实际data_col中的元素个数为channels*ksize*ksize*height_col*width_col，其中channels为data_im的通道数，
**         ksize为卷积核大小，height_col和width_col如下所注。data_col的还是按行排列，只是行数为channels*ksize*ksize,
**         列数为height_col*width_col，即一张特征图总的元素个数，每整列包含与某个位置处的卷积核计算的所有通道上的像素（比如输入图像通道数为3,卷积核尺寸为3*3，则共有27行，每列有27个元素），
**         不同列对应卷积核在图像上的不同位置做卷积
*/
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    // 计算该层神经网络的输出图像尺寸（其实没有必要再次计算的，因为在构建卷积层时，make_convolutional_layer()函数
    // 已经调用convolutional_out_width()，convolutional_out_height()函数求取了这两个参数，
    // 此处直接使用l.out_h,l.out_w即可，函数参数只要传入该层网络指针就可了，没必要弄这么多参数）
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    /// 卷积核大小：ksize*ksize是一个卷积核的大小，之所以乘以通道数channels，是因为输入图像有多通道，每个卷积核在做卷积时，
    /// 是同时对同一位置处多通道的图像进行卷积运算，这里为了实现这一目的，将三通道上的卷积核并在一起以便进行计算，因此卷积核
    /// 实际上并不是二维的，而是三维的，比如对于3通道图像，卷积核尺寸为3*3，该卷积核将同时作用于三通道图像上，这样并起来就得
    /// 到含有27个元素的卷积核，且这27个元素都是独立的需要训练的参数。所以在计算训练参数个数时，一定要注意每一个卷积核的实际
    /// 训练参数需要乘以输入通道数。
    int channels_col = channels * ksize * ksize;
    
    // ******这三层循环之间的逻辑关系，决定了输入图像重排后的格式，更为详细/形象的说明可参考博客*******
    // 外循环次数为一个卷积核的尺寸数，循环次数即为最终得到的data_col的总行数
    for (c = 0; c < channels_col; ++c) {
        // 列偏移，卷积核是一个二维矩阵，并按行存储在一维数组中，利用求余运算获取对应在卷积核中的列数，比如对于
        // 3*3的卷积核（3通道），当c=0时，显然在第一列，当c=5时，显然在第2列，当c=9时，在第二通道上的卷积核的第一列，
        // 当c=26时，在第三列（第三通道上）
        int w_offset = c % ksize;
        // 行偏移，卷积核是一个二维的矩阵，且是按行（卷积核所有行并成一行）存储在一维数组中的，
        // 比如对于3*3的卷积核，处理3通道的图像，那么一个卷积核具有27个元素，每9个元素对应一个通道上的卷积核（互为一样），
        // 每当c为3的倍数，就意味着卷积核换了一行，h_offset取值为0,1,2，对应3*3卷积核中的第1, 2, 3行
        int h_offset = (c / ksize) % ksize;
        // 通道偏移，channels_col是多通道的卷积核并在一起的，比如对于3通道，3*3卷积核，每过9个元素就要换一通道数，
        // 当c=0~8时，c_im=0;c=9~17时，c_im=1;c=18~26时，c_im=2
        int c_im = c / ksize / ksize;

        // 中循环次数等于该层输出图像行数height_col，说明data_col中的每一行存储了一张特征图，这张特征图又是按行存储在data_col中的某行中
        for (h = 0; h < height_col; ++h) {
            // 内循环等于该层输出图像列数width_col，说明最终得到的data_col总有channels_col行，height_col*width_col列
            for (w = 0; w < width_col; ++w) {
                // 由上面可知，对于3*3的卷积核，h_offset取值为0,1,2,当h_offset=0时，会提取出所有与卷积核第一行元素进行运算的像素，
                // 依次类推；加上h*stride是对卷积核进行行移位操作，比如卷积核从图像(0,0)位置开始做卷积，那么最先开始涉及(0,0)~(3,3)
                // 之间的像素值，若stride=2，那么卷积核进行一次行移位时，下一行的卷积操作是从元素(2,0)（2为图像行号，0为列号）开始
                int im_row = h_offset + h * stride;
                // 对于3*3的卷积核，w_offset取值也为0,1,2，当w_offset取1时，会提取出所有与卷积核中第2列元素进行运算的像素，
                // 实际在做卷积操作时，卷积核对图像逐行扫描做卷积，加上w*stride就是为了做列移位，
                // 比如前一次卷积其实像素元素为(0,0)，若stride=2,那么下次卷积元素起始像素位置为(0,2)（0为行号，2为列号）
                int im_col = w_offset + w * stride;

                // col_index为重排后图像中的像素索引，等于c * height_col * width_col + h * width_col +w（还是按行存储，所有通道再并成一行），
                // 对应第c通道，h行，w列的元素
                int col_index = (c * height_col + h) * width_col + w;
                
                // im2col_get_pixel函数获取输入图像data_im中第c_im通道，im_row,im_col的像素值并赋值给重排后的图像，
                // height和width为输入图像data_im的真实高、宽，pad为四周补0的长度（注意im_row,im_col是补0之后的行列号，
                // 不是真实输入图像中的行列号，因此需要减去pad获取真实的行列号）
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

