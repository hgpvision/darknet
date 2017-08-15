#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

/*
** 该函数只是调用了gemm_cpu()函数，并且将参数原封不动的传给gemm_cpu()
*/
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

/*
**  功能：被gemm_cpu()函数调用，实际完成C = ALPHA * A * B + C 矩阵计算，
**       输出的C也是按行存储（所有行并成一行）
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B未转置，故为B的列数
**        K       A的列数（不做转置）或者A'的列数（做转置），B的行数（不做转置）或者B'的行数（做转置），此处A,B均未转置，故为A的列数、B的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的列数
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B未转置，故为B的列数
**        ldc     C的列数
**  说明1：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A,B都不进行转置
**       函数名称gemm_nn()中的两个nn分别表示not transpose， not transpose
*/
void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    // 大循环：遍历A的每一行，i表示A的第i行，也是C的第i行
    for(i = 0; i < M; ++i){
        // 中循环：遍历每一行的所有列，k表示A的第k列，同时表示B的第k行
        for(k = 0; k < K; ++k){
            // 先计算ALPHA * A（A中每个元素乘以ALPHA）
            register float A_PART = ALPHA*A[i*lda+k];
            // 内循环：遍历B中所有列，每次大循环完毕，将计算得到A*B一行的结果
            // j是B的第j列，也是C的第j列
            for(j = 0; j < N; ++j){
                // A中的第i行k列与B中的k行i列对应相乘，因为一个大循环要计算A*B整行之结果，
                // 因此，这里用了一个内循环，并没有直接乘以B[k*ldb+i]
                // 每个内循环完毕，将计算A*B整行的部分结果（A中第i行k列与B所有列第k行所有元素相乘的结果）
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

/*
**  功能：被gemm_cpu()函数调用，实际完成C = ALPHA * A * B' + C矩阵计算，
**       输出的C也是按行存储（所有行并成一行）
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B转置，故为B’的列数
**        K       A的列数（不做转置）或者A'的列数（做转置），B的行数（不做转置）或者B'的行数（做转置），此处A不转置，B转置，故为A的列数、B'的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的列数
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B未转置，故为B'的行数
**        ldc     C的列数
**  说明：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A不进行转置,B转置
**       函数名称gemm_nt()中的nt分别表示not transpose， transpose
*/
void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    // 大循环：遍历A的每一行，i表示A的第i行，也是C的第i行
    for(i = 0; i < M; ++i){
        // 
        for(j = 0; j < N; ++j){
            register float sum = 0;
            // 内循环：每次内循环结束，将计算A中第i行与B中第j列相乘的结果，
            // 也就是得到C[i][j]，因为C也一维化了，且按行存储，所以得到C[i*lda+j]
            // k表示A的第几列，也表示
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

/*
**  功能：矩阵计算，实际完成C = ALPHA * A' * B + BETA * C矩阵计算
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B未转置，故为B的列数
**        K       A的列数（不做转置）或者A'的列数（做转置），B的行数（不做转置）或者B'的行数（做转置），此处A转置，B不转置，故为A'的列数、B的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数  
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B未转置，故为B的列数
**        ldc     C的列数
**  说明：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A进行转置,B不转置
**       函数名称gemm_tn()中的tn分别表示transpose， not transpose
*/
void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

/*
**  功能：矩阵计算，实际完成C = ALPHA * A' * B' + BETA * C矩阵计算
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B转置，故为B'的列数
**        K       A'的列数，B'的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数  
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B转置，故为B'的行数
**        ldc     C的列数
**  说明：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A,B都进行转置
**       函数名称gemm_tt()中的tt分别表示transpose， transpose
*/
void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

/*
**  功能：矩阵计算，完成C = ALPHA * A * B + BETA * C矩阵计算，最后的输出为C
**  输入： 
**        TA,TB   是否需要对A,B做转置操作，是为1,否为0（要不要转置取决于A,B之间维度是否匹配，比如A:3*2,B:4*2，则需要对B转置，才满足矩阵乘法维度匹配规则）
**        M       A,C的行数（若A需要转置，则此处给出转置后的A即A'的行数，而不是转置前的）
**        N       B,C的列数（若B需要转置，则此处给出转置后的B即B'的列数，而不是转置前的）
**        K       A的列数，B的行数（同样，若A与B中的二者或者其中一个需要转置，则不管怎样，转置后的A，B必须行列能够匹配，符合矩阵乘法规则，K也是转置后的值，不是转置前的）
**        A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        lda     A的列数（不做转置）或者行数（做转置，且给的是转置后A即A'的行数）
**        ldb     B的列数（不做转置）或者行数（做转置，且给的是转置后B即B'的行数）
**        ldc     C的列数
**  说明：如果TA = 0, TB = 0，那么计算的是C = ALPHA * A * B + BETA * C,此时M是A,C的行数，N是B,C的列数，K是A的列数、B的行数，lda是A的列数，ldb是B的列数；
**       如果TA = 1, TB = 0，那么计算的是C = ALPHA * A' * B + BETA * C,此时M是A’,C的行数，N是B,C的列数，K是A'的列数、B的行数，lda是A'的行数，ldb是B的列数；
**       如果TA = 0, TB = 1，那么计算的是C = ALPHA * A * B' + BETA * C,此时M是A,C的行数，N是B',C的列数，K是A的列数、B'的行数，lda是A的列数，ldb是B'的行数；
**       如果TA = 1, TB = 1，那么计算的是C = ALPHA * A' * B' + BETA * C,此时M是A’,C的行数，N是B',C的列数，K是A'的列数、B'的行数，lda是A'的行数，ldb是B'的行数；
**       总之，参与计算的矩阵必须满足矩阵行列匹配规则。比如A为2*3，B为3*2，C为2*2，那么就是第一种情况；而如果A为3*2，B为3*2，C为2*2,
**       那么就是第二种情况；如果A为2*3，B为2*3，C为2*2,对应第三种情况；如果A为2*3，B为2*3，C为2*2,对应第四种情况。
**  链接：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**  举例说明： 这个函数比较难以理解的地方在于A,B有没有转置这个问题上。首先要清楚，虽然这里A,B,C都是矩阵，但其实都是用一维数组按行保存的，
**           举个例子，假设： A = [1, 2, 3, 2, 2, 1], B = [2, 0, 1, 1, 2, 1], C = [3, 0, 1, 2] （这些输入是打死不变的，
**           都是一维数组格式），且C为2*2的矩阵，即C = [3, 0; 1, 2]，那么要进行C = ALPHA * A * B + BETA * C的计算，
**           必须满足矩阵乘法行列匹配规则，则参与运算的第一个矩阵只能为2*3，第二个只能为3*2，因为A,B的元素个数已经固定为6个。
**           下面分别说明gemm_nn(),gemm_tn(),gemm_nt,gemm_tt()四个函数对该例子的计算。
**           诚如上所述，不管A, B有没有转置，反正最后参与计算的两个矩阵必须前者为2*3,后者为3*2。如果使用gemm_nn()，A,B都没有转置，
**           那么就要求没有转置的A,B分别为2*3,3*2矩阵，则 A = [ 1, 2, 3; 2, 2, 1], B = [2, 0; 1, 1; 2, 1], 
**           调用gemm_nn(2, 2, 3, 1, A, 3, B, 2, C, 2)计算得到 C = [13, 5; 9, 5]（其中ALPHA = BETA = 1，下同）；

**           如果要用gemm_tn()函数，即A需要进行转置之后才能计算，也即转置之后的维度为2*3,而转置之前的维度为3*2，B没有转置，
**           本身就是3*2的矩阵，这样，A = [ 1, 2; 3, 2; 2, 1], A' = [1, 3, 2; 2, 2, 1], B = [2, 0; 1, 1; 2, 1]，
**           gemm_tn(2, 2, 3, 1, A, 2, B, 2, C, 2)函数实际计算的是A'*B+C的值，注意此时的A与gemm_nn()中的A有什么不同，
**           输入的一维数组还是[1, 2, 3, 2, 2, 1]，如前所述，A是按行保存的，因为此时的A本身是一个3*2的矩阵，按照按行保存规则，
**           就是A = [ 1, 2; 3, 2; 2, 1]，调用gemm_tn()的时候，M, N, K分别为2, 2, 3,都是最终参与计算的矩阵的行列数，
**           因为此处真正参与计算的是A'与B，所以M为A'的行数，即为2,N为B的列数，即为2,K为A'与B的列数，即为3，而此时lda=2，
**           是因为A进行了转置，因此输入的是A'的行数，而不是列数3,ldb=2，为B的列数，最终计算得到C=[12, 5; 9, 5]。
**           对于gemm_nt()与gemm_tt()，与上分析一样，不再赘述了。此部分注释进行了测试，对应测试文件darknet_test_gemm.c。
**  强调： 这一系列的gemm()函数，都带有叠加效果，也即最终的值是保存在C中，但这种保存并不是擦除式的保存，而是叠加式的保存，也就是说，
**        如果进入gemm()函数之前，如果C的元素已经有值了，那么这些值不会被擦除掉，而是会将其叠加，
**        其实看式子就可以看出来：此函数完成的是C = ALPHA * A * B + BETA * C矩阵运算。
**          
*/
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    // 先把BETA * C计算完了，并将结果存在C中，得到的C将为M行，N列（按行存储在一维数组C中）
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    // 根据需要，调用下面四种函数之一
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

/*
** 调用CUDA中cublasSgemm()函数完成C_gpu = ALPHA * A_gpu * B_gpu + BETA * C_gpu的线性矩阵运算，与gemm_cpu()基本类似，输入参数也基本相同，
** 但存在两个不同：
** 1）此处是直接调用CUDA cuBLAS库中的cublasSgemm()函数进行矩阵计算，而无需gemm_cpu()那样，需要自己用循环挨个元素相乘实现；
** 2）在GPU中，默认采用的矩阵存储格式是按列存储，而不是我们之前一度习惯的按行存储，此处调用的cublasSgemm()也不例外，
**    所以下面会有一些不同的操作（由于这个原因，相比于cpu版本的gemm_cpu()，又要复杂了一些，下面举例说明，希望不要绕蒙～举的例子可谓良心啊）
**   （如官网所言：A , B and C are matrices stored in column-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively.
Read more at: http://docs.nvidia.com/cuda/cublas/index.html#ixzz4peJE8c4M 
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook）（在官网上复制了一句话到这，顺便跟了两个尾巴～～挺好，这样网址还不要我粘贴了）
**  输入： 
**        TA,TB   是否需要对A_gpu,B_gpu进行转置操作，是为1,否为0（需不需要转置取决于A_gpu,B_gpu的维度是否匹配，
**                比如A_gpu是2*3矩阵，B_gpu是2*3矩阵，那么必须对转置才能实现维度匹配的矩阵计算，也即最终参与计算的是A_gpu*B_gpu'）
**        M       A,C的行数（若A需要转置，则此处给出转置后的A即A'的行数，而不是转置前的，如果看不懂这里说的是什么，请参考gemm_cpu()注释）
**        N       B,C的列数（若B需要转置，则此处给出转置后的B即B'的列数，而不是转置前的）
**        K       A的列数，B的行数（同样，若A与B中的二者或者其中一个进行了转置，则不管怎样，
**                转置后的A，B必须行列能够匹配，符合矩阵乘法规则，K也是转置后的值，不是转置前的）
**        A_gpu,B_gpu,C_gpu   输入矩阵（一维数组格式），且其内存在GPU设备内存中，不在主机内存中（由cudaMalloc分配，由cudaFree释放）
**        ALPHA   系数
**        BETA    系数
**        lda     A的列数（不做转置）或者行数（做转置，且给的是转置后A即A'的行数）
**        ldb     B的列数（不做转置）或者行数（做转置，且给的是转置后B即B'的行数）
**        ldc     C的列数
** 举例： 不管A_gpu, B_gpu是按行存储的，还是按列存储的，二者都在那里，且都是一维数组（而不是二维矩阵）的格式，
**       为了方便说明，我们假定输入的两个一维数组分别为a = [1,1,2,2,3,3],b=[4,5,7,4,5,7]，
**       我们的真实目标是完成[1,1;2,2;3,3]*[4,5,7;4,5,7]两个矩阵的计算，也即最终得到的矩阵是3*3维的，其值为[8,10,14;16,20,28;24,30,42],
**       这也符合按行存储的习惯，也即我们认为输入到gemm_ongpu()的A_gpu是[1,1;2,2;3,3]，B_gpu = [4,5,7;4,5,7]，但cublasSgemm()不这样认为，
**       它从输入的a,b两个数组中提出的矩阵分别是：A_gpu'=[1,2,3;1,2,3]和B_gpu'=[4,4;5,5;7,7]，第一眼看，感觉有点乱，
**       这是从a,b两个数组中提出的矩阵吗？怎么这么乱？到底是按什么规律排的啊？
**       是的，在此处我们无法像纸上那般方便地写出一个矩阵，你可以试着将这两个矩阵在纸上写成标准矩阵的格式，
**       会立马看出这是按列存储的方式：也即a的前两个数是A_gpu'的第一列，再后两个数是A_gpu’的第二列，依次类推，B_gpu’也一样，
**       是的，我们认为的输入进gemm_ongpu()的A_gpu，B_gpu与cublasSgemm()自己从a,b中提出的矩阵刚好互为转置！
**       如果你将参数一一对应输入进cublasSgemm()函数，它实际上会计算A_gpu' * B_gpu' + C_gpu'，而不是我们期待的
**       C_gpu = A_gpu * B_gpu + C_gpu的值，甚至还可能出现维度不匹配的错误（此处举的例子由于维度特殊，不会出现维度不配的情况），
**       那怎么解决了？总不至于放纵其计算[1,2,3;1,2,3]*[4,4;5,5;7,7]吧，虽然此处没有维度不匹配的错误，但结果是个2*2矩阵，显然不对啊！
**       解决的方法很简单，我们发现如果稍微耍点小聪明，将输入到cublasSgemm()中的A_gpu，B_gpu调换一下位置，这时cublasSgemm()实际会计算
**       B_gpu' * A_gpu' + C_gpu' = (A_gpu * B_gpu)' + C_gpu'，发现没有，此时计算得到的值刚好是我们期待的C_gpu的转置
**       （如果你看不出来，你可以先把后面的加法去掉，只考虑乘法，反正加法没有这个问题），你可以试着在纸上写出：
**       ([1,1;2,2;3,3]*[4,5,7;4,5,7])'，最终得到的值是[8,16,24;10,20,30;14,28,42]，与上面的目标值[8,10,14;16,20,28;24,30,42]对比，
**       刚好是我们期待的值的转置（如果看不出来，请将矩阵写到纸上），你可能会问，我们要的是C_gpu啊，不是其转置啊，
**       不要忘了，C_gpu'是cublasSgemm()输出的，它对C_gpu'的按列存储格式，刚好就是我们期待的并且习惯的按行存储的C_gpu存储格式，
**       它会一列一列的将C_gpu'中的元素放到一维数组中，最终输出的一维数组为[8,10,14,16,20,28,24,30,42],
**       这个数组按我们按行存储的习惯，翻译得到的矩阵就是我们期待的：[8,10,14;16,20,28;24,30,42]。可见，问题的根源在于对一维数组的理解方式不同！
**       输入相同的一维数组，按我们按行存储习惯提取到的矩阵与cublasSgemm()默认按列存储习惯提取到的矩阵刚好互为转置！
**       综述，为了解决与cublasSgemm()按列存储不匹配的问题，我们只需调换A_gpu与B_gpu输入位置即可！
** 综合分析：以上举例是没有考虑对A_gpu或B_gpu转置的情况，如果考虑了，也很简单，既然输入到cublasSgemm()的A_gpu与B_gpu的位置调换了，
**         那对应调换TA,TB的位置即可，关于TA,TB的作用机理，二者的理解还是一致的（关于TA,TB的作用机理，详细请参考gemm_cpu()）。
*/
void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    // GPU使用cuBLAS库中的cublasSgemm()函数进行矩阵乘法计算，参考：http://docs.nvidia.com/cuda/cublas/index.html#ixzz4peJE8c4M ，
    // 这个网址是CUDA关于cuBLAS库的官方文档，此处cublasSgemm()函数在其中的2.7.1节：cublas<t>gemm()，
    // 可以看出cublasSgem()函数完成C_gpu = ALPHA * A_gpu * B_gpu + BETA * C_gpu的线性矩阵运算，
    // 输入： handle    根据官网，这个变量是一个对开发者不透明的变量，也就是里面具体包含什么，开发者一般无法知道，
    //                 只知道里面包含的是cuBLAS库的相关信息，且这个变量是必须的，按照官网的陈述，
    //                 cuBLAS库中所有的函数都需要这个变量参数（且都是作为其第一个参数），该变量由cublasCreate()初始化，
    //                 并由cublasDestroy()销毁。
    //       transa    是否需要转置A_gpu，这里transa = TB ? CUBLAS_OP_T : CUBLAS_OP_N（是个条件表达式），
    //                 如果TB = 1，则取CUBLAS_OP_T，即需要对A_gpu转置
    //       transb    是否需要转置B_gpu，这里transa = TA ? CUBLAS_OP_T : CUBLAS_OP_N（是个条件表达式），
    //                 如果TA = 1，则取CUBLAS_OP_T，即需要对B_gpu转置
    //       N         B,C的列数（若B需要转置，则此处给出转置后的B即B'的列数，而不是转置前的）
    //       M         A,C的行数（若A需要转置，则此处给出转置后的A即A'的行数，而不是转置前的）
    //       K         A的列数，B的行数（同样，若A与B中的二者或者其中一个进行了转置，则不管怎样，
    //                 转置后的A，B必须行列能够匹配，符合矩阵乘法规则，K也是转置后的值，不是转置前的）
    //       ALPHA     实数系数
    //       B_gpu     输入矩阵（一维数组格式）
    //       ldb       B的列数（不做转置）或者行数（做转置，且给的是转置后B即B'的行数）
    //       A_gpu     输入矩阵（一维数组格式）
    //       lda       A的列数（不做转置）或者行数（做转置，且给的是转置后A即A'的行数）
    //       BETA      实数系数（一维数组格式）
    //       C_gpu     计算结果（一维数组格式）
    //       ldc       C的列数
    // 可以看出，如果不是因为存储方式不同，cublasSgemm()函数的结构与darknet自己实现的cpu版gemm_cpu()一模一样；因为二者存储格式的不同，
    // 需要交换A_gpu,B_gpu的位置，对应M与N之间，TB与TA间，ldb与lda之间都要相应交换。
    // 这些函数接口的一致性不是偶然的，应该也是业界不成文的实现标准，这些接口方式，也值得学习统一！（不要问我为什么不统一存储方式，因为我拒绝回答～）
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);

    // 检查cublasSgemm运算是否正常（可以看到，darknet中，cuda的每一步操作，基本都要检查一下运行状态是否正常）
    check_error(status);
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    float *A_gpu = cuda_make_array(A, (TA ? lda*K:lda*M));
    float *B_gpu = cuda_make_array(B, (TB ? ldb*N : ldb*K));
    float *C_gpu = cuda_make_array(C, ldc*M);

    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_ongpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_ongpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,192,729,1600); 
       time_ongpu(0,0,384,196,1728); 
       time_ongpu(0,0,256,196,3456); 
       time_ongpu(0,0,256,196,2304); 
       time_ongpu(0,0,128,4096,12544); 
       time_ongpu(0,0,128,4096,4096); 
     */
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,576,12544); 
    time_ongpu(0,0,256,2304,784); 
    time_ongpu(1,1,2304,256,784); 
    time_ongpu(0,0,512,4608,196); 
    time_ongpu(1,1,4608,512,196); 

    return 0;
}
#endif

