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
**        M       A,C的行数（不做转置）或者A'的行数，此处A未转置，故为A的行数
**        N       B,C的列数（不做转置）或者B'的列数，此处B未转置，故为B的列数
**        K       A的列数（不做转置）或者A'的列数，B的行数（不做转置）或者B'的行数，此处A,B均未转置，故为A的列数、B的行数
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
**        M       A,C的行数（不做转置）或者A'的行数，此处A未转置，故为A的行数
**        N       B,C的列数（不做转置）或者B'的列数，此处B转置，故为B’的列数
**        K       A的列数（不做转置）或者A'的列数，B的行数（不做转置）或者B'的行数，此处A不转置，B转置，故为A的列数、B'的行数
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
**        M       A,C的行数（不做转置）或者A'的行数，此处A转置，故为A'的行数
**        N       B,C的列数（不做转置）或者B'的列数，此处B未转置，故为B的列数
**        K       A的列数（不做转置）或者A'的列数，B的行数（不做转置）或者B'的行数，此处A转置，B不转置，故为A'的列数、B的行数
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
**  功能：矩阵计算，实际完成C = ALPHA * A' * B’ + BETA * C矩阵计算
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数，此处A转置，故为A'的行数
**        N       B,C的列数（不做转置）或者B'的列数，此处B转置，故为B'的列数
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
**  功能：矩阵计算，完成C = ALPHA * A * B + BETA * C矩阵计算
**  输入： 
**        TA,TB   是否需要对A,B做转置操作，是为1,否为0
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
**           举个例子，假设： A = [1, 2, 3, 2, 2, 1], B = [2, 0, 1, 1, 2, 1], C = [3, 0, 1, 2] ，且C为2*2的矩阵，即C = [3, 0; 1, 2]，那么要进行C = ALPHA * A * B + BETA * C的计算，
**           必须满足矩阵乘法行列匹配规则，则参与运算的第一个矩阵只能为2*3，第二个只能为3*2，因为A,B的元素个数已经固定为6个。下面分别说明gemm_nn(),gemm_tn(),gemm_nt,gemm_tt()四个函数对该例子的计算。
**           诚如上所述，不管A, B有没有转置，反正最后参与计算的两个矩阵必须前者为2*3,后者为3*2。如果使用gemm_nn()，A,B都没有转置，那么就要求没有转置的A,B分别为2*3,3*2矩阵，
**           则 A = [ 1, 2, 3; 2, 2, 1], B = [2, 0; 1, 1; 2, 1], 调用gemm_nn(2, 2, 3, 1, A, 3, B, 2, C, 2)计算得到 C = [13, 5; 9, 5]（其中ALPHA = BETA = 1，下同）；
**           如果要用gemm_tn()函数，即A需要进行转置之后才能计算，也即转置之后的维度为2*3,而转置之前的维度为3*2，B没有转置，本身就是3*2的矩阵，这样，
**           A = [ 1, 2; 3, 2; 2, 1], A' = [1, 3, 2; 2, 2, 1], B = [2, 0; 1, 1; 2, 1]，gemm_tn(2, 2, 3, 1, A, 2, B, 2, C, 2)函数实际计算的是A'*B+C的值，注意此时的A与gemm_nn()中的A有什么不同，
**           如前所述，A是按行保存的，因为此时的A本身是一个3*2的矩阵，按照按行保存规则，就是A = [ 1, 2; 3, 2; 2, 1]，调用gemm_tn()的时候，M, N, K分别为2, 2, 3,都是最终参与计算的矩阵的行列数，
**           因为此处真正参与计算的是A'与B，所以M为A'的行数，即为2,N为B的列数，即为2,K为A'与B的列数，即为3，而此时lda=2，是因为A进行了转置，因此输入的是A'的行数，而不是列数3,ldb=2，为B的列数，最终计算得到C=[12, 5; 9, 5]。
**           对于gemm_nt()与gemm_tt()，与上分析一样，不再赘述了。此部分注释进行了测试，对应测试文件darknet_test_gemm.c。
**  强调： 这一系列的gemm()函数，都带有叠加效果，也即最终的值是保存在C中，但这种保存并不是擦除式的保存，而是叠加式的保存，也就是说，如果进入gemm()函数之前，如果C的元素已经有值了，
**        那么这些值不会被擦除掉，而是会将其叠加，其实看式子就可以看出来：此函数完成的是C = ALPHA * A * B + BETA * C矩阵运算。
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

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
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

