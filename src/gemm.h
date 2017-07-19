#ifndef GEMM_H
#define GEMM_H

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

/*
**  功能：矩阵计算，完成C = ALPHA * A * B + BETA * C，
**       输出的C也是按行存储（所有行并成一行）
**  输入： A,B,C   输入矩阵（一维数组格式，按行存储，所有行并成一行）
**        ALPHA   系数
**        BETA    系数
**        TA,TB   是否需要对A,B做转置操作，是为1,否为0
**        M       A,C的行数
**        N       B,C的列数
**        K       A的列数，B的行数
**        lda     A的列数（不做转置）或者行数（做转置）
**        ldb     B的列数（不做转置）或者行数（做转置）
**        ldc     C的列数
**  说明：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
*/
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif
#endif
