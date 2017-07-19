int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include "assert.h"
#include <stdlib.h>
#include <time.h>

/* 
** 设置当前活跃GPU卡号（即设置gpu_index=n，同时调用cudaSetDevice函数设置当前计算的GPU卡号）,
** 并进行错误检查
*/
void cuda_set_device(int n)
{
    gpu_index = n;

    // 设置当前活跃卡号：返回值类型为cudaError_t，当返回cudaSuccess时，说明设置成功，
    // 返回返回其他失败的cudaError_t标志（标志的具体含义通过随后调用的check_error()函数解析）
    // 要说明的是，cudaSetDevice()函数可能会返回之前的API或者异步线程的错误信息，
    // 所以在check_error()中，都会调用cudaGetLastError()函数获取之前的错误信息，
    // （不知道为什么把cudaDeviceSynchronize()注释了，按理这个函数也是需要的，参考http://blog.csdn.net/xiewen_bupt/article/details/47193557）
    // 这样将两段错误信息综合起来，可以更好的定位错误信息（要定位某个API是否调用出错，必须要清楚该API之前的代码有没有错误）
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

/* 
** 获取当前活跃GPU卡号并返回（int类型）, 并进行错误检查
*/
int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

/*
** 该函数来解析cudaError_t对象，获取具体的错误提示信息，因为cudaError_t是枚举类型，
** 并不能直接从中看出具体的错误信息是什么，为此，cuda提供了一些函数来解析具体的错误信息，下面就是其中两个：
** __host__ ​ __device__ ​const char* cudaGetLastError ( cudaError_t error )
** __host__ ​ __device__ ​const char* cudaGetErrorString ( cudaError_t error )
** 参考：http://blog.csdn.net/xiewen_bupt/article/details/47193557
** 本函数主要调用cudaGetErrorString()函数输出打印错误信息，同时调用了cudaGetLastError()函数
** 获取之前的错误信息（要定位某个API是否调用出错，必须要清楚该API之前的代码有没有错误）
** 
*/
void check_error(cudaError_t status)
{
    // cudaGetLastError用来获取之前的API的调用状态
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    // 如若不是成功调用状态，那么就调用cudaGetErrorString()函数打印出具体的错误信息
    // status是当前API（就是调用check_error()之前的API）返回的调用状态，
    // status2是之前的API的调用状态
    // 如果status2没有错误，status有错误，显然可以判断是当前API调用出错；
    // 如果status2有错，说明之前的API调用有错，这时候应该先排除之前的调用错误，
    // 再看看错误情况
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        // TODO(2017/06/23):这个断言放在这？不太理解，难道是上面的理解有误？？
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]){
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}

#endif
