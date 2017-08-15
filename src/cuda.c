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
** 此函数的理解暂时并不透彻！！！后续需进一步理解！！！
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
        assert(0);  // 恒断言，那么后面的语句永远不会执行了。。。。。。这。。。。
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

/*
** 该函数用来计算GPU核函数调用时的grid的配置参数
** 输入： n    元素个数
** 返回： GPU dim3数据类型，共三维，第一维表示每行有多少个block，第二维表示每列有多少个block，
**            第三维表示一个核函数拥有几个grid（目前一个函数只能有一个grid，所以值恒为1）；
**            整个grid中共有Dg.x*Dg.y个block，其中Dg.x和Dg.y最大值为65535（Dg含义见说明1）。  
** 参考：此处涉及的GPU编程（dim3数据类型，grid概念等），可以参考：
**      http://blog.csdn.net/a925907195/article/details/39500915
**      http://blog.csdn.net/hjimce/article/details/51506207
** 说明：GPU核函数（kernel），也即使用__global__限定符定义的函数，在cpu中调用时，必须使用<<<>>>配置参数，
**      调用格式为： Kernel<<<Dg, Db, Ns, S>>>(param list);其中Dg,Db都是dim3数据类型，Dg用来配置grid相关参数，
**      Db用来配置block相关参数（详细见第一个参考链接）
** 说明2：输入n表示总元素个数，或者可以说是总计算次数，本函数的作用就是配置grid的参数，使得总共n次运算能够较为均匀地分到多个block中，
**       进行并行计算。比如blas_kernels.c中fill_ongpu()调用本函数时，输入的n为矩阵总元素个数，需要将包含n个元素的矩阵全部赋初值，
**       这时可以将这n个元素的赋初值操作较为均匀地分到多个block中（每个block又有多个线程），从而并行加快运算。
** 说明3：grid->block->thread：一个Grid代表一块GPU芯片，所有的线程共享显存数据；每个grid就相当于一块显卡。
**       在每一个GPU芯片里面包含着多个block,每个block包含了512或者1024个线程。每个块里各自有一个共享数据存储的区域，只有块内的线程可以访问。
**       每个block包含多个thread。这部分内容可以参考给出的第二个链接。
*/
dim3 cuda_gridsize(size_t n){
    // 计算k:需要的block个数，可以看出，这里是想把这n个元素分得非常非常细啊，每个线程处理一个计算。
    // BOLCK是在cuda.h中定义的宏，值为512，即默认每个block中含有512个线程，此处用(n-1)/BLOCK相当于向下取整，
    // 再加1,即多出来的，不够512的也需要一个block来处理。
    size_t k = (n-1) / BLOCK + 1;
    // 如果k不超过65535，那么grid只需一行就可以了（一行有k个block,x=block），不需要多行了（y=1）
    size_t x = k;
    size_t y = 1;
    // 因为Dg.x和Dg.y都不能超过65535,如果超过，那就直接令x等于k的平方根（向上取整），而后增大y值，总之，使得x*y=k就可以了
    // （当然可能不会严格等于，但差不多接近，这里也没说一定要分配k个block，尽量均匀就够了，实际k的计算也就是一个近似估计值），
    // 你可能会问，要是sqrt(k)还超过65535呢？正常情况下是不会的，想想看，65535*63535*512这个数够吓人的，
    // 正常不会一下遇到这么大的计算量或数据元素个数（假设一张图片1000*1000，那将近得有2.2e10张照片才能有这么多像素。。。）
    // 你要坚持较真，那这话真是没法接（只能临时按需调整取值了～）
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

/*
** 定义一系列的handle（这些handle都是静态变量，只会定义一次，但后来可以重新赋值/初始化），并初始化当前GPU对应的handle 
** 返回： 当前GPU对应的已经赋值的handle
** 说明： 函数中一次定义了16个static handle,但一次只会调用cublasCreate()初始化当前正在参与计算的GPU对应的handle，详见函数内注释
*/
cublasHandle_t blas_handle()
{
    // TODO: 2018.08.15的理解，此时并没有完全洞悉整个代码，可能理解不到位
    // 此处声明为静态变量，这是需要的，因为这里一下子创建了16个关于cuBLAS library context的句柄（这里我将handle翻译成句柄），
    // 这些变量只需定义一次就够了，之后再整个程序中可以长期使用（也就是在本文件中是一个全局变量），避免反复定义，
    // 但要记得在退出整个程序前，用cublasDestory()函数挨个销毁。
    // 参考官方文档：http://docs.nvidia.com/cuda/cublas/index.html#ixzz4peJE8c4M 2.7.1节
    // 目前尚不清楚这个句柄中到底包含了cuBLAS library的哪些背景信息，也不知道这个句柄中包含的信息在整个程序运行中会不会变动，
    // 总之，诚如官方文档所言，cuBLAS库中每个函数都需要这个句柄，且由cublasCreate()初始化，由cublasDestory()最后销毁。
    // 从下面的初始化可以看出，每个句柄都是跟具体的GPU卡号相关的，这里的16,应该也是假定最多有16块GPU吧（如果你有更多，应该要改动值了）
    // 总之，关于句柄的作用机理，此时暂不能多说，日后经验攒够了，再来详述！
    static int init[16] = {0};
    static cublasHandle_t handle[16];

    // 此处并不会一股脑将16个handle都初始化，而是只初始化当前GPU对应的handle。显然，如果有两块GPU，可能现在在使用0号卡计算，
    // 那就初始化对应0号卡的句柄，如果下一个时间，再次使用0号卡计算，是又会调用cublasCreate()函数初始化的（不知道这时的句柄与上次的句柄是否相同，
    // 虽然都是对应同一个GPU卡）（看来这个句柄与GPU卡号存在莫大的联系）
    int i = cuda_get_device();
    if(!init[i]) {
        // 使用cublasCreate()函数初始化（或者说是赋值？）当前GPU卡对应的handle，注意不是定义，
        // 而是用该函数来初始化handle，并且还与GPU卡号相关
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

/*
** 此函数用于创建一个CUDA数组，并用cudaMalloc()分配内存（使用cudaMalloc()分配的内存需要使用cudaFree()释放），
** 此外，如果输入x不为空指针的话，还会将x中的值复制到创建的CUDA数组中（这是考虑有些变量有cpu版本的，这些变量在
** 之前定义cpu版本时已经初始化了，比如l.weights之类的，为了不在重新初始化，直接复制到GPU版本的变量中即可）
** 输入： x    cpu版本的变量（可以参看各层创建函数，比如make_connected_layer()函数）
**       n    CUDA数组含有的元素个数（也是x中含有的元素个数）
** 输出：x_gpu，其指向的内存在GPU设备内存上，不是在主机内存上
*/
float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;

    // status是执行cudaMalloc返回的状态,更为详细的可以参考chek_error()中的注释
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    // 判断是否成功执行cudaMalloc()（也即是否成功分配内存）
    check_error(status);

    // 如果x非空，则将x值复制到x_gpu中，为其初始化值
    // cudaError_t cudaMemcpy (void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    // 类似c语言中的memcpy()函数，将src中的数据从src首地址开始复制count个字节的数据到dst中，最后一个参数是复制方向，
    // 表明是从GPU设备内存拷贝到主机内存还是从主机内存拷贝到GPU设备内存，此处是主机内存拷贝至设备内存
    // cudaMemcpy()就是用来实现主机内存和GPU内存之间互相拷贝数据。
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        // 同样，检查拷贝过程是否正常
        check_error(status);
    }
    // 如果x_gpu为空，说明cuda内存分配失败，则中断程序并打印提示符退出（前面的chek_error应该已经具备这个功能了吧？）
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
