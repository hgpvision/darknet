#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>
#include <limits.h>

#include "utils.h"

int *read_intlist(char *gpu_list, int *ngpus, int d)
{
    int *gpus = 0;
    if(gpu_list){
        int len = strlen(gpu_list);
        *ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++*ngpus;
        }
        gpus = calloc(*ngpus, sizeof(int));
        for(i = 0; i < *ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpus = calloc(1, sizeof(float));
        *gpus = d;
        *ngpus = 1;
    }
    return gpus;
}

int *read_map(char *filename)
{
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    while((str=fgetl(file))){
        ++n;
        map = realloc(map, n*sizeof(int));
        map[n-1] = atoi(str);
    }
    return map;
}

void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections)
{
    size_t i;
    for(i = 0; i < sections; ++i){
        size_t start = n*i/sections;
        size_t end = n*(i+1)/sections;
        size_t num = end-start;
        shuffle(arr+(start*size), num, size);
    }
}

void shuffle(void *arr, size_t n, size_t size)
{
    size_t i;
    void *swp = calloc(1, size);
    for(i = 0; i < n-1; ++i){
        size_t j = i + rand()/(RAND_MAX / (n-i)+1);
        memcpy(swp,          arr+(j*size), size);
        memcpy(arr+(j*size), arr+(i*size), size);
        memcpy(arr+(i*size), swp,          size);
    }
}

void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atof(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

/*
** 用于从文件全路径字符串cfgfile中提取主要信息，比如从cfg/yolo.cfg中，提取出yolo
** 函数中主要用到strchr()函数定位'/'以及','符号
** 返回：c风格字符串，为从输入cfgfile中的深拷贝部分字符串（主要信息）（输出与输入字符串指向不同的地址，二者复制之后不再有关联）
** 说明：此函数没有实质用处，一般用来提取字符串，以打印出主要信息（比如cfg/yolo.cfg，主要信息就是yolo，表示是yolo网络）
*/
char *basecfg(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
    // 定位到'/'字符，让c等于'/'之后的字符，丢掉之前的字符，
    // 比如c='cfg/yolo.cfg'->c='yolo.cfg'
    while((next = strchr(c, '/')))
    {
        c = next+1;
    }
    // copy_string(c)不会改变c的值，但在函数内会重新给c分配一段地址，使得c与cfgfile不再关联，
    // 这样，下面改动c也不会影响cfgfile的值
    c = copy_string(c);
    next = strchr(c, '.');
    // 接着上面的例子，此时c="yolo.cfg"，为了提取yolo，需要有标识符隔开yolo与cfg，
    // 因此需要识别出'.'，并将其代替为'\0'（'\0'的ASCII值就为0），这样，就可以隔开两段内容，
    // 且前一段为c风格字符串，会被系统自动识别断开（严格来说不是真正的提取出来，而只是将两段分开而已，这里理解就行）
    if (next) *next = 0;
    return c;
}

int alphanum_to_int(char c)
{
    return (c < 58) ? c - 48 : c-87;
}
char int_to_alphanum(int i)
{
    if (i == 36) return '.';
    return (i < 10) ? i + 48 : i + 87;
}

void pm(int M, int N, float *A)
{
    int i,j;
    for(i =0 ; i < M; ++i){
        printf("%d ", i+1);
        for(j = 0; j < N; ++j){
            printf("%2.4f, ", A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}

/*
** 在字符串str中查找指定字符串orig，如果没有找到，则直接令output等于str输出（此时本函数相当于没有执行）；
** 如果找到了，即orig是str字符串的一部分，那么用rep替换掉str中的orig，然后再赋给output返回
** 输入： str    原始字符串
**       orig   字符串
**       rep    替代字符串
**       output 输出字符串
** 使用：在读入训练数据时，只给程序输入了图片所在路径，而标签数据的路径并没有直接给，是通过对图片路径进行修改得到的，
**      比如在训练voc数据时，输入的train.txt文件中只包含所有图片的具体路径，如：/home/happy/Downloads/darknet_dataset/VOCdevkit/VOC2007/JPEGImages/000001.jpg
**      而000001.jpg的标签并没有给程序，是通过该函数替换掉图片路径中的JPEGImages为labels，并替换掉后缀.jpg为.txt得到的，最终得到：
**      /home/happy/Downloads/darknet_dataset/VOCdevkit/VOC2007/labels/000001.txt
**      这种替换的前提是，标签数据文件夹labels与图片数据文件夹JPEGImages具有相同的父目录。
*/
void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = {0};
    char *p;

    sprintf(buffer, "%s", str);
    // strstr()用来判断orig是否是buffer字符串的一部分
    // 如果orig不是buffer的字串，则返回的p为NULL指针；如果是buffer的字串，
    // 则返回orig在buffer的首地址（char *类型，注意是buffer中的首地址）
    if(!(p = strstr(buffer, orig))){  // Is 'orig' even in 'str'?
        // 如果不是，则直接将str赋给output返回，此时本函数相当于什么也没做
        sprintf(output, "%s", str);
        return;
    }

    // 运行到这，说明orig是buffer的字串，则首先把orig字符串抹掉，具体做法是将p处（p是orig在buffer的首地址）的字符置为c风格
    // 字符串终止符'\0'，而后，再往buffer添加待替换的字符串rep（会自动识别到'\0'处），以及原本buffer中orig之后的字符串
    *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p+strlen(orig));
}

float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
    int i,j;
    for(j = 0; j < k; ++j) index[j] = -1;
    for(i = 0; i < n; ++i){
        int curr = i;
        for(j = 0; j < k; ++j){
            if((index[j] < 0) || a[curr] > a[index[j]]){
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

/*
**  输入提示动态分配内存失败，exit(-1)，表明异常退出
*/
void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

list *split_str(char *s, char delim)
{
    size_t i;
    size_t len = strlen(s);
    list *l = make_list();
    list_insert(l, s);
    for(i = 0; i < len; ++i){
        if(s[i] == delim){
            s[i] = '\0';
            list_insert(l, &(s[i+1]));
        }
    }
    return l;
}

/*
**  Pyhon中有该函数，此处用C语言实现：删除输入字符数组s中的空白符（包括'\n','\t',' '）
**  此函数用来统一读入的配置文件格式，有些配置文件书写时可能含有空白符（但是不能含有逗号等其他的符号）
*/
void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];

        // offset为要剔除的字符数，比如offset=2，说明到此时需要剔除2个空白符，
        // 剔除完两个空白符之后，后面的要往前补上，不能留空
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;   // 往前补上
    }

    // 依然在真正有效的字符数组最后紧跟一个terminating null-characteristic '\0'
    s[len-offset] = '\0';
}

void strip_char(char *s, char bad)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==bad) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}

/*
**  读取文件中的某一行，通用（仅读取一行，其读取的结果中不再有换行符或者eof，这两个符号在返回之前被处理掉了）
**  输入： *fp     打开的文件流指针
**  输出： 读到的整行字符，返回C风格字符数组指针（最大512字节），如果读取失败，返回0
*/
char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;

    // 默认一行的字符数目最大为512，如果不够，下面会有应对方法
    size_t size = 512;

    // 动态分配整行的内存：C风格字符数组
    char *line = malloc(size*sizeof(char));

    // char* fgets(char * str, int num, FILE *stream)：从stream中读取一行数据存储到str中，最大字符数为num
    // fgets()函数读取数据时会以新行或者文件终止符为断点，也就是遇到新行会停止继续读入，遇到文件终止符号也会停止读入
    // 注意终止符（换行符号以及eof）也会存储到str中。如果数据读取失败，返回空指针。
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    // size_t strlen ( const char * str );返回C风格字符串（C String）即字符数组str的长度
    // （也就是数组中存储的元素个数，不包括C字符数组最后的终止空字符，每记错的话是'\0'， terminating null-character）
    size_t curr = strlen(line);

    // 终止符（换行符号以及eof）也会存储到str中，所以可用作while终止条件
    // 如果一整行数据顺利读入到line中，那么line[curr-1]应该会是换行符或者eof，
    // 这样就可以绕过while循环中的处理；否则说明这行数据未读完，line空间不够，需要重新分配，重新读取
    while((line[curr-1] != '\n') && !feof(fp)){
        
        // 这个if语句略显多余，感觉不可能会出现curr!=size-1的情况，
        // 因为文件流没有问题（否则进入这个函数就返回了），那就肯定是line空间不够，
        // 才导致line的最后一个有效存储元素不是换行符或者eof
        if(curr == size-1){
            // line空间不够，扩容两倍
            size *= 2;

            // void* realloc (void* ptr, size_t size);重新动态分配大小：扩容两倍
            line = realloc(line, size*sizeof(char));

            // 如果动态分配内存失败，就会返回空指针，注意防护
            if(!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }

        // 之前因为line空间不够，没有读完一整行，此处不是从头开始读，
        // 而是接着往下读，并接着往下存（fgets会记着上一次停止读数据的地方）
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    
    // 实际上'\n'并不是我们想要的有效字符，因此，将其置为'\0'，
    // 这样便于以后的处理。'\0'是C风格字符数组的terminating null-character，
    // 识别C字符数组时，以此为终点（不包括此字符）
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

int read_int(int fd)
{
    int n = 0;
    int next = read(fd, &n, sizeof(int));
    if(next <= 0) return -1;
    return n;
}

void write_int(int fd, int n)
{
    int next = write(fd, &n, sizeof(int));
    if(next <= 0) error("read failed");
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) error("read failed");
        n += next;
    }
}

void write_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) error("write failed");
        n += next;
    }
}

/*
** 将输入的字符串s完全拷贝至copy中并返回，注意此处是深拷贝，即是内容上的拷贝，不是指针变量拷贝，
** 拷贝后s与copy指向不同的地址，二者再无瓜葛
*/
char *copy_string(char *s)
{
    char *copy = malloc(strlen(s)+1);
    // strncpy是c标准库函数，定义在string.h中，表示将首地址为s的strlen(s)+1个字符拷贝至copy中，
    // 之所以加1是因为c风格字符串默认以'\0'结尾，但并不记录strlen统计中，如果不将'\0'拷贝过去，
    // 那么得到的copy不会自动以'\0'结尾（除非是这种赋值格式：char *c="happy"，直接以字符串初始化的字符数组会自动添加'\0'）,
    // 就不是严格的c风格字符串（注意：c字符数组不一定要以'\0'结尾，但c风格字符串需要）
    strncpy(copy, s, strlen(s)+1);
    return copy;
}

list *parse_csv_line(char *line)
{
    list *l = make_list();
    char *c, *p;
    int in = 0;
    for(c = line, p = line; *c != '\0'; ++c){
        if(*c == '"') in = !in;
        else if(*c == ',' && !in){
            *c = '\0';
            list_insert(l, copy_string(p));
            p = c+1;
        }
    }
    list_insert(l, copy_string(p));
    return l;
}

int count_fields(char *line)
{
    int count = 0;
    int done = 0;
    char *c;
    for(c = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done) ++count;
    }
    return count;
}

float *parse_fields(char *line, int n)
{
    float *field = calloc(n, sizeof(float));
    char *c, *p, *end;
    int count = 0;
    int done = 0;
    for(c = line, p = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done){
            *c = '\0';
            field[count] = strtod(p, &end);
            if(p == c) field[count] = nan("");
            if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
            p = c+1;
            ++count;
        }
    }
    return field;
}

/*
** 将以a为首地址此后n个元素相加，返回总和
*/
float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}

void mean_arrays(float **a, int n, int els, float *avg)
{
    int i;
    int j;
    memset(avg, 0, els*sizeof(float));
    for(j = 0; j < n; ++j){
        for(i = 0; i < els; ++i){
            avg[i] += a[j][i];
        }
    }
    for(i = 0; i < els; ++i){
        avg[i] /= n;
    }
}

void print_statistics(float *a, int n)
{
    float m = mean_array(a, n);
    float v = variance_array(a, n);
    printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}

float variance_array(float *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}

int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

/*
** 严格限制输入a的值在min~max之间：对a进行边界值检查，如果小于min或者大于max，都直接置为边界值
** 返回： a的值（如若越界，则会改变，若未越界，则保持不变）
*/
float constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; i += sub) sum += pow(a[i]-b[i], 2);
    return sqrt(sum);
}

float mse_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i]*a[i];
    return sqrt(sum/n);
}

void normalize_array(float *a, int n)
{
    int i;
    float mu = mean_array(a,n);
    float sigma = sqrt(variance_array(a,n));
    for(i = 0; i < n; ++i){
        a[i] = (a[i] - mu)/sigma;
    }
    mu = mean_array(a,n);
    sigma = sqrt(variance_array(a,n));
}

void translate_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] += s;
    }
}

float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];   
    }
    return sqrt(sum);
}

void scale_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] *= s;
    }
}

int sample_array(float *a, int n)
{
    float sum = sum_array(a, n);
    scale_array(a, n, 1./sum);
    float r = rand_uniform(0, 1);
    int i;
    for(i = 0; i < n; ++i){
        r = r - a[i];
        if (r <= 0) return i;
    }
    return n-1;
}

/** 找出数组a中的最大元素，返回其索引值.
 * @param a 一维数组，比如检测模型中，可以是包含属于各类概率的数组（数组中最大元素即为物体最有可能所属的类别）
 * @param n a中元素的个数，比如检测模型中，a中包含所有的物体类别，此时n为物体类别总数
 */
int max_index(float *a, int n)
{
    /// 如果a为空，返回-1
    if(n <= 0) return -1;
    
    /// max_i为最大元素的索引，初始为第一个元素，而后遍历整个数组，找出最大的元素，返回其索引值
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

int rand_int(int min, int max)
{
    if (max < min){
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand()%(max - min + 1)) + min;
    return r;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
// Box-Muller transform是一种获取独立，标准正态分布随机数的方法
// 返回标准正态分布随机数（float）
float rand_normal()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    // z0和z1都用了，并不是只用z0或只用z1
    if(haveSpare)
    {
        haveSpare = 0;
        // z1 = sqrt(-2 * log(rand1)) * sin(rand2)
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    // 产生0~1的随机数
    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;  // 不能太小
    rand1 = -2 * log(rand1);
    // 产生0~2*PI之间的随机数
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

    // z0 = sqrt(-2 * log(rand1)) * cos(rand2)
    return sqrt(rand1) * cos(rand2);
}

/*
   float rand_normal()
   {
   int n = 12;
   int i;
   float sum= 0;
   for(i = 0; i < n; ++i) sum += (float)rand()/RAND_MAX;
   return sum-n/2.;
   }
 */

size_t rand_size_t()
{
    return  ((size_t)(rand()&0xff) << 56) | 
            ((size_t)(rand()&0xff) << 48) |
            ((size_t)(rand()&0xff) << 40) |
            ((size_t)(rand()&0xff) << 32) |
            ((size_t)(rand()&0xff) << 24) |
            ((size_t)(rand()&0xff) << 16) |
            ((size_t)(rand()&0xff) << 8) |
            ((size_t)(rand()&0xff) << 0);
}

/*
** 产生(min,max)区间均匀分布的随机数
** 输入： min     区间下限
**       max     区间上限
** 注意:输入的min,max并不一定min<max，所以函数内先比较了二者之间的大小，确保区间上下限无误
*/
float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if(rand()%2) return scale;
    return 1./scale;
}

float **one_hot_encode(float *a, int n, int k)
{
    int i;
    float **t = calloc(n, sizeof(float*));
    for(i = 0; i < n; ++i){
        t[i] = calloc(k, sizeof(float));
        int index = (int)a[i];
        t[i][index] = 1;
    }
    return t;
}

