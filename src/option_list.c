#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

/*
**  读取数据配置文件（.data文件），包含数据所在的路径、名称，其中包含的物体类别数等等
**  返回：list指针，包含所有数据信息。函数中会创建options变量，并返回其指针（若文件打开失败，将直接退出程序，不会返空指针）
*/
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();

    // 读取文件流中的一行数据：返回C风格字符数组指针，不为空有效
    while((line=fgetl(file)) != 0){
        ++ nu;
        // 删除line中的空白符
        strip(line);
        switch(line[0]){
            // 以下面三种字符开头的都是无效行，直接跳过（如注释等）
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

/*
**  解析一行数据的内容，为options赋值，主要调用option_insert()
**  输入：s        从文件读入的某行字符数组指针
**       options    实际输出，解析出的数据将为该变量赋值
**  返回：int类型数据，1表示成功读取有效数据，0表示未能读取有效数据（说明文件中数据格式有问题）
**  流程：从配置（.data或者.cfg，不管是数据配置文件还是神经网络结构数据文件，其读取都需要调用这个函数）
**      文件中读入的每行数据包括两部分，第一部分为变量名称，如learning_rate，
**      第二部分为值，如0.01，两部分由=隔开，因此，要分别读入两部分的值，首先识别出
**      等号，获取等号所在的指针，并将等号替换为terminating null-characteristic '\0'，
**      这样第一部分会自动识别到'\0'停止，而第二部分则从等号下一个地方开始
*/
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        // 找出=的偏移位
        if(s[i] == '='){
            //将等号替换为'\0'，作为第一部分的终止符
            s[i] = '\0';
            // 第二部分字符数组的起始指针
            val = s+i+1;
            break;
        }
    }
    // 如果i==len-1，说明没有找到等号这个符号，那么就直接返回0（文件中还有一些注释，此外还有用[]括起来的字符，这些是网络层的类别或者名字，比如[maxpool]表示这些是池化层的参数）
    if(i == len-1) return 0;
    char *key = s;
    // 调用option_insert为options赋值，类比C++中的map数据结构，key相当是键值（变量名），
    // 而val则是值（变量的值）
    option_insert(options, key, val);
    return 1;
}

/*
**  将输入key和val赋值给kvp结构体对象，最终调用list_insert()将kvp赋值给list对象l，
**  完成最后的赋值（此函数之后，文件中某行的数据真正读入进list变量）
**  说明： 这个函数有点类似C++中按键值插入元素值的功能
**  输入：l        输出，最终被赋值的list变量
**       key      变量的名称，C风格字符数组
**       value    变量的值，C风格字符数组（还未转换成float或者double数据类型）
*/
void option_insert(list *l, char *key, char *val)
{
    // kvp也是一个结构体，包含两个C风格字符数组指针：key和val，对应键值和值，
    // 此处key为变量名，val为变量的值（比如类别数，路径名称，注意都是字符类型数据）
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

/*
**  在l中查找并返回指定键值的值（返回类型统一为C风格字符数组指针，若l中没有该键值，则返回0指针）
**  输入：l        list结构体指针
**       key      指定键值（C风格字符数组指针）
**  返回：char*    l中对应键值key的值（C风格字符数组指针）
**  说明：该函数实现了类似C++中的map数据按键值查找的功能；返回值统一为C风格字符数组，可以进一步转为int,float等
*/
char *option_find(list *l, char *key)
{
    // 获取l中的第一个节点（一个节点的值包含了一条项目的信息：键值与值两部分，
    // 比如classes=80，键值即项目名称classes，值为80，表示该数据集共包含80类物体）
    node *n = l->front;

    // 遍历l中的所有节点，找出节点值的键值等于指定键值key的节点，获取其值的值并返回
    // 这里的注释有些拗口，注意node中含有一个void* val，所以称作节点的值，
    // 而此处节点的val的具体类型为kvp*，该数据中又包含一个key*，一个val*，
    // 因此才称作节点值的值，节点值的键值
    while(n){
        // 获取该节点的值，注意存储到node中的val是void*指针类型，需要强转为kvp*
        kvp *p = (kvp *)n->val;
        // 比较该节点的值的键值是否等于指定的键值key，如果等于，则说明找到了，返回找到的值中（C字符数组指针）
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        // 如当前节点不是指定键值对应的节点，继续查找下一个节点
        n = n->next;
    }
    // 若遍历完l都没找到指定键值的节点的值的值，则返回0指针
    return 0;
}

/*
**  在l中查找并返回指定键值的值（返回类型统一为C风格字符数组指针，若l中没有该键值，返回默认值def）
**  输入：l        list结构体指针
**       key      指定键值（C风格字符数组指针）
**       def      意为default，默认值，如果l中没有找到对应键值key的值，则直接返回def
**  返回：char*    l中对应键值key的值（C风格字符数组指针）或者默认值（未找到key对应的值）
**  说明：C语言不像C++，在声明函数时，是不可以设置默认参数的，但是可以在调用的时候，指定第三个参数def为字面字符数组，
**       这就等同于指定了默认的参数了，就像detectr.c中test_detector()函数调用option_find_str()的那样
*/
char *option_find_str(list *l, char *key, char *def)
{
    // 如果在l中找到了对应key的值，则返回的v不是空指针
    char *v = option_find(l, key);
    if(v) return v;
    // 若没找到，v为空指针，则返回def（默认值），并在屏幕上提示使用默认值
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

/*
**  按给定键值key从l中查找对应的参数值，主要调用option_find()函数。
**  功能和上一个函数option_find_str()基本一样，只不过多了一步处理，将输出转为了int之后再输出。
**  输入：l    list指针，实际为section结构体中的options元素，包含该层神经网络的所有配置参数
**       key  键值，即参数名称，比如卷积核尺寸size，卷积核个数filters，跨度stride等等
**       def  默认值，如果没有找到对应键值的参数值，则使用def作为默认值
**  输出：int类型（该函数专门用来识别整数数值），即参数值，比如filters的值为96，size的值为11等
*/
int option_find_int(list *l, char *key, int def)
{
    // 在l中查找key的值，返回C风格字符数组，若未找到，返回空指针
    char *v = option_find(l, key);
    // 不为空，则调用atoi()函数将v转为整形并返回
    if(v) return atoi(v);
    // 若未空，说明未找到，返回默认值，并输出提示信息
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

/*
**  与上面的option_find_int()函数基本一样，唯一的区别就是使用默认值时，没有在屏幕上输出
**  Using default...字样的提示，因此叫做quiet（就是安安静静的使用默认值，没有提示）
*/
int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

/*
**  与下面的option_find_float()函数基本一样，唯一的区别就是使用默认值时，没有在屏幕上输出
**  Using default...字样的提示，因此叫做quiet（就是安安静静的使用默认值，没有提示）
*/
float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

/*
**  按给定键值key从l中查找对应的参数值，主要调用option_find()函数。
**  功能和函数option_find_int()基本一样，只不过最后调用atof()函数将字符串转为了float类型数据，而不是整型数据。
**  输入：l    list指针，实际为section结构体中的options元素，包含该层神经网络的所有配置参数
**       key  键值，即参数名称，比如卷积核尺寸size，卷积核个数filters，跨度stride等等
**       def  默认值，如果没有找到对应键值的参数值，则使用def作为默认值
**  输出：float类型（该函数专门用来识别浮点型数值），即参数值，比如momentum的值为0.9，decay的值为0.0005等
*/
float option_find_float(list *l, char *key, float def)
{
    // 在l中查找key的值，返回C风格字符数组，若未找到，返回空指针
    char *v = option_find(l, key);
    // 不为空，则调用atoi()函数将v转为整形并返回
    if(v) return atof(v);
    // 若未空，说明未找到，返回默认值，并输出提示信息
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}
