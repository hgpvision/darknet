#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

/*
**	将val指针插入list结构体l中，这里相当于是用C实现了C++中的list的元素插入功能
**	流程：	list中并不直接含有void*类型指针，但list中含有的node指针含有void*类型指针，
**		  因此，将首先创建一个node指针new，而后将val传给new，最后再把new插入list指针l中
**	说明： void*指针可以接收不同数据类型的指针，因此第二个参数具体是什么类型的指针得视情况而定
**	调用： 该函数在众多地方调用，很多从文件中读取信息存入到list变量中，都会调用此函数，
**		  注意此函数类似C++的insert()插入方式；而在option_list.h中的opion_insert()函数，
**		  有点类似C++ map数据结构中的按值插入方式，比如map[key]=value，两个函数操作对象都是list变量，
**		  只是操作方式略有不同。
*/
void list_insert(list *l, void *val)
{
	// 定义一个node指针并动态分配内存
	node *new = malloc(sizeof(node));
	// 将输入的val指针赋值给new中的val元素，注意，这是指针复制，共享地址，二者都是void*类型指针
	new->val = val;
	new->next = 0;

	// 下面的链表嵌套主要注意一下
	// 如果l的back元素为空指针，说明l到目前为止，还没有存入数据（只是在函数外动态分配了内存，并没有赋有效值），
	// 这样，令l的front为new（此后front将不会再变，除非删除），显然此时new的前面没有node，因此new->prev=0
	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		// 如果已经插入了第一个元素，那么往下嵌套，注意这里操作的是指针，互有影响
		// 新插入的node赋给back的下一个node next，
		// 同时对于新的node new来说，其前一个node为l->back
		// 一定注意要相互更新（链表上的数据位置关系都是相对的）
		l->back->next = new;
		new->prev = l->back;
	}
	// 更新back的值
	// 不管前面进行了什么操作，每次插入new，都必须更新l的back为当前new，因为链表结构，
	// 新插入的元素永远在最后，此处back就是链表中的最后一个元素，front是链表中的第一个元素，
	// 链表中的第一个元素在第一次插入元素之后将不会再变（除非删除）
	l->back = new;
	// l中存储的元素个数加1
	++l->size;
}

/*
**	释放节点内存，注意节点node是结构体指针，其中含有自引用，需要递归释放内存
**	输入：	n	需要释放的node指针，其内存是动态分配的
**	注意：	不管什么程序，堆内存的释放都是一个重要的问题，darknet中的内存释放，值得学习！
**		  node结构体含有自引用，输入的虽然是一个节点，但实则是一个包含众多节点的链表结构，
**		  释放内存时，一定要递归释放，且格外注意内存释放的顺序
*/
void free_node(node *n)
{
	// 输入的node n是链表结构中的第一个（最前面的）node，因此是从前到后释放内存的，
	// 在释放每个node的之前，必须首先获取该node的下一个node，否则一旦过早释放，
	// 该节点之后的node将无从访问，无法顺利释放，这就可能造成内存泄漏
	node *next;
	// 遍历链表上所有的node，依次释放，直至n为空指针，说明整个链表上的节点释放完毕
	while(n) {
		// 释放n之前，首先获取n的下一个节点的指针，存至next中
		next = n->next;

		// 释放当前node内存
		free(n);
		// 将next赋值给n，成为下一个释放的节点
		n = next;
	}
}

/*
**	释放链表list变量的内存，主要调用free_node()函数以及C中的free()函数
**	输入：	l	要释放内存的list指针，其内存是动态分配的（堆内存）
**	注意：	list是个结构体，其中又嵌套有node结构体指针，因此需要递归释放内存。
**	      像这样嵌套动态分配内存的指针数据（结构体，还有C++中的类等），
**		  一定要作用到最底层，不能只在表层释放
*/
void free_list(list *l)
{
	// list中含有两种数据类型，其中主要一种是node，由于node存在自引用，组成了一个链表结构，
	// 因此需要调用free_node()函数递归释放内存
	free_node(l->front);
	// 释放完node内存之后，在释放l本身中含有的其他数据类型，即int类型
	// 同样，node的释放必须在释放l之前，否则将访问不到node，无法顺利释放node的内存，
	// 很可能造成内存泄漏
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

/*
**	将list变量l中的每个node的val提取出来，存至一个二维字符数组中，返回
**	输入：	l	list类型变量，此处的list类型变量包含众多节点，每个节点的值就是我们需要提取出来的信息
**	返回：返回类型为void**指针，因此，返回后，需要视具体情况进行指针类型墙转，比如转为char**（二维字符数组）
**  调用：data.c->get_labels()调用，目的是为了从包含数据集所有物体类别名称信息的l中提取所有物体名称，存入二维字符数组中并返回（这样做是为了便于访问）
*/
void **list_to_array(list *l)
{
	// a是一个二维字符数组：第一维可以看作是行，第二维可以看作是列，每一行相当于是一个字符数组（类似C++中的string）
    // a是一个二维字符数组，每个a[i]是一个一维字符数组，不过l中node的值是void*类型的，所以分配内存时，直接获取void*的大小就可以了，
	// calloc()是C语言函数，第一个参数是元素的数目，第二个则是每个元素的大小（字节）
	void **a = calloc(l->size, sizeof(void*));
    int count = 0;

	// 获取l中的首个节点
    node *n = l->front;
	// 遍历l中所有的节点，将各节点的值抓取出来，放到a中
    while(n){
		// 为每一个字符数组赋值：直接用指针赋值（n->val是指针，a[count++]也是指针，且都是void*型），不再需要为每一个a[i]动态分配内存，这里极好地使用了指针带来的便利
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
