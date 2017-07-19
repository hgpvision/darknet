#include "matrix.h"
#include "utils.h"
#include "blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/*
** 深层释放二维数组的堆内存（m.vals是一个float**，即二维数组）
** 说明： 输入虽然是matrix类型数据，但本函数实际是释放其子元素vals的堆内存。
**       因为vals是二维数组，之前使用calloc()为每一行动态分配了内存，
**       此外，vals变量本身也是一个由calloc(rows, sizeof(float*))动态分配的用来存储
**       每一行指针变量的堆内存，因此释放过程首先需要释放每一行的堆内存，最后再直接释放m.vals
*/
void free_matrix(matrix m)
{
    int i;
    // 先逐行释放堆内存
    for(i = 0; i < m.rows; ++i) free(m.vals[i]);
    // 最后直接释放m.vals，清空用于存储每一行指针变量的堆内存
    free(m.vals);
}

float matrix_topk_accuracy(matrix truth, matrix guess, int k)
{
    int *indexes = calloc(k, sizeof(int));
    int n = truth.cols;
    int i,j;
    int correct = 0;
    for(i = 0; i < truth.rows; ++i){
        top_k(guess.vals[i], n, k, indexes);
        for(j = 0; j < k; ++j){
            int class = indexes[j];
            if(truth.vals[i][class]){
                ++correct;
                break;
            }
        }
    }
    free(indexes);
    return (float)correct/truth.rows;
}

void scale_matrix(matrix m, float scale)
{
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.vals[i][j] *= scale;
        }
    }
}

matrix resize_matrix(matrix m, int size)
{
    int i;
    if (m.rows == size) return m;
    if (m.rows < size) {
        m.vals = realloc(m.vals, size*sizeof(float*));
        for (i = m.rows; i < size; ++i) {
            m.vals[i] = calloc(m.cols, sizeof(float));
        }
    } else if (m.rows > size) {
        for (i = size; i < m.rows; ++i) {
            free(m.vals[i]);
        }
        m.vals = realloc(m.vals, size*sizeof(float*));
    }
    m.rows = size;
    return m;
}

void matrix_add_matrix(matrix from, matrix to)
{
    assert(from.rows == to.rows && from.cols == to.cols);
    int i,j;
    for(i = 0; i < from.rows; ++i){
        for(j = 0; j < from.cols; ++j){
            to.vals[i][j] += from.vals[i][j];
        }
    }
}

matrix copy_matrix(matrix m)
{
    matrix c = {0};
    c.rows = m.rows;
    c.cols = m.cols;
    c.vals = calloc(c.rows, sizeof(float *));
    int i;
    for(i = 0; i < c.rows; ++i){
        c.vals[i] = calloc(c.cols, sizeof(float));
        copy_cpu(c.cols, m.vals[i], 1, c.vals[i], 1);
    }
    return c;
}

matrix make_matrix(int rows, int cols)
{
    int i;
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.vals = calloc(m.rows, sizeof(float *));
    for(i = 0; i < m.rows; ++i){
        m.vals[i] = calloc(m.cols, sizeof(float));
    }
    return m;
}

matrix hold_out_matrix(matrix *m, int n)
{
    int i;
    matrix h;
    h.rows = n;
    h.cols = m->cols;
    h.vals = calloc(h.rows, sizeof(float *));
    for(i = 0; i < n; ++i){
        int index = rand()%m->rows;
        h.vals[i] = m->vals[index];
        m->vals[index] = m->vals[--(m->rows)];
    }
    return h;
}

float *pop_column(matrix *m, int c)
{
    float *col = calloc(m->rows, sizeof(float));
    int i, j;
    for(i = 0; i < m->rows; ++i){
        col[i] = m->vals[i][c];
        for(j = c; j < m->cols-1; ++j){
            m->vals[i][j] = m->vals[i][j+1];
        }
    }
    --m->cols;
    return col;
}

matrix csv_to_matrix(char *filename)
{
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);

    matrix m;
    m.cols = -1;

    char *line;

    int n = 0;
    int size = 1024;
    m.vals = calloc(size, sizeof(float*));
    while((line = fgetl(fp))){
        if(m.cols == -1) m.cols = count_fields(line);
        if(n == size){
            size *= 2;
            m.vals = realloc(m.vals, size*sizeof(float*));
        }
        m.vals[n] = parse_fields(line, m.cols);
        free(line);
        ++n;
    }
    m.vals = realloc(m.vals, n*sizeof(float*));
    m.rows = n;
    return m;
}

void matrix_to_csv(matrix m)
{
    int i, j;

    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            if(j > 0) printf(",");
            printf("%.17g", m.vals[i][j]);
        }
        printf("\n");
    }
}

void print_matrix(matrix m)
{
    int i, j;
    printf("%d X %d Matrix:\n",m.rows, m.cols);
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.vals[i][j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}
