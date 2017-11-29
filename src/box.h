#ifndef BOX_H
#define BOX_H

/**
 * 物体检测定位矩形框，即用于定位物体的矩形框，包含矩形框中心坐标x（横坐标）,y（纵坐标）以及矩形框宽w，高h总共4个参数值，
 * 不包含矩形框中所包含的物体的类别编号值。4个值都是比例坐标，也即占整个图片宽高的比例（x为占图宽的比例，y为占图高的比例）。
 */
typedef struct{
    float x, y, w, h;
} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

box float_to_box(float *f, int stride);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
