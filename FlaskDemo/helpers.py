#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Helper classes and functions for detection and tracking
"""

import numpy as np
import cv2
import math


class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;


def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);


def box_iou2(a, b):
    '''


helper函数用于计算相交与并集之间的比率
两个盒子A和B
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''

    w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)


def convert_to_pixel(box_yolo, img, crop_range):
    '''
  转换（缩放）边界框坐标的辅助函数到像素坐标。

    Example (0.89361443264143803, 0.4880486045564924, 0.23544462956491041,
    0.36866588651069609)

    crop_range: specifies the part of image to be cropped
    '''

    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape

    #  计算边界框的左侧、顶部、宽度和高度
    left = int((box.x - box.w / 2.) * (xmax - xmin) + xmin)
    top = int((box.y - box.h / 2.) * (ymax - ymin) + ymin)

    width = int(box.w * (xmax - xmin))
    height = int(box.h * (ymax - ymin))

    # 处理边缘
    if left < 0:  left = 0
    if top < 0:   top = 0

    #  返回坐标（以像素为单位）

    box_pixel = np.array([left, top, width, height])
    return box_pixel


def convert_to_cv2bbox(bbox, img_dim=(1280, 720)):
    '''
     帮助将bbox转换为bbox_cv2的功能
    bbox = [left, top, width, height]
    bbox_cv2 = [left, top, right, bottom]
    img_dim: dimension of the image, img_dim[0]<-> x
    img_dim[1]<-> y
    '''
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])

    return (left, top, right, bottom)


def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    绘制边界框和标签
    bbox_cv2 = [left, top, right, bottom]
    '''
    # box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.9
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # 绘制边界框
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)

    if show_label:
        # 在边界框顶部绘制一个填充框
        cv2.rectangle(img, (left - 2, top - 45), (right + 2, top), box_color, -1, 1)
        # 输出显示边界框中心的坐标
        # if((left+right)/2<700):
        #     text_x = 'x=' + str(700-(left + right) / 2 )
        # else:
        #     text_x= 'x='+str((left+right)/2-700)

        text_y = 'y=' + str(700 - (top + bottom) / 2)
        x = (left + right) / 2 - 700
        x1 = (left + right) / 2
        y = (top + bottom) / 2
        a = 0.00
        #计算车辆的距离
        a = math.sqrt(x * x + y * y)
        #判断车辆的位置
        if (x1 < 500):
            str1 = "左前方"
        elif (x1 > 800):
            str1 = "右前方"
        else:
            str1 = "正前方"
        a = a * 0.03
        b = int(a)
        print("距离" + str(b))
        text_x = str(b) + "m"
        if b < 15:
            str3 = "注意前方车辆！"
        else:
            str3 = ""
        cv2.putText(img, text_x, (left, top), font, font_size, font_color, 1, cv2.LINE_AA)
        str2 = str1 + str(b) + "米处有车辆"
    # print(str2)
    else:
        str2 = " "

    return img, str2, str3
