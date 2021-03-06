#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@author: kyleguan
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
import time
import helpers
import detector
import tracker
import requests

# 用于视频文件功能的全局变量
frame_count = 0  # frame counter
det = detector.CarDetector()
max_age = 4  # 之前连续不匹配检测的数量
# a track is deleted

min_hits = 1  # 建立跟踪所需的连续匹配数

tracker_list = []  # list for trackers
# list for track ID
track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = True


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
 从当前的跟踪器和新检测列表中 未匹配的追踪器，不匹配的检测。
    '''

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        # trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = helpers.box_iou2(trk, det)



    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # 对于创建追踪器，我们考虑重叠小于iou thrd的任何检测，以表示存在未追踪的对象

    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def pipeline(img):
    '''
    用于检测和跟踪的功能
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    global str1
    global str555
    frame_count += 1

    img_dim = (img.shape[1], img.shape[0])
    z_box = det.get_localization(img)  # measurement
    if debug:
        print('Frame:', frame_count)

    x_box = []
    if debug:
        str4 = ""
        str5552 = ""
        stryao=""
        for i in range(len(z_box)):
            img1, str3, str5551 = helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
            str4 = str4 + "\n" + str3
            if(str5551=="注意前方车辆！"):
                stryao="注意前方车辆！"
            plt.imshow(img1)
        print("AAAAaaaaa" + str4)
        str1 = str4
        str555 = str5552
        print("WWWWWaaaaa" + str555)
        plt.show()
    if (z_box == []):
        str1 = " "
        str555 = " "
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
   #当在第一个视频帧中首次检测到车辆时，运行以下代码行分别返回匹配、不匹配和不匹配的空列表、一个元素列表和空列表。
    matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3)
    if debug:
        print('Detection: ', z_box)
        print('x_box: ', x_box)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)

    # 处理匹配的检测
    #当在第二个视频帧中再次检测到车辆时，运行以下assign_detections_to_tracker将分别返回一个元素列表、一个空列表和一个匹配、不匹配和不匹配的_dets的空列表。如下图所示，我们有一个匹配的检测，将由以下代码块处理：
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # 处理不匹配的检测
    #不匹配的检测由以下代码块处理：
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker()  # 创建新的跟踪器
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            try:
               tmp_trk.id = track_id_list.popleft()  # 为跟踪器分配一个id
               tracker_list.append(tmp_trk)
               x_box.append(xx)
            except IndexError:
                tracker_list.append(tmp_trk)
                x_box.append(xx)

    # 处理不匹配的曲目
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    # 要注释的曲目列表
    good_tracker_list = []
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            if debug:
                print('updated box: ', x_cv2)
                print()
            img, str2, str6 = helpers.draw_box_label(img, x_cv2)  # 在图像上绘制边界框

    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))

    return img, str1, stryao


def run(img):
    # 对一系列图像进行测试
    try:
        im2 = cv2.resize(img, (1280, 720), )
        result, str1, str2 = pipeline(im2)
        print(str1)
        print("提示" + str2)
        return result, str1, str2
    except Exception as e:
        str4=" "
        str5=""
        return img,str4,str5
# images = [plt.imread(file) for file in glob.glob('./test/*.jpg')]
#
# for i in range(len(images))[0:7]:
#          image = images[i]
#          im2 = cv2.resize(image, (1280, 720), )
#          image_box,str1,str2 = pipeline(im2)
#          print("BBBBBBBBBB"+str1)
#          print("DDDDDD"+str2)
#          plt.imshow(image_box)
#          plt.show()

