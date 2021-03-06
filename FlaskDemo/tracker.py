#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
实施和测试跟踪器
'''
import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag


class Tracker():  # 基于卡尔曼滤波的跟踪器
    def __init__(self):
        # 初始化跟踪程序的参数
        self.id = 0  # tracker's id
        self.box = []  # 存储边界框坐标的列表
        self.hits = 0  # 检测匹配数
        self.no_losses = 0  # 不匹配的曲目数

        # 初始化卡尔曼滤波参数，卡尔曼滤波器的实现，状态向量有以下八个元素
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state = []
        self.dt = 1.  # time interval

        #  过程矩阵，假设为等速模型
        # 也就是说，我们使用边界框左上角和右下角的坐标及其一阶导数。假设速度恒定（因此没有加速度），过程矩阵为：
        self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        #  测量矩阵，假设我们只能测量坐标

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])

        # 初始化状态协方差
        self.L = 10.0
        self.P = np.diag(self.L * np.ones(8))

        # 初始化进程协方差
        self.Q_comp_mat = np.array([[self.dt ** 4 / 4., self.dt ** 3 / 2.],
                                    [self.dt ** 3 / 2., self.dt ** 2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        # 初始化测量协方差
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)

    def update_R(self):
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)

    def kalman_filter(self, z):
        '''
        利用测量z实现卡尔曼滤波，包括预测和更新阶段。
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S))  # Kalman gain
        y = z - dot(self.H, x)  # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int)  # 转换为整数坐标
        # (pixel values)

    def predict_only(self):
        ''' 只实现预测阶段。用于不匹配的检测和不匹配的轨道
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    import helpers

    # 创建实例
    trk = Tracker()
    # Test R_ratio
    trk.R_scaler = 1.0 / 16
    # 更新测量噪声协方差矩阵
    trk.update_R()
    # Initial state
    x_init = np.array([390, 0, 1050, 0, 513, 0, 1278, 0])
    x_init_box = [x_init[0], x_init[2], x_init[4], x_init[6]]
    # Measurement
    z = np.array([399, 1022, 504, 1256])
    trk.x_state = x_init.T
    trk.kalman_filter(z.T)
    # Updated state
    x_update = trk.x_state
    x_updated_box = [x_update[0], x_update[2], x_update[4], x_update[6]]

    print('The initial state is: ', x_init)
    print('The measurement is: ', z)
    print('The update state is: ', x_update)

    # 可视化卡尔曼滤波过程及测量

    images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
    img = images[3]

    plt.figure(figsize=(10, 14))
    helpers.draw_box_label(img, x_init_box, box_color=(0, 255, 0))
    ax = plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.title('Initial: ' + str(x_init_box))

    helpers.draw_box_label(img, z, box_color=(255, 0, 0))
    ax = plt.subplot(3, 1, 2)
    plt.imshow(img)
    plt.title('Measurement: ' + str(z))

    helpers.draw_box_label(img, x_updated_box)
    ax = plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.title('Updated: ' + str(x_updated_box))
    plt.show()    
