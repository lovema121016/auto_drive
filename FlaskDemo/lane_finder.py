"""Example Google style docstrings.

This module is used to define two classes for advanced lane finding project. First one is LaneFinder, a class used to
find the whole lane on the road.

"""
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import numpy as np
import settings
import math

from PIL import Image, ImageDraw, ImageFont
from settings import CALIB_FILE_NAME, PERSPECTIVE_FILE_NAME


def get_center_shift(coeffs, img_size, pixels_per_meter):
    return np.polyval(coeffs, img_size[1] / pixels_per_meter[1]) - (img_size[0] // 2) / pixels_per_meter[0]


def get_curvature(coeffs, img_size, pixels_per_meter):
    return ((1 + (2 * coeffs[0] * img_size[1] / pixels_per_meter[1] + coeffs[1]) ** 2) ** 1.5) / np.absolute(
        2 * coeffs[0])


class LaneLineFinder:
    def __init__(self, img_size, pixels_per_meter, center_shift):
        self.found = False
        self.poly_coeffs = np.zeros(3, dtype=np.float32)
        self.coeff_history = np.zeros((3, 7), dtype=np.float32)
        self.img_size = img_size
        self.pixels_per_meter = pixels_per_meter
        self.line_mask = np.ones((img_size[1], img_size[0]), dtype=np.uint8)
        self.other_line_mask = np.zeros_like(self.line_mask)
        self.line = np.zeros_like(self.line_mask)
        self.num_lost = 0
        self.still_to_find = 1
        self.shift = center_shift
        self.first = True
        self.stddev = 0

    def reset_lane_line(self):
        self.found = False
        self.poly_coeffs = np.zeros(3, dtype=np.float32)
        self.line_mask[:] = 1
        self.first = True

    def one_lost(self):
        self.still_to_find = 5
        if self.found:
            self.num_lost += 1
            if self.num_lost >= 7:
                self.reset_lane_line()

    def one_found(self):
        self.first = False
        self.num_lost = 0
        if not self.found:
            self.still_to_find -= 1
            if self.still_to_find <= 0:
                self.found = True

    def fit_lane_line(self, mask):
        y_coord, x_coord = np.where(mask)
        y_coord = y_coord.astype(np.float32) / self.pixels_per_meter[1]
        x_coord = x_coord.astype(np.float32) / self.pixels_per_meter[0]
        if len(y_coord) <= 150:
            coeffs = np.array([0, 0, (self.img_size[0] // 2) / self.pixels_per_meter[0] + self.shift], dtype=np.float32)
        else:
            coeffs, v = np.polyfit(y_coord, x_coord, 2, rcond=1e-16, cov=True)
            self.stddev = 1 - math.exp(-5 * np.sqrt(np.trace(v)))

        self.coeff_history = np.roll(self.coeff_history, 1)

        if self.first:
            self.coeff_history = np.reshape(np.repeat(coeffs, 7), (3, 7))
        else:
            self.coeff_history[:, 0] = coeffs

        value_x = get_center_shift(coeffs, self.img_size, self.pixels_per_meter)
        curve = get_curvature(coeffs, self.img_size, self.pixels_per_meter)

        if (self.stddev > 0.95) | (len(y_coord) < 150) | (math.fabs(value_x - self.shift) > math.fabs(0.5 * self.shift)) \
                | (curve < 30):

            self.coeff_history[0:2, 0] = 0
            self.coeff_history[2, 0] = (self.img_size[0] // 2) / self.pixels_per_meter[0] + self.shift
            self.one_lost()
        else:
            self.one_found()

        self.poly_coeffs = np.mean(self.coeff_history, axis=1)

    def get_line_points(self):
        y = np.array(range(0, self.img_size[1] + 1, 10), dtype=np.float32) / self.pixels_per_meter[1]
        x = np.polyval(self.poly_coeffs, y) * self.pixels_per_meter[0]
        y *= self.pixels_per_meter[1]
        return np.array([x, y], dtype=np.int32).T

    def get_other_line_points(self):
        pts = self.get_line_points()
        pts[:, 0] = pts[:, 0] - 2 * self.shift * self.pixels_per_meter[0]
        return pts

    def find_lane_line(self, mask, reset=False):  # 查找路的边界

        n_segments = 16
        window_width = 30
        step = self.img_size[1] // n_segments

        if reset or (not self.found and self.still_to_find == 5) or self.first:
            self.line_mask[:] = 0
            n_steps = 4
            window_start = self.img_size[0] // 2 + int(self.shift * self.pixels_per_meter[0]) - 3 * window_width
            window_end = window_start + 6 * window_width
            sm = np.sum(mask[self.img_size[1] - 4 * step:self.img_size[1], window_start:window_end], axis=0)
            sm = np.convolve(sm, np.ones((window_width,)) / window_width, mode='same')
            argmax = window_start + np.argmax(sm)
            shift = 0
            for last in range(self.img_size[1], 0, -step):
                first_line = max(0, last - n_steps * step)
                sm = np.sum(mask[first_line:last, :], axis=0)
                sm = np.convolve(sm, np.ones((window_width,)) / window_width, mode='same')
                window_start = min(max(argmax + int(shift) - window_width // 2, 0), self.img_size[0] - 1)
                window_end = min(max(argmax + int(shift) + window_width // 2, 0 + 1), self.img_size[0])
                new_argmax = window_start + np.argmax(sm[window_start:window_end])
                new_max = np.max(sm[window_start:window_end])
                if new_max <= 2:
                    new_argmax = argmax + int(shift)
                    shift = shift / 2
                if last != self.img_size[1]:
                    shift = shift * 0.25 + 0.75 * (new_argmax - argmax)
                argmax = new_argmax
                cv2.rectangle(self.line_mask, (argmax - window_width // 2, last - step),
                              (argmax + window_width // 2, last),
                              1, thickness=-1)
        else:
            self.line_mask[:] = 0
            points = self.get_line_points()
            if not self.found:
                factor = 3
            else:
                factor = 2
            cv2.polylines(self.line_mask, [points], 0, 1, thickness=int(factor * window_width))

        self.line = self.line_mask * mask
        self.fit_lane_line(self.line)
        self.first = False
        if not self.found:
            self.line_mask[:] = 1
        points = self.get_other_line_points()
        self.other_line_mask[:] = 0
        cv2.polylines(self.other_line_mask, [points], 0, 1, thickness=int(5 * window_width))


# class that finds the whole lane
class LaneFinder:
    def __init__(self, img_size, warped_size, cam_matrix, dist_coeffs, transform_matrix, pixels_per_meter,
                 warning_icon):
        self.found = False
        self.dist_coeffs = dist_coeffs
        self.cam_matrix = cam_matrix
        self.img_size = img_size
        self.warped_size = warped_size
        self.mask = np.zeros((warped_size[1], warped_size[0], 3), dtype=np.uint8)
        self.roi_mask = np.ones((warped_size[1], warped_size[0], 3), dtype=np.uint8)
        self.total_mask = np.zeros_like(self.roi_mask)
        self.warped_mask = np.zeros((self.warped_size[1], self.warped_size[0]), dtype=np.uint8)
        self.M = transform_matrix
        self.count = 0
        self.left_line = LaneLineFinder(warped_size, pixels_per_meter, -1.8288)  # 6 feet in meters
        self.right_line = LaneLineFinder(warped_size, pixels_per_meter, 1.8288)
        if (warning_icon is not None):
            self.warning_icon = np.array(mpimg.imread(warning_icon) * 255, dtype=np.uint8)
        else:
            self.warning_icon = None

    def undistort(self, img):  # 中心点？不变形，只用直接调用函数就可以得到去畸变的图像
        return cv2.undistort(img, self.cam_matrix, self.dist_coeffs)

    def warp(self, img):  ###################### 对图像进行变换（四点得到一个变换矩阵）# 进行透视变换
        # 可以先用四个点来确定一个3*3的变换矩阵（cv2.getPerspectiveTransform）
        # 然后通过cv2.warpPerspective和上述矩阵对图像进行变换
        return cv2.warpPerspective(img, self.M, self.warped_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.WARP_FILL_OUTLIERS +
                                                                     cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    def equalize_lines(self, alpha=0.9):
        mean = 0.5 * (self.left_line.coeff_history[:, 0] + self.right_line.coeff_history[:, 0])
        self.left_line.coeff_history[:, 0] = alpha * self.left_line.coeff_history[:, 0] + \
                                             (1 - alpha) * (mean - np.array([0, 0, 1.8288], dtype=np.uint8))
        self.right_line.coeff_history[:, 0] = alpha * self.right_line.coeff_history[:, 0] + \
                                              (1 - alpha) * (mean + np.array([0, 0, 1.8288], dtype=np.uint8))

    def find_lane(self, img, distorted=True, reset=False):  # 寻找线的位置
        # undistort, warp, change space, filter
        if distorted:
            img = self.undistort(img)#直接调用函数就可以得到去畸变的图像
        if reset:
            self.left_line.reset_lane_line()
            self.right_line.reset_lane_line()

        img = self.warp(img)
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)# 将彩色图转换为灰度图
        img_hls = cv2.medianBlur(img_hls, 5)#中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术，对脉冲噪声有良好的滤除作用，特别是在滤除噪声的同时，能够保护信号的边缘，使之不被模糊
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab = cv2.medianBlur(img_lab, 5)

        big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))#使用getStructuringElement 定义一个结构元素。
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))#使用getStructuringElement 定义一个结构元素。

        greenery = (img_lab[:, :, 2].astype(np.uint8) > 130) & cv2.inRange(img_hls, (0, 0, 50), (35, 190, 255))

        road_mask = np.logical_not(greenery).astype(np.uint8) & (img_hls[:, :, 1] < 250)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, small_kernel)# 进行各类形态学的变化参数说明:src传入的图片，op进行变化的方式， kernel表示方框的大小
        road_mask = cv2.dilate(road_mask, big_kernel)#参数说明: src表示输入的图片， kernel表示方框的大小， iteration表示迭代的次数膨胀操作原理：存在一个kernel，在图像上进行从左到右，从上到下的平移，如果方框中存在白色，那么这个方框内所有的颜色都是白色
        # cv2.MORPH_OPEN 进行开运算，指的是先进行腐蚀操作，再进行膨胀操作（把图像具体的想要位置清楚化)
        img2, contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # 使用cv2.dilate进行膨胀操作
        biggest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)#计算轮廓面积contourArea
            if area > biggest_area:
                biggest_area = area
                biggest_contour = contour
        road_mask = np.zeros_like(road_mask)
        cv2.fillPoly(road_mask, [biggest_contour], 1)#cv2.fillPoly()函数可以用来填充任意形状的图型.可以用来绘制多边形,填充车道线的图像

        self.roi_mask[:, :, 0] = (self.left_line.line_mask | self.right_line.line_mask) & road_mask
        self.roi_mask[:, :, 1] = self.roi_mask[:, :, 0]
        self.roi_mask[:, :, 2] = self.roi_mask[:, :, 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))#使用getStructuringElement 定义一个结构元素。
        black = cv2.morphologyEx(img_lab[:, :, 0], cv2.MORPH_TOPHAT, kernel)##腐蚀后的图像
        lanes = cv2.morphologyEx(img_hls[:, :, 1], cv2.MORPH_TOPHAT, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        lanes_yellow = cv2.morphologyEx(img_lab[:, :, 2], cv2.MORPH_TOPHAT, kernel)

        self.mask[:, :, 0] = cv2.adaptiveThreshold(black, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -6)#自定义阀值
        self.mask[:, :, 1] = cv2.adaptiveThreshold(lanes, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
        self.mask[:, :, 2] = cv2.adaptiveThreshold(lanes_yellow, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                   13, -1.5)
        self.mask *= self.roi_mask
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.total_mask = np.any(self.mask, axis=2).astype(np.uint8)
        self.total_mask = cv2.morphologyEx(self.total_mask.astype(np.uint8), cv2.MORPH_ERODE, small_kernel)

        left_mask = np.copy(self.total_mask)
        right_mask = np.copy(self.total_mask)
        if self.right_line.found:
            left_mask = left_mask & np.logical_not(self.right_line.line_mask) & self.right_line.other_line_mask
        if self.left_line.found:
            right_mask = right_mask & np.logical_not(self.left_line.line_mask) & self.left_line.other_line_mask
        self.left_line.find_lane_line(left_mask, reset)
        self.right_line.find_lane_line(right_mask, reset)
        self.found = self.left_line.found and self.right_line.found

        if self.found:
            self.equalize_lines(0.875)#平衡线度
        #print("AAAAAAAAAAAA")

    def draw_lane_weighted(self, img, thickness=5, alpha=0.8, beta=1, gamma=0):

        left_line = self.left_line.get_line_points()
        right_line = self.right_line.get_line_points()

        both_lines = np.concatenate((left_line, np.flipud(right_line)), axis=0)
        lanes = np.zeros((self.warped_size[1], self.warped_size[0], 3), dtype=np.uint8)

        if self.found:
            cv2.fillPoly(lanes, [both_lines.astype(np.int32)], (0, 255, 0))
            cv2.polylines(lanes, [left_line.astype(np.int32)], False, (255, 0, 0), thickness=thickness)
            cv2.polylines(lanes, [right_line.astype(np.int32)], False, (0, 0, 255), thickness=thickness)
            cv2.fillPoly(lanes, [both_lines.astype(np.int32)], (0, 255, 0))
            mid_coef = 0.5 * (self.left_line.poly_coeffs + self.right_line.poly_coeffs)
            curve = get_curvature(mid_coef, img_size=self.warped_size, pixels_per_meter=self.left_line.pixels_per_meter)#计算路的曲率
            shift = get_center_shift(mid_coef, img_size=self.warped_size,
                                     pixels_per_meter=self.left_line.pixels_per_meter)#计算本车的位置

            cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同

            pilimg = Image.fromarray(cv2img)
            #print("CCCCCCCCCCCC")
            # PIL图片上打印汉字
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            font = ImageFont.truetype("simhei.ttf", 50, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小

            draw.text((420, 50), "路的曲率: {:6.2f}m".format(curve), (0, 255, 0),
                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

            draw.text((420, 100), "车的位置: {:6.2f}m".format(shift), (0, 255, 0),
                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
            # PIL图片转cv2 图片
            img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            str=" "
            #print("未偏离车道！")
        else:
            warning_shape = self.warning_icon.shape
            corner = (10, (img.shape[1] - warning_shape[1]) // 2)
            patch = img[corner[0]:corner[0] + warning_shape[0], corner[1]:corner[1] + warning_shape[1]]
            patch[self.warning_icon[:, :, 3] > 0] = self.warning_icon[self.warning_icon[:, :, 3] > 0, 0:3]
            img[corner[0]:corner[0] + warning_shape[0], corner[1]:corner[1] + warning_shape[1]] = patch
            # cv2.putText(img, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
            #             thickness=5, color=(255, 255, 255))
            # cv2.putText(img, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
            #             thickness=3, color=(0, 0, 0))
            cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同

            pilimg = Image.fromarray(cv2img)

            # PIL图片上打印汉字
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            font = ImageFont.truetype("simhei.ttf", 50, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
            draw.text((550, 170), "偏离道路！", (255, 255, 255),
                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
            draw.text((550, 170), "偏离道路！", (0, 255, 0),
                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
            # PIL图片转cv2 图片
            img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            str="注意您已偏离车道！"
            #print("注意您已偏离车道！")
        lanes_unwarped = self.unwarp(lanes)
        return cv2.addWeighted(img, alpha, lanes_unwarped, beta, gamma),str
        # try:
        #     cv2.addWeighted(img, alpha, lanes_unwarped, beta, gamma)
        # except Exception as e:
        #     print("000000000000000000" + str(e))
        #     return img
        #     # sys.exit(0)
        # else:
        #     return cv2.addWeighted(img, alpha, lanes_unwarped, beta, gamma)

    def process_image(self, img, lf, reset=False, show_period=10, blocking=False):
      try:
        self.find_lane(img, reset=reset)
        lane_img,str= self.draw_lane_weighted(img)
        self.count += 1
        if show_period > 0 and (self.count % show_period == 1 or show_period == 1):
            start = 231
            plt.clf()
            for i in range(3):
                plt.subplot(start + i)
                # plt.imshow(lf.mask[:, :, i] * 255, cmap='gray')
                plt.subplot(234)
            # plt.imshow((lf.left_line.line + lf.right_line.line) * 255)

            ll = cv2.merge((lf.left_line.line, lf.left_line.line * 0, lf.right_line.line))
            lm = cv2.merge((lf.left_line.line_mask, lf.left_line.line * 0, lf.right_line.line_mask))
            plt.subplot(235)
            # plt.imshow(lf.roi_mask * 255, cmap='gray')
            plt.subplot(236)
            # plt.imshow(lane_img)
            if blocking:
                str="0"
                # plt.show()
            else:
                # plt.draw()
                plt.pause(0.000001)

            return lane_img, str
      except Exception as e:
            str5 = " "
            return img, str5

def run(image):
 with open(CALIB_FILE_NAME, 'rb') as f:
     calib_data = pickle.load(f)
 cam_matrix = calib_data["cam_matrix"]
 dist_coeffs = calib_data["dist_coeffs"]
 img_size = calib_data["img_size"]

 with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
     perspective_data = pickle.load(f)

 perspective_transform = perspective_data["perspective_transform"]
 pixels_per_meter = perspective_data['pixels_per_meter']
 orig_points = perspective_data["orig_points"]
 # 读取图片地址
 lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, cam_matrix, dist_coeffs,
                        perspective_transform, pixels_per_meter, "warning.png")
 im2 = cv2.resize(image, (1280, 720), )
 img,str = lf.process_image(im2, lf, reset=False, show_period=20)
 # plt.imshow(img)
 print(str)
 return img,str

# img=cv2.imread("./static/shishi/a1.png")
# plt.imshow(img)
    # video_files = ['project_video1.mp4']
    # output_path = "output_videos"
    # for file in video_files:
    #     lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, cam_matrix, dist_coeffs,
    #                 perspective_transform, pixels_per_meter, "warning.png")
    #     output = os.path.join(output_path,"lane_"+file)
    #     clip2 = VideoFileClip(file)
    #     challenge_clip = clip2.fl_image(lambda x: lf.process_image(x, reset=False, show_period=20))
    #     challenge_clip.write_videofile(output, audio=False)
