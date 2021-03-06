

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))


class CarDetector(object):
    def __init__(self):

        self.car_boxes = []

        os.chdir(cwd)

        #Tensorflow localization/detection model
        #  基于COCO数据集的移动网络结构单镜头检测
        detect_model_name = 'ssd_mobilenet_v1_coco_11_06_2017'

        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'


        self.detection_graph = tf.Graph()

        #  可能使用GPU的配置
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #加载frozen tensorflow检测模型并初始化
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # 每个框表示图像中检测到特定对象的部分
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # 每个分数表示每个对象
              # 分数和类标签一起显示在结果图像上
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')

    #  用于将图像转换为numpy数组的helper函数
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
    #用于将标准化方框坐标转换为像素的辅助函数
    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def get_localization(self, image, visual=False):

        """确定图像中汽车的位置

        Args:
            image: camera image

        返回:边界框列表 : coordinates [y_up, x_left, y_down, x_right]
        检测器在detector.py中的cardetector类中实现。输出是协调所有检测到的车辆的边界框（格式为[Y轴向上，X轴向左，Y轴向下，X轴向右]）。

        """
        category_index={1: {'id': 1, 'name': u'person'},
                        2: {'id': 2, 'name': u'bicycle'},
                        3: {'id': 3, 'name': u'car'},
                        4: {'id': 4, 'name': u'motorcycle'},
                        5: {'id': 5, 'name': u'airplane'},
                        6: {'id': 6, 'name': u'bus'},
                        7: {'id': 7, 'name': u'train'},
                        8: {'id': 8, 'name': u'truck'},
                        9: {'id': 9, 'name': u'boat'},
                        10: {'id': 10, 'name': u'traffic light'},
                        11: {'id': 11, 'name': u'fire hydrant'},
                        13: {'id': 13, 'name': u'stop sign'},
                        14: {'id': 14, 'name': u'parking meter'}}

        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})
              #这里的boxes, scores, classes, num_detections分别表示与每个检测对应的边界框、置信度级别和类名。

              if visual == True:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      line_thickness=3)

                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()

              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)

              cls = classes.tolist()

              # The ID for car in COCO data set is 3 接下来，我们选择的检测是汽车，并且置信度大于阈值（例如，在本例中为0.3）
              idx_vec = [i for i, v in enumerate(cls) if ((v==3) and (scores[i]>0.3))]

              if len(idx_vec) ==0:
                  print('no detection!')
                  self.car_boxes = []
              else:
                  tmp_car_boxes=[]
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      box = self.box_normal_to_pixel(boxes[idx], dim)
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                      ratio = box_h/(box_w + 0.01)
                      # 为了进一步减少不正确性，我们包括边界框宽度、高度和高宽比的阈值。
                      if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                          tmp_car_boxes.append(box)
                          print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                         
                      else:
                          print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)
                          
                          
                  
                  self.car_boxes = tmp_car_boxes
             
        return self.car_boxes
        
if __name__ == '__main__':
        # Test the performance of the detector
        det =CarDetector()
        os.chdir(cwd)
        TEST_IMAGE_PATHS= glob(os.path.join('test_images/', '*.jpg'))
        
        for i, image_path in enumerate(TEST_IMAGE_PATHS[0:2]):
            print('')
            print('*************************************************')
            
            img_full = Image.open(image_path)
            img_full_np = det.load_image_into_numpy_array(img_full)
            img_full_np_copy = np.copy(img_full_np)
            start = time.time()
            b = det.get_localization(img_full_np, visual=False)
            end = time.time()
            print('Localization time: ', end-start)
#            
            
