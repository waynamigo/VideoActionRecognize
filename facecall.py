#-*-coding:utf8-*-
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys

from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from mtcnn_model import P_Net,R_Net,O_Net
from loader import TestLoader
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import facenet
import requests
# 筛选脸框的threshhold
thresh = [0.6, 0.7, 0.7]
# 最小人脸
min_face_size = 20
detectors = [None, None, None]
# 存放MTCNN的model的路径
model_path = ['./model/PNet/PNet-18', './model/RNet/RNet-14', './model/ONet/ONet-16']
# 存放脸库的目录
face_lib_path = "./face_lib"
# 存放待检测的图片路径
test_image_path = "./test_image"
# 存放facenet的model路径
facenet_model = './model/facenet_model'
epoch = [18, 14, 16]
batch_size = [2048, 64, 16]

face_lib_num = 0
# 先读取脸库中的图片，再读取待检测的图片
image_files = []
for item in os.listdir(face_lib_path):
    image_files.append(os.path.join(face_lib_path, item))
    face_lib_num += 1
for item in os.listdir(test_image_path):
    image_files.append(os.path.join(test_image_path, item))


def face_recognition(coordinate):
    images, face_state = load_and_align_data(coordinate=coordinate)
    pic_num, h, w, c = images.shape

    # 输出图片中所有人脸
    # num, h, w, c = images.shape
    # for i in range(num):
    #     misc.imshow(images[i])

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # 加载模型
            facenet.load_model(facenet_model)

            # 获得input和output的tensor
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 前向计算embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = pic_num

            # 图片距离矩阵
            matrix = np.zeros(shape=(nrof_images, nrof_images))

            # print('Images:')
            # for i in range(nrof_images):
            #     print('%1d: %s' % (i, image_files[i]))
            # print('')
            #
            # # Print distance matrix
            # print('Distance matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            for i in range(nrof_images):
                # print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    matrix[i][j] = dist
                    # print('  %1.4f  ' % dist, end='')
                # print('')
            face_record = []
            for j in range(pic_num - face_lib_num):
                min_index = 0
                index = nrof_images + j - (pic_num - face_lib_num)
                for i in range(face_lib_num):
                    if matrix[i][index] < matrix[min_index][index]:
                        min_index = i
                face_record.append(image_files[min_index])
            # for i in range(nrof_images):
            #     # print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         print('  %1.4f  ' % matrix[i][j], end='')
            #     print('')
            for i in range(len(face_state)):
                if face_state[i]:
                    # print('facerecord',face_record[i])
                    return face_record[i].split('.')[1].split('/')[2]


def load_and_align_data(image_size=160, coordinate=None):
    print('Creating networks and loading parameters')

    # 加载P_Net
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # 加载R_Net
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

    # 加载O_Net
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, threshold=thresh)

    img_list = []

    test_data = TestLoader(image_files)

    all_boxes, landmarks = mtcnn_detector.detect_face(test_data)

    action_num = int(len(coordinate) / 4)
    face_state = []
    action_state = []
    for i in range(len(all_boxes[len(all_boxes) - 1])):
        face_state.append(False)
    for j in range(action_num):
        action_state.append(False)
    for i in range(len(all_boxes[len(all_boxes) - 1])):
        for j in range(action_num):
            if action_num == 1:
                if action_state:
                    continue
                if all_boxes[len(all_boxes) - 1][i][0] > coordinate[4 * j] and all_boxes[len(all_boxes) - 1][i][1] > coordinate[4 * j + 1]\
                        and all_boxes[len(all_boxes) - 1][i][2] < coordinate[4 * j + 2] and all_boxes[len(all_boxes) - 1][i][3] < coordinate[4 * j + 3]:
                    face_state[i] = True
                    action_state = True
            else:
                if action_state[j]:
                    continue
                if all_boxes[len(all_boxes) - 1][i][0] > coordinate[4 * j] and all_boxes[len(all_boxes) - 1][i][1] > coordinate[4 * j + 1]\
                        and all_boxes[len(all_boxes) - 1][i][2] < coordinate[4 * j + 2] and all_boxes[len(all_boxes) - 1][i][3] < coordinate[4 * j + 3]:
                    face_state[i] = True
                    action_state[j] = True

    count = 0
    for image in image_files:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        for bbox in all_boxes[count]:
            # 提取脸框
            cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            # 人脸摆正
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            # 人脸白化
            prewhitened = facenet.prewhiten(aligned)

            img_list.append(prewhitened)
        count += 1
    images = np.stack(img_list)
    return images, face_state
if __name__ == '__main__':
    coordinate = [80, 20, 300, 270, 80, 50, 600, 500]
    face_recognition(coordinate)
