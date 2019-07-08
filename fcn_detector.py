#-*-coding:utf8-*-

import tensorflow as tf
import sys
sys.path.append("./")


class FcnDetector(object):
    # net_factory: P_Net
    # model_path: 模型路径
    def __init__(self, net_factory, model_path):
        # create a graph
        graph = tf.Graph()
        with graph.as_default():
            # define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            # 预测脸框坐标和准确率
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True
                                                         , gpu_options=tf.GPUOptions(allow_growth=True)))
            # 读取模型
            saver = tf.train.Saver()
            # 检查模型是否合法
            model_dict = '/'.join(model_path.split('/')[:-1])
            print(model_dict)
            ckpt = tf.train.get_checkpoint_state(model_dict)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert readstate, "the params dictionary is not valid"
            saver.restore(self.sess, model_path)

    def predict(self, databatch):
        height, width, _ = databatch.shape
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred
