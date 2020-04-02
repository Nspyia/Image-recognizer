import tensorflow.compat.v1 as tf

tf.disable_v2_behavior
import numpy as np
import nodelookup


class TensorflowPredictor():
    def __init__(self):
        self.sess = tf.Session()
        with tf.gfile.FastGFile('./inception_model/classify_image_graph_def.pb', 'rb') as f:
            graph_def = tf.GraphDef()  # 定义一个计算图
            graph_def.ParseFromString(f.read())  #
            tf.import_graph_def(graph_def, name='')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')

    def predict_image(self, image_path):
        # 载入图片
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        predictions = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
        predictions = np.squeeze(predictions)  # 把结果转为1维
        # 打印图片路径及名称
        res_str = ''
        res_str += '图片路径: ...' + image_path[int(len(image_path)/2):] + '\n'
        # 排序
        top_k = predictions.argsort()[-5:][::-1]
        node_lookup = nodelookup.NodeLookup()
        for node_id in top_k:
            # 获取分类名称
            name_str = node_lookup.id_to_string(node_id)
            # 获取该分类的置信度
            score = predictions[node_id] * 100
            res_str += '(%.2f' % (score) + '%), ' + name_str + '\n'
        return res_str
