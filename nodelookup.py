import tensorflow.compat.v1 as tf  # 兼容性

tf.disable_v2_behavior


class NodeLookup(object):
    def __init__(self):
        labels_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uids_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self._node_lookup = self._load(labels_path, uids_path)

    @staticmethod
    def _load(labels_path, uids_path):
        uid_to_human = {}
        for line in tf.gfile.GFile(uids_path).readlines():
            items = line.strip('\n').split('\t')
            uid_to_human[items[0]] = items[1]
        node_id_to_uid = {}
        for line in tf.gfile.GFile(labels_path).readlines():
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            name = uid_to_human[val]
            node_id_to_name[key] = name
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self._node_lookup:
            return ''
        return self._node_lookup[node_id]
