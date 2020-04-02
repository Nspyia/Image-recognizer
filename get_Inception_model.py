import os
import tarfile
import requests


def download_inception_model():
    # inception_v3模型下载
    inception_pre_mod_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # 模型存放地址
    inception_pre_mod_dir = "inception_model"
    if not os.path.exists(inception_pre_mod_dir):
        os.makedirs(inception_pre_mod_dir)
    # 获取文件名，以及文件路径
    filename = inception_pre_mod_url.split('/')[-1]
    filepath = os.path.join(inception_pre_mod_dir, filename)
    # 下载模型
    if not os.path.exists(filepath):
        print('Downloading: ', filename)
        r = requests.get(inception_pre_mod_url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)
    print("Done: ", filename)
    # 解压文件
    tarfile.open(filepath, 'r:gz').extractall(inception_pre_mod_dir)
