# -*- coding: utf-8 -*-
import os
import numpy as np
from keras.preprocessing import image


def load_data(path, img_height, img_width):
       img_names = os.listdir(os.path.join(path))
       img_names = np.array(img_names)
       img = []
       # 把图片读取出来放到列表中
       for i in range(25):
           images = image.load_img(os.path.join(path, img_names[i]), target_size=(img_height, img_width))
           x = image.img_to_array(images)
           x = np.expand_dims(x, axis=0)
           img.append(x)
       
       # 把图片数组联合在一起
       x = np.concatenate([x for x in img])
       # 将像素值缩放到-1~1之间
       x = x / 127.5 - 1.
       print(x.shape)
       return x

if __name__ == "__main__":
    load_data('/home/frank/baidunetdiskdownload/faces', 96, 96)