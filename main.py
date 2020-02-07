# -*- coding:utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.datasets import mnist
from keras.layers import Input,Dense,Reshape,Flatten,Dropout
from keras.layers import BatchNormalization,Activation,ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D,Conv2D
from keras.models import Sequential,Model
from keras.optimizers import Adam,RMSprop
import cv2
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
plt.switch_backend('agg')
import sys
import numpy as np

import tensorflow as tf
from keras.callbacks import TensorBoard
import time



class DCGAN():
   def __init__(self):
       self.callback = TensorBoard('mytensorboard')
       self.img_rows = 96
       self.img_cols = 96
       self.channels = 3
       self.img_shape=(self.img_rows,self.img_cols,self.channels)
       self.latent_dim =100

       optimizer = Adam(0.0002,0.5)
       optimizerD =RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
       optimizerG = RMSprop(lr=0.0004, clipvalue=1.0, decay=6e-8)
       print("-1")

       #对判别器进行构建和编译
       self.discriminator = self.build_discriminator()
       print("0")
       self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
       print("1")
       #对生成器进行构造
       self.generator = self.build_generator()
       print("2")
       # The generator takes noise as input and generates imgs
       z = Input(shape=(self.latent_dim,))
       img = self.generator(z)

       # 总体模型只对生成器进行训练
       self.discriminator.trainable = False

       # 从生成器中生成的图 经过判别器获得一个valid
       valid = self.discriminator(img)
       self.combined = Model(z,valid)
       self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
       # self.callback.set_model(self.combined)

   def build_generator(self):
       model = Sequential()
       model.add(Dense(512*6*6,activation='relu',input_dim=self.latent_dim))  #输入维度为100，输出128*7*7
       model.add(Reshape((6,6,512)))
       model.add(UpSampling2D())  # 进行上采样，变成14*14*128
       model.add(Conv2D(256,kernel_size=5,padding='same'))
       model.add(BatchNormalization(momentum=0.8))
       model.add(Activation("relu"))
       model.add(UpSampling2D())
       model.add(Conv2D(128, kernel_size=5, padding="same"))
       model.add(BatchNormalization(momentum=0.8))
       model.add(Activation("relu"))
       model.add(UpSampling2D())
       model.add(Conv2D(64, kernel_size=5, padding="same"))
       model.add(BatchNormalization(momentum=0.8))
       model.add(Activation("relu"))
       model.add(UpSampling2D())
       model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
       model.add(Activation("tanh"))
       noise = Input(shape=(self.latent_dim,))
       img = model(noise)
       return  Model(noise,img)  #定义一个 一个输入noise一个输出img的模型

   def build_discriminator(self):
       dropout = 0.25
       depth = 32
       model = Sequential()
       
       model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
       model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
       model.add(BatchNormalization(momentum=0.8))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
       model.add(BatchNormalization(momentum=0.8))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Conv2D(512, kernel_size=5, strides=1, padding="same"))
       model.add(BatchNormalization(momentum=0.8))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Flatten())
       model.add(Dense(1, activation='sigmoid'))
       img = Input(shape=self.img_shape)
       validity = model(img)

       return Model(img,validity)

   def train(self,epochs,batch_size=128,count=100,save_interval = 50):
       # valid 真实图片标签 fake生成图片标签
       valid = np.ones((batch_size, 1))
       fake = np.zeros((batch_size, 1))
       imgs = self.load_batch_imgs(batch_size,'/home/frank/baidunetdiskdownload/faces')
       for epoch in range(epochs):
           s = time.time()

           # ---------------------
           #  判别器训练
           # ---------------------
        #    if epoch % count == 0:
            #    imgs = self.load_batch_imgs(batch_size,'/home/frank/baidunetdiskdownload/faces')

           # Sample noise and generate a batch of new images
           noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
           gen_imgs = self.generator.predict(noise)

           # Train the discriminator (real classified as ones and generated as zeros)
           d_loss_real = self.discriminator.train_on_batch(imgs, valid)
           d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
           d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

           # ---------------------
           #  生成器训练
           # ---------------------

           # 记录训练的日志
           # logs = g_loss = self.combined.train_on_batch(noise, valid)
           g_loss = self.combined.train_on_batch(noise, valid)
           # train_names = ['g_loss', 'train_mae']
           # self.write_log(self.callback,logs,epoch)

           e = time.time()
           # Plot the progress
           print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] time: %f" % (epoch, d_loss[0], 100*d_loss[1], g_loss, (e-s)))

           # 每隔save_interval保存头像图片
           if epoch % save_interval == 0:
               self.combined.save('./model/combined_model_%d.h5'%epoch)
               self.discriminator.save('./model/discriminator_model_%d.h5'%epoch )
               self.save_imgs(epoch)


   def load_batch_imgs(self,batch_size,dirName):
       img_names = os.listdir(os.path.join(dirName))
       img_names = np.array(img_names)
       # 随机取一个batch的图片数据
       idx = np.random.randint(0, img_names.shape[0], batch_size)
       img_names = img_names[idx]
       img = []
       # 把图片读取出来放到列表中
       for i in range(len(img_names)):
           images = image.load_img(os.path.join(dirName, img_names[i]), target_size=(96, 96))
           x = image.img_to_array(images)
           x = np.expand_dims(x, axis=0)
           img.append(x)
       
       # 把图片数组联合在一起
       x = np.concatenate([x for x in img])
       # 将像素值缩放到-1~1之间
       x = x / 127.5 - 1.
       return x


   def save_imgs(self, epoch):
       r, c = 5, 5
       noise = np.random.normal(0, 1, (r * c, self.latent_dim))  #高斯分布，均值0，标准差1，size= (5*5, 100)
       gen_imgs = self.generator.predict(noise)
       gen_imgs = 0.5 * gen_imgs + 0.5

       fig, axs = plt.subplots(r, c)
       cnt = 0   #生成的25张图 显示出来
       for i in range(r):
           for j in range(c):
               axs[i,j].imshow(gen_imgs[cnt, :,:,:])
               axs[i,j].axis('off')
               cnt += 1
       fig.savefig("images/mnist_%d.png" % epoch)
       plt.close()
   def loadModel(self):
       self.combined = load_model('./model/combined_model_last.h5')
       self.discriminator = load_model('./model/discriminator_model_last.h5')
if __name__ == '__main__':
    dcgan = DCGAN()
    start = time.time()
    dcgan.train(epochs=10000, batch_size=64, count=100,save_interval=10)
    end = time.time()
    print('cost: ', (end-start))

# 135776s