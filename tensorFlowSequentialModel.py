import os
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import math
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops


img = cv2.imread('./2_BreaKHis 400X/BreaKHis 400X/test/benign/SOB_B_A-14-22549AB-400-005.png',0)
print(img.shape)

x_train=[]
x_test=[]
imageSize = 128
IMAGE_SIZE = (imageSize,imageSize)
train_path='./2_BreaKHis 400X/BreaKHis 400X/train'
test_path='./2_BreaKHis 400X/BreaKHis 400X/test'

c=0 
for folder in os.listdir(train_path):
  sub_path=train_path+"/"+folder
  for img in os.listdir(sub_path):
    image_path=sub_path+"/"+img
    img_arr=cv2.imread(image_path)
    try:
      img_arr=cv2.resize(img_arr,IMAGE_SIZE)
      x_train.append(img_arr)
    except:
      c+=1
      continue
      print("Number of images skipped= ",c)

c=0
for folder in os.listdir(test_path):
  sub_path=test_path+"/"+folder
  for img in os.listdir(sub_path):
    image_path=sub_path+"/"+img
    img_arr=cv2.imread(image_path)
    try:
      img_arr=cv2.resize(img_arr,IMAGE_SIZE)
      x_test.append(img_arr)
    except:
      c+=1
      continue
      print("Number of images skipped= ",c)


x_test=np.array(x_test)
x_train=np.array(x_train)
x_train=x_train/255.0
x_test=x_test/255.0

datagen = ImageDataGenerator()
train_dataset = datagen.flow_from_directory(train_path,
class_mode = 'binary')
test_dataset = datagen.flow_from_directory(test_path,
class_mode = 'binary')
train_dataset.class_indices

y_train=train_dataset.classes
y_test=test_dataset.classes

x_train.shape,y_train.shape
x_test.shape,y_test.shape

index = 12
plt.imshow(x_train[index]) #display sample training image
plt.show()

def BreaKHisModel():
    """
    Binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            
            ## Conv2D with 32 7x7 filters and stride of 1
            
            ## BatchNormalization for axis 3
            
            ## ReLU
            
            ## Max Pooling 2D with default parameters
            
            ## Flatten layer
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
               
        tf.keras.layers.ZeroPadding2D(padding=(3,3),input_shape=(imageSize, imageSize, 3), data_format="channels_last"),  
        tf.keras.layers.Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0'),     
        tf.keras.layers.BatchNormalization(axis = 3, name = 'bn0'),  
        tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0),
        
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pool0'),
        tf.keras.layers.Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1'),
        tf.keras.layers.BatchNormalization(axis = 3, name = 'bn1'), 
        tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0), 
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pool1'),
    
        tf.keras.layers.Flatten(),
    
        tf.keras.layers.Dense(1, activation='sigmoid', name='fc'),         

        ])
    
    return model


BreaKHis_model = BreaKHisModel()
BreaKHis_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
BreaKHis_model.summary()

BreaKHis_model.fit(x_train, y_train, epochs=20, batch_size=16)

BreaKHis_model.evaluate(x_test, y_test)
