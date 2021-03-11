#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

from pathlib import Path
# from fastai import *
# from fastai.vision import *

import tensorflow as tf

print(tf.__version__)


# In[2]:


labels = pd.read_csv("./Downloads/train_master.tsv", sep="\t") #教師ラベルの読み込み


# In[3]:


labels.head()


# In[4]:


master = pd.read_csv("./Downloads/label_master.tsv", sep="\t") #ラベルマスタの読み込み


# In[5]:


master.head()


# In[6]:


sample = pd.read_csv("./Downloads/sample_submit.csv", header=None, sep=",")


# In[55]:


sample.head()


# In[8]:


train_images = []
for fname in labels["file_name"]:
    path = "./Downloads/train_gig/" + fname
    img = imageio.imread(path)
    train_images.append(img)
train_images = np.array(train_images)
print(type(train_images), train_images.shape)


# In[9]:


test_images = []
for fname in sample[0]:
    path = "./Downloads/test_gig/" + fname
    img = imageio.imread(path)
    test_images.append(img)
test_images = np.array(test_images )
print(type(test_images ), test_images.shape)


# In[10]:


train_images = train_images / 255
test_images = test_images / 255


# In[11]:


y = labels["label_id"]

y_categorical = utils.to_categorical(y)
y_categorical[0:10,]


# In[12]:


X_train_image, X_val_image = np.split(train_images, [40000])
y_train_label, y_val_label = np.split(y_categorical, [40000])


# In[13]:


inputs = tf.keras.Input(shape=(None, None, 3))
x = tf.keras.layers.Lambda(lambda img: tf.image.resize(img, (160, 160)))(inputs)
x = tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)(x)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    weights='imagenet', input_tensor=x, input_shape=(160, 160, 3),
    include_top=False, pooling='avg'
)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(20, activation='softmax')
])

model.summary()


# In[19]:


layer_names = [l.name for l in base_model.layers]
idx = layer_names.index('block_12_expand')
print(idx)


# In[20]:


base_model.trainable = True

for layer in base_model.layers[:idx]:
    layer.trainable = False


# In[21]:


model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[22]:


model.fit(X_train_image, y_train_label, epochs=1, validation_split=0.2, batch_size=256)


# In[23]:


print(model.evaluate(X_val_image, y_val_label, verbose=0))


# In[24]:


print(test_images.shape)

predictions = model.predict(test_images)

print(type(predictions))

print(predictions.shape)


# In[26]:


print(predictions[0])


# In[69]:


file_name = sample[0]
df = pd.DataFrame(predictions)


# In[73]:


df_concat = pd.concat([file_name, df],axis=1)


# In[75]:


df_concat.head()


# In[77]:


df_concat.to_csv('./Downloads/sample_submit.csv',index = False, header=None)


# In[ ]:




