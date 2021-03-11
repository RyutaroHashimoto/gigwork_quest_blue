#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

from pathlib import Path
# from fastai import *
# from fastai.vision import *


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


# In[7]:


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


# In[22]:


batch_size=32
epochs=5


# In[23]:


model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", input_shape=(32,32,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GlobalAveragePooling2D())

model.add(Dense(units=20))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
             optimizer=tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6),
              metrics=["accuracy"])

model.fit(X_train_image, y_train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val_image, y_val_label))


# In[30]:


print(model.evaluate(X_val_image, y_val_label, verbose=0))


# In[24]:


print(test_images.shape)

predictions = model.predict(test_images)

print(type(predictions))

print(predictions.shape)


# In[25]:


print(predictions[0])


# In[26]:


file_name = sample[0]
df = pd.DataFrame(predictions)


# In[27]:


df_concat = pd.concat([file_name, df],axis=1)


# In[28]:


df_concat.head()


# In[29]:


df_concat.to_csv('./Downloads/sample_submit_001.csv',index = False, header=None)


# In[ ]:




