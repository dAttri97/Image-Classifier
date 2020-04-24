#!/usr/bin/env python
# coding: utf-8

# In[27]:


import tensorflow
from keras.datasets import cifar10
import keras.utils as util
import numpy as np
(train_img,train_label),(test_img,test_label) = cifar10.load_data()


# In[2]:


train_img.shape


# In[20]:


print(train_label[:5])
train_label.shape


# In[6]:


train_img[0][0]


# In[28]:


labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
train_label = util.to_categorical(train_label)


# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(1,1))
plt.imshow(train_img[0])
plt.show()
ind = np.argmax(train_label[0])
print(labels[ind])


# In[30]:


plt.figure(figsize=(1,1))
plt.imshow(train_img[3])
plt.show()
ind = np.argmax(train_label[3])
print(labels[ind])


# In[31]:


plt.figure(figsize=(1,1))
plt.imshow(train_img[49999])
plt.show()
ind = np.argmax(train_label[49999])
print(labels[ind])


# In[ ]:




