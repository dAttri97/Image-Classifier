#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow
from keras.datasets import cifar10
import keras.utils as util
import numpy as np
from keras.models import load_model


# In[3]:


(_,_),(x_test,y_test) = cifar10.load_data()


# In[4]:


x_test = x_test.astype('float')/255.0
y_test = util.to_categorical(y_test)


# In[5]:


model = load_model('ImageClassifier.h5')


# In[6]:


result = model.evaluate(x_test,y_test)


# In[7]:


print("Test Loss: {}".format(result[0]))
print("Test Accuracy: {}".format(result[1]))


# In[ ]:




