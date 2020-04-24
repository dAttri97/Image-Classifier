#!/usr/bin/env python
# coding: utf-8

# ### Importing Necessary Library and Funtions

# In[37]:


import tensorflow
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.constraints import maxnorm


# ### Intializing the model

# In[23]:


model = Sequential()


# ### Adding the Concolution Layer with the necessary parameters

# In[24]:


model.add(Conv2D(filters = 32, kernel_size=(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))


# ### Adding the MaxPooling with pool_size = (2,2)

# In[25]:


model.add(MaxPooling2D(pool_size=(2,2)))


# ### Converting the input data to 1-D Array

# In[26]:


model.add(Flatten())


# ### Adding the Dense layer on 512 units the _reLU_ Activation Function

# In[27]:


model.add(Dense(units = 512,activation = 'relu', kernel_constraint=maxnorm(3)))


# ### Droping 50% of the Nuerals for better prediction and avoid overfitting

# In[28]:


model.add(Dropout(rate=0.5))


# ### Adding the Dense layer on 512 units the _Softmax_ Activation Function

# In[39]:


model.add(Dense(units=10,activation='softmax'))


# In[30]:


from keras.datasets import cifar10
import keras.utils as util
(x_train,y_train),(x_test,y_test) = cifar10.load_data()


# In[31]:


x_train = x_train.astype('float')/255.0
x_test = x_test.astype('float')/255.0


# In[32]:


y_train = util.to_categorical(y_train)
y_test = util.to_categorical(y_test)


# In[33]:


labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[34]:


from keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.05),loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[40]:


model.fit(x=x_train,y=y_train,epochs=15,batch_size=100)


# In[41]:


model.save('ImageClassifier.h5')

