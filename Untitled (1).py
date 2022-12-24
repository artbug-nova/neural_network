#!/usr/bin/env python
# coding: utf-8

# In[4]:


import keras
import numpy as np
import tensorflow as tf

model = keras.models.Sequential([
                                   keras.layers.Dense(16, input_shape=(2,), activation='relu'),
                                   keras.layers.Dense(8, activation='relu'),
                                   keras.layers.Dense(1, activation='sigmoid') #Sigmoid for probabilistic distribution
])


# In[7]:


def my_loss_fn(y_true, y_pred):
    print('X')
    print(y_true)
    print('Y')
    print(y_pred)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

model.compile(optimizer='sgd', loss=my_loss_fn, metrics=['acc'])# binary cross entropy


# In[8]:


model.fit(np.array([[10.0, 20.0],[20.0,30.0],[30.0,6.0], [8.0, 20.0]]),np.array([1,1,0,1]) ,epochs=10)

