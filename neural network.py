#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')


# In[89]:


# importing the modules for the network
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sn



# In[ ]:





# In[41]:


(X_train , y_train) , ( X_test, y_test) = keras.datasets.mnist.load_data()


# In[42]:


len(y_train)


# In[43]:


X_train[0].shape


# In[44]:


X_train[0]


# In[45]:


plt.matshow(X_train[10])


# In[46]:


y_train[10]


# In[47]:


y_test[10]


# In[49]:


X_train.shape


# In[62]:


#scaling (devide each value bt 255 values will be betwwen zero and one) which will increase accurancy 
X_train = X_train/ 255
X_test = X_test/ 255


# In[63]:


#flattwerning the array 
# did not flattern y_train/y_test because their simple array
X_train_flattened = X_train.reshape(len(X_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)


# In[64]:


X_train_flattened


# In[65]:


X_train_flattened.shape


# In[66]:


X_test_flattened.shape


# In[67]:


#
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')  #connecting all other neurons in the input layer to the second layer
])
# following is a crucial step in keras and tensorflow
#compiling nueral network
#add an optimizer to train the nueral network efficiently others are more efficient than others
model.compile(
    optimizer ='adam',
    loss = 'sparse_categorical_crossentropy',    #loss computes ther erros between the labels and predictions
    metrics = ['accuracy'] # to increase accuracy
)

#perform training 
model.fit(X_train_flattened, y_train, epochs= 10)   #epohs = number of loops for the network    


# In[68]:


#evaluating accuracy on the test data set

model.evaluate(X_test_flattened, y_test)


# In[81]:


plt.matshow(X_test[5019])


# In[82]:


y_predict = model.predict(X_test_flattened)
y_predict[5019]


# In[83]:


np.argmax(y_predict[5019])


# In[87]:


#converting the floats to int values in y_predict array
y_predict_labels =[np.argmax(i) for i in y_predict]
y_predict_labels[:5]


# In[88]:


# confusion matrix
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predict_labels)
cm


# In[90]:


#use seabon so that we can get visually appealing arrays
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:





# In[ ]:





# In[99]:


# adding the hidden layer. which improves perfomance
# the hidden lsyer affects the time
model = keras.Sequential([
    keras.layers.Dense(200, input_shape=(784,),activation='relu') , #relu is best according to the internet
    keras.layers.Dense(10,activation='softmax') 
])

model.compile(
    optimizer ='adam',
    loss = 'sparse_categorical_crossentropy',    #loss computes ther erros between the labels and predictions
    metrics = ['accuracy'] # to increase accuracy
)

#perform training 
model.fit(X_train_flattened, y_train, epochs= 5)   #epohs = number of loops for the network    


# In[97]:


model.evaluate(X_test_flattened, y_test)


# In[98]:


#with the hidden layer accuracy increased from 0.8364999890327454 to 0.9341999888420105
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




