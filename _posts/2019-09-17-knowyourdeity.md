---
title: "Know Your Deity - Image Classification"
date: 2019-09-17
tags: [image classification, transfer learning, CNNs, web scraping, data augmentation]
header:
  image: "/images/perceptron/image00071.jpg"
excerpt: "Binary Image Classification, Buddha/Ganesha, pretrained CNN model, Transfer Learning"
mathjax: "true"
---
# Know your diety

## Binary Image Classification: Ganesha & Buddha

This is related to the work I have been doing in my internship recently. I have been analyzing lots and lots of paintings and collecting metadata about the objects within them using instance segmentation methods (Mask-RCNN model pre-trained on COCO dataset). However, the COCO dataset (Common Objects in COntext) dataset comprises of 80 classes (81 if you count the background class) and flags any portrait of any person as just "person". This limited capacity, although very accurate was not enough for my purposes as it would list all the various portraits of different deities as just "person". Therefore I decided to train a simple classifier model that could take in the portraits and classify the person (in my case the deities of Buddha and Ganesha) to a reasonable degree of accuracy. I modified the VGG16 ConvNet model for binary classification and trained the model on 400 images of each class which I downloaded using the Selenium module and chrome webdriver directly from my Ipython notebook. I used Keras's ImageDataGenerator for image data augmentation, given I had only 400 images of each class to train my model.

## Getting Images:


```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib
import argparse

from imutils import paths
import argparse
import cv2
import requests
```


```python
# searchterm = 'Ganesh Painting' # will also be the name of the folder
# url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
# # NEED TO DOWNLOAD CHROMEDRIVER, insert path to chromedriver inside parentheses in following line
# browser = webdriver.Chrome('webdriver/chromedriver.exe')
# browser.get(url)
# header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

# urlList = []

# if not os.path.exists(searchterm):
#     os.mkdir(searchterm)

# for _ in range(500):
#     browser.execute_script("window.scrollBy(0,10000)")
# for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
#     #print("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
#     urlList.append(json.loads(x.get_attribute('innerHTML'))["ou"])

# browser.close()
# print('Number of URLs tracked: ', len(urlList))

# with open(searchterm+'URLs.txt', 'w') as f:
#     for item in urlList:
#         f.write("%s\n" % item)
```


```python
# searchterm = 'Buddha Painting' # will also be the name of the folder
# url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
# # NEED TO DOWNLOAD CHROMEDRIVER, insert path to chromedriver inside parentheses in following line
# browser = webdriver.Chrome('webdriver/chromedriver.exe')
# browser.get(url)
# header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

# urlList = []

# if not os.path.exists(searchterm):
#     os.mkdir(searchterm)

# for _ in range(500):
#    browser.execute_script("window.scrollBy(0,10000)")

# for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
#     #print("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
#     urlList.append(json.loads(x.get_attribute('innerHTML'))["ou"])

# browser.close()
# print('Number of URLs tracked: ', len(urlList))

# with open(searchterm+'URLs.txt', 'w') as f:
#     for item in urlList:
#         f.write("%s\n" % item)
```


```python
# searchterm = 'Ganesh Painting'

# path_text=searchterm+'URLs.txt'

# o = open(path_text,"r")
# url0 = o.read()
# o.close()

# ## list, containing downloaded files
# urls = url0.split()
# print("The number of urls: {}".format(len(urls)))
# print('-'*50)
# for url in urls[:10]:
#     print(url)

# loc_data = "./data/"+searchterm+"/"
# try:
#     os.makedirs(loc_data)
# except:
#     pass
# iimage = 0
# for url in urls:
#     try:
#         f = open(loc_data + 'image{:05.0f}.jpg'.format(iimage),'wb')
#         f.write(requests.get(url).content)
#         f.close()
#         iimage += 1
#     except Exception as e:
#         print("\n{} {}".format(e,url))
#         pass
```


```python
# searchterm = 'Buddha Painting'

# path_text=searchterm+'URLs.txt'

# o = open(path_text,"r")
# url0 = o.read()
# o.close()

# ## list, containing downloaded files
# urls = url0.split()
# print("The number of urls: {}".format(len(urls)))
# print('-'*50)
# for url in urls[:10]:
#     print(url)

# loc_data = "./data/"+searchterm+"/"
# try:
#     os.makedirs(loc_data)
# except:
#     pass
# iimage = 0
# for url in urls:
#     try:
#         f = open(loc_data + 'image{:05.0f}.jpg'.format(iimage),'wb')
#         f.write(requests.get(url).content)
#         f.close()
#         iimage += 1
#     except Exception as e:
#         print("\n{} {}".format(e,url))
#         pass
```


```python
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

fnames = os.listdir('data/train/Buddha')
fig = plt.figure(figsize=(10,10))
count = 1
for fnm in fnames[:12]:
    img = load_img('data/train/Buddha/' +fnm,target_size=(400,400))
    ax = fig.add_subplot(3,4,count)
    count += 1
    ax.imshow(img)
    ax.axis("off")
plt.show()
```


![png](example-post_files/example-post_8_0.png)



```python
fnames = os.listdir('data/train/Ganesha')
fig = plt.figure(figsize=(10,10))
count = 1
for fnm in fnames[:12]:
    img = load_img('data/train/Ganesha/'+fnm,target_size=(400,400))
    ax = fig.add_subplot(3,4,count)
    count += 1
    ax.imshow(img)
    ax.axis("off")
plt.show()
```


![png](example-post_files/example-post_9_0.png)


The code above lets us download 400 images for the search term we provide. However, a lot of images turn out to be broken links and can't be used. I'm afraid this requires manually going over the images that you have and deleting the ones which are basically empty images. In this case, I got something like 360 good images for both the classes, and therefore I decided to go ahead and use them as my training and testing data. For training I used 300 images of each class and for cross validation, I used 60 images of each class. As I mentioned before, I used image data augmentation techniques as available through the Keras's ImageDataGenrator module. I set up my directory structure to fit the following structure:

**Data:
    *train:
        *Buddha
        *Ganesha
    *cross validation:
        *Buddha
        *Ganesha
    *test:
        *Mixed Images

## Training the binary classification model based on the VGG16 Convnet architecture


```python
# Import the necessary modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model # Model is useful to edit the layers of an existing model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)
```


```python
# Setting the dimensions of our images.

img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 600
nb_validation_samples = 120

epochs = 10   # we won't be using 20 epochs. Here 10 epochs is enough. Due to the complexity of the model, learning is slow
batch_size = 15
```


```python
# Setting the input shape format: 3 is the color channels (RGB)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
```


```python
# Loading the vgg16 model from keras with imagenet weights, setting the input shape to our interests

vgg = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
                               input_tensor=None, input_shape=(224,224,3), pooling=None) #could write input_shape=input_shape
vgg.summary()        
```

    WARNING: Logging before flag parsing goes to stderr.
    W0917 15:52:14.487166 16928 deprecation_wrapper.py:119] From C:\Users\atuls\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

    W0917 15:52:14.510338 16928 deprecation_wrapper.py:119] From C:\Users\atuls\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:529: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

    W0917 15:52:14.510338 16928 deprecation_wrapper.py:119] From C:\Users\atuls\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4420: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

    W0917 15:52:14.556646 16928 deprecation_wrapper.py:119] From C:\Users\atuls\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4255: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

    W0917 15:52:15.999950 16928 deprecation_wrapper.py:119] From C:\Users\atuls\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:178: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

    W0917 15:52:15.999950 16928 deprecation_wrapper.py:119] From C:\Users\atuls\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:185: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.



    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 224, 224, 3)       0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    predictions (Dense)          (None, 1000)              4097000   
    =================================================================
    Total params: 138,357,544
    Trainable params: 138,357,544
    Non-trainable params: 0
    _________________________________________________________________



```python
# Freeze the layers so that they are not trained during model fitting. We want to keep the imagenet weights
for layer in vgg.layers:
    layer.trainable=False
```


```python
# Change the final dense layer to 1 node (sigmoid activation) for binary classification
# could do 2 nodes and determine the probabilities of each class using SoftMax, but we used Sigmoid for our simple ConvNet
x = vgg.layers[-2].output
output_layer = Dense(1, activation='sigmoid', name='predictions')(x)
```


```python
# Combine the output layer to the original model
vgg_binary = Model(inputs=vgg.input, outputs=output_layer)
```


```python
# Sanity check: Print out the model summary. The final layer should have 1 neuron only (again, using sigmoid activation)
vgg_binary.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 224, 224, 3)       0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    predictions (Dense)          (None, 1)                 4097      
    =================================================================
    Total params: 134,264,641
    Trainable params: 4,097
    Non-trainable params: 134,260,544
    _________________________________________________________________



```python
# Compile the modified vgg model with the following hyperparameters (same as simple ConvNet)
# In future try different learning rates for the adam 'adaptive moment estimation'
vgg_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

    W0917 15:52:17.346772 16928 deprecation_wrapper.py:119] From C:\Users\atuls\Anaconda3\lib\site-packages\keras\optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

    W0917 15:52:17.370725 16928 deprecation.py:323] From C:\Users\atuls\Anaconda3\lib\site-packages\tensorflow\python\ops\nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where



```python
# Defining Image transformations: normalization (rescaling) for both training and testing images
# Defining Image transformations: Augmenting the training data with the following transformations
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
```


```python
# Setting up the flow of images in batches for training and validation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Printing out the class labels for both training and validation sets
print(train_generator.class_indices)
print(validation_generator.class_indices)
```

    Found 600 images belonging to 2 classes.
    Found 120 images belonging to 2 classes.
    {'Buddha': 0, 'Ganesha': 1}
    {'Buddha': 0, 'Ganesha': 1}



```python
# Fitting the modified vgg16 model on the image batches set up in the previous step
# Save the model (full model). Save the training history
history = vgg_binary.fit_generator(
        train_generator,
        steps_per_epoch=600 // batch_size,
        epochs=epochs,                           
        validation_data=validation_generator,
        validation_steps=120 // batch_size)

#vgg_binary.save('vgg_binary.h5')
print("Saved vgg16 model to disk") # the modlsize is over 500MB

# As you can see, each epoch is taking ~2mins. The loss is going down with each epoch. The model also generalizes well
# the accuracy on the validation set is mirroring that of training set, actually it is better on the validation set.
```

    Epoch 1/10
    40/40 [==============================] - 156s 4s/step - loss: 0.6066 - acc: 0.6650 - val_loss: 0.5343 - val_acc: 0.7000
    Epoch 2/10
    40/40 [==============================] - 154s 4s/step - loss: 0.5440 - acc: 0.7333 - val_loss: 0.4512 - val_acc: 0.7750
    Epoch 3/10
    40/40 [==============================] - 144s 4s/step - loss: 0.4548 - acc: 0.8067 - val_loss: 0.4023 - val_acc: 0.8167
    Epoch 4/10
    40/40 [==============================] - 141s 4s/step - loss: 0.4310 - acc: 0.8150 - val_loss: 0.3837 - val_acc: 0.8417
    Epoch 5/10
    40/40 [==============================] - 141s 4s/step - loss: 0.4141 - acc: 0.8267 - val_loss: 0.4070 - val_acc: 0.8333
    Epoch 6/10
    40/40 [==============================] - 146s 4s/step - loss: 0.4047 - acc: 0.8167 - val_loss: 0.3621 - val_acc: 0.8500
    Epoch 7/10
    40/40 [==============================] - 139s 3s/step - loss: 0.4099 - acc: 0.8000 - val_loss: 0.4011 - val_acc: 0.8500
    Epoch 8/10
    40/40 [==============================] - 135s 3s/step - loss: 0.4005 - acc: 0.8200 - val_loss: 0.3709 - val_acc: 0.8417
    Epoch 9/10
    40/40 [==============================] - 134s 3s/step - loss: 0.3710 - acc: 0.8550 - val_loss: 0.3977 - val_acc: 0.8500
    Epoch 10/10
    40/40 [==============================] - 150s 4s/step - loss: 0.4047 - acc: 0.8200 - val_loss: 0.3772 - val_acc: 0.8500
    Saved vgg16 model to disk



```python
# Print out the metrics recorded during training (saved in the history)
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# As we can see the accuracy is improving as we train through the epochs. At the end we see it is elbowing around 83%
```

    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])



![png](example-post_files/example-post_24_1.png)



```python
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# As I mentioned before, the loss is going down at each epoch of training, fluctuates around 0.37-0.38
```


![png](example-post_files/example-post_25_0.png)


We see above that the accuracy stabilizes around 83% on the validation set after 10 epochs which is decent enough for my purpose for now. I froze the VGG16 pretrained model (on imagenet labels and data) weights and just popped off the final layers in order to modify it to my purpose of binary Classification. This can be extended to more classes, as we can train the model to identify more portraits of more specific people.
