---
title: "Dogs vs Cats - Binary Image Classification"
date: 2019-09-07
tags: [image classification, data science, dogs vs cats, CNNs]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Binary Image Classification, Dogs v/s Cats, Custom CNN model, Transfer Learning"
mathjax: "true"
---

# Dogs v/s Cats - Binary Image Classification using ConvNets (CNNs)

This is a hobby project I did to jump into the world of deep neural networks. I used Keras with TensorFlow backend to build my custom
convolutional neural network, with 3 subgroups of convolution pooling and activation layers before flattening and adding a couple of fully connected dense layers as well as a dropout layer to prevent over-fitting. I plotted the progression of accuracy and loss during my training epochs on both training and testing sets to monitor the model performance.

The entire code and data, with the directrory structure can be found on my GitHub page here [link](https://github.com/atul6876/Dogs_vs_Cats_CNN_keras)?

The repository linked above contains the code to predict whether the picture contains the image of a dog or a cat using a CNN model trained on the images from the kaggle dataset (not all the images, but I use image augmentation techniques that ensure that the model sees a new "image" at each training epoch. I also use pretrained models with deeper architectures for image classification. I have used Keras's blog on building and compiling a CNN model as a template for most of my code and directory structure.

I have also trained the widely used VGG16 model and modified the model's output layer for binary classification of dogs and cats. I am using the pre-trained weights, and only training the final layer weights at each training epoch. As you'll see, even with very limited training epochs, the VGG model outperforms the simple ConvNet model by 15% (88% accuracy as compared to 73% of the ConvNet).

I have set up the directory structure like this:

1. train
+ dogs
- cats
2. validation
+ dogs
- cats
3. test
+ Mixed images (20)

## Image Augmentation

Given the fact that I was using my laptop to train my convNet model, I couldn't afford to use all the images available in the Kaggle dataset (there are 25000 images available there). Instead, I used 2000 images for training, 1000 each for cats and dogs as well as 800 for validation with 400 each. I used Keras's ImageGenerator functionality to augment my the limited images I had, which ensured that the model was trained on modified images at each training epoch, and they were never trained on the same exact image twice. The code for my transformations is shown below:
```python
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

datagen = ImageDataGenerator(rotation_range=40, # range of rotation angle (could be 0-180 degrees)
    width_shift_range=0.2,                      # portion of the image to shift horizontally
    height_shift_range=0.2,                     # portion of the image to shift vertically
    brightness_range=None,                      # Range of altering brightness levels, no
    shear_range=0.2,                            # range of shearing transformation
    zoom_range=0.2,                             # range of zooming into the image
    horizontal_flip=True,                       # randomly flipping images horizontally
    fill_mode='nearest')                        # filling methodology for missing pixels after aforementioned transformations
 ```

## Custom ConvNet

I designed the following CNN. I based it on some of the common designs avalable online. The basic idea is to start with fewer filters at the beginning, and increasing the number of filters as we go deep into the network. The assumption being that the filters at first learn to identify simple line and shapes, and they get more and more complex and learn to identify complex shapes as we go further down the layers. The code to build my basic net is shown below:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))  
```

After building the ConvNet model, I used the binary crossentropy as the loss metric (we can also use categorial crossentropy here), adam optimizer and I wanted to get back accuracy at each training epoch and validation step as my output. The code to compile the model is as follows:

```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

Now we pass the augmented images for training and validation and save the metrics at each epoch using the history module. We'll use the history module to plot the loss and accuracy curves. We also predict the final model performance on the validation set. In this case the accuracy achieved is ~73%.

```python
train_datagen = ImageDataGenerator(
    rescale=1. / 255,   # rescale the images
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    
test_datagen = ImageDataGenerator(rescale=1./255) # rescale the images

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
    
history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# save model and architecture to single file
model.save("full_model.h5")
print("Saved full model to disk")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Print out the validation accuracy on the validation set
model.evaluate_generator(validation_generator, nb_validation_samples)
```
## Transfer Learning - Using a pre-trained model and its weights

I have used the VGG16 model trained on the imagenet dataset, trained to identify 1000 classes (imagenet data is a labeled dataset of ~1.3 million images belonging to 1000 classes. To use this model and its weights for the purpose of binary classification, we need to modify the VGG16 ConvNet for binary classification. I have included the code for how to load this model, freeze the training weights so that they are not altered during our training, and how to modify the final layer for binary prediction. The model is available in keras and can be imported as is.

```python
# Loading the vgg16 model from keras with imagenet weights, setting the input shape to our interests 

vgg = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
                               input_tensor=None, input_shape=(224,224,3), pooling=None) #could write input_shape=input_shape
vgg.summary()             # print out the model summary

# Freeze the layers so that they are not trained during model fitting. We want to keep the imagenet weights
for layer in vgg.layers: 
    layer.trainable=False
    
# Change the final dense layer to 1 node (sigmoid activation) for binary classification
# could do 2 nodes and determine the probabilities of each class using SoftMax, but we used Sigmoid for our simple ConvNet
x = vgg.layers[-2].output
output_layer = Dense(1, activation='sigmoid', name='predictions')(x)

# Combine the output layer to the original model
vgg_binary = Model(inputs=vgg.input, outputs=output_layer)

# Sanity check: Print out the model summary. The final layer should have 1 neuron only (again, using sigmoid activation)
vgg_binary.summary()

# Compile the modified vgg model with the following hyperparameters (same as simple ConvNet)
# In future try different learning rates for the adam 'adaptive moment estimation'
vgg_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Defining Image transformations: normalization (rescaling) for both training and testing images
# Defining Image transformations: Augmenting the training data with the following transformations 
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

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

# Fitting the modified vgg16 model on the image batches set up in the previous step
# Save the model (full model). Save the training history
history = vgg_binary.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=5,                           # changed epochs=epochs to 5, larger model and thus takes more time to train
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

vgg_binary.save('vgg_binary.h5')
print("Saved vgg16 model to disk") # Caution: the model size is over 500MB

# Print out the performance over the validation set (Caution: it takes a long time, run it at your own expense)
# The model does a much better job than the simple ConvNet. With accuracy of ~88%

vgg_binary.evaluate_generator(validation_generator, nb_validation_samples)
```

By using a pretrained model (VGG16), which is a deeper ConvNet than the one I designed and which has also been trained on millions of images performs much better even when modified to act as a binary classifier. The accuracy jumps from ~73% for my custom built simple ConvNet to ~88% for the modified VGG16 ConvNet model. The modeified model ensures that we don't tinker with the model's original weights, but inly train the final layer modified for binary prediction. In this hobby project, I also ensured that I kept my dataset balanced, with equal number of dog and cat images. We often don't have such luxury with real world data, and there are many solutions to tackle imbalanced datasets such as oversampling the minority classes or undersampling the majority class, or a combination of both, data augmentation for minority class, ignoring accuracy and focusing on presicion and recall as your performance metric depending what matters more in the problem case, adding penalty for misclassification etc. 

So, this wraps up the project for now. Going forward, I am going to use more images for training my model and I am going to use some GPU power to back my computations. For now, I am going to try Google's Colab Jupyter Notebooks tool as they offer free GPU capabilities and come with a lot of libraries such as TensorFlow and Keras preinstalled.
