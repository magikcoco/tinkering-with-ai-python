import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras import layers
from keras import models

###################################################################
##                 Get and Prepare the Dataset                   ##
###################################################################
# get the data from within keras
(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()
# normalize the data, so that all the values are between 0 and 1. The values start from 0 to 255.
training_images, testing_images = training_images / 255, testing_images / 255

###################################################################
##                       Visualize the Dataset                   ##
###################################################################
# in the current training and testing datasets, labels are just numbers
# need to assign names to the labels with this list, order is important
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
'''
# Visualize the data
for i in range(16):
    # a 4x4 grid, and choose a space in the grid to put the next image
    plt.subplot(4, 4, i+1)
    # remove coordinate system
    plt.xticks([])
    plt.yticks([])
    # show an image and its label
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()
'''
###################################################################
##               Building and Training the model                 ##
###################################################################
# Optionally limit the testing and training data
'''
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]
# Building the model
model = models.Sequential()
# Convolutional layers filter for features in an image
# MaxPooling layers reduce the image to the essential information
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# This layer flattens the result of the above
model.add(layers.Flatten())
# Output layer scales the results so that we get percentages
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Default batch size will be 32
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss = {loss}\nAccuracy = {accuracy}")
model.save("imageclassifier.model")
'''
# works well with obvious examples
model = models.load_model('imageclassifier.model')
img = cv.imread('resources/imageclassifier/car.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
plt.show()

