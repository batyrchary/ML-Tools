#ML Basics with Keras

#This guide trains a neural network model to classify images of clothing, like sneakers and shirts.
#https://www.tensorflow.org/tutorials/keras/classification


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



#dataset which contains 70,000 grayscale images in 10 categories. 
#The images show individual articles of clothing at low resolution (28 by 28 pixels)

#60,000 images are used to train the network and 10,000 images to evaluate the accuracy
#Loading the dataset returns four NumPy arrays
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)  #there are 60,000 images in the training set, with each image represented as 28 x 28 pixels
print(len(train_labels))   #there are 60,000 labels in the training set
print(train_labels)        #Each label is an integer between 0 and 9

print(test_images.shape)   #There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels
print(len(test_labels))    #And the test set contains 10,000 images labels


#Preprocess the data: the pixel values fall in the range of 0 to 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Scale these values to a range of 0 to 1 before feeding them to the neural network model.
train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
# tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional 
#array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). 

#the network consists of a sequence of two tf.keras.layers.Dense layers. 
#These are densely connected, or fully connected, neural layers
#The first Dense layer has 128 nodes (or neurons). The second (and last) 
#layer returns a logits array with length of 10. 
#Each node contains a score that indicates the current image belongs to one of the 10 classes.



#########Compile the model
#Optimizer —This is how the model is updated based on the data it sees and its loss function.
#Loss function —This measures how accurate the model is during training. 
#You want to minimize this function to "steer" the model in the right direction.
#Metrics —Used to monitor the training and testing steps. 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#Train the model
#1.Feed the training data to the model. 
#The model learns to associate images and labels.
#You ask the model to make predictions about a test set—in this example, the test_images array.
#Verify that the predictions match the labels from the test_labels array.
model.fit(train_images, train_labels, epochs=10)



#Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


#Make predictions
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
#Here, the model has predicted the label for each image in the testing set.
#Let's take a look at the first prediction:
print(predictions[0])
#array([2.4707775e-08, 6.4802941e-09, 5.7907998e-09, 2.2175234e-10,
#       2.9547037e-08, 2.6677480e-05, 2.0438068e-07, 2.8663729e-03,
#       4.4412984e-09, 9.9710673e-01], dtype=float32)
#A prediction is an array of 10 numbers. 
#They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing.

print(np.argmax(predictions[0]))   #=>9



#Define functions to graph the full set of 10 class predictions.

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Verify predictions
#Correct prediction labels are blue and incorrect prediction labels are red.

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()



#Use the trained model to make a prediction about a single image.
# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

#tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. 
#Accordingly, even though you're using a single image, you need to add it to a list

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

#Now predict the correct label for this image
predictions_single = probability_model.predict(img)
print(predictions_single)

#tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data. 
#Grab the predictions for our (only) image in the batch
np.argmax(predictions_single[0])

