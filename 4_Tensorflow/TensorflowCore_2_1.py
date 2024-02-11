#ML Basics with Keras

#This guide trains a neural network model to classify images of clothing, like sneakers and shirts.
#https://www.tensorflow.org/tutorials/keras/classification


# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#dataset contains 70,000 grayscale images(28x28) in 10 categories  
#60,000 images for train and 10,000 for test, each label is int between 0-9
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape, test_images.shape, len(train_labels), len(test_labels), train_labels[:3]) 
    
#just visualize first image
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#Scale values of images 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#Visualize 25 images
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()


########################--Build the model 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#---tf.keras.layers.Flatten, transforms the format of the images from a 2D
#array (of 28 by 28 pixels) to a 1D array (of 28 * 28 = 784 pixels). 
#---the network consists of a sequence of 2 Dense layers (these layers densely (fully) connnected)
#The 1st Dense layer has 128 nodes (or neurons). 
#The 2nd layer returns a logits array with length of 10. 
#Each node contains a score that indicates the current image belongs to one of the 10 classes.


########################--Compile the model
#--Optimizer—how the model is updated based on the data it sees and its loss function.
#--Loss function—measures how accurate the model is during training. 
#You want to minimize this function to "steer" the model in the right direction.
#--Metrics—Used to monitor the training and testing steps. 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


########################--Train and Evaluate the model
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(predictions[0])
#array([2.4707775e-08, 6.4802941e-09, 5.7907998e-09, 2.2175234e-10,
#       2.9547037e-08, 2.6677480e-05, 2.0438068e-07, 2.8663729e-03,
#       4.4412984e-09, 9.9710673e-01], dtype=float32)
#A prediction is an array of 10 numbers. 
#They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing.
print(np.argmax(predictions[0]))   #=>9


########################--Plot the Results

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


#Verify predictions-Correct prediction labels are blue and incorrect prediction labels are red.
for i in []:#[0, 12]:
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(1,2,2)
  plot_value_array(i, predictions[i],  test_labels)
  plt.show()

########################--Prediction on Single image
#Use the trained model to make a prediction about a single image.
img = test_images[1]
print(img.shape)

#tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. 
#Accordingly, even though you're using a single image, you need to add it to a list

#Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

#tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data. 
#Grab the predictions for our (only) image in the batch
print(np.argmax(predictions_single[0]))
