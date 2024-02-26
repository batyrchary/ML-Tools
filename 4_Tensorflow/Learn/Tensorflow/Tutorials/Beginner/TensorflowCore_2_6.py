#ML Basics with Keras

#Model progress can be saved during/after training. 
#Save and load models 
#https://www.tensorflow.org/tutorials/keras/save_and_load

#pip install pyyaml h5py  # Required to save models in HDF5 format


import os
import tensorflow as tf
from tensorflow import keras
import math


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

model = create_model()    # Create a basic model instance
print(model.summary())


#Save checkpoints during training
#The tf.keras.callbacks.ModelCheckpoint callback allows you to 
#continually save the model both during and at the end of training.

#Create a tf.keras.callbacks.ModelCheckpoint callback that saves weights only during training:
checkpoint_path = "./modelSaving/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
#This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  


# Create a basic model instance and Evaluate the model, model will perform poor
model = create_model()
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


#Then load the weights from the checkpoint and re-evaluate
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


print("------------------------------------------------")

#Checkpoint callback options, Include the epoch in the file name (uses `str.format`)
checkpoint_path = "./modelSaving/training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Calculate the number of batches per epoch
batch_size = 32
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*n_batches)

# Create a model and save the weights using the `checkpoint_path` format
model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

#Now, review the resulting checkpoints and choose the latest one:
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# Create a new model instance and Load the previously saved weights, and re-evaluate
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

print("------------------------------------------------")
#Manually save weights with use tf.keras.Model.save_weights. 
#By default, it uses format with a .ckpt extension. 
#To save in the HDF5 format with a .h5 extension, refer to the Save and load models guide.

model.save_weights('./modelSaving/checkpoints/my_checkpoint')
model = create_model()
model.load_weights('./modelSaving/checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


print("------------------------------------------------")
#Save the entire model with tf.keras.Model.save to save a model's 
#architecture, weights, and training configuration in a single model.keras zip archive.
#An entire model can be saved in three different file formats (the new .keras format and two legacy formats: SavedModel, and HDF5). 

#You can switch to the SavedModel format by:
#Passing save_format='tf' to save()
#Passing a filename without an extension
#You can switch to the H5 format by:
#Passing save_format='h5' to save()
#Passing a filename that ends in .h5

#Saving a fully-functional model is very useful—you can load them in TensorFlow.js (Saved Model, HDF5) 
#and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite (Saved Model, HDF5)


#New high-level .keras format Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('./modelSaving/entiremodel/my_model.keras')  # Save the entire model as a `.keras` zip archive.

#Reload a fresh Keras model from the .keras zip archive:
new_model = tf.keras.models.load_model('./modelSaving/entiremodel/my_model.keras')
print(new_model.summary())

#Try running evaluate and predict with the loaded model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
print(new_model.predict(test_images).shape)


#SavedModel format: is another way to serialize models. 
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
model.save('./modelSaving/entiremodel/saved_model/my_model')

new_model = tf.keras.models.load_model('./modelSaving/entiremodel/saved_model/my_model')
print(new_model.summary())

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
print(new_model.predict(test_images).shape)


#HDF5 format
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file. The '.h5' extension indicates that the model should be saved to HDF5.
model.save('./modelSaving/entiremodel/my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('./modelSaving/entiremodel/my_model.h5')

print(new_model.summary())
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
