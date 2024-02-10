#ML Basics with Keras

#Save and load models 
#https://www.tensorflow.org/tutorials/keras/save_and_load



#Model progress can be saved during and after training. 
#This means a model can resume where it left off and avoid long training times. 

#pip install pyyaml h5py  # Required to save models in HDF5 format


import os
import tensorflow as tf
from tensorflow import keras


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# Define a simple sequential model
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

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()


#Save checkpoints during training
#The tf.keras.callbacks.ModelCheckpoint callback allows you to continually save the model both during and at the end of training.



#Checkpoint callback usage
#Create a tf.keras.callbacks.ModelCheckpoint callback that saves weights only during training:
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


#This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:
os.listdir(checkpoint_dir)



# Create a basic model instance
model = create_model()

# Evaluate the model, model will perform poor
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


#Then load the weights from the checkpoint and re-evaluate:
# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


#Checkpoint callback options
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Calculate the number of batches per epoch
import math
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*n_batches)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
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
os.listdir(checkpoint_dir)


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))



#Manually save weights
#To save weights manually, use tf.keras.Model.save_weights. 
#By default, tf.keras—and the Model.save_weights method in particular—uses 
#the TensorFlow Checkpoint format with a .ckpt extension. 
#To save in the HDF5 format with a .h5 extension, refer to the Save and load models guide.


# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))



#Save the entire model
#Call tf.keras.Model.save to save a model's architecture, weights, and training configuration in a single model.keras zip archive.
#An entire model can be saved in three different file formats (the new .keras format and two legacy formats: SavedModel, and HDF5). Saving a model as path/to/model.keras automatically saves in the latest format.
'''You can switch to the SavedModel format by:

Passing save_format='tf' to save()
Passing a filename without an extension
You can switch to the H5 format by:

Passing save_format='h5' to save()
Passing a filename that ends in .h5
Saving a fully-functional model is very useful—you can load them in TensorFlow.js (Saved Model, HDF5) and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite (Saved Model, HDF5)

*Custom objects (for example, subclassed models or layers) require special attention when saving and loading. Refer to the Saving custom objects section below.
'''

#New high-level .keras format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')


#Reload a fresh Keras model from the .keras zip archive:
new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()


#Try running evaluate and predict with the loaded model:
# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

#SavedModel format
#The SavedModel format is another way to serialize models. Models saved in this format can be restored using tf.keras.models.load_model and are compatible with TensorFlow Serving. The SavedModel guide goes into detail about how to serve/inspect the SavedModel. The section below illustrates the steps to save and restore the model.
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')

#The SavedModel format is a directory containing a protobuf binary and a TensorFlow checkpoint. Inspect the saved model directory:
# my_model directory
ls saved_model

# Contains an assets folder, saved_model.pb, and variables folder.
ls saved_model/my_model

#Reload a fresh Keras model from the saved model:
new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

#The restored model is compiled with the same arguments as the original model. Try running evaluate and predict with the loaded model:
# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)


#HDF5 format
#Keras provides a basic legacy high-level save format using the HDF5 standard.
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')


# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


#Keras saves models by inspecting their architectures. This technique saves everything:

#The weight values
#The model's architecture
#The model's training configuration (what you pass to the .compile() method)
#The optimizer and its state, if any (this enables you to restart training where you left off)
#Keras is not able to save the v1.x optimizers (from tf.compat.v1.train) since they aren't compatible with checkpoints. For v1.x optimizers, you need to re-compile the model after loading—losing the state of the optimizer.


'''
Saving custom objects
If you are using the SavedModel format, you can skip this section. The key difference between high-level .keras/HDF5 formats and the low-level SavedModel format is that the .keras/HDF5 formats uses object configs to save the model architecture, while SavedModel saves the execution graph. Thus, SavedModels are able to save custom objects like subclassed models and custom layers without requiring the original code. However, debugging low-level SavedModels can be more difficult as a result, and we recommend using the high-level .keras format instead due to its name-based, Keras-native nature.

To save custom objects to .keras and HDF5, you must do the following:

Define a get_config method in your object, and optionally a from_config classmethod.
get_config(self) returns a JSON-serializable dictionary of parameters needed to recreate the object.
from_config(cls, config) uses the returned config from get_config to create a new object. By default, this function will use the config as initialization kwargs (return cls(**config)).
Pass the custom objects to the model in one of three ways:
Register the custom object with the @tf.keras.utils.register_keras_serializable decorator. (recommended)
Directly pass the object to the custom_objects argument when loading the model. The argument must be a dictionary mapping the string class name to the Python class. E.g., tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})
Use a tf.keras.utils.custom_object_scope with the object included in the custom_objects dictionary argument, and place a tf.keras.models.load_model(path) call within the scope.
Refer to the Writing layers and models from scratch tutorial for examples of custom objects and get_config.
'''





