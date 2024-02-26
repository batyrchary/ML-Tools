#ML Basics with Keras

#Text classification with TensorFlow Hub: Movie reviews 
#https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
#Classifies movie reviews as positive or negative using the text of the review

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

#pip install tensorflow_datasets
#pip install tensorflow_hub
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")


########################--Dataset

#Split with 60% and 40%: train (15,000), validation (10,000), testing(25,000)
#Label is 0 (neg review) or 1 (pos review) 
train_data, validation_data, test_data = tfds.load(name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
#print(train_examples_batch)
#print(train_labels_batch)

#There are many other pre-trained text embeddings from TFHub that can be used
#Create a Keras layer that uses a TensorFlow Hub model to embed the sentences
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
print(hub_layer(train_examples_batch[:3]))
#Note--it may give connection error


########################--Build The Model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
print(model.summary())

#The first layer is a TensorFlow Hub layer. 
#This layer uses a pre-trained Saved Model (google/nnlm-en-dim50/2) to map a sentence into its embedding vector. T
#Splits the sentence into tokens, embeds each token and then combines the embedding. 
#The resulting dimensions are: (num_examples, embedding_dimension). 
#For this NNLM model, the embedding_dimension is 50.
#This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
#The last layer is densely connected with a single output node.

#Compile
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

#Train
history = model.fit(train_data.shuffle(10000).batch(512), epochs=10, validation_data=validation_data.batch(512),verbose=1)

#Evaluate
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

