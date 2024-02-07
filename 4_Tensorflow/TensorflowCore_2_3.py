#ML Basics with Keras

#Text classification with TensorFlow Hub: Movie reviews 
#https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses




#Sentiment analysis
#trains a sentiment analysis model to classify movie reviews as positive or negative, based on the text of the review. 
#Movie Review Dataset that contains the text of 50,000 movie reviews from the Internet Movie Database. 
#These are split into 25,000 reviews for training and 25,000 reviews for testing. 
#The training and testing sets are balanced, meaning they contain an equal number of positive and negative reviews.
#Download and explore the IMDB dataset

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.',cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

#print one review 
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())

#Load the dataset
#IMPORTANT
#To prepare a dataset for binary classification, you will need two folders on disk, 
#corresponding to class_a and class_b. These will be the positive and negative movie reviews, 
#which can be found in aclImdb/train/pos and aclImdb/train/neg. As the IMDB dataset contains 
#additional folders, you will remove them before using this utility

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)


#The IMDB dataset has already been divided into train and test, but it lacks a validation set. 
#Let's create a validation set using an 80:20 split of the training data by using the validation_split argument below.

batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)
#25,000 examples in the training folder, of which you will use 80% (or 20,000) for training

for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])


print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

#create a validation and test dataset. You will use the remaining 5,000 reviews from the training set for validation.

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/test', batch_size=batch_size)


#Prepare the dataset for training
#standardize, tokenize, and vectorize the data using tf.keras.layers.TextVectorization layer.


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')


#create a TextVectorization layer.
#You will use this layer to standardize, tokenize, and vectorize our data.

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

#call adapt to fit the state of the preprocessing layer to the dataset. This will cause the model to build an index of strings to integers.
# Make a text-only dataset (without labels), then call adapt
#Note: It's important to only use your training data when calling adapt (using the test set would leak information).
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)




def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

#each token has been replaced by an integer. You can lookup the token (string) 
#that each integer corresponds to by calling .get_vocabulary() on the layer.

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

#apply the TextVectorization layer you created earlier to the train, validation, and test dataset.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)



#.cache() keeps data in memory after it's loaded off disk. 
#This will ensure the dataset does not become a bottleneck while training your model
#.prefetch() overlaps data preprocessing and model execution while training.


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Create the model
embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()


The layers are stacked sequentially to build the classifier:

#Embedding layer takes the integer-encoded reviews and looks up an embedding vector for each word-index. 
#These vectors are learned as the model trains. The vectors add a dimension to the output array. 
#The resulting dimensions are: (batch, sequence, embedding). 

#GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging 
#over the sequence dimension. This allows the model to handle input of variable length, 
#in the simplest way possible.
#densely connected with a single output node.



model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


#Train the model
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

#Evaluate the model
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


#Create a plot of accuracy and loss over time
history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


#Export the model

#If you want to make your model capable of processing raw strings 
#(for example, to simplify deploying it), you can include the TextVectorization 
#layer inside your model. To do so, you can create a new model using the weights you just trained.

export_model = tf.keras.Sequential([vectorize_layer, model, layers.Activation('sigmoid')])

export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

#Inference on new data
examples = ["The movie was great!", "The movie was okay.", "The movie was terrible..."]

export_model.predict(examples)




