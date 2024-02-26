#Load and preprocess data

#pandas
#https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

import pandas as pd
import tensorflow as tf
import pprint


SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
df = pd.read_csv(csv_file)
target = df.pop('target')


###########################--A DataFrame as an array
#If your data has a uniform datatype, or dtype, it's possible to use a pandas DataFrame anywhere you could use a NumPy array. 

numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]


#The DataFrame can be converted to a NumPy array using the DataFrame.values property or numpy.array(df). 
#To convert it to a tensor, use tf.convert_to_tensor
#if an object can be converted to a tensor with tf.convert_to_tensor it can be passed anywhere you can pass a tf.Tensor.
tf.convert_to_tensor(numeric_features)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)
normalizer(numeric_features.iloc[:3])

def get_basic_model():
  model = tf.keras.Sequential([normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)])

  model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
  return model

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)
numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))
#for row in numeric_dataset.take(3):
#  print(row)

numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)
model = get_basic_model()
model.fit(numeric_batches, epochs=15)


###########################--A DataFrame as a dictionary
#When you start dealing with heterogeneous data, it is no longer possible to treat the DataFrame 
#as if it were a single array. TensorFlow tensors require that all elements have the same dtype.
numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))
#for row in numeric_dict_ds.take(3):
#  print(row)

#Dictionaries with Keras: two ways you can write a Keras model that accepts a dictionary as input.

'''
#1. The Model-subclass style
#You write a subclass of tf.keras.Model (or tf.keras.Layer). You directly handle the inputs, and create the outputs
def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)
numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)
model.predict(dict(numeric_features.iloc[:3]))



#2. The Keras functional style
inputs = {}
for name, column in numeric_features.items():
  inputs[name] = tf.keras.Input(
      shape=(1,), name=name, dtype=tf.float32)

x = stack_dict(inputs, fun=tf.concat)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, x)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'], run_eagerly=True)

#tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)
model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)
numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)
'''


binary_feature_names = ['sex', 'fbs', 'exang']
categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']

inputs = {}
for name, column in df.items():
  if type(column[0]) == str:
    dtype = tf.string
  elif (name in categorical_feature_names or
        name in binary_feature_names):
    dtype = tf.int64
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

preprocessed = []
for name in binary_feature_names:
  inp = inputs[name]
  inp = inp[:, tf.newaxis]
  float_value = tf.cast(inp, tf.float32)
  preprocessed.append(float_value)


#normalizer = tf.keras.layers.Normalization(axis=-1)
#normalizer.adapt(stack_dict(dict(numeric_features)))

#The code below stacks the numeric features and runs them through the normalization layer.
#numeric_inputs = {}
#for name in numeric_feature_names:
#  numeric_inputs[name]=inputs[name]

#numeric_inputs = stack_dict(numeric_inputs)  ##gives error because of stack function
#numeric_normalized = normalizer(numeric_inputs)

#preprocessed.append(numeric_normalized)


###########################--Categorical features
#To use categorical features you'll first need to encode them into either binary vectors or embeddings. 

vocab = ['a','b','c']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
#lookup(['c','a','a','b','zzz'])

vocab = [1,4,7,99]
lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
#lookup([-1,4,1])

#To determine the vocabulary for each input, create a layer to convert that vocabulary to a one-hot vector:
for name in categorical_feature_names:
  vocab = sorted(set(df[name]))
  #print(f'name: {name}')
  #print(f'vocab: {vocab}\n')

  if type(vocab[0]) is str:
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
  else:
    lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

  x = inputs[name][:, tf.newaxis]
  x = lookup(x)
  preprocessed.append(x)

preprocessed_result = tf.concat(preprocessed, axis=-1)

preprocessor = tf.keras.Model(inputs, preprocessed_result)
#tf.keras.utils.plot_model(preprocessor, rankdir="LR", show_shapes=True)

preprocessor(dict(df.iloc[:1]))

body = tf.keras.Sequential([ tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)])
x = preprocessor(inputs)
result = body(x)

model = tf.keras.Model(inputs, result)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE)

ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
ds = ds.batch(BATCH_SIZE)

#for x, y in ds.take(1):
#  pprint.pprint(x)
#  print()
#  print(y)

history = model.fit(ds, epochs=5)
