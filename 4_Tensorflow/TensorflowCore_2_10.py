#Load and preprocess data

#csv
#https://www.tensorflow.org/tutorials/load_data/csv


#There are two main parts to this:

#Loading the data off disk
#Pre-processing it into a form suitable for training.




import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers




#In memory data
#For any small CSV dataset the simplest way to train a TensorFlow model on it is to load it into memory as a pandas Dataframe or a NumPy array.

#A relatively simple example is the abalone dataset.

#The dataset is small.
#All the input features are all limited-range floating point values.
#Here is how to download the data into a pandas DataFrame:


abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()


abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

#For this dataset you will treat all features identically. Pack the features into a single NumPy array.:
abalone_features = np.array(abalone_features)
abalone_features


#Next make a regression model predict the age. Since there is only a single input tensor, a tf.keras.Sequential model is sufficient here.

abalone_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

abalone_model.fit(abalone_features, abalone_labels, epochs=10)


#Basic preprocessing

#It's good practice to normalize the inputs to your model. The Keras preprocessing layers provide a convenient way to build this normalization into your model.

#The tf.keras.layers.Normalization layer precomputes the mean and variance of each column, and uses these to normalize the data.


normalize = layers.Normalization()

#Then, use the Normalization.adapt method to adapt the normalization layer to your data.


normalize.adapt(abalone_features)

norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                           optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)


#Mixed data types
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

#Because of the different data types and ranges, you can't simply stack the features into a NumPy array and pass it to a tf.keras.Sequential model. Each column needs to be handled individually.

#As one option, you could preprocess your data offline (using any tool you like) to convert categorical columns to numeric columns, then pass the processed output to your TensorFlow model. The disadvantage to that approach is that if you save and export your model the preprocessing is not saved with it. The Keras preprocessing layers avoid this problem because they're part of the model.

#In this example, you'll build a model that implements the preprocessing logic using Keras functional API. You could also do it by subclassing.

#The functional API operates on "symbolic" tensors. Normal "eager" tensors have a value. In contrast these "symbolic" tensors do not. Instead they keep track of which operations are run on them, and build a representation of the calculation, that you can run later. Here's a quick example:


# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# Perform a calculation using the input
result = 2*input + 1

# the result doesn't have a value
result

calc = tf.keras.Model(inputs=input, outputs=result)

print(calc(1).numpy())
print(calc(2).numpy())


inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs

#The first step in your preprocessing logic is to concatenate the numeric inputs together, and run them through a normalization layer:

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs


#Collect all the symbolic preprocessing results, to concatenate them later:

preprocessed_inputs = [all_numeric_inputs]


#For the string inputs use the tf.keras.layers.StringLookup function to map from strings to integer indices in a vocabulary. Next, use tf.keras.layers.CategoryEncoding to convert the indexes into float32 data appropriate for the model.
#The default settings for the tf.keras.layers.CategoryEncoding layer create a one-hot vector for each input. A tf.keras.layers.Embedding would also work. Check out the Working with preprocessing layers guide and the Classify structured data using Keras preprocessing layers tutorial for more on this topic.


for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)


#With the collection of inputs and preprocessed_inputs, you can concatenate all the preprocessed inputs together, and build a model that handles the preprocessing:



preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)


#This model just contains the input preprocessing. You can run it to see what it does to your data. Keras models don't automatically convert pandas DataFrames because it's not clear if it should be converted to one tensor or to a dictionary of tensors. So, convert it to a dictionary of tensors:

titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}

#Slice out the first training example and pass it to this preprocessing model, you see the numeric features and string one-hots all concatenated together:

features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)



#Now, build the model on top of this:

def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
  return model

titanic_model = titanic_model(titanic_preprocessing, inputs)


titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

titanic_model.save('test')
reloaded = tf.keras.models.load_model('test')


features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}

before = titanic_model(features_dict)
after = reloaded(features_dict)
assert (before-after)<1e-3
print(before)
print(after)


'''
Using tf.data
In the previous section you relied on the model's built-in data shuffling and batching while training the model.

If you need more control over the input data pipeline or need to use data that doesn't easily fit into memory: use tf.data.

For more examples, refer to the tf.data: Build TensorFlow input pipelines guide.

On in memory data
As a first example of applying tf.data to CSV data, consider the following code to manually slice up the dictionary of features from the previous section. For each index, it takes that index for each feature:
'''

import itertools

def slices(features):
  for i in itertools.count():
    # For each feature take index `i`
    example = {name:values[i] for name, values in features.items()}
    yield example

for example in slices(titanic_features_dict):
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break



#The most basic tf.data.Dataset in memory data loader is the Dataset.from_tensor_slices constructor. This returns a tf.data.Dataset that implements a generalized version of the above slices function, in TensorFlow.

features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)



for example in features_ds:
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break



#The from_tensor_slices function can handle any structure of nested dictionaries or tuples. The following code makes a dataset of (features_dict, labels) pairs:
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))

#To train a model using this Dataset, you'll need to at least shuffle and batch the data.

titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)

#Instead of passing features and labels to Model.fit, you pass the dataset:

titanic_model.fit(titanic_batches, epochs=5)


#From a single file
#So far this tutorial has worked with in-memory data. tf.data is a highly scalable toolkit for building data pipelines, and provides a few functions for dealing loading CSV files.

titanic_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

#Now read the CSV data from the file and create a tf.data.Dataset.
#(For the full documentation, see tf.data.experimental.make_csv_dataset)


titanic_csv_ds = tf.data.experimental.make_csv_dataset(
    titanic_file_path,
    batch_size=5, # Artificially small to make examples easier to show.
    label_name='survived',
    num_epochs=1,
    ignore_errors=True,)



#This function includes many convenient features, so the data is easy to work with. This includes:

#Using the column headers as dictionary keys.
#Automatically determining the type of each column.


for batch, label in titanic_csv_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value}")
  print()
  print(f"{'label':20s}: {label}")


traffic_volume_csv_gz = tf.keras.utils.get_file(
    'Metro_Interstate_Traffic_Volume.csv.gz', 
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    cache_dir='.', cache_subdir='traffic')

#Set the compression_type argument to read directly from the compressed file:



traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
    traffic_volume_csv_gz,
    batch_size=256,
    label_name='traffic_volume',
    num_epochs=1,
    compression_type="GZIP")

for batch, label in traffic_volume_csv_gz_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value[:5]}")
  print()
  print(f"{'label':20s}: {label[:5]}")





#Caching
#There is some overhead to parsing the CSV data. For small models this can be the bottleneck in training.

#Depending on your use case, it may be a good idea to use Dataset.cache or tf.data.Dataset.snapshot, so that the CSV data is only parsed on the first epoch.

#The main difference between the cache and snapshot methods is that cache files can only be used by the TensorFlow process that created them, but snapshot files can be read by other processes.

#For example, iterating over the traffic_volume_csv_gz_ds 20 times may take around 15 seconds without caching, or about two seconds with caching.




#%%time
for i, (batch, label) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
  if i % 40 == 0:
    print('.', end='')
print()



#%%time
caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)

for i, (batch, label) in enumerate(caching.shuffle(1000).repeat(20)):
  if i % 40 == 0:
    print('.', end='')
print()


#%%time
snapshotting = traffic_volume_csv_gz_ds.snapshot('titanic.tfsnap').shuffle(1000)

for i, (batch, label) in enumerate(snapshotting.shuffle(1000).repeat(20)):
  if i % 40 == 0:
    print('.', end='')
print()



'''
If your data loading is slowed by loading CSV files, and Dataset.cache and tf.data.Dataset.snapshot are insufficient for your use case, consider re-encoding your data into a more streamlined format.

Multiple files
All the examples so far in this section could easily be done without tf.data. One place where tf.data can really simplify things is when dealing with collections of files.

For example, the character font images dataset is distributed as a collection of csv files, one per font.
'''
fonts_zip = tf.keras.utils.get_file(
    'fonts.zip',  "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir='.', cache_subdir='fonts',
    extract=True)


import pathlib
font_csvs =  sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))

font_csvs[:10]

#When dealing with a bunch of files, you can pass a glob-style file_pattern to the tf.data.experimental.make_csv_dataset function. The order of the files is shuffled each iteration.
#Use the num_parallel_reads argument to set how many files are read in parallel and interleaved together.


fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)


#These CSV files have the images flattened out into a single row. The column names are formatted r{row}c{column}. Here's the first batch:


for features in fonts_ds.take(1):
  for i, (name, value) in enumerate(features.items()):
    if i>15:
      break
    print(f"{name:20s}: {value}")
print('...')
print(f"[total: {len(features)} features]")



#Optional: Packing fields
#You probably don't want to work with each pixel in separate columns like this. Before trying to use this dataset be sure to pack the pixels into an image-tensor.

#Here is code that parses the column names to build images for each example:


import re

def make_images(features):
  image = [None]*400
  new_feats = {}

  for name, value in features.items():
    match = re.match('r(\d+)c(\d+)', name)
    if match:
      image[int(match.group(1))*20+int(match.group(2))] = value
    else:
      new_feats[name] = value

  image = tf.stack(image, axis=0)
  image = tf.reshape(image, [20, 20, -1])
  new_feats['image'] = image

  return new_feats



#Apply that function to each batch in the dataset:

fonts_image_ds = fonts_ds.map(make_images)

for features in fonts_image_ds.take(1):
  break

#Plot the resulting images:

from matplotlib import pyplot as plt

plt.figure(figsize=(6,6), dpi=120)

for n in range(9):
  plt.subplot(3,3,n+1)
  plt.imshow(features['image'][..., n])
  plt.title(chr(features['m_label'][n]))
  plt.axis('off')


'''
Lower level functions
So far this tutorial has focused on the highest-level utilities for reading csv data. There are other two APIs that may be helpful for advanced users if your use-case doesn't fit the basic patterns.

tf.io.decode_csv: a function for parsing lines of text into a list of CSV column tensors.
tf.data.experimental.CsvDataset: a lower-level CSV dataset constructor.
This section recreates functionality provided by tf.data.experimental.make_csv_dataset, to demonstrate how this lower-level functionality can be used.

tf.io.decode_csv
This function decodes a string, or list of strings into a list of columns.

Unlike tf.data.experimental.make_csv_dataset this function does not try to guess column data-types. You specify the column types by providing a list of record_defaults containing a value of the correct type, for each column.

To read the Titanic data as strings using tf.io.decode_csv you would say:
'''


text = pathlib.Path(titanic_file_path).read_text()
lines = text.split('\n')[1:-1]

all_strings = [str()]*10
all_strings


features = tf.io.decode_csv(lines, record_defaults=all_strings) 

for f in features:
  print(f"type: {f.dtype.name}, shape: {f.shape}")




#To parse them with their actual types, create a list of record_defaults of the corresponding types:


print(lines[0])

titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
titanic_types



features = tf.io.decode_csv(lines, record_defaults=titanic_types) 

for f in features:
  print(f"type: {f.dtype.name}, shape: {f.shape}")


'''
tf.data.experimental.CsvDataset
The tf.data.experimental.CsvDataset class provides a minimal CSV Dataset interface without the convenience features of the tf.data.experimental.make_csv_dataset function: column header parsing, column type-inference, automatic shuffling, file interleaving.

This constructor uses record_defaults the same way as tf.io.decode_csv:
'''

simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)

for example in simple_titanic.take(1):
  print([e.numpy() for e in example])




def decode_titanic_line(line):
  return tf.io.decode_csv(line, titanic_types)

manual_titanic = (
    # Load the lines of text
    tf.data.TextLineDataset(titanic_file_path)
    # Skip the header row.
    .skip(1)
    # Decode the line.
    .map(decode_titanic_line)
)

for example in manual_titanic.take(1):
  print([e.numpy() for e in example])




'''
Multiple files
To parse the fonts dataset using tf.data.experimental.CsvDataset, you first need to determine the column types for the record_defaults. Start by inspecting the first row of one file:
'''

font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)


#Only the first two fields are strings, the rest are integers or floats, and you can get the total number of features by counting the commas:



num_font_features = font_line.count(',')+1
font_column_types = [str(), str()] + [float()]*(num_font_features-2)


#The tf.data.experimental.CsvDataset constructor can take a list of input files, but reads them sequentially. The first file in the list of CSVs is AGENCY.csv:

font_csvs[0]

#So, when you pass the list of files to CsvDataset, the records from AGENCY.csv are read first:

simple_font_ds = tf.data.experimental.CsvDataset(
    font_csvs, 
    record_defaults=font_column_types, 
    header=True)

for row in simple_font_ds.take(10):
  print(row[0].numpy())




#To interleave multiple files, use Dataset.interleave.

#Here's an initial dataset that contains the CSV file names:


font_files = tf.data.Dataset.list_files("fonts/*.csv")


#This shuffles the file names each epoch:

print('Epoch 1:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')
print()

print('Epoch 2:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')





#The interleave method takes a map_func that creates a child-Dataset for each element of the parent-Dataset.

#Here, you want to create a tf.data.experimental.CsvDataset from each element of the dataset of files:


def make_font_csv_ds(path):
  return tf.data.experimental.CsvDataset(
    path, 
    record_defaults=font_column_types, 
    header=True)

#
#The Dataset returned by interleave returns elements by cycling over a number of the child-Datasets. Note, below, how the dataset cycles over cycle_length=3 three font files:


font_rows = font_files.interleave(make_font_csv_ds,
                                  cycle_length=3)

fonts_dict = {'font_name':[], 'character':[]}

for row in font_rows.take(10):
  fonts_dict['font_name'].append(row[0].numpy().decode())
  fonts_dict['character'].append(chr(row[2].numpy()))

pd.DataFrame(fonts_dict)




#Performance
#Earlier, it was noted that tf.io.decode_csv is more efficient when run on a batch of strings.

#It is possible to take advantage of this fact, when using large batch sizes, to improve CSV loading performance (but try caching first).

#With the built-in loader 20, 2048-example batches take about 17s.


BATCH_SIZE=2048
fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=BATCH_SIZE, num_epochs=1,
    num_parallel_reads=100)

#%%time
for i,batch in enumerate(fonts_ds.take(20)):
  print('.',end='')

print()



#Passing batches of text lines todecode_csv runs faster, in about 5s:


fonts_files = tf.data.Dataset.list_files("fonts/*.csv")
fonts_lines = fonts_files.interleave(
    lambda fname:tf.data.TextLineDataset(fname).skip(1), 
    cycle_length=100).batch(BATCH_SIZE)

fonts_fast = fonts_lines.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))


%%time
for i,batch in enumerate(fonts_fast.take(20)):
  print('.',end='')

print()

#For another example of increasing CSV performance by using large batches, refer to the Overfit and underfit tutorial.

#This sort of approach may work, but consider other options like Dataset.cache and tf.data.Dataset.snapshot, or re-encoding your data into a more streamlined format.











