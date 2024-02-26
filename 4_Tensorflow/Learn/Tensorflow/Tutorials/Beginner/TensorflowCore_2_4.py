#ML Basics with Keras

#Basic regression: Predict fuel efficiency 
#https://www.tensorflow.org/tutorials/keras/regression

#Uses seaborn for pairplot.
#pip install -q seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

########################--Get The Data and Clean
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
#print(dataset.tail())

#Clean the data
dataset.isna().sum()
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
#print(dataset.tail())


########################--Split into Train and Test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Inspect data
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
#print(train_dataset.describe().transpose())


train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#Normalization
#print(train_dataset.describe().transpose()[['mean', 'std']])
normalizer = tf.keras.layers.Normalization(axis=-1)
#normalizer.adapt(np.array(train_features)) #Gives an error --ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
#print(normalizer.mean.numpy())



#When the layer is called, it returns the input data, with each feature independently normalized
#first = np.array(train_features[:1])
#with np.printoptions(precision=2, suppress=True):
#  print('First example:', first)
#  print()
#  print('Normalized:', normalizer(first).numpy()) #Gives an error --ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).



horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

#Build the Keras Sequential model
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

print(horsepower_model.summary())
print(horsepower_model.predict(horsepower[:10]))
#Compile
horsepower_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

#Use Keras Model.fit to execute the training for 100 epochs
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,  # Suppress logging.
    validation_split = 0.2)   # Calculate validation results on 20% of the training data.


#Visualize the model's training progress using the stats stored in the history object
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()
#plot_loss(history)


test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.show()
#plot_horsepower(x, y)


#TODO
#########################--Linear regression with multiple inputs
linear_model = tf.keras.Sequential([normalizer,layers.Dense(units=1)])
#print(linear_model.predict(train_features[:10])) #--Failed to convert a NumPy array to a Tensor (Unsupported object type int).
#print(linear_model.layers[1].kernel)

linear_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error')

#--Failed to convert a NumPy array to a Tensor (Unsupported object type int).
#history = linear_model.fit(
#    train_features,
#    train_labels,
#    epochs=100,
#    verbose=0,
#    validation_split = 0.2)

#plot_loss(history)
#test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=0)




#########################--Regression with a deep neural network (DNN)

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
print(dnn_horsepower_model.summary())

history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
#plot_loss(history)
#plot_horsepower(x, y)

#--TODO
'''
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(test_features['Horsepower'], test_labels,verbose=0)
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

#Performance
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')

dnn_model.save('dnn_model.keras')

#then reload if needed
reloaded = tf.keras.models.load_model('dnn_model.keras')
test_results['reloaded'] = reloaded.evaluate(test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
'''
