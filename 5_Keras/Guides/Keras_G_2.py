#The Sequential model

import keras
from keras import layers
from keras import ops

#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

# Define Sequential model with 3 layers
model = keras.Sequential(
    [layers.Dense(2, activation="relu", name="layer1"),
    layers.Dense(3, activation="relu", name="layer2"),
    layers.Dense(4, name="layer3"),]
)

x = ops.ones((3, 3)) # Call model on a test input
y = model(x)

#Above is equivalent to this function:

# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = ops.ones((3, 3))
y = layer3(layer2(layer1(x)))


#A Sequential model is not appropriate when:
#Your model has multiple inputs or multiple outputs
#Any of your layers has multiple inputs or multiple outputs
#You need to do layer sharing
#You want non-linear topology (e.g. a residual connection, a multi-branch model)

##############--Creating a Sequential model
#You can create a Sequential model by passing a list of layers to the Sequential constructor

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)

print(model.layers)

#create a Sequential model incrementally via the add() method:
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

model.pop()     #pop() method to remove layers
print(len(model.layers))  # 2



#Specifying the input shape in advance
layer = layers.Dense(3)
print(layer.weights)  # Empty

# Call layer on a test input
x = ops.ones((1, 4))
y = layer(x)
layer.weights  # Now it has weights, of shape (4, 3) and (3,)



#When you instantiate a Sequential model without an input shape, it isn't "built": 
#it has no weights (and calling model.weights results in an error stating just this). 
#The weights are created when the model first sees some input data:

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)  # No weights at this stage!

print(model.weights)
print(model.summary())

# Call the model on a test input
x = ops.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 6


###########---Transfer learning with a Sequential model
#Transfer learning consists of freezing the bottom layers in a model and only training the top layers.
#freeze all layers except the last one. In this case, you would simply iterate over 
#model.layers and set layer.trainable = False on each layer, except the last one.

model = keras.Sequential([
    keras.Input(shape=(250, 250, 3)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

#model.load_weights(...) # Presumably you would want to first load pre-trained weights.

# Freeze all layers except the last one.
for layer in model.layers[:-1]:
  layer.trainable = False

#model.compile(...)
#model.fit(...)


#Another common approach for transfer learning with Sequential model to stack a 
#pre-trained model and some freshly initialized classification layers.

# Load a convolutional base with pre-trained weights
base_model = keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
base_model.trainable = False    ## Freeze the base model

# Use a Sequential model to add a trainable classifier on top
model = keras.Sequential([ base_model, layers.Dense(1000),])




