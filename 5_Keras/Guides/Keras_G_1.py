#The Functional API

import numpy as np
import keras
from keras import layers
from keras import ops
#import tensorflow as tf


#The Keras functional API is a way to create models that are more 
#flexible than the keras.Sequential API. The functional API can handle models 
#with non-linear topology, shared layers, and even multiple inputs or outputs.

#Consider the following model:
#(input: 784-dimensional vectors)
#[Dense (64 units, relu activation)]
#[Dense (64 units, relu activation)]
#[Dense (10 units, softmax activation)]
#(output: logits of a probability distribution over 10 classes)

#Lets build this
inputs = keras.Input(shape=(784,)) #The shape of the data is set as a 784-dimensional vector
dense = layers.Dense(64, activation="relu")
x = dense(inputs) #The "layer call" action is like drawing an arrow from "inputs" to this layer you created. You're "passing" the inputs to the dense layer, and you get x as the output.
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x) 	#At this point, you can create a Model by specifying its inputs and outputs

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
print(model.summary())

#You can also plot the model as a graph:
#keras.utils.plot_model(model, "my_first_model.png") #couldnt install library


##############--Training, evaluation, and inference

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


##############--Save and serialize
model.save("./models/my_model.keras")
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model("./models/my_model.keras")

##############--Use the same graph of layers to define multiple models
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
print(encoder.summary())

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
print(autoencoder.summary())



##############--All models are callable, just like layers
#You can treat any model as if it were a layer by invoking it on an Input or on the output of another layer. 

encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
print(encoder.summary())

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
print(autoencoder.summary())

#As you can see, the model can be nested: a model can contain sub-models (since a model is just like a layer). A common use case for model nesting is ensembling.
#Ensembling
def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)


##############--Manipulate complex graph topologies

#building a system for ranking customer issue tickets by priority and 
#routing them to the correct department, then the model will have three 
#inputs:
#the title of the ticket (text input),
#the text body of the ticket (text input), and
#any tags added by the user (categorical input)
#This model will have two outputs:
#the priority score between 0 and 1 (scalar sigmoid output), and
#the department that should handle the ticket (softmax output over the set of departments).

num_tags = 12  			# Number of unique issue tags
num_words = 10000  		# Size of vocabulary obtained when preprocessing text data
num_departments = 4  	# Number of departments for predictions

title_input = keras.Input(shape=(None,), name="title")  	# Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  		# Variable-length sequence of ints
tags_input = keras.Input(shape=(num_tags,), name="tags")  	# Binary vectors of size `num_tags`

title_features = layers.Embedding(num_words, 64)(title_input)	# Embed each word in the title into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)		# Embed each word in the text into a 64-dimensional vector

title_features = layers.LSTM(128)(title_features)		# Reduce sequence of embedded words in the title into a single 128-dimensional vector
body_features = layers.LSTM(32)(body_features)			# Reduce sequence of embedded words in the body into a single 32-dimensional vector

x = layers.concatenate([title_features, body_features, tags_input])		# Merge all available features into a single large vector via concatenation

priority_pred = layers.Dense(1, name="priority")(x)						# Stick a logistic regression for priority prediction on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)	# Stick a department classifier on top of the features

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs={"priority": priority_pred, "department": department_pred},)


#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),],
    loss_weights=[1.0, 0.2],)


#Since the output layers have different names, you could also specify the losses and loss weights with the corresponding layer names
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={"priority": keras.losses.BinaryCrossentropy(from_logits=True),
    "department": keras.losses.CategoricalCrossentropy(from_logits=True),},
    loss_weights={"priority": 1.0, "department": 0.2},)

# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,batch_size=32,)



###############--A toy ResNet model
inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
print(model.summary())


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],)



model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2,)


###############--Shared layers
#Shared layers are layer instances that are reused multiple times in the same model – 
#they learn features that correspond to multiple paths in the graph-of-layers.

#Shared layers are often used to encode inputs from similar spaces 
#(say, two different pieces of text that feature similar vocabulary). 
#They enable sharing of information across these different inputs, and 
#they make it possible to train such a model on less data. 
#If a given word is seen in one of the inputs, that will benefit the processing 
#of all inputs that pass through the shared layer.

#Embedding layer shared across two different text inputs
shared_embedding = layers.Embedding(1000, 128)				# Embedding for 1000 unique words mapped to 128-dimensional vectors
text_input_a = keras.Input(shape=(None,), dtype="int32")	# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype="int32")	# Variable-length sequence of integers

encoded_input_a = shared_embedding(text_input_a)			# Reuse the same layer to encode both inputs
encoded_input_b = shared_embedding(text_input_b)


###############--Extend the API using custom layers
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)

        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)


class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)

        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
new_model = keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})



#RNN, written from scratch, being used in a functional model:
units = 32
timesteps = 10
input_dim = 5
batch_size = 16


class CustomRNN(layers.Layer):
    def __init__(self):
        super().__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        self.classifier = layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = ops.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = ops.stack(outputs, axis=1)
        return self.classifier(features)


# Note that you specify a static batch size for the inputs with the `batch_shape`
# arg, because the inner computation of `CustomRNN` requires a static batch size
# (when you create the `state` zeros tensor).
inputs = keras.Input(batch_shape=(batch_size, timesteps, input_dim))
x = layers.Conv1D(32, 3)(inputs)
outputs = CustomRNN()(x)

model = keras.Model(inputs, outputs)

rnn_model = CustomRNN()
_ = rnn_model(ops.zeros((1, 10, 5)))


