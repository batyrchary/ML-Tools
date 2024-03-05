#Making new layers and models via subclassing

import numpy as np
import keras
from keras import ops
from keras import layers


#A layer encapsulates both a state (the layer's "weights") and a transformation from 
#inputs to outputs (a "call", the layer's forward pass).
#Here's a densely-connected layer. It has two state variables: the variables w and b.

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b


linear_layer = Linear(4, 2)
x = ops.ones((2, 2))
y = linear_layer(x)
print(y)



#Layers can have non-trainable weights:Such weights are meant not to be taken into account during backpropagation, when you are training the layer.
class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.total = self.add_weight(
            initializer="zeros", shape=(input_dim,), trainable=False)

    def call(self, inputs):
        self.total.assign_add(ops.sum(inputs, axis=0))
        return self.total


x = ops.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())

print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))
print("trainable_weights:", my_sum.trainable_weights)   # It's not included in the trainable weights:



#Best practice: deferring weight creation until the shape of the inputs is known
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer="random_normal",
            trainable=True,)

        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

class Linear(keras.layers.Layer):
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


linear_layer = Linear(32)   # At instantiation, we don't know on what inputs this is going to get called
y = linear_layer(x)         # The layer's weights are created dynamically the first time the layer is called




#Layers are recursively composable
#If you assign a Layer instance as an attribute of another Layer, the outer layer will start tracking the weights created by the inner layer.

class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = keras.activations.relu(x)
        x = self.linear_2(x)
        x = keras.activations.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(ops.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))



#The add_loss() method
#When writing the call() method of a layer, you can create loss tensors that you will want to use later, 
#when writing your training loop. This is doable by calling self.add_loss(value)

# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * ops.mean(inputs))
        return inputs


#These losses (including those created by any inner layer) can be retrieved via layer.losses. 
#This property is reset at the start of every __call__() to the top-level layer, so that layer.losses 
#always contains the loss values created during the last forward pass.
class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # No losses yet since the layer has never been called

_ = layer(ops.zeros((1, 1)))
assert len(layer.losses) == 1  # We created one loss value

# `layer.losses` gets reset at the start of each __call__
_ = layer(ops.zeros((1, 1)))
assert len(layer.losses) == 1  # This is the loss created during the call above



class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(1e-3))

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayerWithKernelRegularizer()
_ = layer(ops.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel ** 2)`,created by the `kernel_regularizer` above.
print(layer.losses)


inputs = keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = keras.Model(inputs, outputs)

# If there is a loss passed in `compile`, the regularization
# losses get added to it
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# It's also possible not to pass any loss in `compile`,
# since the model already has a loss to minimize, via the `add_loss`
# call during the forward pass!
model.compile(optimizer="adam")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))


#If you need your custom layers to be serializable as part of a Functional model, you can optionally implement a get_config() method:
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


# Now you can recreate the layer from its config:
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)


#Note that the __init__() method of the base Layer class takes some keyword arguments, in particular a name and a dtype. It's good practice to pass these arguments to the parent class in __init__() and to include them in the layer config:
class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)


####################----End-to-end example
#Here's what you've learned so far:
#A Layer encapsulate a state (created in __init__() or build()) and some computation (defined in call()).
#Layers can be recursively nested to create new, bigger computation blocks.
#Layers are backend-agnostic as long as they only use Keras APIs. You can use backend-native APIs (such as jax.numpy, torch.nn or tf.nn), but then your layer will only be usable with that specific backend.
#Layers can create and track losses (typically regularization losses) via add_loss().
#The outer container, the thing you want to train, is a Model. A Model is just like a Layer, but with added training and serialization utilities.

#Let's put all of these things together into an end-to-end example: we're going to implement a Variational AutoEncoder (VAE) in a backend-agnostic fashion – so that it runs the same with TensorFlow, JAX, and PyTorch. We'll train it on MNIST digits.
#Our VAE will be a subclass of Model, built as a nested composition of layers that subclass Layer. It will feature a regularization loss (KL divergence).

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * ops.mean(z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

original_dim = 784
vae = VariationalAutoEncoder(784, 64, 32)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=2, batch_size=64)


