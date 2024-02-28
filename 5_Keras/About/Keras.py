import keras
from keras import layers


model = keras.Sequential()
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


#further configure the compile process
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))

#we dont have data so it will through error
# x_train and y_train are Numpy arrays
#model.fit(x_train, y_train, epochs=5, batch_size=32)
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#classes = model.predict(x_test, batch_size=128)