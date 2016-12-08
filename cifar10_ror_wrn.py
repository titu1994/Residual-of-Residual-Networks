import ror_wrn as ror

import numpy as np
import sklearn.metrics as metrics

import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from keras import backend as K

batch_size = 64
nb_epoch = 300
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX /= 255
testX /= 255

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32)

generator.fit(trainX, seed=0)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

# For RoR-WRN-16-8 put N = 2, k = 8
# For RoR-WRN-28-10 put N = 4, k = 10
# For RoR-WRN-40-2 put N = 6, k = 2
model = ror.create_pre_residual_of_residual(init_shape, nb_classes=10, N=6, k=2, dropout=0.0)

#model.summary()

optimizer = Adam(lr=1e-3)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
print("Finished compiling")
print("Allocating GPU memory")

model.load_weights("weights/RoR-WRN-40-2 Weights.h5")
print("Model loaded.")

# model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
#                    callbacks=[callbacks.ModelCheckpoint("weights/RoR-WRN-40-2 Weights.h5", monitor="val_acc", save_best_only=True)],
#                    validation_data=(testX, testY),
#                    nb_val_samples=testX.shape[0], verbose=2)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = tempY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)