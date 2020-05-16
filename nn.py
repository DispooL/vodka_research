import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.minivggnet import MiniVGGNet

print(tf.__version__)

BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "images",
    target_size=(90,90),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    "images",
    target_size=(90,90),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

miniVGG = MiniVGGNet.build(90,90,3)
miniVGG.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=["accuracy"])
miniVGG.fit_generator(generator=train_generator,
                      validation_data=validation_generator,
                      steps_per_epoch=train_generator.n // BATCH_SIZE,
                      validation_steps=validation_generator.n // BATCH_SIZE,
                      epochs=10)