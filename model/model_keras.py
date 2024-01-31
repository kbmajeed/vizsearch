import os
import math
import datetime
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt

from utils import initialize
from etl.data_prep import cat_dogs_dataset

from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_images = np.array([img_i[0].numpy().reshape((100, 100, -1)) for img_i, _ in cat_dogs_dataset])
train_labels = np.array([img_l[0].numpy().flatten() for _, img_l in cat_dogs_dataset])


if __name__ == '__main__':
    history = model.fit(train_images, train_labels, epochs=30, batch_size=64, shuffle=True)
    model.save('keras_model.h5')
    model.summary()

    #model.predict(train_images[0].reshape((-1, 50, 50, 3)))
    #plt.imshow(train_images[0].reshape((-1, 100, 100, 3))); plt.show();

    #plt.imshow(cat_dogs_dataset[0][0].permute(1, 2, 0).numpy().reshape((100, 100, 3))); plt.show();
    plt.imshow(train_images[0]); plt.show();


    print(0)