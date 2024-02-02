import os
import datetime
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt

from utils import initialize
from etl.data_prep import cat_dogs_dataset, tensor_transform

from keras import layers
from keras import models
from keras import optimizers


initialize.load_logging()
config = initialize.load_config()
warnings.filterwarnings("ignore")


img_input_dim = (config.etl.image_resize, config.etl.image_resize, 3)

model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=img_input_dim, name='Conv2D_1'))
model.add(layers.MaxPooling2D((2, 2), name='MaxPooling2D_1'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', name='Conv2D_2'))
# model.add(layers.MaxPooling2D((2, 2), name='MaxPooling2D_2'))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', name='Conv2D_3'))
# model.add(layers.MaxPooling2D((2, 2), name='MaxPooling2D_3'))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', name='Conv2D_4'))
# model.add(layers.MaxPooling2D((2, 2), name='MaxPooling2D_4'))
model.add(layers.Flatten(name='Flatten_1'))
model.add(layers.Dense(16, activation='relu', name='Dense_1'))
model.add(layers.Dense(1, activation='sigmoid', name='Dense_2'))

model.compile(optimizer=optimizers.legacy.RMSprop(learning_rate=config.keras_model.learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
logging.info('Keras CNN model built and compiled')

train_images = np.array([tensor_transform(img_i) for img_i, _ in cat_dogs_dataset])
train_labels = np.array([img_l.numpy().flatten() for _, img_l in cat_dogs_dataset])
logging.info('Training images and labels extracted for model fitting')


def model_infer(test_loader, index, test_model, dataset):
    """
    Utility function to view test image and class prediction
    :test_loader: dataset loader object
    :index: index to inference
    :test_model: trained model
    :img_listing: training dataset
    :return: prediction, test image, and filename
    """
    test_img = test_loader[index].reshape((-1, config.etl.image_resize, config.etl.image_resize, 3))
    pred = test_model.predict(test_img)
    pred = pred.flatten()
    test_name = dataset.img_listing[index]
    print(pred)
    print('Dog!') if pred > 0.5 else print('Cat!')
    print(test_name)
    plt.imshow(test_img.squeeze())
    plt.show()

    #TODO: return embeddings as well
    return pred, test_img, test_name


if __name__ == '__main__':

    # Train Model
    logging.info(f'Model training started: START : {datetime.datetime.now()}')
    history = model.fit(train_images,
                        train_labels,
                        epochs=config.keras_model.epochs,
                        batch_size=config.keras_model.batch_size,
                        shuffle=True)
    logging.info(f'Model training ended: END : {datetime.datetime.now()}')
    model.save('keras_model.h5')
    logging.info(f'Model saved to location: DIR : {os.getcwd()}')
    model.summary()

    # Test Model
    model_infer(train_images, 0, model, cat_dogs_dataset)
    print(0)