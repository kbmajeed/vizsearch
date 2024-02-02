import os
import datetime
import logging
import pickle
import warnings

import keras
import numpy as np
import matplotlib.pyplot as plt

from utils import initialize
from etl.data_prep import cat_dogs_dataset, tensor_transform


initialize.load_logging()
config = initialize.load_config()
warnings.filterwarnings("ignore")


def model_infer(index,
                test_model,
                dataset,
                verbose=True,
                show_img=True):
    """
    Utility function to view test image and class prediction
    :test_loader: dataset loader object
    :index: index to inference
    :test_model: trained model
    :img_listing: training dataset
    :return: prediction, test image, and filename
    """
    test_img = np.array([tensor_transform(cat_dogs_dataset[index][0])])
    pred = test_model.predict(test_img)
    pred = pred.flatten()
    pred_name = 'Dog!' if pred > 0.5 else 'Cat!'
    test_name = dataset.img_listing[index]

    if verbose:
        print(f'{pred_name} :: {pred} :: {test_name}')

    if show_img:
        plt.imshow(test_img.squeeze())
        plt.show()

    #model_embedding = keras.Model(inputs=model.input, outputs=model.layers[8].output)
    model_embedding = keras.Model(inputs=model.input, outputs=model.get_layer('Dense_1').output)
    embedding_vector = model_embedding(test_img)

    return (pred, test_img, test_name, embedding_vector)


def model_get_embedding(n_samples,
                        test_model,
                        dataset,
                        verbose=True):

    embedding_matrix = {}
    for index in range(n_samples):
        test_img = np.array([tensor_transform(cat_dogs_dataset[index][0])])
        test_name = dataset.img_listing[index]
        #model_embedding = keras.Model(inputs=test_model.input, outputs=test_model.layers[8].output)
        model_embedding = keras.Model(inputs=test_model.input, outputs=test_model.get_layer('Dense_1').output)
        embedding_vector = model_embedding(test_img)
        embedding_matrix[test_name] = embedding_vector
        logging.info(f"Extracting embedding {index} for : {test_name} => Tensor{embedding_vector.shape}")

    return embedding_matrix


# favorite_color = pickle.load(open("save.p", "rb"))
# # favorite_color is now {"lion": "yellow", "kitty": "red"}


if __name__ == '__main__':
    model = keras.saving.load_model('keras_model.h5', compile=True, safe_mode=True)
    tmp_results = model_infer(np.random.randint(0,24999), model, cat_dogs_dataset)
    embedding_matrix = model_get_embedding(n_samples=25000, test_model=model, dataset=cat_dogs_dataset)
    pickle.dump(embedding_matrix, open('embedding_matrix.emb', 'wb'))
    print(0)