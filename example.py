#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-11 of 2020
"""
import numpy as np
import keras

import efficientnet.keras as efn

# For activation definition
from keras.backend import sigmoid
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Swish(swish)})

if __name__ == '__main__':
    model = keras.models.load_model('./EfficientCovidNetB0-500.hdf5')
    img = np.zeros(shape=(1, 500, 500, 3))
    prediction = model.predict(img)
