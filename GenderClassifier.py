import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

class GenderClassifier(object):

    def __init__(self,img_width,img_height):
        self.img_width = img_width
        self.img_height = img_height

    def create_model(self):
        network = input_data(shape=[None, self.img_width, self.img_height,1])

        network = conv_2d(network, 64, 3, strides=2, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = conv_2d(network, 64, 3, strides=2, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = fully_connected(network, 128, activation='relu')
        network = fully_connected(network, 2, activation='softmax')
        model = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)

        return model

    def get_model(self):
        model = self.create_model()
        model = tflearn.DNN(model)
        model.load('models/tflearn_genderClassification_model.model')

        return model
