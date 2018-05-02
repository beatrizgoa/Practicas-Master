__author__ = 'Bea'
import cv2
import numpy
import cPickle, gzip, numpy

# Open the images with gzip in read binary mode
images = gzip.open('train_images.gz', 'rb')
labels = gzip.open('train_labels.gz', 'rb')

input_data=numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]], numpy.float32)
output_data=numpy.array([[-1], [-1], [1], [1]], numpy.float32)

layers = numpy.array([100, 4, 4, 10])

layers = numpy.array([2, 4, 4, 1])
nnet = cv2.ml.ANN_MLP_create()
nnet.setLayerSizes(layers)
nnet.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
nnet.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
nnet.setTermCriteria((cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,1000,0.001))

nnet.train(input_data, cv2.ml.ROW_SAMPLE, output_data)

la, prediction =nnet.predict(input_data)

print prediction