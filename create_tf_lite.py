#!/usr/bin/env python3.6

import argparse
from datetime import date
import os
import sys
import tensorflow as tf
import numpy as np

# import keras
# import keras.preprocessing.image
# import keras.backend as K
# from keras.optimizers import Adam, SGD

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD

from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet
from losses import smooth_l1, focal
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

#for defining the new custom layers
from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization
from initializers import PriorProbability
from losses import *

#from the losses.py
alpha=0.25
gamma=2.0
def _focal(y_true, y_pred):
    """
    Compute the focal loss given the target tensor and the predicted tensor.

    As defined in https://arxiv.org/abs/1708.02002

    Args
        y_true: Tensor of target data from the generator with shape (B, N, num_classes).
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

    Returns
        The focal loss of y_pred w.r.t. y_true.
    """
    labels = y_true[:, :, :-1]
    # -1 for ignore, 0 for background, 1 for object
    anchor_state = y_true[:, :, -1]
    classification = y_pred

    # filter out "ignore" anchors
    indices = tf.where(keras.backend.not_equal(anchor_state, -1))
    labels = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    # compute the focal loss
    alpha_factor = keras.backend.ones_like(labels) * alpha
    alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
    focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(keras.backend.equal(anchor_state, 1))
    normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
    normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

    return keras.backend.sum(cls_loss) / normalizer

    return _focal
sigma_squared = 3.0
def _smooth_l1(y_true, y_pred):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

    Args
        y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
        y_pred: Tensor from the network of shape (B, N, 4).

    Returns
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    # separate target and state
    regression = y_pred
    regression_target = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]

    # filter out "ignore" anchors
    indices = tf.where(keras.backend.equal(anchor_state, 1))
    regression = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    regression_diff = keras.backend.abs(regression_diff)
    regression_loss = tf.where(
        keras.backend.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    # compute the normalizer: the number of positive anchors
    normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
    normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
    return keras.backend.sum(regression_loss) / normalizer






def generator():
    # for item in dataset:
    for i in range(1):
        # image = # get image from dataset item
        # yield [np.array([image.astype(np.float32)])]
        yield [np.zeros((1,512,512,3),dtype=np.float32)] #sizes are from model.py where each phi parameter of the model has different sizes of input

    


phi=0
model, prediction_model = efficientdet(phi, num_classes=3,
                                        weighted_bifpn=True,
                                        freeze_bn=True)


##load the hd5 and save as pb model
snapshot_path="/media/rosu/Data/phd/c_ws/src/mbzirc_2020/EfficientDet/checkpoints/2020-01-06/coco_01_1.0186_0.4586.h5"
# model.load_weights(snapshot_path, by_name=True) #loads the model normally
pb_model_path="/media/rosu/Data/phd/c_ws/src/mbzirc_2020/EfficientDet/dump"
# tf.saved_model.save(model, pb_model_path)


#convert to tflite  https://www.tensorflow.org/lite/guide/get_started#2_convert_the_model_format
# converter = tf.lite.TFLiteConverter.from_saved_model(pb_model_path)
#need custom objects https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/lite/python/lite.py#L560-L613
custom_objects={
    "_focal": _focal,
    "focal": focal,
    "_smooth_l1": _smooth_l1,
    "smooth_l1": smooth_l1,
    "PriorProbability": PriorProbability,
    "ClipBoxes": ClipBoxes,
    "RegressBoxes": RegressBoxes,
    "FilterDetections": FilterDetections,
    "wBiFPNAdd": wBiFPNAdd,
    "BatchNormalization": BatchNormalization,
}
converter = tf.lite.TFLiteConverter.from_keras_model_file(snapshot_path, custom_objects=custom_objects )
# converter = tf.compact.v1.lite.TFLiteConverter.from_keras_model_file(snapshot_path, custom_objects=custom_objects )
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
#QUANTIZE https://www.tensorflow.org/lite/guide/get_started#2_convert_the_model_format
#https://stackoverflow.com/a/59239513
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = tf.lite.RepresentativeDataset(generator)
# converter.experimental_new_converter = True
tflite_quantized_model = converter.convert()
open("converted_model_quantized.tflite", "wb").write(tflite_quantized_model)

#COMPILE TO TPU https://coral.ai/docs/edgetpu/models-intro/#quantization

