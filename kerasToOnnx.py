# from keras.models import load_model
# import onnx
# import keras2onnx
#
# model = load_model(r'C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5')
#
# onnx_model_name = 'fish-resnet50.onnx'
#
# onnx_model = keras2onnx.convert_keras(model, model.name)
# onnx.save_model(onnx_model, onnx_model_name)
import os

import tf2onnx
from keras.models import load_model


model = load_model(r'C:\\Users\\Admin\\Desktop\\Кандидат\\neural_network\\model_best_best (2).h5')
(onnx_model_proto, storage) = tf2onnx.convert.from_keras(model)
with open("C:\\Users\\Admin\\Desktop\\Кандидат\\neural_network\\modelData5.onnx", "wb") as f:
    f.write(onnx_model_proto.SerializeToString())
# onnx_model = keras2onnx.convert_keras(model, model.name)
#
# file = open("C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\modelSample_model.onnx", "wb")
# file.write(onnx_model.SerializeToString())
# file.close()


#AttributeError: module 'tensorflow.python.keras' has no attribute 'applications'