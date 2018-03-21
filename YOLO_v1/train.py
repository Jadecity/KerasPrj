# -*- coding=utf-8 -*-
"""
Train YOLO v1 model using ImageNet-1000.
"""

from YOLO_v1 import yolo_v1
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from keras.callbacks import TensorBoard, ModelCheckpoint
import os

"""
Set train param.
"""
max_epoch = 30
batch_size = 64
momentum = 0.9
weight_decay = 0.0005

"""
Prepare training dataset.
"""
#x_train, y_train =

"""
Prepare training dataset.
"""
#x_val, y_val =


"""
Create YOLO network model
"""
input_imgs = Input(shape=(448, 448, 3))
model = yolo_v1.YOLOV1(input_imgs)

optimizer = Adam()
loss = yolo_v1.loss()
model.compile(optimizer=optimizer, loss=loss)

"""
Using TensorBoard to analyse
"""
tb_callback = TensorBoard(write_graph=True, write_images=False)

"""
Using checkpoint
"""
checkpoint = ModelCheckpoint(os.path.join('./models', 'model-{epoch:02d}.h5'), )

"""
Train model
"""
model.fit(x_train,
          y_train,
          batch_size,
          max_epoch,
          verbose=1,
          validation_data=(x_val, y_val),
          shuffle=True)

"""
Save model
"""
weight_path = './models'
weight_name = 'final_model_weight.h5'
model.save_weights(os.path.join(weight_path, weight_name))