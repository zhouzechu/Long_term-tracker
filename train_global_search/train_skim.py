import tensorflow as tf
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import onnx
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

input_z = tf.keras.Input(shape=(140, 140, 3))
z = tf.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=(140, 140, 3), weights=None)(input_z)
z = tf.keras.layers.AveragePooling2D((4, 4), strides=1)(z)
branch_z = tf.keras.Model(inputs=input_z, outputs=z)  # branch_z是个模型

def tile(embed1):
    embed = tf.keras.backend.tile(embed1, [1, 5, 5, 1])
    return embed

inputs_x = tf.keras.Input(shape=(256, 256, 3))
inputs_z_ = tf.keras.Input(shape=(1, 1, 1024))
z_ = tf.keras.layers.Lambda(tile)(inputs_z_)

x = tf.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=(256, 256, 3), weights=None)(inputs_x)
x = tf.keras.layers.AveragePooling2D((4, 4), strides=1)(x)
x = tf.keras.layers.Multiply()([x, z_])
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # MaxPooling

z_in = tf.keras.layers.Flatten()(inputs_z_)
x = tf.keras.layers.Concatenate()([x, z_in])
x = tf.keras.layers.Dropout(0.5)(x)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(x)

branch_search = tf.keras.Model(inputs=[inputs_z_, inputs_x], outputs=pred)

inputs_1 = tf.keras.Input(shape=(140, 140, 3))
inputs_2 = tf.keras.Input(shape=(256, 256, 3))

output = branch_search([branch_z(inputs_1), inputs_2])

model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=output)

fdata = h5py.File('E:/same_type_negtive_data_large.h5', 'r')
search = fdata['search']
template = fdata['template']
labels = fdata['label']

t_data = fdata['template'][()]
s_data = fdata['search'][()]
label_data = fdata['label'][()]

train_t_data, valid_t_data, train_label,  valid_label = \
    train_test_split(t_data, label_data, test_size=0.1, random_state=1)

train_s_data, valid_s_data, train_label_,  valid_label_ = \
    train_test_split(s_data, label_data, test_size=0.1, random_state=1)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001, decay=1e-6, name='global_dect'),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([train_t_data, train_s_data], train_label, batch_size=32, epochs=15, shuffle=True,
          validation_data=([valid_t_data, valid_s_data], valid_label))

model.save('E:/final_model3.h5', include_optimizer=False)
