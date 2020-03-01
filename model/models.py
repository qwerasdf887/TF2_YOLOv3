import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

#yolov3 一個區塊為 Conv2D -> BN -> leakyReLU
#padding != same : ((1,0),(1,0))
#kernel_regularizer
#use bias = False
def DarknetConv(x, filters, kernel_size, strides=1):
    #在v3中，下採樣採用Conv2D完成，strieds為2，則可形成與maxpooling相同大小feature map
    if strides == 1:
        padding = 'same'
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding, use_bias=False,
                               kernel_regularizer=l2(0.0005))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

#Residual block
#兩層DarknetConv之後與輸入相加
def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = tf.keras.layers.Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

#output layers
#不同層feature之間計算最後的output
def output_layer(x, filters, out_filters, name=None):
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters*2, 3)
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters*2, 3)
    x = DarknetConv(x, filters, 1)
    
    y = DarknetConv(x, filters*2, 3)
    y = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=1, padding='same',
                               kernel_regularizer=l2(0.0005), name=name)(y)
    
    return x, y



def Darknet_body(input_size, num_anchors, num_classes):
    num_anchors = num_anchors // 3
    x = inputs = tf.keras.layers.Input([input_size[0], input_size[1], 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)
    #三個不同尺度輸出，分別用來檢測不同大小的物體
    x = x3 = DarknetBlock(x, 256, 8)
    x = x2 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    
    x, y1 = output_layer(x, 512, num_anchors*(num_classes+5))
    y1 = tf.keras.layers.Reshape((input_size[0]//32, input_size[1]//32, num_anchors,num_classes+5), name='shape//32')(y1)
    
    #最後一層先計算完再往回upsampling
    x = DarknetConv(x, 256, 1)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, x2])
    
    x, y2 = output_layer(x, 256, num_anchors*(num_classes+5))
    y2 = tf.keras.layers.Reshape((input_size[0]//16, input_size[1]//16, num_anchors,num_classes+5), name='shape//16')(y2)
    
    x = DarknetConv(x, 128, 1)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, x3])
    
    x, y3 = output_layer(x, 128, num_anchors*(num_classes+5))
    y3 = tf.keras.layers.Reshape((input_size[0]//8, input_size[1]//8, num_anchors,num_classes+5), name='shape//8')(y3)
    return tf.keras.Model(inputs, [y1, y2, y3])


def DarknetTiny_body(input_size, num_anchors, num_classes):
    num_anchors = num_anchors // 2
    x = inputs = tf.keras.layers.Input([input_size[0], input_size[1], 3])
    x = DarknetConv(x, 16, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = x2 = DarknetConv(x, 256, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = tf.keras.layers.MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 256, 1)
    
    y1 = DarknetConv(x, 512, (3,3))
    y1 = tf.keras.layers.Conv2D(filters=num_anchors*(num_classes+5), kernel_size=1,
                                padding='same', use_bias=True, kernel_regularizer=l2(0.0005))(y1)
    y1 = tf.keras.layers.Reshape((input_size[0]//32, input_size[1]//32, num_anchors,num_classes+5), name='shape//32')(y1)
    
    x = DarknetConv(x, 128, 1)
    x = tf.keras.layers.UpSampling2D(2)(x)
    
    y2 = tf.keras.layers.Concatenate()([x, x2])
    y2 = DarknetConv(y2, 256, (3,3))
    y2 = tf.keras.layers.Conv2D(filters=num_anchors*(num_classes+5), kernel_size=1,
                                padding='same', use_bias=True, kernel_regularizer=l2(0.0005))(y2)
    y2 = tf.keras.layers.Reshape((input_size[0]//16, input_size[1]//16, num_anchors,num_classes+5), name='shape//16')(y2)
    
    
    return tf.keras.Model(inputs, (y1, y2))

