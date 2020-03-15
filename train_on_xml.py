
# coding: utf-8

# In[1]:

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import cv2
from model.models import Darknet_body, DarknetTiny_body
from model.loss import YoLoLoss
from model.data_aug import get_random_data
from model.load_xml_data import load_data, preprocess_true_boxes
import os


# In[2]:

def main():
    #input shape要為32的倍數，因為5次downsampling
    input_shape = (416, 416)
    annotation_path = 'annotation path'
    log_dir = 'save model path'
    classes_path = 'classes path'
    anchors_path = 'anchors path'
    class_name = get_classes(classes_path)
    #class數量
    num_classes = len(class_name)
    anchors = get_anchors(anchors_path) / input_shape[::-1]
    num_anchors = len(anchors)
    is_tiny_version = True
    batch_size = 40

    #creat model
    if is_tiny_version:
        model = DarknetTiny_body(input_shape, num_anchors, num_classes)
        anchor_mask = [[3,4,5], [0,1,2]]
        
    else:
        model = Darknet_body(input_shape, num_anchors, num_classes)
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        
    model.summary()
    
    xmls = os.listdir(annotation_path)
    total_train = len(xmls)
    
    print('train data:', total_train)
    print('anchors:', anchors)
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath= log_dir + '/best_loss.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    verbose=1)]
    
    loss = [YoLoLoss(input_shape, anchors[mask], classes=num_classes) for mask in anchor_mask]
    
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=loss)
    
    model.fit(img_generator(xmls, annotation_path, batch_size, input_shape, anchors, anchor_mask, num_classes, is_tiny_version),
              steps_per_epoch = total_train//batch_size, callbacks=callbacks, epochs=200)

    model.save_weights(log_dir + '/finall.h5')

#讀取類別，回傳類別List
def get_classes(classes_path):
    with open(classes_path) as f:
        class_name = f.readlines()
    class_name = [c.strip() for c in class_name]
    return class_name

#讀取anchors
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    #str to float
    anchors = [float(x) for x in anchors.split(',')]
    #回傳[[w1,h1],....,[wn,hn]]形式
    return np.array(anchors).reshape(-1,2)

def img_generator(xml_name, ann_path, batch_size, input_shape, anchors, anchor_mask, num_classes, is_tiny_version):
    n = len(xml_name)
    i = 0
    while True:
        image_data = []
        box_data = []
        for r in range(batch_size):
            if i==0:
                np.random.shuffle(xml_name)
            image, box = load_data(ann_path + xml_name[i])
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n

        image_data = (np.array(image_data) / 255).astype(np.float32)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, anchor_mask, num_classes, tiny=is_tiny_version)
        
        yield (image_data, y_true)


# In[6]:

main()



