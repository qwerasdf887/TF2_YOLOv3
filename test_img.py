import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import cv2
from model.models import Darknet_body, DarknetTiny_body
from model.utils import predict_box, draw_box, get_classes, get_anchors

#創建每個類別不同顏色
def color_list(num_classes):
    hsv_range = np.linspace(0, 180, num_classes, endpoint=False, dtype=np.uint8)
    hsv_list = np.ones((num_classes, 1, 3), dtype=np.uint8) * 255
    
    hsv_list[:,:,0] = np.expand_dims(hsv_range, axis=1)
    bgr_list = cv2.cvtColor(hsv_list, cv2.COLOR_HSV2BGR)
    bgr_list = np.reshape(bgr_list, (-1,3))
    
    return bgr_list.tolist()

#計算等比縮放置416*416的大小
def resize_img(ori_img, input_size):
    h, w, _ = ori_img.shape
    scale = min( input_size[0] / h, input_size[1] / w)
    
    h = int(h * scale)
    w = int(w * scale)
    
    img = cv2.resize(ori_img, (w,h))
    
    return_img = np.ones((input_size[0], input_size[1], 3), np.uint8) * 127
    return_img[:h,:w,:] = img
    
    return return_img

#還原原圖座標
def reduction_img(ori_img, input_size, loc):
    h, w, _ = ori_img.shape
    scale = min( input_size[0] / h, input_size[1] / w)
    
    loc = loc / scale
    
    return loc
    
if __name__ == '__main__':
    
    input_shape = (416,416)
    classes_path = './data/coco_classes.txt'
    class_name = get_classes(classes_path)
    num_classes = len(class_name)
    is_tiny = False
    color = color_list(num_classes)
    
    if is_tiny:
        anchors_path = './data/tiny_anchors.txt'
        anchors = get_anchors(anchors_path) / input_shape[::-1]
        num_anchors = len(anchors)
        anchor_mask = [[3,4,5], [0,1,2]]
        model = DarknetTiny_body(input_shape, num_anchors, num_classes)
        model.load_weights('./tiny.h5', by_name=True)
    else:
        anchors_path = './data/yolo_anchors.txt'
        anchors = get_anchors(anchors_path) / input_shape[::-1]
        num_anchors = len(anchors)
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        model = Darknet_body(input_shape, num_anchors, num_classes)
        model.load_weights('./yolo.h5', by_name=True)
        
    model.summary()
    
    img = cv2.imread('./test.jpg')
    res_img = resize_img(img, input_shape)
    pred_img = (np.expand_dims(res_img, axis=0) / 255).astype(np.float32)
    result = model.predict(pred_img)
    boxes, cls, score = predict_box(result, input_shape, anchors, anchor_mask)
    boxes = reduction_img(img, input_shape, boxes)
    draw_box(img, boxes, cls, class_name,score, color)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    