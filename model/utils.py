import tensorflow as tf
import numpy as np
import cv2

def box_iou(box_1, box_2):
    #box_1: (grid, grid, anchor數, (x,y,w,h))
    #box_2: (N, (x,y,w,h))
    
    #擴展後可利用broadcasting機制，並且計算(x1,y1,x2,y2)
    box_1 = tf.expand_dims(box_1, -2)
    box_1_xy = box_1[...,:2]
    box_1_wh = box_1[...,2:4]
    box_1_wh_half = box_1_wh / 2
    box_1_mins = box_1_xy - box_1_wh_half
    box_1_maxes = box_1_xy + box_1_wh_half
    
    
    #擴展後可利用broadcasting機制，同理，算出(x1,y1,x2,y2)
    box_2 = tf.expand_dims(box_2, 0)
    box_2_xy = box_2[...,:2]
    box_2_wh = box_2[...,2:4]
    box_2_wh_half = box_2_wh / 2
    box_2_mins = box_2_xy - box_2_wh_half
    box_2_maxes = box_2_xy + box_2_wh_half
    
    #計算IOU
    intersect_mins = tf.maximum(box_1_mins, box_2_mins)
    intersect_maxes = tf.minimum(box_1_maxes, box_2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_1_area = box_1_wh[..., 0] * box_1_wh[..., 1]
    box_2_area = box_2_wh[..., 0] * box_2_wh[..., 1]
    
    iou = intersect_area / (box_1_area + box_2_area - intersect_area)
    
    #回傳每個gird的每個anchor與y_true的IOU值
    return iou
    
def predict_raw_xywh(y_pred, input_shape, anchors):
    #y_pred:model output, (batch, grid, grid, number of anchors, cls)
    #將model output轉換為每個grid對應的實際數值
    #box_xy: (batch size, grid, grid, number of anchors, (x1,y1))
    #box_wh: (batch size, grid, grid, number of anchors, (w,h))
    #創建gird的 index，也就是論文中的cx,cy
    
    grid_shape = tf.shape(y_pred)[1:3]
    grid = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    grid = tf.cast(grid, tf.float32)
    
    #output的0.1維度取sigmoid + cx,cy，再除以該尺度做歸一化
    box_xy = (tf.sigmoid(y_pred[...,:2]) + grid) / tf.cast(grid_shape[::-1], tf.float32)
    #2.3維度定義為寬高，取exp再乘以anchors得到對應box大小
    box_wh = tf.exp(y_pred[...,2:4]) * anchors
    #4為confidence
    box_confidence = tf.sigmoid(y_pred[...,4:5])
    #5以後為分類機率
    box_class_prob = tf.sigmoid(y_pred[...,5:])
    return box_xy, box_wh, box_confidence, box_class_prob

    
    
def predict_box(y_pred, input_shape, anchors, anchor_mask, score_threshold=.6, iou_threshold=.5):
    #y_pred: list, (yolo_outputs, grid, grid, anchors數, (x,y,w,h,cls..))
    
    box = []
    cls = []
    score = []
    
    for i in range(len(y_pred)):
        grid_shape = tf.shape(y_pred[i])[1:3]
        grid = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        grid = tf.cast(grid, tf.float32)
        
        scale = input_shape / grid_shape
        
        box_xy = (tf.sigmoid(y_pred[i][...,:2]) + grid) * tf.cast(scale[::-1], tf.float32)
        box_wh = tf.exp(y_pred[i][...,2:4]) * anchors[anchor_mask[i]] * tf.cast(input_shape[::-1], tf.float32)
        box_xy_mins = box_xy - (box_wh / 2)
        box_xy_maxes = box_xy + (box_wh / 2)
        
        #形成(y1,x1,y2,x2)格式
        
        boxes = tf.concat([box_xy_mins[...,1:2],
                           box_xy_mins[...,0:1],
                           box_xy_maxes[...,1:2],
                           box_xy_maxes[...,0:1]], axis = -1)
        
        boxes = tf.reshape(boxes, (-1,4))
        
        #confidence
        box_confidence = tf.sigmoid(y_pred[i][...,4:5])
        box_confidence = tf.reshape(box_confidence, (-1,1))
        box_confidence = tf.squeeze(box_confidence, axis=-1)
        
        #class prob
        box_class_prob = tf.nn.softmax(y_pred[i][...,5:])
        box_class_prob = tf.reshape(box_class_prob, (-1,box_class_prob.shape[-1]))
        
        #mask
        mask = ( box_confidence > score_threshold)
        
        box_confidence = tf.boolean_mask(box_confidence, mask)
        box_class_prob = tf.boolean_mask(box_class_prob, mask)
        boxes = tf.boolean_mask(boxes, mask)
        
        box.extend(boxes)
        score.extend(box_confidence)
        cls.extend(box_class_prob)
        
    box = np.array(box)
    score = np.array(score)
    cls = np.array(cls)
    
    if len(box) == 0:
        return box, cls, score
    else:
        nms_index = tf.image.non_max_suppression(box, score, 40, iou_threshold = iou_threshold)
        selected_boxes = tf.gather(box, nms_index)
        selected_cls = tf.math.argmax(tf.gather(cls, nms_index), -1)
        selected_score = tf.gather(score, nms_index)     
        return selected_boxes, selected_cls, selected_score

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

def draw_box(img, box, cls, cls_name, scores, color):
    print('result:')
    for i in range(len(box)):
        cv2.rectangle(img, (box[i,1], box[i,0]), (box[i,3], box[i,2]), color[cls[i]], 2)
        label = cls_name[cls[i]] + ':{0:3.2f}'.format(np.array(scores[i]) * 100)
        cv2.putText(img, label, (box[i,1], box[i,0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color[cls[i]], 2, cv2.LINE_AA)
        print('{}%'.format(label))