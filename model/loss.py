import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from .utils import box_iou, predict_raw_xywh

#loss 計算
def YoLoLoss(input_shape, anchors, classes=4, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        #y_pred: 該層的model output(batch_size, grid, grid,  anchor數, (5 + cls))
        #y_true: (batch_size, grid, grid, anchor數, (5+cls))
        batch_size = tf.shape(y_pred)[0]
        bf = tf.cast(batch_size, tf.float32)
        grid_shape = tf.shape(y_pred)[1:3]
        pred_xy, pred_wh, pred_cf, pred_cls = predict_raw_xywh(y_pred, input_shape, anchors)
        
        box_loss_scale = 2 - y_true[...,2:3]*y_true[...,3:4]
        object_mask = y_true[...,4:5]
        #對於每張圖輸出ignore mask
        ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        def loop_box(index, ignore_mask):
            #從y_true中找出有值得列
            #shape: (N, (x,y,w,h))
            true_box = tf.boolean_mask(y_true[index,...,0:4], tf.squeeze(object_mask[index],-1))
            #每個grid與true_box的IOU，shape:(gird,grid,anchors數,trub_box數)
            iou = box_iou(tf.concat((pred_xy[index], pred_wh[index]), axis=-1), true_box)
            #每個anchors最大的IOU，shape:(grid,grid,anchor數)
            best_iou = tf.reduce_max(iou, axis=-1)
            #根據ignore_thresh添加ignore mask
            ignore_mask = ignore_mask.write(index, tf.cast(best_iou<ignore_thresh, tf.float32))
            return index+1, ignore_mask    
        _, ignore_mask = tf.while_loop(lambda index,*args: index < batch_size, loop_box, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
    
    
        #loss
        xy_loss = object_mask * box_loss_scale * tf.square(y_true[...,0:2] - pred_xy)
        xy_loss = tf.reduce_sum(xy_loss) / bf
    
        wh_loss = object_mask * box_loss_scale * tf.square(y_true[...,2:4] - pred_wh)
        wh_loss = tf.reduce_sum(wh_loss) / bf
        
        
        #print(tf.boolean_mask(y_true[...,0:4], tf.squeeze(object_mask,-1)))
        
        #obj loss分為兩種
        #是obj，計算confidence loss
        #不是obj，如果與GT的 IOU > thread，則不計算損失，以一個ignore mask計算
        
        cf_loss = tf.squeeze(object_mask,-1) * binary_crossentropy(y_true[...,4:5], pred_cf)
        cf_loss = cf_loss + tf.squeeze((1 - object_mask),-1) * binary_crossentropy(y_true[...,4:5], pred_cf) * ignore_mask
        cf_loss = tf.reduce_sum(cf_loss) / bf
    
        cls_loss = tf.squeeze(object_mask,-1) * binary_crossentropy(y_true[...,5:],pred_cls)
        cls_loss = tf.reduce_sum(cls_loss) / bf
        return xy_loss + wh_loss + cf_loss + cls_loss
    return yolo_loss