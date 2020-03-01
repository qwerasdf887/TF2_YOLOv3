# TF2_YOLOv3
Tensorflow 2.1版的YOLOv3，參考許多前人的寫法並且修改一些個人感覺奇怪的部分。
權重轉換的converty.py由[qqwweee](https://github.com/qqwweee/keras-yolo3) 修改而來。
裡面包含一些詳細註解。

## References:

[qqwweee](https://github.com/qqwweee/keras-yolo3)

[zzh8829](https://github.com/zzh8829/yolov3-tf2)

[YunYang1994](https://github.com/YunYang1994/tensorflow-yolov3)

## 環境:

1. Tensorflow 2.1
2. Python 3.5~3.7
3. OpenCV 3~4

## 使用訓練好的weights

至[YOLO](https://pjreddie.com/darknet/yolo/) 下載對應的權重，並且使用converty.py轉換為.h5檔案。

```bashrc
python convert.py yolov3.cfg yolov3.weights yolo.h5

python python test_img.py
```

則可看到下圖結果

<p align="center">
    <img width="100%" src="https://github.com/qwerasdf887/TF2_YOLOv3/blob/master/result.jpg?raw=true" style="max-width:100%;">
    </a>
</p>

## training

需修改train.py。
使用[labelImg](https://github.com/tzutalin/labelImg) 生成的xml檔即可。
如要使用YOLOv3提供的weights，類別又不同時，在convert時，要先修改yolo.cfg相關層。