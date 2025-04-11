import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/chen/Desktop/driver-status-detection/YOLOv8/runs/detect/train6/weights/best.pt') # select your model.pt path
    model.predict(source='/home/chen/Downloads/Modified distracted driver dataset.v2-augmented-dataset.yolov8-obb/test/images',
                  imgsz=640,
                  project='/home/chen/Desktop/driver-status-detection/YOLOv8/runs/',
                  name='预测结果',
                  save=True,
                  # classes=0, 是否指定检测某个类别.
                )
