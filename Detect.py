import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('/home/chen/Desktop/yolo-V8/runs/detect/v8s-p2/weights/best.pt') # select your model.pt path
    model = YOLO('/home/chen/Desktop/yolo-V8/runs/detect/fire/YOLOV8n/weights/best.pt') # select your model.pt path
    model.predict(source='/home/chen/Desktop/Fire_Smoke/fire_smoke_datasets/images/test',
                  imgsz=640,
                  project='/home/chen/Desktop/yolo-V8/runs/detect/fire/YOLOV8n/',
                  name='预测结果',
                  save=True,
                  # classes=0, 是否指定检测某个类别.
                )
