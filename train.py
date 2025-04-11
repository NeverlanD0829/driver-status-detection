from ultralytics import YOLO

if __name__ == '__main__':
    # 从 YAML 创建模型，并加载预训练权重
    model = YOLO("/home/chen/Desktop/driver-status-detection/YOLOv8/ultralytics/cfg/models/v8/yolov8n.yaml")  
    # model = YOLO("/home/chen/Desktop/driver-status-detection/YOLOv8/ultralytics/cfg/models/v8/yolov8n-C2f-FasterBlock.yaml")  
    # model home/chen/Desktop/driver-status-detection/YOLOv8/ultralytics/cfg/models/v8/yolov8n-C2f-FasterBlock-BiFPN.yaml")  
    # model = YOLO("/home/chen/Desktop/driver-status-detection/YOLOv8/ultralytics/cfg/models/v8/yolov8n-C2f-FasterBlock-BiFPN+SE.yaml")  

    # model.load("/home/chen/Desktop/driver-status-detection/YOLOv8/yolov8n.pt")  

    # 训练模型
    model.train(
        data="ultralytics/cfg/datasets/Modified distracted driver dataset.yaml",
        cache=False,
        epochs=100,
        imgsz=640,
        batch=64,
        optimizer='Adam',
        workers=12,
        amp=True,
        project= '/home/chen/Desktop/driver-status-detection/YOLOv8/runs/detect/'
    )





