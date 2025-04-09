from ultralytics import YOLO
 
# 加载模型

model = YOLO('/home/chen/Desktop/yolo-V8/runs/detect/fire/YOLOV8n/weights/best.pt')  # 加载自定义训练模型（示例）
 
# 导出模型
model.export(format='onnx')