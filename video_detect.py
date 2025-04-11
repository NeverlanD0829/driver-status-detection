# import cv2
# from ultralytics import YOLO

# # 加载YOLOv8模型（你可以替换为自定义模型路径）
# model = YOLO('/home/chen/Desktop/driver-status-detection/YOLOv8/runs/detect/train6/weights/best.pt')  # 例如 'best.pt' 替换为你的模型

# # 打开视频文件 或 摄像头（0为默认摄像头）
# video_path = 'home/chen/video_test.mp4'  # 替换为你的文件路径，或设置为 0 读取摄像头
# cap = cv2.VideoCapture(video_path)

# # 获取视频属性
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps    = int(cap.get(cv2.CAP_PROP_FPS))

# # 可选：保存检测后的视频
# save_path = 'home/chen/output_detected.mp4'
# out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 模型推理（可选设置 conf=0.5）
#     results = model(frame, conf=0.5)

#     # 可视化检测框
#     annotated_frame = results[0].plot()

#     # 显示画面
#     cv2.imshow("YOLOv8 Detection", annotated_frame)

#     # 写入输出视频
#     out.write(annotated_frame)

#     # 按下 q 退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放资源
# cap.release()
# out.release()
# cv2.destroyAllWindows()




import cv2
from ultralytics import YOLO

# 加载模型（可根据自己模型路径修改）
model = YOLO('/home/chen/Desktop/driver-status-detection/YOLOv8/runs/detect/train6/weights/best.pt')  # 或 'best.pt' 等

# 读取视频
video_path = '/home/chen/video_test.mp4'  # 替换成你的视频路径
cap = cv2.VideoCapture(video_path)

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置输出路径和编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
out = cv2.VideoWriter('/home/chen/output_detected.mp4', fourcc, fps, (width, height))

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8 目标检测
    results = model(frame)

    # 绘制检测结果（在 results[0].plot() 中）
    annotated_frame = results[0].plot()

    # 写入到输出视频中
    out.write(annotated_frame)

    frame_idx += 1
    print(f"处理第 {frame_idx} 帧...")

# 释放资源
cap.release()
out.release()
print("检测完成，视频已保存为 output_video.mp4")
