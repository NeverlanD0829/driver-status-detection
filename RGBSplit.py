

# 读取RGB图片
image_path = '/home/chen/Desktop/IMG_20230716_130333.jpg'  # 替换为你的图片路径



import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取RGB图片
# image_path = 'input_image.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)

# 转换为RGB格式（OpenCV 默认使用BGR格式）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 分离RGB通道
r, g, b = cv2.split(image_rgb)

# 创建全零通道的图像（大小和原图一样）
r_image = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)])  # 红色通道
g_image = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])  # 绿色通道
b_image = cv2.merge([np.zeros_like(b), np.zeros_like(b), b])  # 蓝色通道

# 显示各通道
plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.imshow(r_image)
plt.title('Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(g_image)
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(b_image)
plt.title('Blue Channel')
plt.axis('off')

plt.show()

# 保存每个通道的图片
cv2.imwrite('red_channel.jpg', cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('green_channel.jpg', cv2.cvtColor(g_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('blue_channel.jpg', cv2.cvtColor(b_image, cv2.COLOR_RGB2BGR))

print("通道图像已保存")
