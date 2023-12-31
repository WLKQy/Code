"""
Author:XiaoMa
date:2021/11/2
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img0 = cv2.imread("mao2.jpg")
img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
h, w = img1.shape[:2]
print(h, w)
cv2.namedWindow("W0")
cv2.imshow("W0", img1)
cv2.waitKey(delay=0)

# 分水岭算法
ret1, img10 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # （图像阈值分割，将背景设为黑色）
cv2.namedWindow("W3")
cv2.imshow("W3", img10)
cv2.waitKey(delay=0)
##noise removal（去除噪声，使用图像形态学的开操作，先腐蚀后膨胀）
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(img10, cv2.MORPH_OPEN, kernel, iterations=2)
# sure background area(确定背景图像，使用膨胀操作)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Finding sure foreground area（确定前景图像，也就是目标）
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret2, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
# Finding unknown region（找到未知的区域）
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret3, markers = cv2.connectedComponents(sure_fg)  # 用0标记所有背景像素点
# Add one to all labels so that sure background is not 0, but 1（将背景设为1）
markers = markers + 1
##Now, mark the region of unknown with zero（将未知区域设为0）
markers[unknown == 255] = 0
markers = cv2.watershed(img1, markers)  # 进行分水岭操作
img1[markers == -1] = [0, 0, 255]  # 边界区域设为-1，颜色设置为红色
cv2.namedWindow("W4")
cv2.imshow("W4", img1)
cv2.waitKey(delay=0)