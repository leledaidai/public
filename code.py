import cv2#opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import cv2

img = cv2.imread('/Users/user/Desktop/ball.png')

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义颜色范围
lower_color = (0,43,46)
upper_color = (10, 255, 255)
# 颜色范围之内的为白色  颜色范围之外的为黑色
mask_img = cv2.inRange(hsv_img, lower_color, upper_color)
kernel = np.ones((7,7),np.uint8)
opening = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

#cv2.imshow('img', img)
#cv2.imshow('mask_img', mask_img)
cv2.imshow('opening',opening)
cv2.waitKey()


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
ret, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)
binary,contours,hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv2.putText(img,"positon:"+str(x)+"   "+str(y),(150,400),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
cv_show('img',img)
