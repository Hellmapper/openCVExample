import cv2
import numpy as np

my_photo = cv2.imread('lena.jpg')
img_grey = cv2.cvtColor(my_photo,cv2.COLOR_BGR2GRAY)
thresh = 100 #зададим порог
#получим картинку, обрезанную порогом
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#надем контуры
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#создадим пустую картинку
img_contours = np.zeros((my_photo.shape[0], my_photo.shape[1], 3), np.uint8)
cnt = len(contours); #отобразим контуры
for x in range(cnt):
    color = list(np.random.random(size=3) * 256 + 30)
    cv2.drawContours(img_contours, contours, x, color, 1)
cv2.imshow('contours', img_contours) # выводим итоговое изображение в окно
cv2.waitKey()