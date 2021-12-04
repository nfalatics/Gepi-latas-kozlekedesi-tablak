#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2

# Ezen a képen keresünk mintát/mintákat
I = cv2.imread("/home/noemi/Documents/Suli/Gépi_látás/Valos_tablak/55286_source.jpg")
# A keresett első minta
melsobbseg = cv2.imread("/home/noemi/Documents/Suli/Gépi_látás/Tablak_minta/elsobbsegadas.png")



hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

#hsv hue érték
lower_red = np.array([150,150,20])
upper_red =np.array([180,255,255])
lower_orange = np.array([0,150,20])
upper_orange = np.array([15,255,255])
#maszk a piros szűrésre
redmask1 = cv2.inRange(hsv,lower_red, upper_red)
redmask2 = cv2.inRange(hsv,lower_orange, upper_orange)
redmask = redmask1 + redmask2
kernel = np.ones((4,4), np.uint8)
redmask = cv2.erode(redmask, kernel)



x = 0
y = 0
w = 100
h = 100

#kontúr keresés
contours, hierarchy = cv2.findContours(redmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
	area = cv2.contourArea(cnt)
	if area>1000:
		cv2.drawContours(I, cnt,-1, (255, 0, 255), 5)
		peri = cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,0.02*peri, True)
		print(len(approx))
		x, y, w, h = cv2.boundingRect(approx)
		cv2.rectangle(I, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255,0), 5)

#print('Original Dimensions: ', melsobbseg.shape)
dim = (w, h)
melsobbseg = cv2.resize(melsobbseg, dim, interpolation = cv2.INTER_AREA)
#print('resized dimensions: ', melsobbseg.shape)

M = cv2.matchTemplate(I, melsobbseg, cv2.TM_CCOEFF)
(_, _, _, maxloc) = cv2.minMaxLoc(M)
cv2.rectangle(I, (maxloc[0]-20, maxloc[1]-20),  (maxloc[0] +  melsobbseg.shape[1]+20,  maxloc[1] + melsobbseg.shape[0]+20), (255, 0, 0), 3)

if (maxloc[0]-x) < 15 and len(approx) == 3:
	print('Elsőbbség adás tábla van a képen')

cv2.imshow("eredeti", I)
cv2.imshow("piros_maszk", redmask)
cv2.waitKey(0)
cv2.destroyAllWindows()
