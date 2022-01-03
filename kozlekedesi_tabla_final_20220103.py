#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2

# Ezen a képen keresünk mintát/mintákat
I = cv2.imread("/home/noemi/Documents/Suli/Gépi_látás/Valos_tablak/kep47.jpg")
# A keresett minták
mzsak = cv2.imread("/home/noemi/Documents/Suli/Gépi_látás/Tablak_minta/zsakutca.png")
mkorforgalom = cv2.imread("/home/noemi/Documents/Suli/Gépi_látás/Tablak_minta/korforgalom.png")

#kép átméretezése túl nagy képfelbontás esetén
while I.shape[1] > 1800 or I.shape[0] > 1800:
	scale_percent = 50 # percent of original size
	width = int(I.shape[1] * scale_percent / 100)
	height = int(I.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize original image
	I = cv2.resize(I, dim, interpolation = cv2.INTER_AREA)

tablakszama = 0

hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

#maszk a piros szűrésre
lower_red = np.array([150,100,20])
upper_red =np.array([180,255,255])
lower_orange = np.array([0,140,20])
upper_orange = np.array([10,255,255])
redmask1 = cv2.inRange(hsv,lower_red,upper_red)
redmask2 = cv2.inRange(hsv,lower_orange,upper_orange)
redmask = redmask1 + redmask2
#maszk a kék szűrésre
lower_blue = np.array([95,100,20])
upper_blue = np. array([130,255,255])
bluemask = cv2.inRange(hsv,lower_blue,upper_blue)
#maszk a sárga szűrésre
lower_yellow = np.array([17,100,20])
upper_yellow = np.array([32,255,255])
yellowmask = cv2.inRange(hsv,lower_yellow,upper_yellow)
#maszk a fehér szűrésre
lower_white = np.array([0,0,20])
upper_white = np.array([180,45,255])
whitemask = cv2.inRange(hsv, lower_white,upper_white)


x = 0
y = 0
w = 100
h = 100
zsak = False

#kontúr keresés piros színre
contours, hierarchy = cv2.findContours(redmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
	area = cv2.contourArea(cnt)
	if area>500:
		cv2.drawContours(I, cnt,-1, (0, 0, 0), 1)
		peri = cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,0.02*peri, True)
		#print(len(approx))
		if len(approx) == 3 or len(approx) == 8:
			x, y, w, h = cv2.boundingRect(approx)
			if len(approx) == 3:
				wcontours, hierarchy = cv2.findContours(whitemask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				for wcnt in wcontours:
					warea = cv2.contourArea(wcnt)
					if warea>300:
						cv2.drawContours(I, wcnt, -1, (0, 0, 0), 1)
						wperi = cv2.arcLength(wcnt,True)
						wapprox = cv2.approxPolyDP(wcnt,0.02*wperi,True)
						if len(wapprox) == 3:
								wx, wy, ww, wh = cv2.boundingRect(wapprox)
								if ww<w or wh<h:
									cv2.rectangle(I, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255,0), 5)
									print('Elsőbbség adás tábla van a képen (türkiz színnel bekeretezve)')
									tablakszama = tablakszama+1
			elif len(approx) == 8 and area>1500:
				cv2.rectangle(I, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255,0), 5)	
				print('Stop tábla van a képen (zöld színnel bekeretezve)')
				tablakszama = tablakszama+1
	if area>200:
		cv2.drawContours(I, cnt,-1, (0, 0, 0), 1)
		peri = cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,0.02*peri, True)
		if len(approx) == 4:
			zsak=True

#kontúr keresés kék színre
contours2, hierarchy = cv2.findContours(bluemask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt2 in contours2:
	area = cv2.contourArea(cnt2)
	if area>500:
		cv2.drawContours(I, cnt2,-1, (0, 0, 0), 1)
		peri = cv2.arcLength(cnt2,True)
		approx = cv2.approxPolyDP(cnt2,0.02*peri, True)
		if len(approx) == 4 or len(approx) == 8:
			x, y, w, h = cv2.boundingRect(approx)
			dim = (w, w)
			if len(approx) == 4 and zsak:
				mzsak = cv2.resize(mzsak, dim, interpolation = cv2.INTER_AREA)
				M = cv2.matchTemplate(I, mzsak, cv2.TM_CCOEFF)
				(_, _, _, maxloc) = cv2.minMaxLoc(M)
				if abs(maxloc[0]-x) < 25 :
					cv2.rectangle(I, (maxloc[0]-20, maxloc[1]-20),  (maxloc[0] +  mzsak.shape[1]+20,  maxloc[1] + mzsak.shape[0]+20), (255, 0, 0), 3)
					print('Zsákutca tábla van a képen (kék színnel bekeretezve)')
					tablakszama = tablakszama+1
			elif len(approx) == 8:
				mkorforgalom = cv2.resize(mkorforgalom, dim, interpolation = cv2.INTER_AREA)
				M = cv2.matchTemplate(I, mkorforgalom, cv2.TM_CCOEFF)
				(_, _, _, maxloc) = cv2.minMaxLoc(M)
				if abs(maxloc[0]-x) < 25 :
					cv2.rectangle(I, (maxloc[0]-20, maxloc[1]-20),  (maxloc[0] +  mkorforgalom.shape[1]+20,  maxloc[1] + mkorforgalom.shape[0]+20), (0, 0, 255), 3)
					print('Körforgalom tábla van a képen (piros színnel bekeretezve)')
					tablakszama = tablakszama+1
		

#kontúr keresés sárga színre
contours3, hierarchy = cv2.findContours(yellowmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt3 in contours3:
	area = cv2.contourArea(cnt3)
	if area>500:
		cv2.drawContours(I, cnt3,-1, (0, 0, 0), 1)
		peri = cv2.arcLength(cnt3,True)
		approx = cv2.approxPolyDP(cnt3,0.02*peri, True)
		if len(approx) == 4:
			x, y, w, h = cv2.boundingRect(approx)
			wcontours, hierarchy = cv2.findContours(whitemask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			for wcnt in wcontours:
					warea = cv2.contourArea(wcnt)
					if warea>300:
						cv2.drawContours(I, wcnt, -1, (0, 0, 0), 1)
						wperi = cv2.arcLength(wcnt,True)
						wapprox = cv2.approxPolyDP(wcnt,0.02*wperi,True)
						if len(wapprox) == 4:
							wx, wy, ww, wh = cv2.boundingRect(wapprox)
							if ww>w or wh>h:
								cv2.rectangle(I, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0,255), 5)
								print('Főútvonal tábla van a képen (lila színnel bekeretezve)')
								tablakszama = tablakszama+1



if tablakszama == 0 :
	print('A következő táblák közül egyet sem talált a program a képen: Elsőbbségadás kötelező, Stop, Zsákutca, Körforgalom, Főút')
cv2.imshow("eredeti", I)
#cv2.imshow("piros_maszk", redmask)
#cv2.imshow("kek_maszk", bluemask)
#cv2.imshow("sarga_maszk", yellowmask)
cv2.waitKey(0)
cv2.destroyAllWindows()
