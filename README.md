#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ezen a képen keresünk mintát/mintákat
I = cv2.imread("/home/noemi/behajtani-tilos-rendor.jpg")
# A keresett első minta
T = cv2.imread("/home/noemi/behajtanitilos.png")

M = cv2.matchTemplate(I, T, cv2.TM_CCOEFF)
(_, _, _, maxloc) = cv2.minMaxLoc(M)

# Körrel jelöljük a legjobban illeszkedő pontot
cv2.circle(I, (maxloc[0] + T.shape[1] // 2, maxloc[1] + T.shape[0] // 2), 90, (0, 255, 0), 3)

plt.figure()
plt.plot()
plt.title("Behajtanitilos")
plt.imshow(I[:, :, ::-1])
plt.axis("off")

plt.show()
