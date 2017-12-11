import sys

import numpy as np
import cv2

im = cv2.imread('Cropped_data/' + 'out' + str(3) + '.png')
#im = cv2.imread('out' + str(3) + '.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray,(5,5),0)
#thresh, _ = cv2.threshold(im,255,0,cv2.THRESH_BINARY)
thresh = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
im_resize = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100))
responses = []
keys = [i for i in range(1114032, 1114042)]

for cnt in contours:
    if cv2.contourArea(cnt)>2:
        [x,y,w,h] = cv2.boundingRect(cnt)
        # h>12, h<25, w<80
        print(w, h)
        if h == 24 and w > 5:
            cv2.rectangle(im_resize,(x,y),(x+w,y+h),(0,0,255),1)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi, (10, 10))
            cv2.imshow('norm',im_resize)
            key = cv2.waitKey(0)

            if key == 1048603:  # (escape to quit)
                sys.exit()
            elif key in keys:
                print(int(chr(key - 1113984)))
                responses.append(int(chr(key - 1113984)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

np.savetxt('generalsamples.data', samples)
np.savetxt('generalresponses.data', responses)