import cv2
import numpy as np

#######   training part    ###############
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

############################# testing part  #########################

for index in range(1, 11):
    data = []
    im = cv2.imread('Cropped_data/' + 'out' + str(index) + '.png')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    im_resize = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    out = np.zeros(im_resize.shape, np.uint8)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 2:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h == 24 and w > 5:
                cv2.rectangle(im_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.find_nearest(roismall, k=1)
                string = str(int((results[0][0])))
                #data = np.append(data, int((results[0][0])))
                data.append(int((results[0][0])))
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
            #data.append(\n)
    print(data)
    np.savetxt('result/' + 'result' + str(index) + '.txt', data)

    cv2.imshow('im', im_resize)
    cv2.imshow('out', out)
    cv2.waitKey(0)

