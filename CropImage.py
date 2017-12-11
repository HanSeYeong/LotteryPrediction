import numpy as np
import cv2

def cropimage(num):
    image = cv2.imread('original_data/' + str(num) + '.png')
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = imgGray.shape

    # increase contrast with 0 and 255
    for y in range(h):
        for x in range(w):
            tp = imgGray[y, x]
            if tp < 127:
                tp = 255
            else:
                tp = 0
            imgGray[y, x] = tp

    imgCrop_left = imgGray[500:890, 405:439]
    imgCrop_right = imgGray[500:890, 597:640]

    return imgCrop_left, imgCrop_right

for i in range(1, 11):
    append_im = np.concatenate((cropimage(i)[0], cropimage(i)[1]), axis=1)
    cv2.imwrite('Cropped_data/' + 'append_out' + str(i) + '.png', append_im)

#cv2.imshow("Cropped image", cropimage(3)[0])
#cv2.waitKey(0)
#cv2.imshow("Cropped image", cropimage(3)[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()