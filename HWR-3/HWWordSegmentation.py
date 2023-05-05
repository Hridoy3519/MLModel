import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import shutil


def crop_words(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape

    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    plt.imshow(img);

    def thresholding(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
        plt.imshow(thresh, cmap='gray')
        return thresh

    thresh_img = thresholding(img);

    # dilation
    kernel = np.ones((8, 100), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)
    plt.imshow(dilated, cmap='gray');

    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    sorted_contours_lines = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

    # word dilation
    kernel = np.ones((4, 30), np.uint8)
    dilated2 = cv2.dilate(thresh_img, kernel, iterations=1)
    plt.imshow(dilated2, cmap='gray')

    img3 = img.copy()
    words_list = []
    before_sort_word_list = []

    for line in sorted_contours_lines:

        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y + h, x:x + w]

        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

        for word in cnt:
            if cv2.contourArea(word) < 400:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(word)

            before_sort_word_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])

        x_list = sorted(before_sort_word_list, key=lambda x: x[0])
        words_list.extend(x_list)
        before_sort_word_list = []

    if os.path.exists('cropped_images'):
        shutil.rmtree('cropped_images')
    os.mkdir('cropped_images')

    textfile = open("cropped_images/img-sequence.txt", 'w')
    for i in range(len(words_list)):
        word = words_list[i]
        roi_9 = img[word[1]:word[3], word[0]:word[2]]

        # Generate the filename and save the image
        filename = f"word{i}.png"
        filepath = os.path.join('cropped_images', filename)
        cv2.imwrite(filepath, roi_9)
        textfile.write(filepath + '\n')
    textfile.close()