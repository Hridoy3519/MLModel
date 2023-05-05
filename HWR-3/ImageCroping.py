from word_detector import prepare_img, detect, sort_multiline
import matplotlib.pyplot as plt
import cv2
from skimage import io


def cropImage():
    img = cv2.imread('upload/image.JPEG')

    h, w, c = img.shape
    print(img.shape)
    new_h = h
    new_w = w
    ar = w / h
    if w > 1000:
        new_w = 1000
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(img.shape)
    print(new_h)

    img = prepare_img(img, new_h)
    # (2) detect words in image
    detections = detect(img,
                        kernel_size=25,
                        sigma=11,
                        theta=ar,
                        min_area=100)
    # (3) sort words in line
    lines = sort_multiline(detections)
    print(detections)
    # plot results
    plt.imshow(img, cmap='gray')
    num_colors = 7
    colors = plt.cm.get_cmap('rainbow', num_colors)
    for line_idx, line in enumerate(lines):
        for word_idx, det in enumerate(line):
            xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
            ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
            plt.plot(xs, ys, c=colors(line_idx % num_colors))
            plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

    # plt.show()

    list_img_names_serial = []
    num_colors = 7

    colors = plt.cm.get_cmap('rainbow', num_colors)

    for line_idx, line in enumerate(lines):
        print("line index and line is printed", line_idx)
        for word_idx, det in enumerate(line):
            # print("word index in line is printed", word_idx)
            xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
            ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
            plt.plot(xs, ys, c=colors(line_idx % num_colors))
            plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
            # print(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)
            crop_img = img[det.bbox.y:det.bbox.y + det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]
            cv2.imwrite("detect/line" + str(line_idx) + "word" + str(word_idx) + ".jpg", crop_img)
            full_img_path = "detect/line" + str(line_idx) + "word" + str(word_idx) + ".jpg"
            list_img_names_serial.append(full_img_path)
            # print(list_img_names_serial)
            list_img_names_serial_set = set(list_img_names_serial)

            textfile = open("detect/img-sequence.txt", 'w')
            for element in list_img_names_serial:
                textfile.write(element + '\n')
            textfile.close()
