import cv2
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')


def switch(argument):

    switcher = {
        0: '/Users/michal/XcodeProjects/OpenCVTempl/opencv_templ/lena.png',
        1: '1/sIMG_2528.jpg',
        2: '1/sIMG_2529.jpg',
        3: '1/sIMG_2530.jpg',
        4: '1/sIMG_2531.jpg',
        5: '1/sIMG_2532.jpg',
        6: '1/sIMG_2533.jpg',
    }
    return switcher.get(argument)


def negative(array):

    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            if array[i][j] == 250:
                array[i][j] = 0
            else:
                array[i][j] = 250
    return array


def plot_step_by_step(images, name):

    img_start = images[0]
    img_blurred = images[1]
    img_thresholded = images[2]
    img_opened = images[3]
    img_contoured = images[4]
    img_final = images[5]
    img_final = img_final[:, :, ::-1]  # zmiana BGR (openCV) na RGB (matplotlib)

    fig = plt.figure()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                    wspace=0.05, hspace=0.05)

    plt.subplot(231), plt.imshow(img_start, cmap='gray', interpolation='bicubic'), plt.title('1, Orginal')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.subplot(232), plt.imshow(img_blurred, cmap='gray', interpolation='bicubic'), plt.title('2. Blurred')
    plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(img_thresholded, cmap='gray', interpolation='bicubic'), plt.title('2. Thresholded')
    plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(img_opened, cmap='gray', interpolation='bicubic'), plt.title('4. Opened')
    plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(img_contoured, cmap='gray', interpolation='bicubic'), plt.title('5. Contoured')
    plt.xticks([]), plt.yticks([])

    plt.subplot(236), plt.imshow(img_final, cmap='cool', interpolation='bicubic'), plt.title('6. Final')
    plt.xticks([]), plt.yticks([])

    name = "results/algorithm" + str(name)
    plt.savefig(name)
    plt.close(fig)


def plot_final(image, name):

    image = image[:, :, ::-1]  # zmiana BGR (openCV) na RGB (matplotlib)
    fig = plt.figure()
    plt.imshow(image, cmap='cool', interpolation='bicubic'), plt.title('Result of recognition')
    plt.xticks([]), plt.yticks([])
    fig.tight_layout()
    name = "results/final" + str(name)
    plt.savefig(name, dpi=500)
    plt.close(fig)


def main(snapshot):

    font = cv2.FONT_HERSHEY_SIMPLEX

    img = cv2.imread(switch(snapshot), 0)
    img_colour = cv2.imread(switch(snapshot), 1)

    height, width = img.shape[:2]
    if width > 800:
        ratio = (width-800)/width
        height = int(ratio*height)
        width = int(ratio*width)
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
        img_colour = cv2.resize(img_colour, (width, height), interpolation = cv2.INTER_CUBIC)
    # cv2.imshow('Bazowy',img_colour)
    # cv2.waitKey(0)

    blur = np.array(img)
    blur = cv2.GaussianBlur(blur, (15, 15), 0)
    # cv2.imshow('wygladzony',blur)
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(blur, 127, 250, cv2.THRESH_OTSU)
    # cv2.imshow('progowany',thresh)
    # cv2.waitKey(0)

    thresh = negative(thresh)
    kernel1 = np.ones((5, 5), np.uint8)  # noise removal
    opening = cv2.erode(thresh, kernel1, iterations=1)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel1, iterations=1)
    opening = negative(opening)
    # cv2.imshow('dylatacja+erozja', opening)
    # cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(opening, 1, 2)  # (_, mode, method)
    contoured = np.array(img)
    cv2.drawContours(contoured, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('kontury',contoured)
    # cv2.waitKey(0)

    n_barley = 0;
    n_halves = 0;
    n_groups = 0;
    for i in contours:
        size, _, _ = np.shape(i)
        if size > 5:
            ellipse = cv2.fitEllipse(i)
            area = cv2.contourArea(i)
            x, y, w, h = cv2.boundingRect(i)
            aspect_ratio = float(w) / h
            # text = "Area" + str(area) + "aspect" + str(aspect_ratio)
            '''assign ellipse to class'''
            if 900 < area < 1500:
                n_barley += 1
                cv2.ellipse(img_colour, ellipse, (0, 255, 0), 2)
            elif area <= 900:
                n_halves += 1
                cv2.ellipse(img_colour, ellipse,  (0, 255, 255), 2)
            elif 1500 < area < 8000:
                n_groups += 1
                cv2.ellipse(img_colour, ellipse, (0, 0, 255), 2)

    cv2.putText(img_colour, "Recognised " + str(n_barley) + " barley seeds", (width-370, height-80),
                font, 0.65, (0, 0, 0), 1)
    cv2.putText(img_colour, "Recognised " + str(n_halves) + " halves barley seeds", (width - 370, height - 55),
                font, 0.65, (0, 0, 0), 1)
    cv2.putText(img_colour, "Marked " + str(n_groups) + " other objects", (width-370, height-30),
                font, 0.65, (0, 0, 0), 1)

    # cv2.imshow('image', img_colour)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plot_step_by_step([img, blur, thresh, opening, contoured, img_colour], snapshot)
    plot_final(img_colour, snapshot)
    print("All done!")

if __name__ == "__main__":
    # execute only if run as a script
    main(1)

