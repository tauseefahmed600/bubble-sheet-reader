import csv
import logging
import multiprocessing
import pickle
import traceback

import cv2
import imreg_dft
import imutils
import imutils.contours
import numpy as np
import sys

import os
from imutils.perspective import four_point_transform
from multiprocessing import Pool

base_template = None
base_tables = None


def load_grayscale_image(image_path):
    """
        This function loads image from hard disk then converts it into gray scale image.
        Returns:
             Gray Scale Image.
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def resize_image(image):
    """
        This function scales image up or down to 779x566 resolution.
        Returns:
             Scaled Image.
    """
    desired_height, desired_width = 779, 566
    height, width = image.shape
    height_ratio = height / desired_height
    width_ratio = width / desired_width
    if height_ratio > 1 and width_ratio > 1:
        image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    elif height_ratio < 1 and width_ratio < 1:
        image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_CUBIC)
    return image


def get_table_contours(img):
    """
        This function applies canny edge detector to find borders for each table in a given image.
        Returns:
            An array of points of each table's borders.
    """
    (high_thresh, im_bw) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    edged = cv2.Canny(img, 0.5 * high_thresh, high_thresh)
    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = sorted(contours, key=cv2.boundingRect)
    portions = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        (x, y, w, h) = cv2.boundingRect(c)
        if (120 / 1.5) < w < (200 / 1.5) < h < (900 / 1.5) and len(approx) == 4:
            portions.append(approx)
    table_contours = imutils.contours.sort_contours(portions)[0]
    return table_contours


def apply_dfft_algorithm(image):
    """
        This function applies dfft based image registration algorithm to
        align image to base image.
        Base Image is deserialized from "base_template.pkl" file.
        Returns:
             Aligned Image.
    """
    global base_tables, base_template
    if base_template is None:
        with open('base_template.pkl', 'rb') as f:
            base_template = pickle.load(f)

    similarity = imreg_dft.imreg.similarity(base_template, image)

    image = np.array(similarity, dtype=np.uint8)
    return image


def get_base_table_contours():
    """
        This function loads points of borders for each table.
        Points of Base Tables are deserialized from "tables.pkl" file.
        Returns:
            An array of points of each base table's borders.
    """
    global base_tables, base_template
    with open('tables.pkl', 'rb') as f:
        base_tables = pickle.load(f)
    return base_tables


def get_roll_number(roll_table, dilate=False):
    """
        Given a table of roll number, this function finds roll number by identifying filled bubbles.
        Returns:
            String of Roll Number.
    """
    if dilate is True:
        dilate = cv2.dilate(roll_table, (1, 1), iterations=2)
        thresh = cv2.threshold(dilate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        thresh = cv2.threshold(roll_table, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = imutils.contours.sort_contours(contours, method='left-to-right')[0]

    contours = [(contour, cv2.boundingRect(contour)) for contour in contours if cv2.boundingRect(contour)[2] >= 11
                and 16 >= 11 <= cv2.boundingRect(contour)[3] <= 16 and 0.7 <= float(
        cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]) <= 1.3]
    roll_number = ocr_number(contours, thresh)
    return roll_number


def ocr_number(contours, thresh):
    roll_number = ''
    for contour, (x, y, w, h) in contours:
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        if total > 100:
            if 20 <= y <= 30:
                roll_number += '0'
            elif 35 <= y <= 45:
                roll_number += '1'
            elif 55 <= y <= 65:
                roll_number += '2'
            elif 72 <= y <= 82:
                roll_number += '3'
            elif 90 <= y <= 100:
                roll_number += '4'
            elif 108 <= y <= 118:
                roll_number += '5'
            elif 125 <= y <= 135:
                roll_number += '6'
            elif 140 <= y <= 150:
                roll_number += '7'
            elif 160 <= y <= 170:
                roll_number += '8'
            elif 175 <= y <= 185:
                roll_number += '9'
    return roll_number


def get_selected_answers(tables, dilate=False):
    """
        Given tables of questions, this function finds all filled bubbles from each table.
        Then returns each marked bubble from each question.
        Returns:
            An array of string.
    """
    table_number = 0
    selections = [''] * 100
    for table in tables[1:]:
        q_num = 0
        if dilate:
            dilated = cv2.dilate(table.copy(), (2, 2), iterations=4)
            thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        else:
            thresh = cv2.threshold(table, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
        contours = imutils.contours.sort_contours(contours, method='top-to-bottom')[0]
        cropped_height = [[ind, int(ind * (table.shape[0] / 25)),
                           int((ind + 1) * (table.shape[0] / 25))] for ind in range(25)]
        contours = [contour for contour in contours
                    if 8 <= cv2.boundingRect(contour)[2] <= 50
                    and 8 <= cv2.boundingRect(contour)[3] <= 50 and
                    0.5 <= float(cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]) <= 1.5]
        shape = thresh.shape
        for contour in contours:
            mask = np.zeros(shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            threshold_value = 95
            if dilate:
                threshold_value = 55
            if total > threshold_value:
                question_number, bubble = ocr_question(contour, cropped_height, q_num, table_number)
                if question_number is None:
                    continue
                q_num = question_number - (25 * table_number)
                selections[question_number - 1] += bubble
        table_number += 1
    return selections


# noinspection PyPep8Naming
def ocr_question(contour, cropped_height, q_num, table_number):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    question_number = 0
    for r in range(q_num - 1, len(cropped_height)):
        if r >= cropped_height[r][0]:
            min_height = cropped_height[r][1]
            max_height = cropped_height[r][2]
            if min_height < cY < max_height:
                question_number = (r + 1) + (table_number * 25)
                break
    if question_number == 0:
        return None, None
    bubble = ''
    if 25 <= cX < 40:
        bubble = 'A'
    elif 45 <= cX < 60:
        bubble = 'B'
    elif 60 <= cX < 78:
        bubble = 'C'
    elif cX > 78:
        bubble = 'D'
    return question_number, bubble


def sub_image(image, center, theta, width, height):
    """Extract a rectangle from the source image.

    image - source image
    center - (x,y) tuple for the centre point.
    theta - angle of rectangle.
    width, height - rectangle dimensions.
    """

    if 45 < theta <= 90:
        theta -= 90
        width, height = height, width
    import math
    theta *= math.pi / 180  # convert to rad
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def get_cropped_tables(img, table_contours, reverse, imgdff):
    """
        Given tables of questions, this function finds all filled bubbles from each table.
        Then returns each marked bubble from each question.
		Parameters:
			img: Source image
			table_contours: An array of points for region of interest i.e. Tables
			reverse: Boolean value to reverse the cropped image.
			imgdff: Boolean value indicating whether dfft based image registration is already applied or not
        Returns:
            An array of cropped tables.
    """
    tables = []
    for contour_number in range(len(table_contours)):
        table_contour = table_contours[contour_number]
        if contour_number == 0:
            table = four_point_transform(img, table_contour.reshape(len(table_contour), 2))
        else:
            centre, dimensions, theta = cv2.minAreaRect(table_contour)
            rect = cv2.minAreaRect(table_contour)
            width = int(dimensions[0])
            height = int(dimensions[1])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            M = cv2.moments(box)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            table = sub_image(img, (cx, cy), theta + 90, height, width)
        if reverse:
            table = np.rot90(table, 2)
        if imgdff and is_photocopy:
            table = table[1:-12, 3:-1]
        else:
            table = table[3:-1, 3:-1]
        if contour_number == 0:
            table = cv2.resize(table, (80, 201))
        else:
            table = cv2.resize(table, (98, 538))
        tables.append(table)
    return tables


def get_result(selections, selections2, answers):
    correct = 0
    wrong = 0
    missing = 0
    for i in range(len(selections)):
        if len(selections[i]) > 1 \
                or (len(selections[i]) == 1 and len(selections2[i]) > 1) \
                or (len(selections2[i]) != 1 and len(selections[i]) < 1):
            missing += 1
            continue
        if len(selections[i]) == 0:
            if len(selections2[i]) == 1:
                if str(selections2[i]) == answers[i]:
                    correct += 1
                else:
                    wrong += 1
        if len(selections[i]) == 1:
            if str(selections[i]) == answers[i]:
                correct += 1
            else:
                wrong += 1
    return correct, wrong, missing


def main(data):
    img_path = data["Image"]
    answers = data["Answers"]
    try:
        img = load_grayscale_image(img_path)
        img = resize_image(img)
        reverse = False
        table_contours = []
    except Exception:
        print('Error:' + img_path)
        return
    try:
        table_contours = get_table_contours(img)
    except:
        traceback.print_exc()
    imgdff = False
    if len(table_contours) == 5:
        _, _, w, h = cv2.boundingRect(table_contours[0])
        if h > 250:
            reverse = True
            table_contours = imutils.contours.sort_contours(table_contours, method='right-to-left')[0]
    else:
        try:
            img = apply_dfft_algorithm(img)
            table_contours = get_base_table_contours()
            imgdff = True
        except Exception:
            print('Error:' + img_path)
            return
    tables = get_cropped_tables(img, table_contours, reverse, imgdff)
    roll_number = get_roll_number(tables[0])
    if len(roll_number) != 3:
        roll_number = get_roll_number(tables[0], True)
    selections = get_selected_answers(tables)
    selections2 = get_selected_answers(tables, True)

    correct, wrong, missing = get_result(selections, selections2, answers)

    if len(roll_number.strip()) != 3:
        print('Error:' + img_path)
    else:
        print('RollNo:' + str(roll_number) + ',Correct:' + str(correct) + ',Wrong:' + str(wrong) +
              ',Missing:' + str(missing) + ',Total:' + str(correct))
    sys.stdout.flush()


def load_answers():
    with open(answers_path, 'r') as f:
        lines = csv.reader(f)
        return [l[0].strip()[0].upper() for l in lines]


def get_files():
    image_files = []
    for image_file in os.listdir(base_path):
        abs_path = os.path.join(base_path, image_file)
        if os.path.isfile(abs_path):
            image_files.append(abs_path)
    return image_files


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        base_path = sys.argv[1]
        answers_path = sys.argv[2]
        answers = load_answers()
        is_photocopy = False
        if len(sys.argv) > 3:
            is_photocopy = True
        files = get_files()
        data = []
        for file in files:
            if file[-3:] == 'jpg' or file[-3:] == 'bmp' or file[-4:] == 'jpeg':
                data.append({"Image": file, "Answers": answers})

        concurrency = multiprocessing.cpu_count()
        p = Pool(concurrency)
        p.map(main, data)
    except Exception as e:
        print(traceback.print_exc(), file=sys.stderr)
