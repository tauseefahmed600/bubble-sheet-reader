"""

 The MIT License (MIT)
 Copyright (c) 2018 Tauseef

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 OR OTHER DEALINGS IN THE SOFTWARE.

"""
import csv
import logging
import multiprocessing
import os
import pickle
import sys
import traceback

import cv2
import imutils
import imutils.contours
import numpy as np
from imutils.perspective import four_point_transform

import register_image

base_template = None
base_tables = None
DESIRED_HEIGHT = 779
DESIRED_WIDTH = 566
MINIMUM_TABLE_WIDTH = 80
MAXIMUM_TABLE_WIDTH = 133
MINIMUM_TABLE_HEIGHT = 133
MAXIMUM_TABLE_HEIGHT = 600
BASE_TEMPLATE = 'base_template.pkl'
BASE_TABLE_CONTOURS = 'tables.pkl'


def load_grayscale_image(image_path):
    """Loads a grayscale image.

    Loads an image with a given path then converts it into grayscale image.

    Args:
        image_path: Path to image file.

    Returns:
        A grayscale image.

    Raises:
        IOError: An error occurred loading the image.
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def resize_image(image):
    """Resizes a given image.

    This function scales an image up or down to 779x566 resolution.

    Args:
        image: An image object of answer sheet.

    Returns:
        Resized image with a resolution of 779x566
    """
    height, width = image.shape
    height_ratio = height / DESIRED_HEIGHT
    width_ratio = width / DESIRED_WIDTH
    if height_ratio > 1 and width_ratio > 1:
        image = cv2.resize(image, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_AREA)
    elif height_ratio < 1 and width_ratio < 1:
        image = cv2.resize(image, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return image


def get_table_contours(image):
    """Gets contours of tables.

    This function applies canny edge detector to find borders for each table in a given image.

    Args:
        image: An image object of answer sheet.

    Returns:
        An array of tables containing each table's contours.
    """
    high_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edged = cv2.Canny(image, low_thresh, high_thresh)
    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = sorted(contours, key=cv2.boundingRect)
    tables = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        _, _, width, height = cv2.boundingRect(c)
        if MINIMUM_TABLE_WIDTH < width < MAXIMUM_TABLE_WIDTH and MINIMUM_TABLE_HEIGHT < height < MAXIMUM_TABLE_HEIGHT \
                and len(approx) == 4:
            tables.append(approx)
    table_contours = imutils.contours.sort_contours(tables)[0]
    return table_contours


def apply_template_matching_algorithm(image):
    """Applies template matching algorithm.

    This function applies Discrete Fourier Transform based image registration algorithm to
    pixel to pixel alignment of the answer sheet image to base template image.
    Base Image is deserialized from "base_template.pkl" file.

    Args:
        image: An image object of answer sheet.

    Returns:
        Skew and Rotation corrected image of answer sheet.

    Raises:
        ValueError: Image must be based on template answer sheet.
    """
    global base_template
    if base_template is None:
        with open(BASE_TEMPLATE, 'rb') as f:
            base_template = pickle.load(f)

    similarity = register_image.similarity(base_template, image)

    image = np.array(similarity, dtype=np.uint8)
    return image


def get_base_table_contours():
    """Retrieves table contours from base template.

    This function loads points of borders for each table.
    Points of Base Tables are deserialized from "tables.pkl" file.

    Returns:
        An array of tables containing each table's contours.

    Raises:
        IOError: An error occurred accessing the base template.
    """
    global base_tables, base_template
    if base_tables is None:
        with open(BASE_TABLE_CONTOURS, 'rb') as f:
            base_tables = pickle.load(f)
    return base_tables


def recognize_roll_number(roll_table, dilate=False):
    """Recognizes roll number from roll numbers table.

    Given a table of roll number, this function finds roll number by identifying filled bubbles.

    Args:
        roll_table: An image of roll numbers table.
        dilate: boolean value indicating whether to dilate the table or not.

    Returns:
        String representing roll number.
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
    roll_number = get_roll_number(contours, thresh)
    return roll_number


def get_roll_number(bubbles, image):
    """Retrieves roll number by their co-ordinates.

    Given an image of roll numbers table and contours of bubbles , recognizes filled bubbles based on
    number of pixels, it's x and y co-ordinates.

    Args:
        bubbles: Contours of possible bubbles in roll numbers table.
        image: An image of roll numbers table.

    Returns:
        String representing roll number.
    """
    roll_number = ''
    for bubble, (x, y, w, h) in bubbles:
        mask = np.zeros(image.shape, dtype="uint8")
        cv2.drawContours(mask, [bubble], 0, 255, -1)
        mask = cv2.bitwise_and(image, image, mask=mask)
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


def recognize_selected_answers(tables, dilate=False):
    """Recognizes selected answers from questions tables.

    Given tables of questions, this function finds selected answers by identifying filled bubbles.

    Args:
        tables: Images of tables of questions.
        dilate: boolean value indicating whether to dilate the table or not.

    Returns:
        String representing roll number.

        Given images containing tables of questions, this function finds all filled bubbles from each table.
        Then returns each marked bubble from each question.
        Returns:
            An array of strings, each index representing question number and
            each value at each index representing the selected answers.
            Note:
                Indexes start from 0 and index 0 represents the question number 1.
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
        cropped_question = [[ind, int(ind * (table.shape[0] / 25)),
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
                question_number, bubble = get_selected_answers(contour, cropped_question, q_num, table_number)
                if question_number is None:
                    continue
                q_num = question_number - (25 * table_number)
                selections[question_number - 1] += bubble
        table_number += 1
    return selections


def get_selected_answers(bubbles, cropped_question, question_number, table_number):
    """Retrieves roll number by their co-ordinates.
    
    Given a cropped section of a question and contours of bubbles , recognizes the filled bubbles based on
    number of pixels, it's x and y co-ordinates.

    Args:
        bubbles: Contours of possible bubbles in roll numbers table.
        cropped_question: An image containing the cropped section of the question.
        question_number: Number representing the question.
        table_number: Number representing the questions table

    Returns:
        A tuple containing question number and selected answers.
    """
    M = cv2.moments(bubbles)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    question_number = 0
    for r in range(question_number - 1, len(cropped_question)):
        if r >= cropped_question[r][0]:
            min_height = cropped_question[r][1]
            max_height = cropped_question[r][2]
            if min_height < cY < max_height:
                question_number = (r + 1) + (table_number * 25)
                break
    if question_number == 0:
        return None, None
    selected_anwers = ''
    if 25 <= cX < 40:
        selected_anwers = 'A'
    elif 45 <= cX < 60:
        selected_anwers = 'B'
    elif 60 <= cX < 78:
        selected_anwers = 'C'
    elif cX > 78:
        selected_anwers = 'D'
    return question_number, selected_anwers


def crop_skew_corrected_table(image, center, theta, width, height):
    """Crops a skew corrected table from an image of the answer sheet.

    Crops a table of the given height, width and center point, than performs skew correction
    Finally crops the table from the image.

    Args:
        image: Contours of possible bubbles in roll numbers table.
        center: Tuple (x,y) for the centre point of the rectangle.
        theta: Angle of the rectangle representing the table.
        width: Width of the rectangle representing the table.
        height: Height of the rectangle representing the table.

    Returns:
        Crops a skew corrected table from answer sheet image.
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


def get_cropped_tables(image, table_contours, reverse):
    """Crops tables from an image of the answer sheet.

    Crops each table from an image of the answer sheet.

    Args:
        image: An image of the answer sheet
        table_contours: Contours of the tables.
        reverse: A boolean value representing whether to reverse the image or not.

    Returns:
        An array of cropped tables.
    """
    tables = []
    for contour_number in range(len(table_contours)):
        table_contour = table_contours[contour_number]
        if contour_number == 0:
            table = four_point_transform(image, table_contour.reshape(len(table_contour), 2))
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
            table = crop_skew_corrected_table(image, (cx, cy), theta + 90, height, width)
        if reverse:
            table = np.rot90(table, 2)
        table = table[3:-1, 3:-1]
        if contour_number == 0:
            table = cv2.resize(table, (80, 201))
        else:
            table = cv2.resize(table, (98, 538))
        tables.append(table)
    return tables


def get_result(selections, selections2, answers):
    """Retrieves the final grades.

    Generates the final result by comparing the answers from given correct answers.

    Args:
        selections: An image of the answer sheet
        selections2: Contours of the tables.
        answers: A boolean value representing whether to reverse the image or not.

    Returns:
        A tuple containing amount of correct, wrong and missed answers
    """
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
    if len(table_contours) == 5:
        _, _, w, h = cv2.boundingRect(table_contours[0])
        if h > 250:
            reverse = True
            table_contours = imutils.contours.sort_contours(table_contours, method='right-to-left')[0]
    else:
        try:
            img = apply_template_matching_algorithm(img)
            table_contours = get_base_table_contours()
        except Exception:
            traceback.print_exc()
            print('Error:' + img_path)
            return
    tables = get_cropped_tables(img, table_contours, reverse)
    roll_number = recognize_roll_number(tables[0])
    if len(roll_number) != 3:
        roll_number = recognize_roll_number(tables[0], True)
    selections = recognize_selected_answers(tables)
    selections2 = recognize_selected_answers(tables, True)

    correct, wrong, missing = get_result(selections, selections2, answers)

    if len(roll_number.strip()) != 3:
        print('Error:' + img_path)
    else:
        pass
        print('RollNo:' + str(roll_number) + ',Correct:' + str(correct) + ',Wrong:' + str(wrong) +
              ',Missing:' + str(missing) + ',Total:' + str(correct))
    sys.stdout.flush()


def load_answers():
    """Fetches correct answers.

    Reads a csv file containing correct answers in first column.

    Returns:
        A list containing correct answers.
    """
    with open(answers_path, 'r') as f:
        lines = csv.reader(f)
        return [l[0].strip()[0].upper() for l in lines]


def get_path_of_answer_sheet_images():
    """Fetches paths of answer sheet images.

    Iterates the directory containing answer sheet images and
    returns an array containing paths of each answer sheet.

    Returns:
        An array of  containing paths of each answer sheet.
    """
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
        files = get_path_of_answer_sheet_images()
        data = []
        for file in files:
            if file[-3:] == 'jpg' or file[-3:] == 'bmp' or file[-4:] == 'jpeg':
                data.append({"Image": file, "Answers": answers})

        from multiprocessing import Pool

        concurrency = multiprocessing.cpu_count()
        p = Pool(concurrency)
        p.map(main, data)

    except Exception as e:
        print(traceback.print_exc(), file=sys.stderr)
