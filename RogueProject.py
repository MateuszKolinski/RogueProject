# Unnamed Rogue-like Card Game Project
# By Mateusz Kolinski MateuszPKolinski@gmail.com
# The code below generates an image of a playing card according to provided data in database
# Image is generated from a few border templates and the character image associated with the database data
# Each card can be painted in one, two or three colors depending on the data in database
# It is simply easier to use this code to swiftly generate hundreds of cards instead of doing it with a manually one by one

import cv2 as cv
import numpy as np
import sqlite3
import os
import scipy
import math
import traceback
from PIL import Image, ImageDraw, ImageFont

# Scales all objects in the card image
SCALING_FACTOR = 2

# Constant card and object dimension settings
CARD_WIDTH = 500 * SCALING_FACTOR
CARD_HEIGHT = 700 * SCALING_FACTOR

# Ability text
ABILITY_TEXT_WIDTH_START = int(9/100 * CARD_WIDTH)
ABILITY_TEXT_WIDTH_END = int(91/100 * CARD_WIDTH)
ABILITY_TEXT_HEIGHT_START = int(86 / 140 * CARD_HEIGHT)
ABILITY_TEXT_SIZE = 15 * SCALING_FACTOR
ABILITY_TEXT_COLOR = (1, 1, 1, 255)

# Power number position
POWER_NUMBER_WIDTH_START = int(500 / 1000 * CARD_WIDTH)
POWER_NUMBER_HEIGHT_START = int(1242 / 1400 * CARD_HEIGHT)
POWER_NUMBER_SIZE = 75 * SCALING_FACTOR
POWER_NUMBER_COLOR = (9, 9, 235, 255)

# Mana number position
# Position x (width) isn't absolute. It needs to be adjusted by taking into account the width of an image
MANA_NUMBER_WIDTH_START = int(925/1000 * CARD_WIDTH)
MANA_NUMBER_HEIGHT_START = int(1242 / 1400 * CARD_HEIGHT)
MANA_NUMBER_SIZE = 75 * SCALING_FACTOR
MANA_NUMBER_COLOR = (255, 150, 0, 255)

# Cost number position
# Position x (width) isn't absolute. It needs to be adjusted by taking into account the width of an image
COST_NUMBER_WIDTH_START = int(75 / 1000 * CARD_WIDTH)
COST_NUMBER_HEIGHT_START = int(1242 / 1400 * CARD_HEIGHT)
COST_NUMBER_SIZE = 75 * SCALING_FACTOR
COST_NUMBER_COLOR = (28, 157, 255, 255)

# Card name position
NAME_TEXT_WIDTH_START = int(63/1000 * CARD_WIDTH)
NAME_TEXT_WIDTH_END = int(87/100 * CARD_WIDTH)
NAME_TEXT_HEIGHT_START = int(96/1400 * CARD_HEIGHT)
NAME_TEXT_SIZE = 30 * SCALING_FACTOR
NAME_COLOR = (1, 1, 1, 255)

# Main character image
CREATURE_IMAGE_WIDTH = 448 * SCALING_FACTOR
CREATURE_IMAGE_HEIGHT = 343 * SCALING_FACTOR
CREATURE_IMAGE_HEIGHT_START = 71 * SCALING_FACTOR
CREATURE_IMAGE_WIDTH_START = 27 * SCALING_FACTOR
CREATURE_IMAGE_ROUNDING_RADIUS = 30

# Allegience text
ALLEGIENCE_TEXT_HEIGHT = int(55/1400 * CARD_HEIGHT)
ALLEGIENCE_TEXT_SIZE = 9.5 * SCALING_FACTOR
ALLEGIENCE_TEXT_COLOR = (1, 1, 1, 255)

# Allegience logos
LOGO_WIDTH = int(27.5 * SCALING_FACTOR)
LOGO_HEIGHT = int(27.5 * SCALING_FACTOR)
LOGO_BORDER_WIDTH = int(3 * SCALING_FACTOR)
LOGO_WIDTH_START = int(946/1000 * CARD_WIDTH)
LOGO_HEIGHT_START = int(81/1400 * CARD_HEIGHT) - LOGO_BORDER_WIDTH

# Space between next line of text
LINE_SPACE_HEIGHT = 1

# Heights of vertixes making a color triangle
TRICOLOR_TRIANGLE_VERTICE_HEIGHT1 = 450 * SCALING_FACTOR
TRICOLOR_TRIANGLE_VERTICE_HEIGHT2 = 325 * SCALING_FACTOR

# Fonts used
NUMBER_FONT = "Ancient Medium.ttf"
TEXT_FONT = "ButlerModified.otf"

# Constant card colors assigned to database input
COLOUR_DICT = {"Vampire": (42, 42, 42),
                "Dragon": (21, 21, 200),
                "Human": (166, 158, 204),
                "Horror": (175, 38, 83),
                "Demon": (31, 0, 119),
                "Undead": (43, 8, 84),
                "Construct": (113, 113, 113),
                "Angel": (204, 204, 204),
                "Warrior": (41, 57, 67),
                "Mage": (204, 204, 0),
                "Beast": (0, 82, 0),
                "Knight": (138, 41, 41),
                "Hunter": (61, 173, 61),
                "Noble": (21, 175, 204)
                }


# Main card class
class Card:
    def __init__(self, name, ability_text, power, mana, cost, creature_path, **allegiances):
        self.name = name
        self.ability_text = ability_text
        self.mana = mana
        self.power = power
        self.cost = cost
        self.creature_path = creature_path

        # Dynamically assign allegiance attributes using the allegiances dictionary
        for key in allegiances:
            setattr(self, key.lower(), allegiances.get(key.lower(), 0))

    # Allegiences define card colors
    # Using COLOUR_DICT since it has all the allegiences names already
    def get_allegiences(self):
        allegiences = [key for key, _ in COLOUR_DICT.items() if getattr(self, key.lower(), 0) in (1, "1")]

        return allegiences
    

# Adds two images, one on top of another with an option of displacing the top one
def add_two_images(bottom, top, displacement):
    # Limiting dimensions so that the output image doesn't have to be enlarged
    # Also figuring out new image's dimensions
    if displacement[0] + top.shape[1] > bottom.shape[1]:
        temp_width = displacement[0] + top.shape[1]
        right_border = temp_width - bottom.shape[1]
    else:
        temp_width = bottom.shape[1]
        right_border = 0

    if displacement[1] + top.shape[0] > bottom.shape[0]:
        temp_height = displacement[1] + top.shape[0]
        bottom_border = temp_height - bottom.shape[0]
    else:
        temp_height = bottom.shape[0]
        bottom_border = 0

    # Creating a new image
    output_image = np.zeros((temp_height, temp_width, 4), np.uint8)

    # Enlarging bottom image with makeborder
    if temp_height != bottom.shape[0] or temp_width != bottom.shape[1]:
        output_image = cv.copyMakeBorder(bottom, 0, bottom_border, 0, right_border, cv.BORDER_CONSTANT, None, (0, 0, 0, 0))
    else:
        output_image = bottom.copy()

    displaced_top = np.zeros((temp_height, temp_width, 4), np.uint8)
    
    # Displace top image
    for y in range(top.shape[0]):
        for x in range(top.shape[1]):
            displaced_top[y + displacement[1], x + displacement[0]] = top[y, x]

    output_image[np.where(displaced_top[:, :, 3] >= 0.001)] = displaced_top[np.where(displaced_top[:, :, 3] >= 0.001)]

    return output_image


def create_text_image_PIL(text, size, color, font_path):
    font = ImageFont.truetype(font_path, size)

    # Create a temporary ImageDraw instance to get text size
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    
    # Get the bounding box (top-left to bottom-right coordinates)
    bbox = draw.textbbox((0, 0), text, font=font)
    
    # Extract width and height with precise calculation
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Create an image based on the correct width and height
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image_pil = Image.fromarray(image)

    # Draw the text on the image with adjusted positioning
    draw = ImageDraw.Draw(image_pil)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=color)

    # Convert back to OpenCV format
    image = np.array(image_pil)

    return image


def getTextWidth(text, font):
    # Create a temporary ImageDraw instance to get text size
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    # Get the bounding box (top-left to bottom-right coordinates)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    return text_width


def wrap_text(text, font, max_width):
    if getTextWidth(text, font) <= max_width:
        lines = [text]
    else:
        textSplit = text.split(" ")
        textSplitLen = []
        textLenSum = 0
        lines = []
        line = ""
        i = 0

        spaceLen = getTextWidth(" ", font)

        for i in range(len(textSplit)):
            textSplitLen.append(getTextWidth(textSplit[i], font))

        for i in range(len(textSplitLen)):
            if textLenSum + textSplitLen[i] + spaceLen > max_width:
                lines.append(line)
                line = ""
                textLenSum = 0

            textLenSum = textLenSum + textSplitLen[i] + spaceLen

            if line == "":
                line = textSplit[i]
            else:
                line = line + " " + textSplit[i]

        if lines[-1] != line:
            lines.append(line)
    
    return lines


def create_text_image(text, font_path, fontSize, color):
    font = ImageFont.truetype(font_path, fontSize)
    lines = wrap_text(text, font, ABILITY_TEXT_WIDTH_END - ABILITY_TEXT_WIDTH_START)
    text_image = create_text_image_PIL(lines[0], fontSize, color, font_path)
    for i in range(len(lines)-1):
        image = create_text_image_PIL(lines[i+1], fontSize, color, font_path)
        text_image = add_two_images(text_image, image, (0, LINE_SPACE_HEIGHT + text_image.shape[0]))

    return text_image


# Cropping an image to content, used for morphological operations on angled gradients in tricolor card images
def crop_to_content(image):
    image_data = np.asarray(image)
    image_data_bw = image_data.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

    return image_data_new


# Rounding corners of an image, used to round character image only
def round_corners(image, r):
    h, w = image.shape[:2]
    t = 1
    c = (255, 255, 255)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw four quarter-circles in corners
    mask = cv.ellipse(mask, (int(r+t/2), int(r+t/2)), (r, r), 180, 0, 90, c, t)
    mask = cv.ellipse(mask, (int(w-r-t/2-1), int(r+t/2)), (r, r), 270, 0, 90, c, t)
    mask = cv.ellipse(mask, (int(r+t/2), int(h-r-t/2-1)), (r, r), 90, 0, 90, c, t)
    mask = cv.ellipse(mask, (int(w-r-t/2-1), int(h-r-t/2-1)), (r, r), 0, 0, 90, c, t)

    # Draw borders between quarter-circles
    mask = cv.line(mask, (int(r+t/2), int(t/2)), (int(w-r+t/2-1), int(t/2)), c, t)
    mask = cv.line(mask, (int(t/2), int(r+t/2)), (int(t/2), int(h-r+t/2)), c, t)
    mask = cv.line(mask, (int(r+t/2), int(h-t/2)), (int(w-r+t/2-1), int(h-t/2)), c, t)
    mask = cv.line(mask, (int(w-t/2), int(r+t/2)), (int(w-t/2), int(h-r+t/2)), c, t)

    # Fill the mask
    cv.floodFill(mask, None, (w//2, h//2), c)

    # Apply to mask
    image = cv.bitwise_and(image, image, mask=mask)

    return image


def create_angled_gradients(card_colour_image, card_image, rotation_sign, x_placement, second_color_width):
    # Get gradient starting colors from our image
    gradient_start_colour = card_colour_image[1, second_color_width, :4]
    gradient_end_colour = card_colour_image[CARD_HEIGHT-1, 1, :4]

    # Create a gradient
    gradient = np.linspace(gradient_start_colour, gradient_end_colour, card_colour_image.shape[0]//6, axis=0).astype(np.uint8)

    # Calculate the angle between card's bottom and triangle's hypotenuse
    angle = math.degrees(math.atan((TRICOLOR_TRIANGLE_VERTICE_HEIGHT1 - TRICOLOR_TRIANGLE_VERTICE_HEIGHT2) / (CARD_WIDTH/2)))

    # Set 2D gradient's dimensions
    gradient_length = int((CARD_WIDTH / 2) / math.cos(math.radians(angle)))
    gradient_width = card_colour_image.shape[0]//6

    # Create an image and fill it with our gradient
    gradient_image = np.zeros((gradient_length, gradient_width, 4), dtype=np.uint8)
    gradient_image[:, :] = gradient.copy()

    # Rotate the gradient by the calculated angle and then crop resulting image to fit only the content
    gradient_full_image = scipy.ndimage.rotate(gradient_image, rotation_sign*angle-90, reshape=True, order=5)
    gradient_full_image = crop_to_content(gradient_full_image)

    # Erode gradient to get rid of all border inconsistencies resulting in interpolation from rotation
    kernel = np.ones((2,2), np.uint8)
    gradient_full_image = cv.erode(gradient_full_image, kernel, iterations=1)

    # Crop to content again to cut borders removed by eroding
    gradient_full_image = crop_to_content(gradient_full_image)

    # Find lowest non-black pixel
    i = 0
    non_black_pixel = np.argwhere(gradient_full_image[gradient_full_image.shape[0]-1, :, :3] > 0)
    while len(non_black_pixel) == 0:
        i = i + 1
        non_black_pixel = np.argwhere(gradient_full_image[gradient_full_image.shape[0]-1-i, :, :3] > 0)

    # Cut everything to the left of the lowest non-black pixel, so that the left side of our image matches the left card side
    if rotation_sign == 1:
        gradient_full_image[:, :non_black_pixel[0][0]] = 0
    elif rotation_sign == -1:
        gradient_full_image[:, non_black_pixel[0][0]:] = 0

    gradient_full_image = crop_to_content(gradient_full_image)

    # Combine gradient image on top of card image right in between the two colours
    card_colour_image = add_two_images(card_colour_image, gradient_full_image, (x_placement, TRICOLOR_TRIANGLE_VERTICE_HEIGHT2 - gradient_width//2 + gradient_width//8))

    # Add card borders
    card_colour_image[card_image.copy()[:, :, 3] == 0] = [0, 0, 0, 0]

    return card_colour_image


# Main card generation function
def create_card(card_template_path, border_template_path, creature_image_path, sparks_path, logos_path, power, mana, cost, ability_text, name_text, allegiences, output_path):
    # Essentially sanitizing database output
    if power == "None":
        power = "1"
    if mana == "None":
        mana = "1"
    if cost == "None":
        cost = "1"
    if ability_text == "None":
        ability_text = "Hey"
    if name_text == "None":
        name_text = "MK"

    # Loading template images
    card_image = cv.imread(card_template_path, cv.IMREAD_UNCHANGED)
    card_image = cv.resize(card_image, (CARD_WIDTH, CARD_HEIGHT), interpolation = cv.INTER_AREA)

    colour_image = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), np.uint8)

    sparks_image = cv.imread(sparks_path, cv.IMREAD_UNCHANGED)
    sparks_image = cv.resize(sparks_image, (CARD_WIDTH, CARD_HEIGHT))

    border_template = cv.imread(border_template_path, cv.IMREAD_UNCHANGED)
    border_template = cv.resize(border_template, (CARD_WIDTH, CARD_HEIGHT), interpolation = cv.INTER_AREA)

    creature_image = cv.imread(creature_image_path, cv.IMREAD_UNCHANGED)
    creature_image = cv.resize(creature_image, (CREATURE_IMAGE_WIDTH, CREATURE_IMAGE_HEIGHT), interpolation = cv.INTER_AREA)

    mask = np.all(card_image[:, :, :3] != 255, axis=-1)
    sparks_image[mask] = [0, 0, 0, 0]

    sparks_image_inner = sparks_image.copy()
    sparks_image_outer = sparks_image.copy()

    # Extract and threshold the alpha channel to create a binary mask, find its largest contour and create a mask from that
    # It separates sparks_image on the borders from its inner counterpart, because they will have different opacity
    alpha_channel = border_template[:, :, 3]

    _, binary = cv.threshold(alpha_channel, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv.contourArea)
    border_template_mask = np.zeros_like(alpha_channel)
    cv.drawContours(border_template_mask, [largest_contour], -1, 255, thickness=cv.FILLED)

    reverse_border_template_mask = cv.bitwise_not(border_template_mask)

    sparks_image_inner = cv.bitwise_and(sparks_image_inner, sparks_image_inner, mask=border_template_mask)
    sparks_image_outer = cv.bitwise_and(sparks_image_outer, sparks_image_outer, mask=reverse_border_template_mask)

    # Adding alpha channel if it's not present yet
    if creature_image.shape[2] != 4:
        # First create the image with alpha channel
        creature_image = cv.cvtColor(creature_image, cv.COLOR_RGB2RGBA)

        # Then assign the mask to the last channel of the image
        creature_image[:, :, 3] = np.ones((creature_image.shape[0], creature_image.shape[1]), np.uint8)

    creature_image[:, :, 3] = 255
    creature_image = round_corners(creature_image, CREATURE_IMAGE_ROUNDING_RADIUS)

    # One allegience which means one color card
    if len(allegiences) == 1:
        # Change color of any non-black pixel to a color associated with card's allegience
        colour_image[np.any(card_image.copy()[:, :, :3] != [0, 0, 0], axis=2)] = (COLOUR_DICT[allegiences[0]][0], COLOUR_DICT[allegiences[0]][1], COLOUR_DICT[allegiences[0]][2], 255)
        card_colour_image = colour_image
    else:
        # Two allegiences which means two color card
        if len(allegiences) == 2:
            # Divide the card vertically into two, then apply color associated with card's allegience
            left_image = card_image.copy()[:, :CARD_WIDTH//2, :]
            left_image[np.any(left_image.copy()[:, :, :3] != [0, 0, 0], axis=2)] = (COLOUR_DICT[allegiences[0]][0], COLOUR_DICT[allegiences[0]][1], COLOUR_DICT[allegiences[0]][2], 255)

            right_image = card_image.copy()[:, CARD_WIDTH//2:, :]
            right_image[np.any(right_image[:, :, :3] != [0, 0, 0], axis=2)] = (COLOUR_DICT[allegiences[1]][0], COLOUR_DICT[allegiences[1]][1], COLOUR_DICT[allegiences[1]][2], 255)

            card_colour_image = cv.hconcat([left_image.copy(), right_image.copy()])

            gradient_start_colour = left_image[-100 * SCALING_FACTOR, 100 * SCALING_FACTOR, :4]
            gradient_end_colour = right_image[-100 * SCALING_FACTOR, -100 * SCALING_FACTOR, :4]
            gradient = np.linspace(gradient_start_colour, gradient_end_colour, card_colour_image.shape[1]//3, axis=0).astype(np.uint8)

            gradient_image = np.zeros((CARD_HEIGHT, card_colour_image.shape[1]//3, 4), dtype=np.uint8)
            gradient_image[:, :] = gradient.copy()

            card_colour_image[:, 2*CARD_WIDTH//6:CARD_WIDTH*4//6, :] = gradient_image.copy()[:, :, :]

            card_colour_image[card_image.copy()[:, :, 3] == 0] = [0, 0, 0, 0]
        else:
            if len(allegiences) == 3:
                mask_triangle = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)
                mask_trapeze_left = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)
                mask_trapeze_right = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)

                triangle_vertices = np.array([(0, TRICOLOR_TRIANGLE_VERTICE_HEIGHT1), (CARD_WIDTH, TRICOLOR_TRIANGLE_VERTICE_HEIGHT1), (CARD_WIDTH // 2, TRICOLOR_TRIANGLE_VERTICE_HEIGHT2)])
                mask_triangle = cv.drawContours(mask_triangle.copy(), [triangle_vertices], 0, ([COLOUR_DICT[allegiences[2]][0], COLOUR_DICT[allegiences[2]][1], COLOUR_DICT[allegiences[2]][2], 255]), -1)
                mask_triangle = cv.drawContours(mask_triangle.copy(), [np.array([(0, TRICOLOR_TRIANGLE_VERTICE_HEIGHT1), (0, CARD_HEIGHT), (CARD_WIDTH, CARD_HEIGHT), (CARD_WIDTH, TRICOLOR_TRIANGLE_VERTICE_HEIGHT1)])], 0, ([COLOUR_DICT[allegiences[2]][0], COLOUR_DICT[allegiences[2]][1], COLOUR_DICT[allegiences[2]][2], 255]), -1)

                mask_trapeze_left[:, :CARD_WIDTH // 2] = [255, 255, 255, 255]
                mask_trapeze_right[:, CARD_WIDTH // 2:] = [255, 255, 255, 255]

                left_indices = np.where((mask_trapeze_left.copy() == [255, 255, 255, 255]) & (mask_triangle.copy() != [COLOUR_DICT[allegiences[2]][0], COLOUR_DICT[allegiences[2]][1], COLOUR_DICT[allegiences[2]][2], 255]))
                right_indices = np.where((mask_trapeze_right.copy() == [255, 255, 255, 255]) & (mask_triangle.copy() != [COLOUR_DICT[allegiences[2]][0], COLOUR_DICT[allegiences[2]][1], COLOUR_DICT[allegiences[2]][2], 255]))
                triangle_indices = np.where(mask_triangle.copy() == [COLOUR_DICT[allegiences[2]][0], COLOUR_DICT[allegiences[2]][1], COLOUR_DICT[allegiences[2]][2], 255])
                colour_image_1 = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)
                colour_image_2 = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)
                colour_image_3 = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)

                colour_image_1[left_indices[0], left_indices[1], :] = [COLOUR_DICT[allegiences[0]][0], COLOUR_DICT[allegiences[0]][1], COLOUR_DICT[allegiences[0]][2], 255]
                colour_image_2[right_indices[0], right_indices[1], :] = [COLOUR_DICT[allegiences[1]][0], COLOUR_DICT[allegiences[1]][1], COLOUR_DICT[allegiences[1]][2], 255]
                colour_image_3 = mask_triangle.copy()

                card_colour_image_1 = colour_image_1
                card_colour_image_2 = colour_image_2
                card_colour_image_3 = colour_image_3
                
                left_mask = np.ones((CARD_HEIGHT, CARD_WIDTH), dtype=bool)
                right_mask = np.ones((CARD_HEIGHT, CARD_WIDTH), dtype=bool)
                triangle_mask = np.ones((CARD_HEIGHT, CARD_WIDTH), dtype=bool)

                left_mask[left_indices[0], left_indices[1]] = False
                right_mask[right_indices[0], right_indices[1]] = False
                triangle_mask[triangle_indices[0], triangle_indices[1]] = False

                card_colour_image_1[left_mask] = [0, 0, 0, 0]
                card_colour_image_2[right_mask] = [0, 0, 0, 0]
                card_colour_image_3[triangle_mask] = [0, 0, 0, 0]

                card_colour_image = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), np.uint8)
                mask_1 = np.any(card_colour_image_1[:, :, :3] != 0, axis=2)
                mask_2 = np.any(card_colour_image_2[:, :, :3] != 0, axis=2)
                mask_3 = np.any(card_colour_image_3[:, :, :3] != 0, axis=2)

                # Combine images based on non-black pixel masks
                card_colour_image = np.zeros_like(card_colour_image_1)
                card_colour_image[mask_1] = card_colour_image_1[mask_1]
                card_colour_image[mask_2] = card_colour_image_2[mask_2]

                card_colour_image[np.where(mask_3 == True)] = card_colour_image_3[np.where(mask_3 == True)]
                card_colour_image[np.where(card_colour_image[:, :, :3] == (0, 0, 0))] = card_colour_image_3[np.where(card_colour_image[:, :, :3] == (0, 0, 0))]

                # Gradient left-right
                # Get gradient starting colors from our image
                gradient_start_colour = card_colour_image[1, 1, :4]
                gradient_end_colour = card_colour_image[1, CARD_WIDTH-1, :4]

                # Create a gradient
                gradient = np.linspace(gradient_start_colour, gradient_end_colour, card_colour_image.shape[1]//6, axis=0).astype(np.uint8)

                # Create an image and fill it with our gradient
                gradient_image = np.zeros((TRICOLOR_TRIANGLE_VERTICE_HEIGHT2, card_colour_image.shape[1]//6, 4), dtype=np.uint8)
                gradient_image[:, :] = gradient.copy()

                # Place gradient onto the card
                card_colour_image[:TRICOLOR_TRIANGLE_VERTICE_HEIGHT2, 5*CARD_WIDTH//12:CARD_WIDTH*7//12-1, :] = gradient_image.copy()[:, :, :]

                # Create left-bottom gradient
                card_colour_image = create_angled_gradients(card_colour_image, card_image, 1, 0, 1)

                # Create right-bottom gradient
                card_colour_image = create_angled_gradients(card_colour_image, card_image, -1, CARD_WIDTH//2, CARD_WIDTH-1)

    card_colour_image = cv.addWeighted(card_colour_image.copy(), 1, sparks_image_inner.copy(), 0.025, 0.0)
    card_colour_image = cv.addWeighted(card_colour_image.copy(), 1, sparks_image_outer.copy(), 0.15, 0.0)

    power_number_image = create_text_image(power, os.path.join(logos_path, NUMBER_FONT), POWER_NUMBER_SIZE, POWER_NUMBER_COLOR)
    power_number_image = image_outline(power_number_image)

    mana_number_image = create_text_image(mana, os.path.join(logos_path, NUMBER_FONT), MANA_NUMBER_SIZE, MANA_NUMBER_COLOR)
    mana_number_image = image_outline(mana_number_image)

    cost_number_image = create_text_image(cost, os.path.join(logos_path, NUMBER_FONT), COST_NUMBER_SIZE, COST_NUMBER_COLOR)
    cost_number_image = image_outline(cost_number_image)

    ability_text_image = create_text_image(ability_text, os.path.join(logos_path, TEXT_FONT), ABILITY_TEXT_SIZE, ABILITY_TEXT_COLOR)

    name_text_image = create_text_image(name_text, os.path.join(logos_path, TEXT_FONT), NAME_TEXT_SIZE, NAME_COLOR)

    allegience_text_images = []
    for i in range(len(allegiences)):
        allegience_text_images.append(create_text_image(allegiences[i], os.path.join(logos_path, TEXT_FONT), ALLEGIENCE_TEXT_SIZE, ALLEGIENCE_TEXT_COLOR))

    creature_image = cv.copyMakeBorder(creature_image.copy(), CREATURE_IMAGE_HEIGHT_START, 0, CREATURE_IMAGE_WIDTH_START, 0, cv.BORDER_CONSTANT, None, (0, 0, 0, 0))

    image = border_template
    for i in range(len(allegiences)):
        logo_template = os.path.join(logos_path, ("Logo" + allegiences[i] + ".png"))
        logo = cv.imread(logo_template, cv.IMREAD_UNCHANGED)
        logo = replace_color(logo, [8, 255, 0], [0, 0, 0], 3)
        logo = replace_color(logo, [255, 255, 255], [COLOUR_DICT[allegiences[i]][0], COLOUR_DICT[allegiences[i]][1], COLOUR_DICT[allegiences[i]][2]], 3)
        logo = cv.resize(logo, (LOGO_WIDTH, LOGO_HEIGHT), interpolation = cv.INTER_AREA)
        logo = cv.copyMakeBorder(logo.copy(), LOGO_BORDER_WIDTH, 0, LOGO_BORDER_WIDTH, 0, cv.BORDER_CONSTANT, None, (0, 0, 0, 255))
        image = add_two_images(image, logo, (LOGO_WIDTH_START - (len(allegiences) - i) * (LOGO_WIDTH + LOGO_BORDER_WIDTH), LOGO_HEIGHT_START))

    # Adding border template once again, because we want it on top of allegience symbols
    # This technically should be fixed in add_two_images, but that function isn't doing great when the bottom image is smaller than the top one
    image = add_two_images(image, border_template, (0, 0))

    creature_n_border = add_two_images(creature_image, image, (0, 0))

    creature_n_border_n_colour = add_two_images(card_colour_image.copy(), creature_n_border, (0, 0))
    creature_n_border_n_colour[:, :, 3] = 255

    image3 = add_two_images(creature_n_border_n_colour, power_number_image, (POWER_NUMBER_WIDTH_START - power_number_image.shape[1]//2, POWER_NUMBER_HEIGHT_START))
    image4 = add_two_images(image3, mana_number_image, (MANA_NUMBER_WIDTH_START - mana_number_image.shape[1], MANA_NUMBER_HEIGHT_START))
    image6 = add_two_images(image4, cost_number_image, (COST_NUMBER_WIDTH_START, COST_NUMBER_HEIGHT_START))
    image7 = add_two_images(image6, ability_text_image, (ABILITY_TEXT_WIDTH_START, ABILITY_TEXT_HEIGHT_START))
    image = add_two_images(image7, name_text_image, (int(NAME_TEXT_WIDTH_START), NAME_TEXT_HEIGHT_START - name_text_image.shape[0]//2))
    for i in range(len(allegiences)):
        image = add_two_images(image, allegience_text_images[i], (LOGO_WIDTH_START + (LOGO_WIDTH + 2 * LOGO_BORDER_WIDTH)//2 - allegience_text_images[i].shape[1]//2 - (len(allegiences) - i) * (LOGO_WIDTH + LOGO_BORDER_WIDTH), ALLEGIENCE_TEXT_HEIGHT))

    image[:, :, 3] = 255

    cv.imwrite(os.path.join(output_path, str(name_text) + ".png"), image)


# Draws a 2 pixel outline around an image
def image_outline(input):
    # Pad the image so that dilatation fits in it
    image = cv.copyMakeBorder(input.copy(), 2, 2, 2, 2, cv.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # Turn all non-transparent pixels fully opaque
    mask = image[:, :, 3] > 0
    image[mask, :3] = 255

    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

    # Turn all non-black pixels white
    gray_image[gray_image > 0] = 255

    # Dilate white area
    dilated = cv.dilate(gray_image, np.ones((5, 5), np.uint8), iterations=1)

    # Switch colors, black to white and white to black
    dilated_corrected = 255 - dilated

    # Convert to rgba
    color_image = cv.cvtColor(dilated_corrected, cv.COLOR_GRAY2BGRA)

    # Turn all black pixels transparent
    mask = np.all(color_image[:, :, :3] == [255, 255, 255], axis=-1)
    color_image[mask, 3] = 0

    # Overlay input image on top of dilated image
    output = add_two_images(color_image, input, (2, 2))

    return output


def replace_color(image, target_color, replacement_color, threshold=40):
    # If input image has alpha, split it and bgr channels
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image

    # Iterate over every pixel and replace color if it matches target color within threshold
    for i in range(bgr.shape[0]):
        for j in range(bgr.shape[1]):
            pixel = bgr[i, j]
            
            # Calculate the absolute difference for each channel
            diff = np.abs(pixel - target_color)
            
            # Sum the differences
            distance = np.sum(diff)

            # If the distance is less than the threshold, replace the color
            if distance < threshold:
                bgr[i, j] = replacement_color

    output = bgr

    # If the image had an alpha channel, merge it back
    if image.shape[2] == 4:
        output = cv.merge((output, alpha))

    return output


def read_database(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # reading all table names
    record_list = [a for a in cursor.execute("SELECT * FROM CARDS")]
    cards = []
    for record in record_list:
        cards.append(Card(record[1], record[2], record[3], record[4], record[5], record[6], vampire=record[7], dragon=record[8], human=record[9], horror=record[10], demon=record[11], undead=record[12], construct=record[13], angel=record[14], warrior=record[15], mage=record[16], beast=record[17], knight=record[18], hunter=record[19], noble=record[20]))

    connection.close()

    return cards
            

def main():
    default_path = os.path.dirname(os.path.abspath(__file__))

    card_template = "WhiteTemplate.png"
    border_template = "InnerBorderTemplate.png"
    sparks_template = "Web.png"
    characters_directory = "Characters"
    output_directory = "Output"

    os.makedirs(os.path.join(default_path, "Templates"), exist_ok=True)
    os.makedirs(os.path.join(default_path, characters_directory), exist_ok=True)
    os.makedirs(os.path.join(default_path, output_directory), exist_ok=True)

    card_template_path = os.path.join(default_path, "Templates", card_template)
    border_template_path = os.path.join(default_path, "Templates", border_template)
    sparks_template_path = os.path.join(default_path, "Templates", sparks_template)

    cards = read_database(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RogueProjectDB.db"))
    
    for card in cards:
        try:
            create_card(card_template_path,
                        border_template_path,
                        os.path.join(default_path, characters_directory, card.creature_path),
                        sparks_template_path,
                        os.path.join(default_path, "Templates"),
                        str(card.power),
                        str(card.mana),
                        str(card.cost),
                        str(card.ability_text),
                        str(card.name),
                        card.get_allegiences(),
                        os.path.join(default_path, output_directory)
                        )
            
        except Exception as e:
            print(f"Error processing card '{card.name}': {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()