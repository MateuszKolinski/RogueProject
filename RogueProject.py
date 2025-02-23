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

# Scales all objects in the card image
SCALING_FACTOR = 2

# Constant card and object dimension settings
CARD_WIDTH = 500 * SCALING_FACTOR
CARD_HEIGHT = 700 * SCALING_FACTOR

COLOUR_THRESHOLD = (0.001, 0.001, 0.001)

NUMBER_WIDTH = 100 * SCALING_FACTOR
NUMBER_HEIGHT = 100 * SCALING_FACTOR

TEXT_WIDTH_START = int(9/100 * CARD_WIDTH)
TEXT_WIDTH_END = int(91/100 * CARD_WIDTH)

TEXT_HEIGHT_START = int(89 / 140 * CARD_HEIGHT)

ATTACK_NUMBER_WIDTH = int(75 / 1000 * CARD_WIDTH)
ATTACK_NUMBER_HEIGHT = int(1250 / 1400 * CARD_HEIGHT)

MANA_NUMBER_WIDTH = int(925/1000 * CARD_WIDTH)
MANA_NUMBER_HEIGHT = int(65/1400 * CARD_HEIGHT)

HEALTH_NUMBER_WIDTH = int(925/1000 * CARD_WIDTH)
HEALTH_NUMBER_HEIGHT = ATTACK_NUMBER_HEIGHT

COST_NUMBER_WIDTH = ATTACK_NUMBER_WIDTH
COST_NUMBER_HEIGHT = int(65/1400 * CARD_HEIGHT)

NAME_WIDTH_START = int(13/100 * CARD_WIDTH)
NAME_WIDTH_END = int(87/100 * CARD_WIDTH)
NAME_HEIGHT = COST_NUMBER_HEIGHT

CREATURE_WIDTH = 448 * SCALING_FACTOR
CREATURE_HEIGHT = 343 * SCALING_FACTOR

ALLEGIENCE_HEIGHT = int(130/1400 * CARD_HEIGHT)

TRICOLOR_TRIANGLE_VERTICE_HEIGHT1 = 450 * SCALING_FACTOR
TRICOLOR_TRIANGLE_VERTICE_HEIGHT2 = 325 * SCALING_FACTOR

LOGO_WIDTH = int(55.5 * SCALING_FACTOR)
LOGO_HEIGHT = int(55.5 * SCALING_FACTOR)

LOGO_POSITION_H = int(1235 / 1400 * CARD_HEIGHT)

LINE_SPACE_HEIGHT = 1

# Constant card colors assigned to database input
COLOUR_DICT = {"Vampire": (1, 1, 1, 0.2),
                "Dragon": (0, 0, 250, 0.1),
                "Human": (203, 192, 255, 0.1),
                "Horror": (215, 24, 87, 0.1),
                "Demon": (39, 0, 149, 0),
                "Undead": (105, 10, 54, 0),
                "Construct": (128, 128, 128, 0.1),
                "Angel": (255, 255, 255, 0.7),
                "Warrior": (0, 25, 41, 0.2),
                "Mage": (255, 255, 0, 0),
                "Beast": (0, 102, 0, 0),
                "Knight": (153, 0, 0, 0.2),
                "Hunter": (0, 200, 0, 0.3),
                "Noble": (0, 215, 255, 0.1)
                }


class Card:
    def __init__(self, name, ability_text, mana, attack, health, cost, creature_path, **allegiances):
        self.name = name
        self.ability_text = ability_text
        self.mana = mana
        self.attack = attack
        self.health = health
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
    # output_image[np.where(displaced_top[:, :, :3] >= COLOUR_THRESHOLD)] = displaced_top[np.where(displaced_top[:, :, 3] >= COLOUR_THRESHOLD)]

    return output_image


# Creates an image containing desired text
def create_text_image(text, size, thickness, colour, font, width):
    lines = wrap_text(text, size, thickness, font, width)
    line_images = []
    for line in lines:
        temp_image = putText_MK(line, size, thickness, font, colour)
        temp_image = cv.cvtColor(temp_image, cv.COLOR_RGB2RGBA)

        # Then assign the mask to the last channel of the image
        temp_image[:, :, 3] = np.ones((temp_image.shape[0], temp_image.shape[1]), np.uint8)
        
        temp_image[np.all(temp_image == (0, 0, 0, 1), axis=-1)] = (0, 0, 0, 0)
        line_images.append(temp_image)

    temp = line_images[0]
    for i in range(len(line_images)-1):
        temp = add_two_images(temp, line_images[i+1], (0, LINE_SPACE_HEIGHT + line_images[i].shape[0]*(i+1)))

    return temp


# A function cropping a text image to its size, since cv.puttext doesn't do that automatically
def putText_MK(text, scale, thickness, font, colour):
    base = np.zeros((2000, 2000, 3), np.uint8)
    text_img = cv.putText(base, text, (1000, 1000), font, scale, colour, thickness, cv.FILLED)

    # Greyscale
    img = cv.cvtColor(text_img, cv.COLOR_BGR2GRAY)

    # Horizontal close
    kernel = np.ones((5, 191), np.uint8)
    morph = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    # Find contours and bounding rectangle
    contours, hierarchy = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv.boundingRect(cnt)

    # Crop
    crop = text_img[y:y+h,x:x+w]
        
    return crop


# Wrapping image text lines
def wrap_text(text, scale, thickness, font, width):
    textSize = cv.getTextSize(text, font, scale, thickness)
    spaceSize = cv.getTextSize(" ", font, scale, thickness)
    if textSize[0][0] > width:
        textSplit = text.split(" ")
        textSplitLen = []
        for i in range(len(textSplit)):
            textSplitLen.append(cv.getTextSize(textSplit[i], font, scale, thickness))

        textLenSum = 0
        lines = []
        line = ""
        for i in range(len(textSplitLen)):
            if textSplitLen[i][0][0] > width:
                print("You don goofed up")

            if textLenSum + textSplitLen[i][0][0] + spaceSize[0][0] > width:
                lines.append(line)
                line = ""
                textLenSum = 0

            textLenSum = textLenSum + textSplitLen[i][0][0] + spaceSize[0][0]
            line = line + " " + textSplit[i]

        if lines[-1] != line:
            lines.append(line)

        return lines
    
    text_list = [text]
    return text_list


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
def create_card(card_template_path, border_template_path, creature_image_path, sparks_path, logos_path, attack, mana, health, cost, ability_text, name_text, allegiences, output_path):
    # Essentially sanitizing database output
    if attack == "None":
        attack = "1"
    if health == "None":
        health = "1"
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
    creature_image = cv.resize(creature_image, (CREATURE_WIDTH, CREATURE_HEIGHT), interpolation = cv.INTER_AREA)

    # Adding alpha channel if it's not present yet
    if creature_image.shape[2] != 4:
        # First create the image with alpha channel
        creature_image = cv.cvtColor(creature_image, cv.COLOR_RGB2RGBA)

        # Then assign the mask to the last channel of the image
        creature_image[:, :, 3] = np.ones((creature_image.shape[0], creature_image.shape[1]), np.uint8)

    creature_image[:, :, 3] = 255
    creature_image = round_corners(creature_image, 30)

    # One allegience which means one color card
    if len(allegiences) == 1:
        # Change color of any non-black pixel to a color associated with card's allegience
        colour_image[np.any(card_image.copy()[:, :, :3] != [0, 0, 0], axis=2)] = (COLOUR_DICT[allegiences[0]][0], COLOUR_DICT[allegiences[0]][1], COLOUR_DICT[allegiences[0]][2], 255)

        # White-blend an image over it to 'mellow' the color down a little
        alpha = COLOUR_DICT[allegiences[0]][3]
        card_colour_image = cv.addWeighted(card_image.copy(), alpha, colour_image.copy(), 1-alpha, 0.0)
    else:
        # Two allegiences which means two color card
        if len(allegiences) == 2:
            # Divide the card vertically into two, then apply color associated with card's allegience
            left_image = card_image.copy()[:, :CARD_WIDTH//2, :]
            left_image[np.any(left_image.copy()[:, :, :3] != [0, 0, 0], axis=2)] = (COLOUR_DICT[allegiences[0]][0], COLOUR_DICT[allegiences[0]][1], COLOUR_DICT[allegiences[0]][2], 255)

            right_image = card_image.copy()[:, CARD_WIDTH//2:, :]
            right_image[np.any(right_image[:, :, :3] != [0, 0, 0], axis=2)] = (COLOUR_DICT[allegiences[1]][0], COLOUR_DICT[allegiences[1]][1], COLOUR_DICT[allegiences[1]][2], 255)

            colour_image = cv.hconcat([left_image.copy(), right_image.copy()])

            alpha = COLOUR_DICT[allegiences[0]][3]
            left_card_colour_image = cv.addWeighted(card_image.copy()[:, :CARD_WIDTH//2, :], alpha, colour_image.copy()[:, :CARD_WIDTH//2, :], 1-alpha, 0.0)
            alpha = COLOUR_DICT[allegiences[1]][3]
            right_card_colour_image = cv.addWeighted(card_image.copy()[:, CARD_WIDTH//2:, :], alpha, colour_image.copy()[:, CARD_WIDTH//2:, :], 1-alpha, 0.0)
            card_colour_image = cv.hconcat([left_card_colour_image.copy(), right_card_colour_image.copy()])

            gradient_start_colour = left_card_colour_image[-100 * SCALING_FACTOR, 100 * SCALING_FACTOR, :4]
            gradient_end_colour = right_card_colour_image[-100 * SCALING_FACTOR, -100 * SCALING_FACTOR, :4]
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

                card_colour_image[mask_1] = cv.addWeighted(card_image[mask_1].copy(), COLOUR_DICT[allegiences[0]][3], card_colour_image[mask_1].copy(), 1 - COLOUR_DICT[allegiences[0]][3], 0.0)
                card_colour_image[mask_2] = cv.addWeighted(card_image[mask_2].copy(), COLOUR_DICT[allegiences[1]][3], card_colour_image[mask_2].copy(), 1 - COLOUR_DICT[allegiences[1]][3], 0.0)
                card_colour_image[mask_3] = cv.addWeighted(card_image[mask_3].copy(), COLOUR_DICT[allegiences[2]][3], card_colour_image[mask_3].copy(), 1 - COLOUR_DICT[allegiences[2]][3], 0.0)

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


    card_colour_image = cv.addWeighted(card_colour_image.copy(), 0.8, sparks_image.copy(), 0.2, 0.0)

    attack_number_image = create_text_image(attack, 4 * SCALING_FACTOR, 6, (1, 1, 1), cv.FONT_HERSHEY_PLAIN, int(CARD_WIDTH/3))
    mana_number_image = create_text_image(mana, 4 * SCALING_FACTOR, 6, (1, 1, 1), cv.FONT_HERSHEY_PLAIN, int(CARD_WIDTH/3))
    health_number_image = create_text_image(health, 4 * SCALING_FACTOR, 6, (1, 1, 1), cv.FONT_HERSHEY_PLAIN, int(CARD_WIDTH/3))
    cost_number_image = create_text_image(cost, 4 * SCALING_FACTOR, 6, (1, 1, 1), cv.FONT_HERSHEY_PLAIN, int(CARD_WIDTH/3))

    ability_text_image = create_text_image(ability_text, 0.5 * SCALING_FACTOR, 2, (1, 1, 1), cv.FONT_HERSHEY_COMPLEX, TEXT_WIDTH_END - TEXT_WIDTH_START)
    name_text_image = create_text_image(name_text, 0.7 * SCALING_FACTOR, 3, (1, 1, 1), cv.FONT_HERSHEY_COMPLEX, NAME_WIDTH_END - NAME_WIDTH_START)
    allegience_text_image = create_text_image(" ".join(allegiences), 0.5 * SCALING_FACTOR, 2, (1, 1, 1), cv.FONT_HERSHEY_COMPLEX, NAME_WIDTH_END - NAME_WIDTH_START)

    creature_image = cv.copyMakeBorder(creature_image.copy(), 86 * SCALING_FACTOR, 0, 27 * SCALING_FACTOR, 0, cv.BORDER_CONSTANT, None, (0, 0, 0, 0))

    creature_n_border = add_two_images(creature_image, border_template, (0, 0))

    creature_n_border_n_colour = add_two_images(card_colour_image.copy(), creature_n_border, (0, 0))
    creature_n_border_n_colour[:, :, 3] = 255

    image3 = add_two_images(creature_n_border_n_colour, attack_number_image, (ATTACK_NUMBER_WIDTH, ATTACK_NUMBER_HEIGHT))
    image4 = add_two_images(image3, mana_number_image, (MANA_NUMBER_WIDTH - int(mana_number_image.shape[1]), MANA_NUMBER_HEIGHT))
    image5 = add_two_images(image4, health_number_image, (HEALTH_NUMBER_WIDTH - health_number_image.shape[1], HEALTH_NUMBER_HEIGHT))
    image6 = add_two_images(image5, cost_number_image, (COST_NUMBER_WIDTH, COST_NUMBER_HEIGHT))
    image7 = add_two_images(image6, ability_text_image, (TEXT_WIDTH_START, TEXT_HEIGHT_START))
    image8 = add_two_images(image7, name_text_image, (int(NAME_WIDTH_START + (NAME_WIDTH_END-NAME_WIDTH_START)/2 - name_text_image.shape[1]/2), NAME_HEIGHT))
    image9 = add_two_images(image8, allegience_text_image, (int(NAME_WIDTH_START + (NAME_WIDTH_END-NAME_WIDTH_START)/2 - allegience_text_image.shape[1]/2), ALLEGIENCE_HEIGHT))

    for i in range(len(allegiences)):
        logo_template = os.path.join(logos_path, ("Logo" + allegiences[i] + ".png"))
        logo = cv.imread(logo_template, cv.IMREAD_UNCHANGED)
        logo = replace_color(logo, [8, 255, 0], [0, 0, 0], 3)
        logo = replace_color(logo, [255, 255, 255], [COLOUR_DICT[allegiences[i]][0], COLOUR_DICT[allegiences[i]][1], COLOUR_DICT[allegiences[i]][2]], 3)
        logo = cv.resize(logo, (LOGO_WIDTH, LOGO_HEIGHT), interpolation = cv.INTER_AREA)
        image9 = add_two_images(image9, logo, (int(CARD_WIDTH // 2 - len(allegiences) * LOGO_WIDTH//2 + i * LOGO_WIDTH), LOGO_POSITION_H))

    image9[:, :, 3] = 255

    cv.imwrite(os.path.join(output_path, str(name_text) + ".png"), image9)


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
        cards.append(Card(record[1], record[2], record[3], record[4], record[5], record[6], record[7], vampire=record[8], dragon=record[9], human=record[10], horror=record[11], demon=record[12], undead=record[13], construct=record[14], angel=record[15], warrior=record[16], mage=record[17], beast=record[18], knight=record[19], hunter=record[20], noble=record[21]))

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
                        str(card.attack),
                        str(card.mana),
                        str(card.health),
                        str(card.cost),
                        str(card.ability_text),
                        str(card.name),
                        card.get_allegiences(),
                        os.path.join(default_path, output_directory)
                        )
            
        except Exception as e:
            print(f"Error processing card '{card.name}': {e}")

if __name__ == "__main__":
    main()