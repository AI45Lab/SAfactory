import xmltodict
import re
import math
import csv
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import numpy as np
import io

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def shortest_distance_to_corners(x, y):
    # Coordinates of the four corners
    corners = [(0, 0), (1920, 0), (0, 1080), (1920, 1080)]
    
    # Calculate distances to each corner
    distances = [euclidean_distance(x, y, corner[0], corner[1]) for corner in corners]
    
    # Return the shortest distance
    return min(distances)

def find_files_with_prefix(directory, prefix):
    def find_shortest_string(strings):
        if not strings:  # Check if the list is empty
            return None
    
        shortest = strings[0]  # Assume the first string is the shortest
        for string in strings:
            if len(string) < len(shortest):
                shortest = string  # Update the shortest string if a shorter one is found
    
        return shortest
    
    matching_files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)) and file.startswith(prefix):
            matching_files.append(os.path.join(directory, file))

    # return the shorted match
    return find_shortest_string(matching_files)

def ensure_list(variable):
    if not isinstance(variable, list):
        variable = [variable]
    return variable

def extract_coordinates(code_string):
    # Pattern to match coordinates in pyautogui.click(980, 160
    click_pattern = re.compile(r'pyautogui\.click\(\s*(\d+)\s*,\s*(\d+)\s*')
    
    # Pattern to match coordinates in pyautogui.click(x=250, y=750
    click_keyword_pattern = re.compile(r'pyautogui\.click\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*')
    
    # Pattern to match coordinates in pyautogui.moveTo(500, 300
    move_pattern = re.compile(r'pyautogui\.moveTo\(\s*(\d+)\s*,\s*(\d+)\s*')
    
    # Pattern to match coordinates in pyautogui.moveTo(x=250, y=750
    move_keyword_pattern = re.compile(r'pyautogui\.moveTo\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*')
    
    # Pattern to match coordinate assignment like x, y = 500, 300
    assignment_pattern = re.compile(r'x\s*,\s*y\s*=\s*(\d+)\s*,\s*(\d+)')
    
    # Search for coordinates in the code string
    click_match = click_pattern.search(code_string)
    click_keyword_match = click_keyword_pattern.search(code_string)
    move_match = move_pattern.search(code_string)
    move_keyword_match = move_keyword_pattern.search(code_string)
    assignment_match = assignment_pattern.search(code_string)

    # Extract coordinates from the matches
    if click_match:
        x, y = click_match.groups()
    elif click_keyword_match:
        x, y = click_keyword_match.groups()
    elif move_match:
        x, y = move_match.groups()
    elif move_keyword_match:
        x, y = move_keyword_match.groups()
    elif assignment_match:
        x, y = assignment_match.groups()
    else:
        return None  # No coordinates found

    # Convert coordinates to integers
    return int(x), int(y)

def extract_coordinate_list(code_string):
    # Update patterns to match either numbers or variable names
    coordinate_pattern = r'(\d+|[a-zA-Z_][a-zA-Z_0-9]*)'
    
    # Pattern to match coordinates in pyautogui.click(980, 160) or pyautogui.click(variable_x, variable_y)
    click_pattern = re.compile(rf'pyautogui\.click\(\s*{coordinate_pattern}\s*,\s*{coordinate_pattern}\s*')
    
    # Pattern to match coordinates in pyautogui.click(x=250, y=750) or pyautogui.click(x=var_x, y=var_y)
    click_keyword_pattern = re.compile(rf'pyautogui\.click\(\s*x\s*=\s*{coordinate_pattern}\s*,\s*y\s*=\s*{coordinate_pattern}\s*')
    
    # Pattern to match coordinates in pyautogui.moveTo(500, 300) or pyautogui.moveTo(variable_x, variable_y)
    move_pattern = re.compile(rf'pyautogui\.moveTo\(\s*{coordinate_pattern}\s*,\s*{coordinate_pattern}\s*')
    
    # Pattern to match coordinates in pyautogui.moveTo(x=250, y=750) or pyautogui.moveTo(x=var_x, y=var_y)
    move_keyword_pattern = re.compile(rf'pyautogui\.moveTo\(\s*x\s*=\s*{coordinate_pattern}\s*,\s*y\s*=\s*{coordinate_pattern}\s*')
    
    # Pattern to match assignments like tag_1 = (1006, 571)
    assignment_pattern = re.compile(rf'([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)')
    
    # Find all variable assignments
    assignment_matches = assignment_pattern.findall(code_string)
    
    # Store the variable assignments in a dictionary
    variable_map = {var: (int(x), int(y)) for var, x, y in assignment_matches}
    
    # Replace pyautogui.click(tag_X) with pyautogui.click(X, Y)
    for var, (x, y) in variable_map.items():
        click_var_pattern = re.compile(rf'pyautogui\.click\(\s*{var}\s*\)')
        code_string = click_var_pattern.sub(f'pyautogui.click({x}, {y})', code_string)
        # Replace pyautogui.moveTo(tag_X) with pyautogui.moveTo(X, Y)
        move_var_pattern = re.compile(rf'pyautogui\.moveTo\(\s*{var}\s*\)')
        code_string = move_var_pattern.sub(f'pyautogui.moveTo({x}, {y})', code_string)
    # Now that the code_string is updated, run the original pattern matching logic
    
    # Find all matches
    click_matches = click_pattern.findall(code_string)
    click_keyword_matches = click_keyword_pattern.findall(code_string)
    move_matches = move_pattern.findall(code_string)
    move_keyword_matches = move_keyword_pattern.findall(code_string)
    
    # Combine all matches into one list
    all_matches = click_matches + click_keyword_matches + move_matches + move_keyword_matches
    
    if len(all_matches) == 0 and "click" in code_string:
        # In case of direct assignment like x, y = 100, 200 or x, y = var_x, var_y
        preassignment_pattern = re.compile(rf'x\s*,\s*y\s*=\s*{coordinate_pattern}\s*,\s*{coordinate_pattern}')
        preassignment_matches = preassignment_pattern.findall(code_string)
        all_matches += preassignment_matches
    
    # Process results: numbers as integers, variables as strings
    coordinates = [(int(x) if x.isdigit() else x, int(y) if y.isdigit() else y) for x, y in all_matches]
    
    return coordinates

def is_point_in_bbox(x, y, bboxes):
    def point_in_bbox(px, py, bbox):
        return bbox['xmin'] <= px <= bbox['xmax'] and bbox['ymin'] <= py <= bbox['ymax']
    
    def distance_to_bbox(px, py, bbox):
        if px < bbox['xmin']:
            if py < bbox['ymin']:
                return math.sqrt((px - bbox['xmin'])**2 + (py - bbox['ymin'])**2)
            elif py > bbox['ymax']:
                return math.sqrt((px - bbox['xmin'])**2 + (py - bbox['ymax'])**2)
            else:
                return bbox['xmin'] - px
        elif px > bbox['xmax']:
            if py < bbox['ymin']:
                return math.sqrt((px - bbox['xmax'])**2 + (py - bbox['ymin'])**2)
            elif py > bbox['ymax']:
                return math.sqrt((px - bbox['xmax'])**2 + (py - bbox['ymax'])**2)
            else:
                return px - bbox['xmax']
        else:
            if py < bbox['ymin']:
                return bbox['ymin'] - py
            elif py > bbox['ymax']:
                return py - bbox['ymax']
            else:
                return 0
    
    in_bbox = False
    min_distance = float('inf')
    
    for bbox in bboxes:
        bbox = {k: float(v) for k, v in bbox.items()}
        if point_in_bbox(x, y, bbox):
            in_bbox = True
            break
        else:
            distance = distance_to_bbox(x, y, bbox)
            if distance < min_distance:
                min_distance = distance
                
    return in_bbox, min_distance if not in_bbox else 0.0

def mark_bounding_boxes(image, bounding_boxes):
    """
    Mark the pixels in the image matrix that are occupied by bounding boxes.
    
    Args:
    image (numpy.ndarray): 2D binary image where 1 represents occupied and 0 represents empty.
    bounding_boxes (list): List of bounding boxes in the format [x, y, w, h].
    """
    for box in bounding_boxes:
        x, y, w, h = box
        image[y:y+h, x:x+w] = 1  # Mark the area covered by each bounding box

def largest_empty_rectangle(image, squareness_preference_factor=1.0):
    """
    Find the largest empty rectangle in a 2D binary matrix with a preference for square-like rectangles.
    
    Args:
    image (numpy.ndarray): 2D binary matrix representing occupied (1) and empty (0) cells.
    squareness_preference_factor (float): A factor that adjusts the score for squareness.
    
    Returns:
    tuple: The coordinates of the top-left corner and the dimensions of the largest empty rectangle (x, y, w, h).
    """
    rows, cols = image.shape
    height = np.zeros((rows, cols), dtype=int)
    
    # Calculate the height of consecutive empty cells in each column
    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 0:
                height[i, j] = height[i-1, j] + 1 if i > 0 else 1
    
    # Now find the largest rectangle in each row using the height histogram method
    max_score = 0
    best_rectangle = (0, 0, 0, 0)  # (x, y, w, h)

    for i in range(rows):
        stack = []
        for j in range(cols + 1):
            cur_height = height[i, j] if j < cols else 0  # Sentinel value for the last column
            while stack and cur_height < height[i, stack[-1]]:
                h = height[i, stack.pop()]
                w = j if not stack else j - stack[-1] - 1
                area = h * w

                squareness_score = 1.0
                # we prefer wider with enough height
                if h > w:
                    squareness_score = min(h, w) / max(h, w)
                if h < 100:
                    squareness_score = min(squareness_score, 0.5)

                score = area * (squareness_score ** squareness_preference_factor)
                
                if score > max_score:
                    max_score = score
                    best_rectangle = (stack[-1] + 1 if stack else 0, i - h + 1, w, h)
            stack.append(j)
    
    return best_rectangle

def extract_bounding_boxes_from_image(image_input):
    import pytesseract
    from PIL import Image
    import numpy as np
    import io
    
    boxes = []

    # Convert input to PIL Image if it's in bytes
    if isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        print("Invalid input: Input must be either bytes or a PIL Image object.")
        return boxes
    
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    try:
        # Get OCR data with bounding box information
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extract bounding boxes
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            # Filter out empty text results
            if int(ocr_data['conf'][i]) > 0:  # Only consider results with confidence > 0
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Only add non-zero width and height boxes
                if w > 0 and h > 0:
                    boxes.append([int(x), int(y), int(w), int(h)])
    
    except Exception as e:
        print(f"An error occurred during OCR processing: {e}")
    
    return boxes

def find_largest_non_overlapping_box(image_size, bounding_boxes, squareness_preference_factor=1.0):
    """
    Finds the largest non-overlapping bounding box in a given image with preference for square-like rectangles.
    
    Args:
    image_size (tuple): The size of the image (width, height).
    bounding_boxes (list): List of bounding boxes in the format [x, y, w, h].
    squareness_preference_factor (float): A factor that adjusts the score for squareness.
    
    Returns:
    tuple: Coordinates and size of the largest bounding box that can be drawn without overlap (x, y, w, h).
    """
    width, height = image_size
    image = np.zeros((height, width), dtype=int)  # Create an empty grid representing the image
   
    # Mark occupied areas
    mark_bounding_boxes(image, bounding_boxes)

    # Find the largest empty rectangle
    largest_box = largest_empty_rectangle(image, squareness_preference_factor)
   
    return largest_box

def draw_som_for_attack_osworld(image, x, y, w, h, index, color=None):
    try:
        # Adjust the path to the font file you have or use a default one
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # Fallback to a basic font if the specified font can't be loaded
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    coords = (x, y)
    size = (w, h)

    # Check for negative sizes
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Size must be positive, got: {size}")

    # Calculate the bottom-right corner of the bounding box
    bottom_right = (coords[0] + size[0], coords[1] + size[1])

    # Check that bottom_right > coords (x1 >= x0, y1 >= y0)
    if bottom_right[0] < coords[0] or bottom_right[1] < coords[1]:
        raise ValueError(f"Invalid coordinates or size, coords: {coords}, size: {size}")

    # Check if the area only contains one color
    cropped_image = image.crop((*coords, *bottom_right))
    if len(set(list(cropped_image.getdata()))) == 1:
        return

    # Draw rectangle on image
    draw.rectangle([coords, bottom_right], outline="red", width=1)

    # Draw index number at the bottom left of the bounding box with black background
    text_position = (coords[0], bottom_right[1])  # Adjust Y to be above the bottom right
    text_bbox: Tuple[int, int, int, int] = draw.textbbox(text_position, str(index), font=font, anchor="lb")
    # offset: int = bottom_right[1]-text_bbox[3]
    # text_bbox = (text_bbox[0], text_bbox[1]+offset, text_bbox[2], text_bbox[3]+offset)

    # draw.rectangle([text_position, (text_position[0] + 25, text_position[1] + 18)], fill='black')
    draw.rectangle(text_bbox, fill='black')
    draw.text(text_position, str(index), font=font, anchor="lb", fill="white")

    return image