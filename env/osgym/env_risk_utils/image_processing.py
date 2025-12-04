import xmltodict
from PIL import Image, ImageDraw, ImageFont

def ensure_list(variable):
    if not isinstance(variable, list):
        variable = [variable]
    return variable

def fill_bounding_box_with_text(image, bounding_box, text, init_font_size=20, fill_color="lightgray", text_color="red", edge_thickness=2):
    draw = ImageDraw.Draw(image)
    
    # Define the bounding box coordinates
    xmin, ymin, xmax, ymax = (float(bounding_box['xmin']), float(bounding_box['ymin']), 
                              float(bounding_box['xmax']), float(bounding_box['ymax']))
    
    # Calculate the bounding box dimensions
    box_width = xmax - xmin - 2 * edge_thickness
    box_height = ymax - ymin - 2 * edge_thickness

    # Function to calculate if text fits in the box
    def text_fits(draw, text, font, box_width, box_height):
        text_bbox = draw.textbbox((0, 0), text=text, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # right - left
        text_height = text_bbox[3] - text_bbox[1]  # bottom - top
        return text_width <= box_width and text_height <= box_height

    def calculate_area(text_bbox):
        width = text_bbox[2] - text_bbox[0]  # right - left
        height = text_bbox[3] - text_bbox[1]  # bottom - top
        return width * height

    # Initial horizontal fitting
    font_size = init_font_size
    font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
    while not text_fits(draw, text, font, box_width, box_height) and font_size > 1:
        font_size -= 1
        font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)

    # Calculate text size and area for initial horizontal fit
    bbox_horiz = draw.textbbox((0, 0), text=text, font=font)
    area_horiz = calculate_area(bbox_horiz)

    # Attempt to remove '\n' and fit again
    font_size = init_font_size
    text_no_newline = text.replace("\n", " ")
    font_no_newline = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
    while not text_fits(draw, text_no_newline, font_no_newline, box_width, box_height) and font_size > 1:
        font_size -= 1
        font_no_newline = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
    
    bbox_no_newline = draw.textbbox((0, 0), text=text_no_newline, font=font_no_newline)
    area_no_newline = calculate_area(bbox_no_newline)

    # Determine the best fit and draw the text accordingly
    best_text = text
    best_font = font
    best_bbox = bbox_horiz
    best_area = area_horiz
    
    if area_no_newline > best_area:
        best_text = text_no_newline
        best_font = font_no_newline
        best_bbox = bbox_no_newline
        best_area = area_no_newline

    # Calculate the width and height of the text
    text_width = best_bbox[2] - best_bbox[0]  # right - left
    text_height = best_bbox[3] - best_bbox[1]  # bottom - top
    
    # Calculate the starting position of the drawn text to achieve a centered effect
    x = xmin + edge_thickness + (box_width - text_width) / 2
    y = ymin + edge_thickness + (box_height - text_height) / 2
    
    # Adjust the y-coordinate to compensate for font baseline offset and improve vertical centering accuracy
    # Some fonts may need to fine tune this offset for optimal results
    baseline_offset = text_height * 0.1  # Estimated value, may need to be adjusted according to font
    y -= baseline_offset
    
    # Draw text within the bounding box
    if fill_color is not None:
        draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)
        draw.text((x, y), best_text, font=best_font, fill=text_color)
    else:
        draw.text((x, y), best_text, font=best_font, fill=text_color)
    
    return image


def get_largest_bounding_box_center(bounding_boxes):
    def get_area(box):
        return (float(box['xmax']) - float(box['xmin'])) * (float(box['ymax']) - float(box['ymin']))
    
    largest_box = max(bounding_boxes, key=get_area)
    
    center_x = (float(largest_box['xmin']) + float(largest_box['xmax'])) / 2
    center_y = (float(largest_box['ymin']) + float(largest_box['ymax'])) / 2
    
    return int(center_x), int(center_y)


def fill_bounding_box_with_image(current_observation, attack_bounding_box, image_path_or_obj, button_info, resize_mode="fit",  add_close_button=True, close_button_color="red", close_button_size_ratio=0.05):
    """
    put the popup image in the largest bounding box
    
    Parameters:
    - current_observation: screenshot
    - attack_bounding_box: xmin, ymin, box_width, box_height
    - image_path_or_obj: path of popup image
    - resize_mode: image adjust mode ("fit", "fill", "stretch")
    
    Returns:
    - modified screenshot
    """    
    if isinstance(image_path_or_obj, str):
        try:
            insert_img = Image.open(image_path_or_obj).convert('RGBA')
        except Exception as e:
            print(f"Error Loading Image: {e}")
            return current_observation
    elif isinstance(image_path_or_obj, Image.Image):
        insert_img = image_path_or_obj.convert('RGBA')
    else:
        print("invalid Image Type. Must PIL image or str path")
        return current_observation
 
    # Calculate bounding box dimensions
    xmin, ymin, xmax, ymax = attack_bounding_box['xmin'], attack_bounding_box['ymin'], attack_bounding_box['xmax'], attack_bounding_box['xmax']
    box_width = int(xmax - xmin)
    box_height = int(ymax - ymin)
    
    # Adjust the image size according to the adjustment mode
    if resize_mode == "fit":
        # Maintain aspect ratio and adapt to bounding boxes
        img_ratio = insert_img.width / insert_img.height
        box_ratio = box_width / box_height
        if img_ratio > box_ratio:  # The image is wider than the bounding box
            new_width = box_width
            new_height = int(new_width / img_ratio)
        else:
            new_height = box_height
            new_width = int(new_height * img_ratio)
        insert_img = insert_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Place the image in the center within the bounding box
        paste_x = xmin
        paste_y = ymin

        new_bbx = [xmin, ymin, new_width, new_height]
        # Calculate the actual position of the button
        button_x = int(attack_bounding_box['xmin'] + (attack_bounding_box['xmax'] - attack_bounding_box['xmin']) * button_info['rel_x'])
        button_y = int(attack_bounding_box['ymin'] + (attack_bounding_box['ymax'] - attack_bounding_box['ymin']) * button_info['rel_y'])
        button_xy = {"x": button_x, "y": button_y}

    elif resize_mode == "fill":
        # Filling the bounding box may crop the image
        img_ratio = insert_img.width / insert_img.height
        box_ratio = box_width / box_height
        
        if img_ratio > box_ratio:
            new_height = box_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = box_width
            new_height = int(new_width / img_ratio)
            
        insert_img = insert_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Crop to fit the bounding box
        left = (new_width - box_width) // 2 if new_width > box_width else 0
        top = (new_height - box_height) // 2 if new_height > box_height else 0
        right = left + box_width if new_width > box_width else new_width
        bottom = top + box_height if new_height > box_height else new_height
        
        insert_img = insert_img.crop((left, top, right, bottom))
        paste_x = xmin
        paste_y = ymin

        new_bbx = [xmin, ymin, new_width, new_height]
        
    else:
        insert_img = insert_img.resize((box_width, box_height), Image.LANCZOS)
        paste_x = xmin
        paste_y = ymin
        new_bbx = [xmin, ymin, box_width, box_height]
    
    # If the image has an alpha channel, create a mask
    mask = insert_img if insert_img.mode == 'RGBA' else None

    # Paste the image onto the original image
    current_observation.paste(insert_img, (int(paste_x), int(paste_y)), mask)

    return current_observation, button_xy, new_bbx