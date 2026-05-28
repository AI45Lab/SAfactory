import logging
import xml.etree.ElementTree as ET
from io import BytesIO

from PIL import Image

from .accessibility_tree_wrap.heuristic_retrieve import (
    draw_bounding_boxes,
    filter_nodes,
)

logger = logging.getLogger("desktopenv.agent")


def log_info(*args):
    logger.info(" ".join(map(str, args)))


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform=platform, check_image=True)
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(nodes, screenshot)
    return marks, drew_nodes, tagged_screenshot, element_list


def filter_bounding_boxes(bounding_boxes, nodes, max_width=1920, max_height=1080):
    def get_node_text(node):
        if node.text:
            node_text = node.text if '"' not in node.text else '"{:}"'.format(node.text.replace('"', '""'))
        elif node.get("{uri:deskat:uia.windows.microsoft.org}class", "").endswith("EditWrapper") and node.get(
            "{uri:deskat:value.at-spi.gnome.org}value"
        ):
            value = node.get("{uri:deskat:value.at-spi.gnome.org}value")
            node_text = value if '"' not in value else '"{:}"'.format(value.replace('"', '""'))
        else:
            node_text = '""'
        return node.tag + " " + node.get("name", "") + " " + node_text

    filtered_boxes = []
    for idx, box in enumerate(bounding_boxes):
        if box[2] <= max_width or box[3] <= max_height:
            filtered_boxes.append(box)
        else:
            node_text = get_node_text(nodes[idx]) if idx < len(nodes) else "<missing node>"
            logger.debug("%s %s %s removed", idx, box, node_text)
    return filtered_boxes


def agent_attack_wrapper(
    attack_config,
    instruction,
    button_name_dict,
    current_boundingbox,
    nodes,
    current_observation,
    som=False,
    reserved_index=-1,
    content_str="",
):
    from ..env_risk_utils.attack import agent_attack, adversarial_text_agent_attack
    from ..env_risk_utils.general_attack_utils import (
        draw_som_for_attack_osworld,
        extract_bounding_boxes_from_image,
        find_largest_non_overlapping_box,
    )

    ocr_boxes = extract_bounding_boxes_from_image(current_observation)
    current_observation = Image.open(BytesIO(current_observation))
    current_boundingbox = filter_bounding_boxes(current_boundingbox, nodes)
    current_boundingbox = current_boundingbox + ocr_boxes

    largest_non_overlapping_box = find_largest_non_overlapping_box((1920, 1080), current_boundingbox)

    attack_func = adversarial_text_agent_attack if attack_config else agent_attack
    (
        current_observation,
        skip,
        button_name_dict,
        whole_attack_bounding_box,
        x_tgt,
        y_tgt,
        attack_string,
    ) = attack_func(
        attack_config=attack_config,
        current_observation=current_observation,
        largest_non_overlapping_box=largest_non_overlapping_box,
        reserved_index=reserved_index,
        reserved_color=None,
        customized_draw_som_for_attack=draw_som_for_attack_osworld,
        som=som,
        button_name_dict=button_name_dict,
        instruction=instruction,
        customized_print=log_info,
        content_str=content_str,
    )

    image_bytes_io = BytesIO()
    current_observation.save(image_bytes_io, format="PNG")
    return (
        image_bytes_io.getvalue(),
        skip,
        button_name_dict,
        whole_attack_bounding_box,
        x_tgt,
        y_tgt,
        attack_string,
    )
