from embodiedbench.planner.planner_utils import local_image_to_data_url
from copy import deepcopy
import os
# System prompt for robot task generation

# Template for examples showing successful task completion
EXAMPLE_TEMPLATE = """
Task Description: {task_description}
Reasoning and reflection:{reasoning}
Executable Plan:{output}
Feedback: {env_feedback}
"""

EXAMPLE_PATH=[
    os.path.join(os.path.dirname(__file__), f'example{i}.jsonl') for i in range(1)
]

def create_example(
    i,
    example_dict_list,
):

    # Format the action list for better readability
    contents=[
         {
            "type": "text",
            "text": f"## Example {i} of a successful task completion"
        },
    ]
    for example_dict in example_dict_list:
        img_url=local_image_to_data_url(os.path.join(os.path.dirname(__file__), example_dict["image_path"]))
        contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": img_url
                }
            }
        )
        example_text = EXAMPLE_TEMPLATE.format(
            task_description=example_dict["task_description"],
            reasoning=example_dict["reasoning"],
            output=example_dict["output"],
            env_feedback=example_dict["env_feedback"]
        )
        contents.append(
            {
                "type": "text",
                "text": example_text
            }
        )
    return contents
    
def create_example_no_image(
    i,
    example_dict_list,
):

    # Format the action list for better readability
    contents=[
         {
            "type": "text",
            "text": f"## Example {i} of a successful task completion"
        },
    ]
    for example_dict in example_dict_list:
        # img_url=local_image_to_data_url(os.path.join(os.path.dirname(__file__), example_dict["image_path"]))
        # contents.append(
        #     {
        #         "type": "image",
        #         "url": img_url
        #     }
        # )
        example_text = EXAMPLE_TEMPLATE.format(
            task_description=example_dict["task_description"],
            reasoning=example_dict["reasoning"],
            output=example_dict["output"],
            env_feedback=example_dict["env_feedback"]
        )
        contents.append(
            {
                "type": "text",
                "text": example_text
            }
        )
    return contents

import json
def create_example_json_list(include_image=True):
    example_content=[]
    for i, path in enumerate(EXAMPLE_PATH):
        # load jsonl as a list of dict
        with open(path, 'r') as f:
            example_dict_list = [json.loads(line) for line in f]
        if include_image:
            example_content.extend(create_example(i, example_dict_list))
        else:
            example_content.extend(create_example_no_image(i, example_dict_list))
    return example_content
    

if __name__ == "__main__":
    print(create_example_json_list())