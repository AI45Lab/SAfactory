import os
import re
import base64
import copy
from mimetypes import guess_type
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
import typing_extensions as typing
from pydantic import BaseModel, Field

template_lang = '''\
The output json format should be {'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \
2. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \
3. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!
'''

template = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_lang_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available. 
2. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name. 
3. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. visual_state_description: Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input. 
2. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available. 
3. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name. 
4. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

def fix_json(json_str):
    """
    Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
    and escapes any inner double quotes that are not already escaped.
    
    The regex uses a positive lookahead to stop matching when reaching the delimiter for the next key.
    """
    # first fix common errors
    json_str = json_str.replace("'",'"') 
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # Then fix some situations. Pattern explanation:
    # 1. ("reasoning_and_reflection"\s*:\s*") matches the key and the opening quote.
    # 2. (?P<value>.*?) lazily captures everything in a group named 'value'.
    # 3. (?=",\s*"language_plan") is a positive lookahead that stops matching before the closing quote
    #    that comes before the "language_plan" key.
    pattern = r'("reasoning_and_reflection"\s*:\s*")(?P<value>.*?)(?=",\s*"language_plan")'
    
    def replacer(match):
        prefix = match.group(1)            # Contains the key and the opening quote.
        value = match.group("value")         # The raw value that might contain unescaped quotes.
        # Escape any double quote that is not already escaped.
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value

    # Use re.DOTALL so that newlines in the value are included.
    fixed_json = re.sub(pattern, replacer, json_str, flags=re.DOTALL)
    return fixed_json


class ExecutableAction_1(typing.TypedDict): 
    action_id: int = Field(
        description="The action ID to select from the available actions given by the prompt"
    )
    action_name: str = Field(
        description="The name of the action"
    )
class ActionPlan_1(BaseModel):
    visual_state_description: str = Field(
        description="Description of current state from the visual image"
    )
    reasoning_and_reflection: str = Field(
        description="summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task"
    )
    language_plan: str = Field(
        description="The list of actions to achieve the user instruction. Each action is started by the step number and the action name"
    )
    executable_plan: list[ExecutableAction_1] = Field(
        description="A list of actions needed to achieve the user instruction, with each action having an action ID and a name."
    )

class ActionPlan_1_manip(BaseModel):
    visual_state_description: str = Field(
        description="Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input."
    )
    reasoning_and_reflection: str = Field(
        description="Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available."
    )
    language_plan: str = Field(
        description="A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name."
    )
    executable_plan: str = Field(
        description="A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action."
    )

def convert_format_2claude(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":
            new_content = []
    
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages

def convert_format_2gemini(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":

            new_content = []
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}"
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)
        
    return new_messages



class ExecutableAction(typing.TypedDict): 
    action_id: int
    action_name: str
class ActionPlan(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction]

class ActionPlan_manip(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

class ExecutableAction_lang(typing.TypedDict): 
    action_id: int
    action_name: str
class ActionPlan_lang(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction_lang]

class ActionPlan_lang_manip(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def truncate_message_prompts(message_history: list):
    """
    Traverse the message list and truncate the part before "------------" in the text content of all messages except the last one
    
    Args:
        message_history: Message list, each message contains role and content
        
    Returns:
        list: Processed message list
    """
    if not message_history:
        return message_history
        
    # Create a deep copy to avoid modifying the original data
    processed_messages = []
    
    # Process all messages except the last one
    for i, message in enumerate(message_history):
        if i == len(message_history) - 1:
            # Keep the last message unchanged
            processed_messages.append(message)
        else:
            # Process current message
            processed_message = {
                "role": message.get("role", ""),
                "content": []
            }
            
            # Traverse content list
            for content_item in message.get("content", []):
                if content_item.get("type") == "text":
                    # Process text type content
                    text_content = content_item.get("text", "")
                    
                    # Look for "----------" separator
                    if "----------" in text_content:
                        # Truncate content before separator, keep content after separator
                        truncated_text = text_content.split("----------")[1]
                    else:
                        # If no separator found, keep original text
                        truncated_text = text_content
                        
                    processed_content_item = content_item.copy()
                    processed_content_item["text"] = truncated_text
                    processed_message["content"].append(processed_content_item)
                else:
                    # Directly copy non-text type content
                    processed_message["content"].append(content_item.copy())
            
            processed_messages.append(processed_message)
    
    return processed_messages