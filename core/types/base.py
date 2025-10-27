from typing import Dict, Any, Tuple, Optional, List, Union
from pydantic import BaseModel, RootModel
import base64
import json
import numpy as np

# 定义环境相关的数据类型
class ResetOutput(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]

class StepOutput(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: Dict[str, str]

class TextContent(BaseModel):
    type: str = "text"
    text: str

class MessageContent(RootModel):
    root: Union[TextContent, ImageContent]

class OpenAIMessage(BaseModel):
    role: str
    content: List[MessageContent]

class PromptOutput(BaseModel):
    system_message: OpenAIMessage
    user_message: OpenAIMessage

class RenderOutput(BaseModel):
    # 图片二进制数据（可选，适合内存中直接传递）
    image_data: Optional[bytes] = None
    # Base64编码的图片字符串（可选，适合网络传输或JSON序列化）
    image_base64: Optional[str] = None
    # 图片保存路径（可选，适合需要持久化存储的场景）
    image_path: Optional[str] = None
    step: int

    class Config:
        arbitrary_types_allowed = True  # 允许bytes类型
        json_encoders = {
            bytes: lambda v: base64.b64encode(v).decode('utf-8')  # 自动将bytes转为Base64字符串用于JSON输出
        }

    def __init__(self, **data):
        # 确保至少提供一种图片数据形式
        if not any(key in data for key in ['image_data', 'image_base64', 'image_path']):
            raise ValueError("RenderOutput must contain either image_data, image_base64, or image_path")
        
        # 自动转换：如果提供了image_data，自动生成image_base64（方便序列化）
        if 'image_data' in data and 'image_base64' not in data:
            data['image_base64'] = base64.b64encode(data['image_data']).decode('utf-8')
        
        super().__init__(** data)

def serialize_prompt_output(prompt_output: PromptOutput) -> str:
    """将PromptOutput序列化为JSON字符串（兼容RootModel）"""
    # 使用model_dump(mode='json')确保RootModel正确序列化
    prompt_dict = prompt_output.model_dump(mode='json')
    return json.dumps(prompt_dict, ensure_ascii=False, indent=2)


def deserialize_prompt_output(json_str: str) -> PromptOutput:
    """将JSON字符串反序列化为PromptOutput对象（兼容RootModel）"""
    prompt_dict = json.loads(json_str)
    # 递归处理MessageContent的root字段（Pydantic RootModel需要显式传入root键）
    def _fix_root_fields(data: Dict) -> Dict:
        if isinstance(data, dict):
            # 处理OpenAIMessage中的content列表（元素为MessageContent）
            if 'content' in data:
                data['content'] = [
                    # MessageContent是RootModel，需要用root键包裹内容
                    {'root': _fix_root_fields(item['root'])} 
                    for item in data['content']
                ]
            return data
        return data  # 非字典类型直接返回

    # 修复嵌套结构中的root字段格式
    fixed_dict = _fix_root_fields(prompt_dict)
    return PromptOutput(**fixed_dict)