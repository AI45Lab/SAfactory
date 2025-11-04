from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List, Union
from pydantic import BaseModel, RootModel, ConfigDict, field_validator, model_validator
import base64
import json
import numpy as np
import pandas as pd


def _safe(v: Any) -> Any:
    # primitives
    if v is None or isinstance(v, (bool, int, float, str)):
        return v

    # numpy scalars & arrays
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()

    # pandas
    if pd is not None:
        if isinstance(v, pd.DataFrame):
            return v.to_dict(orient="records")
        if isinstance(v, pd.Series):
            return v.to_dict()

    # bytes -> base64 string
    if isinstance(v, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(v)).decode("ascii")

    # mappings / sequences
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]

    # fallback: leave as-is; if it shows up, it will fail fast and we can add a case
    return v

# 定义环境相关的数据类型
class ResetOutput(BaseModel):
    """
    observation/info may contain pandas/np/bytes; we convert them to JSON-safe
    values *before* the model is created, so every instance is safe to serialize.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: Dict[str, Any]
    info: Dict[str, Any]

    @field_validator('observation', 'info', mode='before')
    @classmethod
    def _sanitize(cls, v):
        return _safe(v)

class StepOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    @field_validator('observation', 'info', mode='before')
    @classmethod
    def _sanitize(cls, v):
        return _safe(v)

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
    # 生成步骤编号
    step: int

    # 图片相关字段
    image_data: Optional[bytes] = None  # 图片二进制数据（内存传递）
    image_base64: Optional[str] = None  # Base64编码图片（网络传输）
    image_path: Optional[str] = None    # 图片保存路径（持久化）
    
    # 文本相关字段
    text_content: Optional[str] = None  # 单段文本内容
    text_list: Optional[List[str]] = None  # 多段文本列表
    text_dict: Optional[Dict[str, Any]] = None # 字典格式文本

    class Config:
        arbitrary_types_allowed = True  # 允许bytes类型
        json_encoders = {
            bytes: lambda v: base64.b64encode(v).decode('utf-8')  # bytes自动转Base64
        }

    def __init__(self, **data):
        # 确保至少提供一种数据形式（图片或文本）
        has_image = any(key in data for key in ['image_data', 'image_base64', 'image_path'])
        has_text = any(key in data for key in ['text_content', 'text_list'])
        
        if not (has_image or has_text):
            raise ValueError("RenderOutput must contain either image data (image_data/image_base64/image_path) or text data (text_content/text_list)")
        
        # 图片数据自动转换：如果提供image_data且无image_base64，自动生成
        if 'image_data' in data and 'image_base64' not in data:
            data['image_base64'] = base64.b64encode(data['image_data']).decode('utf-8')
        
        # 文本数据校验：避免同时提供单段文本和文本列表（可选约束，根据需求调整）
        if 'text_content' in data and 'text_list' in data:
            raise ValueError("Cannot provide both text_content and text_list simultaneously")
        
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

def dumps_json_bytes(obj: Union[BaseModel, Dict[str, Any]]) -> bytes:
    """
    Serialize a pydantic model (v2 or v1) or plain dict to JSON bytes.
    Assumes your ResetOutput/StepOutput/RenderOutput already sanitize pandas/np/bytes
    at construction (as you just implemented).
    """
    if isinstance(obj, BaseModel):
        if hasattr(obj, "model_dump_json"):  # Pydantic v2
            return obj.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
        # Pydantic v1 fallback
        return obj.json(by_alias=True, exclude_none=True, ensure_ascii=False,
                        separators=(",", ":")).encode("utf-8")
    # plain dict
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
