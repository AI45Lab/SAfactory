import asyncio
import json
import logging
import os
import time
import uuid
import copy
import re
import base64
import hashlib
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime

# 复用已有的数据模型 (仅用于接口传参兼容)
from core.data_manager.models import EnvironmentConfig, InteractionSession, InteractionStep
from core.data_manager.strategy.base_strategy import StorageStrategy

# 引入 Cloud SDK
from wt_sdk import WTGatewayClient, GatewayConfig, EnvConfigManager
from wt_sdk.models import LandingRecord, ChatMessage, ContentItem
from wt_sdk.utils import generate_deterministic_id, S3Uploader

log = logging.getLogger("cloud_strategy")

class CloudStrategy(StorageStrategy):
    """
    将交互数据直接存储到云端数据库 (WTGateway)，环境配置存储到 S3。
    """

    def __init__(self,job_session: str ,db_url: str, enable_buffer: bool = False, buffer_size: int = 1, flush_interval: float = 1.0):
        self.db_url = db_url
        
        self.client: Optional[WTGatewayClient] = None
        self.env_manager: Optional[EnvConfigManager] = None
        self.initialized = False
        
        # 内存缓存：仅用于 interactor 运行时保持对象引用，不持久化
        self._env_configs: Dict[str, EnvironmentConfig] = {} 
        self._sessions: Dict[str, InteractionSession] = {}

        #job session is used for distinguishing each simulation task
        self.job_session = job_session

    async def init(self):
        if self.initialized:
            return

        # TODO After the test passed, it was modified to the production configuration.
        # 1. 初始化 WTGatewayClient
        config = GatewayConfig() 
        config.tables.landing_table = "landing_test"

        try:
            self.client = WTGatewayClient(config)
            log.info(f"CloudStrategy initialized with table: {config.tables.landing_table}")
        except Exception as e:
            log.error(f"Failed to initialize WTGatewayClient: {e}")
            raise

        # 2. 初始化 EnvConfigManager (S3)
        try:
            # 这里使用默认配置或从环境变量读取 AWS key
            self.env_manager = EnvConfigManager() 
        except Exception as e:
            log.error(f"Failed to initialize EnvConfigManager: {e}")
            raise

        self.initialized = True

    async def add_environment_config(self, env_name: str, env_params: dict, image: str, group_id: str):
        env_id = str(uuid.uuid4())
        
        # 1. 构造 Config 字典
        config_dict = {
            "session": self.job_session,
            "env_id": env_id,
            "env_name": env_name,
            "env_params": env_params,
            "image": image,
            "created_at": int(time.time()),
            "group_id": group_id,
        }
        
        # 2. 上传到 S3 (同步方法，建议 wrap 一下防止阻塞 loop，虽然 boto3 也是阻塞的)
        try:
            await asyncio.to_thread(self.env_manager.save_config, config_dict)
            log.info(f"Environment config saved to S3: {env_id}")
        except Exception as e:
            log.error(f"Failed to save env config to S3: {e}")
            
        config_obj = EnvironmentConfig(
            env_name=env_name,
            env_id=env_id,
            env_params=env_params,
            image=image,
            group_id=group_id
        )
        self._env_configs[env_id] = config_obj

    async def get_all_environments(self):
        pass

    async def create_session(self, env_id: str, llm_model: str, group_id: str = "") -> InteractionSession:
        """
        仅在内存创建 Session 对象，不进行云端交互
        """
        # 尝试关联环境配置
        env_config = self._env_configs.get(env_id)
        
        session = InteractionSession(
            env_id=env_id,
            llm_model=llm_model,
            group_id=group_id
        )
        session.env_cache = env_config 
        
        self._sessions[session.session_id] = session
        return session


    async def update_session(self, session: InteractionSession, trajectory: str, total_reward: float, is_completed: bool = True) -> InteractionSession:
        """
        仅更新内存对象，云端无 Session 表
        """
        session.total_reward = total_reward
        session.trajectory = trajectory
        session.is_completed = is_completed
        session.end_time = datetime.now()
        
        # 更新缓存
        self._sessions[session.session_id] = session
        return session

    async def record_step(self, session: InteractionSession, step_id: int, prompt: list, response: str, reward: float, env_key: str, env_state: Optional[str] = None, done: bool = False, truncated: bool = False):
        """
        处理单步数据并立即发送到云端 Landing Table
        """
        await self.init()
        
        if session.reward_count is None:
            session.reward_count = 0.0
            
        session.reward_count += reward

        # 1. 图片处理
        clean_prompt, _ = self.process_message(prompt, env_key, step_id)

        # 2. 构造 LandingRecord 所需数据
        # 转换 Prompt (OpenAI format) -> Messages (SDK format)
        messages = self._convert_to_chat_messages(clean_prompt)
        
        # 构造 Response Message
        response_msg = ChatMessage(
            role="assistant",
            content=[ContentItem(type="text", text=response)]
        )

        # 获取环境名称
        if hasattr(session, 'env_cache') and session.env_cache:
            env_name = session.env_cache.env_name
        else:
            env_name = env_key.split('_')[0]
        
        # 生成确定性 ID
        record_id = generate_deterministic_id(
            {
                "env_name": env_name,
                "messages": messages,
                "response_msg": response_msg,
            }
        )

        # 3. 创建 Record 对象
        current_ts = int(time.time())
        record = LandingRecord(
            dataset_type="Test", # 可根据需求修改
            dt=datetime.now().strftime("%Y-%m-%d"),
            id=record_id,
            session_id=str(session.env),
            created_at=current_ts,
            step_id=step_id,
            is_terminal=done,
            step_reward=reward, # 假设传入的是单步奖励
            reward=session.reward_count,      # 此处根据业务定义，可能有累积奖励需求，这里暂存单步
            messages=messages,  # 包含完整的历史 Prompt
            response=response_msg,
            ground_truth_answer=None,
            reference_answer=None,
            agent_model=session.llm_model,
            env_name=env_name,
            is_session_completed=done or truncated,
            meta_json=json.dumps({
                "source": "AIEvoBox",
                "group_id": session.group_id,
            })
        )

        # 4. 直接上传 (无缓冲)
        try:
            # 使用 to_thread 避免阻塞事件循环
            await asyncio.to_thread(self.client.ingest_landing, record)
            log.debug(f"Step {step_id} ingested to cloud: {record_id}")
        except Exception as e:
            log.error(f"Failed to ingest step {step_id} to cloud: {e}")

    async def close(self):
        if self.client:
            if hasattr(self.client, 'close'):
                self.client.close()
        if self.env_manager:
            if hasattr(self.env_manager, 'close'):
                self.env_manager.close()
                
    def get_sync_connection(self):
        pass
    
    def process_message(self, messages: List[Dict[str, Any]], env_key: str, step_id: int, root_dir: str = "saved_images") -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        处理 OpenAI 消息列表中的图片：
        1. 提取 Base64 图片
        2. 保存为本地文件，
        3. 尝试上传到S3存储中，
        4. 将消息中的 Base64 替换为 S3 URL （如果失败则退回为本地路径）
        
        Returns:
            processed_messages: 处理后的消息列表 (深拷贝)
            saved_urls: 本次处理保存的文件路径列表
        """
        # 初始化
        try:
            uploader = S3Uploader()
        except Exception as e:
            log.error(f"[{env_key}] Failed to initialize S3Uploader: {e}")
            uploader = None
        
        test_prefix = f"test/test_s3_util_{int(time.time())}"
        save_dir = os.path.join(root_dir, f"{env_key}")
        
        processed_messages = []
        saved_urls = []
        image_count = 0
        
        for msg_idx, message in enumerate(messages):
            content = message.get("content")
            
            # 如果 content 不是列表，或者没有 image_url，直接复用原对象引用
            if not isinstance(content, list) or not any(item.get("type") == "image_url" for item in content):
                processed_messages.append(message)
                continue
            
            # 如果包含图片，创建 Message 的浅拷贝，避免修改原始对象
            new_message = message.copy()
            new_content = []
            
            for item_idx, item in enumerate(content):
                # 如果不是图片，直接复用 item 引用
                if item.get("type") != "image_url":
                    new_content.append(item)
                    continue
                
                image_url = item.get("image_url", {}).get("url", "")
                
                # 检查是否为 Base64
                match = re.match(r"data:image/(\w+);base64,(.+)", image_url)
                if not match:
                    # 如果已经是 URL，直接保留
                    new_content.append(item)
                    continue
                
                # --- 开始处理图片 ---
                ext = match.group(1)
                b64_str = match.group(2)
                
                file_name = f"step_{step_id}_m{msg_idx}_i{image_count}.{ext}"
                file_path = os.path.join(save_dir, file_name)
                final_url = None
                
                # 创建 item 的浅拷贝，准备修改 url
                new_item = item.copy()
                new_item["image_url"] = item["image_url"].copy()

                # A. 尝试保存到本地
                local_save_success = False
                try:
                    # 延迟创建目录
                    os.makedirs(save_dir, exist_ok=True)
                    img_data = base64.b64decode(b64_str)
                    
                    with open(file_path, "wb") as f:
                        f.write(img_data)
                    local_save_success = True
                    final_url = file_path # 默认值：本地路径
                    
                except Exception as e:
                    log.error(f"[{env_key}][Step {step_id}] Save Local Failed. Msg: {msg_idx}, Item: {item_idx}. Error: {e}")
                    # 如果本地都保存失败，保留原始 Base64 防止数据丢失，或者标记错误
                    new_item["image_url"]["url"] = b64_str
                    new_content.append(new_item)
                    continue # 跳过后续 S3 逻辑

                # B. 尝试上传到 S3 (仅在本地保存成功后)
                if local_save_success and uploader:
                    try:
                        key = f"{test_prefix}/{file_name}"
                        # upload_file 可能会失败，我们需要捕获它
                        s3_url = uploader.upload_file(file_path=file_path, key=key)
                        
                        if s3_url:
                            final_url = s3_url
                        else:
                            # 即使没有抛出异常但返回空，也视为失败
                            log.warning(f"[{env_key}][Step {step_id}] S3 Upload returned empty URL. Using local path.")

                    except Exception as e:
                        # 4. S3 失败时的降级处理：仅打印日志，不中断，继续使用 final_url (即本地路径)
                        log.error(f"[{env_key}][Step {step_id}] S3 Upload Error. Key: {key}. Error: {e}")
                        # 保持 final_url 为本地路径，确保 Base64 被替换
                
                # C. 更新 Item 内容
                new_item["image_url"]["url"] = final_url
                new_content.append(new_item)
                
                saved_urls.append(final_url)
                image_count += 1
            
            # 将重构后的 content 赋值给新消息
            new_message["content"] = new_content
            processed_messages.append(new_message)
        
        return processed_messages, saved_urls

    # --- Helpers ---

    def _convert_to_chat_messages(self, prompt: List[Dict[str, Any]]) -> List[ChatMessage]:
        """将 OpenAI 格式的 messages 转换为 SDK 的 ChatMessage 对象"""
        out = []
        if not prompt:
            return out
            
        for msg in prompt:
            role = msg.get("role", "user")
            content_raw = msg.get("content", "")
            
            content_items = []
            if isinstance(content_raw, str):
                content_items.append(ContentItem(type="text", text=content_raw))
            elif isinstance(content_raw, list):
                for item in content_raw:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_items.append(ContentItem(type="text", text=item.get("text", "")))
                        elif item.get("type") == "image_url":
                             url = item.get("image_url", {}).get("url", "")
                             content_items.append(ContentItem(type="text", text=url))
            
            out.append(ChatMessage(role=role, content=content_items))
        return out

    async def fetch_done_steps_with_context(self, after_id: int = 0, limit: int = 100) -> List[Dict]:
        return []

    async def get_max_step_id(self) -> int:
        return 0