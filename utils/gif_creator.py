from PIL import Image
import os
from typing import List, Optional

class StepGifCreator:
    @staticmethod
    def create_gif(image_paths: List[str], 
                  output_path: str, 
                  duration: int = 200,  # 每帧持续时间（毫秒）
                  loop: int = 0,       # 0表示无限循环
                  sort_by_step: bool = True) -> str:
        """
        将步骤图片合并为GIF
        
        参数:
            image_paths: 图片路径列表
            output_path: 输出GIF路径
            duration: 每帧持续时间（毫秒）
            loop: 循环次数，0表示无限循环
            sort_by_step: 是否按步骤编号排序（根据我们的命名规则）
            
        返回:
            生成的GIF文件路径
        """
        if not image_paths:
            raise ValueError("没有图片路径可用于生成GIF")
            
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 根据文件名中的步骤编号排序（按我们的命名规则：step_xxx_0001.png）
        if sort_by_step:
            def extract_step(path: str) -> int:
                """从文件名中提取步骤编号"""
                filename = os.path.basename(path)
                # 分割文件名获取步骤号（例如 "step_abcd_0005.png" -> 5）
                parts = filename.split("_")
                for part in parts:
                    if part.isdigit() or (part.endswith('.png') and part[:-4].isdigit()):
                        return int(part.replace('.png', ''))
                return 0  # 无法提取时返回0
            
            # 按步骤编号排序
            image_paths.sort(key=extract_step)
        
        # 打开所有图片
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = Image.open(path)
                images.append(img.convert('RGB'))  # 确保统一格式
        
        if not images:
            raise ValueError("没有找到有效的图片文件")
            
        # 保存为GIF
        images[0].save(
            output_path,
            format='GIF',
            append_images=images[1:],
            save_all=True,
            duration=duration,
            loop=loop,
            disposal=2  # 每帧显示后清除
        )
        
        return output_path
