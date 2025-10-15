import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
from typing import List, Dict, Any
import uuid

class StepMultiPlotSaver:
    def __init__(self, output_dir: str = None):
        """初始化多子图步骤保存器"""
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "step_plots")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.session_id = str(uuid.uuid4())[:6]
        self.image_paths: List[str] = []
        
        # 初始化图形和子图
        self.fig, self.axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
        plt.close(self.fig)  # 不显示窗口，只保存图片

    def render_step(self, step: int, 
                   price_data: np.ndarray, 
                   price_dates,
                   positions: List[int], 
                   step_rewards: List[float],
                   total_rewards: List[float],
                   total_profits: List[float],
                   current_total_reward: float,
                   current_total_profit: float) -> str:
        """绘制单步的多子图并保存"""
        # 清除所有子图
        for ax in self.axes:
            ax.clear()
            
        days = np.arange(len(price_data))
        
        # 子图1：价格 + 持仓
        self.axes[0].plot(days, price_data, label="Close Price", color="black")
        
        # 标记多空仓位
        long_days = [i for i, pos in enumerate(positions) if pos == 1]  # 1表示多仓
        short_days = [i for i, pos in enumerate(positions) if pos == 0]  # 0表示空仓/短仓
        
        self.axes[0].scatter(long_days, price_data[long_days], 
                           marker="^", color="green", label="Long")
        self.axes[0].scatter(short_days, price_data[short_days], 
                           marker="v", color="red", label="Short")
        
        self.axes[0].set_ylabel("Price")
        self.axes[0].set_title("Price and Positions")
        self.axes[0].legend()
        
        # 子图2：每日 Reward
        if step_rewards:  # 确保有数据
            self.axes[1].bar(days, step_rewards, 
                           color=["green" if r >= 0 else "red" for r in step_rewards])
        self.axes[1].set_ylabel("Step Reward")
        self.axes[1].set_title("Daily Reward")
        self.axes[1].grid(True)
        
        # 子图3：累积 Reward
        if total_rewards:  # 确保有数据
            self.axes[2].plot(days, total_rewards, color="blue", label="Total Reward")
        self.axes[2].set_ylabel("Reward")
        self.axes[2].set_title("Total Reward")
        self.axes[2].legend()
        self.axes[2].grid(True)
        
        # 子图4：累积 Profit
        if total_profits:  # 确保有数据
            self.axes[3].plot(days, total_profits, color="blue", label="Profit")
        self.axes[3].set_ylabel("Profit")
        self.axes[3].set_title("Cumulative Profit")
        self.axes[3].legend()
        self.axes[3].grid(True)
        
        # 设置x轴和总标题
        self.axes[3].set_xlabel("Day")
        self.fig.suptitle(f"Step: {step} | Total Reward: {current_total_reward:.6f} | Total Profit: {current_total_profit:.6f}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
        
        # 保存图片
        img_filename = f"step_{self.session_id}_{step:04d}.png"
        img_path = os.path.join(self.output_dir, img_filename)
        self.fig.savefig(img_path, dpi=100)
        self.image_paths.append(img_path)
        
        return img_path

    def get_image_paths(self) -> List[str]:
        """获取所有图片路径"""
        return self.image_paths.copy()

    def cleanup(self) -> None:
        """清理生成的图片"""
        for img_path in self.image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        self.image_paths.clear()
