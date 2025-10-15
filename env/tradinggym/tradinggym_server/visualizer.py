import os
import uuid
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import List

class StepMultiPlotSaver:
    def __init__(self, output_dir: str = None, max_log_entries: int = 10):
        """初始化多子图步骤保存器，包含左侧日志区域"""
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "step_plots")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.session_id = str(uuid.uuid4())[:6]
        self.image_paths: List[str] = []
        self.log_entries: List[str] = []  # 存储日志条目
        self.max_log_entries = max_log_entries  # 日志区域最多显示的条目数
        
        # 初始化图形和子图 - 优化宽度比例，确保右侧图表有足够空间
        # 整体宽度比例：日志占20%，间距占5%，图表占75%
        self.fig = plt.figure(figsize=(28, 12))  # 适度增加整体宽度到28
        self.log_ax = self.fig.add_axes([0.02, 0.05, 0.18, 0.9])  # 日志区域宽度缩减为0.18
        # 图表区域起始位置调整为0.25，宽度增加到0.73，确保足够宽
        self.axes = [
            self.fig.add_axes([0.25, 0.7, 0.73, 0.25]),  # 价格图表
            self.fig.add_axes([0.25, 0.45, 0.73, 0.2]),  # 每日Reward
            self.fig.add_axes([0.25, 0.2, 0.73, 0.2]),   # 累积Reward
            self.fig.add_axes([0.25, 0.05, 0.73, 0.1])   # 累积Profit
        ]
        
        # 设置共享x轴
        for ax in self.axes[1:]:
            ax.sharex(self.axes[0])
        
        plt.close(self.fig)  # 不显示窗口，只保存图片

    def add_log_entry(self, step: int, action: str, explanation: str = ""):
        """添加日志条目"""
        # 格式化日志条目，包含步骤和动作信息
        log_entry = f"Step {step}:\n"
        log_entry += f"Action: {action}\n"
        if explanation:
            # 拆分长解释为多行，每行不超过45个字符（适应稍窄的日志区域）
            explanation_lines = []
            for i in range(0, len(explanation), 45):
                explanation_lines.append(explanation[i:i+45])
            log_entry += "Explanation: " + "\n             ".join(explanation_lines)
        
        self.log_entries.append(log_entry)
        
        # 始终保持日志条目数量不超过最大值，实现滚动效果
        if len(self.log_entries) > self.max_log_entries:
            self.log_entries = self.log_entries[-self.max_log_entries:]

    def render_step(self, step: int, 
                   price_data: np.ndarray, 
                   price_dates,
                   positions: List[int], 
                   step_rewards: List[float],
                   total_rewards: List[float],
                   total_profits: List[float],
                   current_total_reward: float,
                   current_total_profit: float,
                   action: str,  # 当前步骤的动作
                   explanation: str = ""  # 动作的解释
                   ) -> str:
        """绘制单步的多子图并保存，包含日志区域"""
        # 添加当前步骤的日志
        self.add_log_entry(step, action, explanation)
        
        # 清除所有子图
        self.log_ax.clear()
        for ax in self.axes:
            ax.clear()
            
        days = np.arange(len(price_data))
        
        # ----------------------
        # 左侧日志区域
        # ----------------------
        self.log_ax.axis('off')  # 关闭坐标轴
        
        # 准备日志文本，条目间添加分隔线
        log_text = "\n" + "\n" + "="*35 + "\n".join(self.log_entries)  # 适应宽度减少分隔线长度
        
        # 设置等宽字体以获得更好的显示效果
        font = FontProperties(family='monospace', size=10)
        
        # 文本位置调整，确保在窄日志区域内显示正常
        self.log_ax.text(0.02, 0.98, log_text, fontproperties=font, 
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(facecolor='white', alpha=0.9, pad=10),
                       wrap=True)
        self.log_ax.set_title("Interaction Log", fontsize=12, pad=10)
        
        # ----------------------
        # 右侧图表区域（更宽）
        # ----------------------
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
        self.fig.suptitle(f"Step: {step} | Total Reward: {current_total_reward:.6f} | Total Profit: {current_total_profit:.6f}",
                         fontsize=14)
        
        # 保存图片
        img_filename = f"step_{self.session_id}_{step:04d}.png"
        img_path = os.path.join(self.output_dir, img_filename)
        self.fig.savefig(img_path, dpi=100, bbox_inches='tight')
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
    