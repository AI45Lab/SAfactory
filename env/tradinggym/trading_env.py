import re
import os
import uuid
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import gymnasium as gym
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from time import time
from matplotlib.font_manager import FontProperties

from core.env.base_env import BaseEnv
from core.env.env_register import register_env


class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

@register_env("trading_gym")
class TradingGym(BaseEnv):

    def __init__(
        self,
        env_id: str,
        data_dir: str,
        visual_save_path: str,
        price_filename: str,
        tweet_filename: Optional[str] = None,
        window_size: int = 7, # 智能体能看到过去window_size天的历史数据
    ):
        super().__init__()
        self.env_id = env_id
        self.price_df = self._read_csv(os.path.join(data_dir, price_filename))
        self.visual_save_path = visual_save_path
        self.window_size = window_size

        # 初始化价格，文本内容
        self.price_merge, self.price_date = self._process_data(data_dir=data_dir, tweet_filename=tweet_filename)
        self.shape = (window_size,)

        # 动作与观测空间定义
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Dict({
            "prices_features": gym.spaces.Box(
                low=-1e10, high=1e10, 
                shape=self.shape, 
                dtype=np.float32
            ),
            "text": gym.spaces.Text(max_length=100_000_000)
        })

        # 初始化状态变量
        self._reset_state()

        # 可视化工具
        self.render_step = StepMultiPlotSaver()

        # 映射字典
        self.action_dict = {0: 'Sell', 1: 'Buy'}
        self.position_dict = {0: 'Short', 1: 'Long'}

    def reset(self, seed: Optional[int] = None):
        self._reset_state()
        
        observation = self._get_observation_with_text()
        info = self._get_info()

        return observation, info
    
    def step(self, action: Any) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步环境交互"""
        self.current_action, self.current_explanation = self.parse_llm_response(action)
        self._print_step_info(self.current_action)
        
        # 检查终止条件
        self._truncated = self._current_date == self._end_date
        
        # 计算奖励与更新状态
        step_reward = self._calculate_reward(self.current_action)
        trade = self._update_position(self.current_action)
        if trade:
            self._update_profit()
            self._position = self._position.opposite()
            self._last_trade_index = self._current_index
        
        # 推进时间步
        self._current_index += 1
        self._current_date = self.price_date[self._current_index]
        self.current_step += 1
        
        # 更新历史记录
        self._update_history(step_reward)
        self._record_step_data(self.current_action, step_reward)

        return (
            self._get_observation_with_text(),
            step_reward,
            False,  # terminated始终为False（由truncated控制结束）
            self._truncated,
            self._get_info()
        )
    
    def get_task_prompt(self):
        return self._build_stock_sentiment_prompt()
    
    def render(self, img_filename) -> None:
        """渲染环境状态"""
        step_fig = self.render_step.render_step(
            step=self.current_step,
            price_data=np.array(self.price_history),
            price_dates=self.price_dates[:self.current_step+1],
            positions=self.position_history,
            step_rewards=self.step_reward_history,
            total_rewards=self.total_reward_history,
            total_profits=self.total_profit_history,
            current_total_reward=self.total_reward,
            current_total_profit=self._total_profit,
            action=self.action_dict[self.current_action],
            explanation=self.current_explanation
        )

        img_path = os.path.join(self.visual_save_path, img_filename)
        img_dir = os.path.dirname(img_path)
        os.makedirs(img_dir, exist_ok=True)
        step_fig.savefig(img_path, dpi=100, bbox_inches='tight')

    def close(self) -> None:
        """关闭环境，释放资源"""
        super().close()
        plt.close('all')  # 确保关闭所有matplotlib图表

    def parse_llm_response(self, response_text: str) -> Tuple[int, str]:
        """解析LLM输出，提取动作和解释"""
        response_text = str(response_text).strip()
        action = None
        explanation = ""
        
        # 提取趋势预测
        trend_match = re.search(r"TRENDS\s*[:\-]\s*([01])", response_text, re.IGNORECASE)
        if trend_match:
            action = int(trend_match.group(1))
        else:
            # 容错：提取第一个0/1
            nums = re.findall(r"(?<!\d)(0|1)(?!\d)", response_text)
            action = int(nums[0]) if nums else 1  # 默认买
        
        # 提取解释
        exp_match = re.search(r"EXPLANATION\s*[:\-]\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()
        else:
            explanation = "No explanation provided."
        
        return action, explanation

    def _read_csv(self, path):
        return pd.read_csv(path)

    def _process_data(self, data_dir: str, tweet_filename: Optional[str]) -> Tuple[pd.DataFrame, List]:
        """处理数据，按日期合并为一个DataFrame"""
        # 处理价格数据
        self.price_df['Date'] = pd.to_datetime(self.price_df['Date']).dt.date
        self.price_df = self.price_df.sort_values(by='Date')
        price_date = sorted(self.price_df['Date'].tolist())

        # 处理情绪数据
        if tweet_filename is not None:
            tweet_df = self._read_csv(os.path.join(data_dir, tweet_filename))
            tweet_df['date'] = pd.to_datetime(tweet_df['date']).dt.date
            self.ticker_name = tweet_df['ticker'][0] # 记录股票名
        
            # 按时间聚合与合并
            tweet_grouped = tweet_df.groupby('date')['senti_label'].apply(list).reset_index()
            processed_text_grouped = tweet_df.groupby('date')['processed'].apply(list).reset_index()
            tweet_text_merged = pd.merge(tweet_grouped, processed_text_grouped, left_on='date', right_on='date', how='left')

            # 合并价格与情绪数据
            merged = pd.merge(self.price_df, tweet_text_merged, left_on='Date', right_on='date', how='outer')
            merged['Date'] = merged['Date'].combine_first(merged['date'])
            merged = merged.drop(columns=['date']).sort_values(by='Date')
        
        return merged, price_date
    
    def _reset_state(self) -> None:
        """重置环境状态变量"""
        self._start_index = self.window_size
        self._current_index = self._start_index
        self._end_date = self.price_date[-2] if self.price_date else None
        self._current_date = self.price_date[self._start_index] if self.price_date else None
        self._last_trade_date = self.price_date[self._start_index - 1] if self.price_date else None
        self._last_trade_index = None
        self._position = None
        self._truncated = False
        
        # 历史记录
        self.current_step = 0
        self.current_action = None
        self.current_explanation = None
        self.price_history: List[float] = []
        self.position_history: List[int] = []
        self.step_reward_history: List[float] = []
        self.total_reward_history: List[float] = []
        self.total_profit_history: List[float] = []
        self.total_reward = 0.0
        self._total_profit = 1.0  # 初始利润
        
        # 价格日期列表
        self.price_dates = self.price_df.index.tolist() if not self.price_df.empty else []

    def _get_observation_with_text(self) -> Dict[str, Any]:
        """获取包含文本提示的观测"""
        return {
            "prices_history": self._get_observation(),
            "text": self._build_stock_sentiment_prompt()
        }

    def _build_stock_sentiment_prompt(self) -> str:
        """构建股票情绪分析提示"""
        prompt = [
            f"You are a financial analysis assistant. Today is {self._current_date}. "
            f"Analyze stock {self.ticker_name} using the past {self.window_size} days of prices and tweets. "
            "Predict if the price will rise (1) or fall (0) tomorrow.",
            "Historical data:"
        ]
        
        # 添加历史数据
        start_idx = self._current_index - self.window_size
        for _, row in self.price_merge.iloc[start_idx:self._current_index+1].iterrows():
            prompt.append(self._format_row_data(row))
        
        # 添加输出格式要求
        prompt.extend([
            "\nPredict the next day's trend (up=1, down=0) with reasoning.",
            "IMPORTANT: Output MUST be in this format:",
            "LINE1: TRENDS: <0 or 1>",
            "LINE2: EXPLANATION: <concise reason>",
            "Only output these 2 lines."
        ])
        
        return "\n".join(prompt)
    
    def _format_row_data(self, row: pd.Series) -> str:
        """格式化行数据为字符串"""
        # 价格信息
        if not pd.isna(row['Open']):
            price_str = (f"Date: {row['Date']} | Open: {row['Open']:.2f}, "
                         f"Close: {row['Close']:.2f}, High: {row['High']:.2f}, "
                         f"Low: {row['Low']:.2f}, Volume: {int(row['Volume'])}")
        else:
            price_str = f"Date: {row['Date']} | No transactions today"
        
        # 情绪信息
        if isinstance(row.get('senti_label'), list) and isinstance(row.get('processed'), list):
            tweets = [f"（{i+1}）Tweet: {t}, sentiment: {s}" 
                     for i, (t, s) in enumerate(zip(row['processed'], row['senti_label']))]
            senti_str = " | Today's tweets: " + " ".join(tweets)
        else:
            senti_str = " | No tweets today"
            
        return price_str + senti_str
    
    def _get_observation(self) -> pd.DataFrame:
        """获取价格观测"""
        return self.price_df.iloc[
            (self._current_index - self.window_size + 1): self._current_index + 1
        ][['Date', 'Close']]
    
    def _get_current_price(self) -> float:
        """获取当前价格"""
        return self.price_df.iloc[self._current_index]['Close']

    def _get_info(self) -> Dict[str, Any]:
        """获取当前信息字典"""
        return {
            'total_reward': self.total_reward,
            'step_reward': self.step_reward_history[-1] if self.step_reward_history else 0,
            'total_profit': self._total_profit,
            'position': self._position
        }
    
    # -------------------------
    # 交易逻辑
    # -------------------------
    def _calculate_reward(self, action: int) -> float:
        """计算单步奖励"""
        current_price = self._get_current_price()
        next_price = self.price_df.iloc[self._current_index + 1]['Close']
        
        if action == Actions.Buy.value:
            return next_price - current_price
        return current_price - next_price  # Sell

    def _update_position(self, action: int) -> bool:
        """更新持仓状态，返回是否发生交易"""
        if self._position is None:
            # 初始持仓
            self._position = Positions.Long if action == Actions.Buy.value else Positions.Short
            self._last_trade_index = self._current_index
            return False
            
        # 检查是否需要平仓
        if (action == Actions.Buy.value and self._position == Positions.Short) or \
           (action == Actions.Sell.value and self._position == Positions.Long):
            return True
        return False

    def _update_profit(self) -> None:
        """更新累计利润"""
        current_price = self._get_current_price()
        last_trade_price = self.price_df.iloc[self._last_trade_index]['Close']
        
        if self._position == Positions.Long:
            shares = (self._total_profit * 0.9999) / last_trade_price  # 手续费0.01%
            self._total_profit = shares * 0.9995 * current_price  # 手续费0.05%
        else:  # Short
            shares = (self._total_profit * 0.9995) / last_trade_price
            self._total_profit = shares * 0.9999 * current_price

    # -------------------------
    # 辅助方法
    # -------------------------
    def _print_step_info(self, action: int) -> None:
        """打印步骤信息"""
        print("-" * 80)
        print(f"Env id: {self.env_id}")
        print(f"Current date: {self._current_date}")
        print(f"Current position: {self._position}")
        print(f"Today's action: {self.action_dict[action]}")

    def _update_history(self, step_reward: float) -> None:
        """更新历史记录"""
        self.step_reward_history.append(step_reward)
        self.total_reward += step_reward
        self.total_reward_history.append(self.total_reward)
        self.total_profit_history.append(self._total_profit)

    def _record_step_data(self, action: int, step_reward: float) -> None:
        """记录步骤数据"""
        self.price_history.append(self._get_current_price())
        self.position_history.append(action)


class StepMultiPlotSaver:
    def __init__(self, max_log_entries: int = 10):
        """初始化多子图步骤保存器，包含左侧日志区域"""
        
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
        
        
        return self.fig

    def cleanup(self) -> None:
        """清理生成的图片"""
        for img_path in self.image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        self.image_paths.clear()