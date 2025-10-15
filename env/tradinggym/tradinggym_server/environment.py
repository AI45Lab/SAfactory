from time import time
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import re
import os
from .utils import Actions, Positions
from .visualizer import StepMultiPlotSaver
from .gif_creator import StepGifCreator

class TradingGym(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(
            self,
            df,                                     # 包含Close列的价格DataFrame（按时间升序）
            window_size: int = 7,
            senti_df: Optional[Any] = None,
            render_mode = None,
            max_text_history_days: int = 7
    ):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.df = df
        self.window_size = window_size
        self.render_mode = render_mode

        # 外部文本数据（可以为DataFrame、list 或 None）。要求与df对齐（同长度可索引）
        self.senti_df = senti_df
        self.max_text_history_days = int(max_text_history_days)

        # 初始化价格/特征（与原 StocksEnv _process_data 类似）
        self.price_merge, self.price_date = self._process_data()
        self.shape = (window_size,)

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions)) # 0=Sell, 1=Buy
        INF = 1e10

        # 改成 Dict observation，包含原始 price 和 text prompt（便于 LLM 使用）
        self.observation_space = gym.spaces.Dict({
            "prices_features": gym.spaces.Box(low=-INF, high=INF, shape=self.shape, dtype=np.float32),
            "text": gym.spaces.Text(max_length=10000000000) # 用gym.spaces.Text？
        })
        
        # epoisode state
        self._start_index = self.window_size
        self._current_index = None
        self._end_date = self.price_date[-2]
        self._truncated = None
        self._current_date = None
        self._last_trade_date = None
        self._position = None
        self._position_history = None
        self._step_reward = None
        self._step_reward_history = None
        self._total_reward_history = None
        self._total_reward = None
        self._total_profit = None
        self._total_profit_history = None
        self.history = None
        self.action_dict = {
            0: 'Sell',
            1: 'Buy'
        }
        self.position_dict = {
            0: 'Short',
            1: 'Long'
        }

        # 存储价格日期数据
        self.price_dates = self.price_df.index.tolist()
        self.image_saver = StepMultiPlotSaver(output_dir="/mnt/shared-storage-user/chenxinquan/ai_sandbox/visualize")

    # -------------------------
    # 处理数据，按日期合并为一个DataFrame
    # -------------------------
    def _process_data(self):
        self.df['Date'] = pd.to_datetime(self.df['Date']).dt.date
        self.price_df = self.df.sort_values(by='Date')
        price_date = self.df['Date'].tolist()
        price_date.sort()
        self.senti_df['date'] = pd.to_datetime(self.senti_df['date']).dt.date
        self.ticker = self.senti_df['ticker'][0]
        
        # 按时间聚合与合并
        senti_grouped = self.senti_df.groupby('date')['senti_label'].apply(list).reset_index()
        processed_text_grouped = self.senti_df.groupby('date')['processed'].apply(list).reset_index()
        senti_text_merged = pd.merge(senti_grouped, processed_text_grouped, left_on='date', right_on='date', how='left')
        merged = pd.merge(self.df, senti_text_merged, left_on='Date', right_on='date', how='outer')
        merged['Date'] = merged['Date'].combine_first(merged['date'])
        merged = merged.drop(columns=['date'])
        merged = merged.sort_values(by='Date')
        
        return merged, price_date
    
    # -------------------------
    # reset / step （保持原接口，但观测变为 dict 且包含 prompt）
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._truncated = False
        self._current_index = self._start_index
        self._current_date = self.price_date[self._start_index]
        self._last_trade_date = self.price_date[self._start_index - 1]
        self._last_trade_index = None
        self._position = None
        self._position_history = self.window_size * [None]
        self._step_reward_history = self.window_size * [0]
        self._total_reward_history = self.window_size * [0]
        self._total_profit_history = self.window_size * [1]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.current_step = 0
        self.price_history: List[float] = []  # 价格历史
        self.position_history: List[int] = []  # 仓位历史 (0=空仓/短仓, 1=多仓)
        self.step_reward_history: List[float] = []  # 单步奖励历史
        self.total_reward_history: List[float] = []  # 累积奖励历史
        self.total_profit_history: List[float] = []  # 累积利润历史
        self.total_reward = 0.0
        self.total_profit = 0.0

        observation = self._get_observation_with_text()
        info = self._get_info()
        if self.render_mode == 'human':
            self._render_frame()
        return observation, info

    def step(self, action):
        action, explanation = self.parse_llm_response(action)
        print("-" * 80)
        print("The current date is: ", self._current_date)
        print("The current position is: ", self._position)
        print("Today's action is: ", self.action_dict[action])
        self._truncated = False

        if self._current_date == self._end_date:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._step_reward = step_reward
        self._step_reward_history.append(step_reward)
        self._total_reward += step_reward
        self._total_reward_history.append(self._total_reward)

        trade = self._update_position(action)
        self._position_history.append(self._position)
        if trade:
            self._update_profit()
            self._total_profit_history.append(self._total_profit)
            self._position = self._position.opposite()
            self._last_trade_index = self._current_index
            print("Position opposite!")
        else:
            self._total_profit_history.append(self._total_profit)
            print("Position maintenance!")

        self._current_index += 1
        self._current_date = self.price_date[self._current_index]
        observation = self._get_observation_with_text()
        info = self._get_info()
        self._update_history(info)

        # 获取当前价格
        current_price = self._get_current_price()
        self.price_history.append(current_price)
        
        # 处理动作（0=空仓/短仓，1=多仓）
        self.position_history.append(action)

        self.total_reward += step_reward
        # 记录历史数据
        self.step_reward_history.append(step_reward)
        self.total_reward_history.append(self.total_reward)
        self.total_profit_history.append(self._total_profit)

        self.image_saver.render_step(
            step=self.current_step,
            price_data=np.array(self.price_history),
            price_dates=self.price_dates[:self.current_step+1],
            positions=self.position_history,
            step_rewards=self.step_reward_history,
            total_rewards=self.total_reward_history,
            total_profits=self.total_profit_history,
            current_total_reward=self.total_reward,
            current_total_profit=self.total_profit
        )

        self.current_step += 1

        return observation, step_reward, False, self._truncated, info
    
    def generate_gif(self, 
                    output_path: Optional[str] = None, 
                    duration: int = 200, 
                    loop: int = 0) -> str:
        """
        将所有步骤图片合并为GIF
        
        参数:
            output_path: 输出GIF路径，None则自动生成
            duration: 每帧持续时间（毫秒）
            loop: 循环次数，0表示无限循环
            
        返回:
            生成的GIF文件路径
        """
        if not self.image_saver:
            raise ValueError("未启用图片保存模式，请在初始化时设置render_mode='save_images'")
            
        image_paths = self.image_saver.get_image_paths()
        if not image_paths:
            raise ValueError("没有保存的步骤图片，无法生成GIF")
            
        # 自动生成输出路径（如果未指定）
        if not output_path:
            # 使用图片保存目录作为GIF输出目录
            if self.image_saver.output_dir:
                output_dir = self.image_saver.output_dir
            else:
                output_dir = os.path.dirname(image_paths[0])
                
            output_path = os.path.join(output_dir, f"trading_simulation_{self.image_saver.session_id}.gif")
        
        # 生成GIF
        return StepGifCreator.create_gif(
            image_paths=image_paths,
            output_path=output_path,
            duration=duration,
            loop=loop
        )
    
    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            step_reward=self._step_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        return self.price_df.iloc[(self._current_index-self.window_size+1): self._current_index+1][['Date','Close']]
    
    def _get_current_price(self):
        return self.current_price

    # -------------------------
    # 新：把 text prompt 加入观测
    # -------------------------
    def _get_observation_with_text(self) -> Dict[str, Any]:
        """返回 dict ，包含数值 features（原始）和 text prompt（string）"""
        obs_price = self._get_observation()
        prompt = self._build_stock_sentiment_prompt()
        # 为兼容 observation_space 我们返回 'text' 字段，但实际 PPOTrainer 只会取 obs['text'] 的字符串
        obs = {"prices_history": obs_price, "text": prompt}
        return obs
    
    def _build_stock_sentiment_prompt(self):
        prompt_lines = []
        # prompt_lines.append(f"你是一名金融分析助手，今天是{self._current_date}，你将看到过去{self.window_size}天的历史股价和金融情绪推文，请根据这些资料预测下一日股价是上涨(1)还是下跌(0)。\n")
        prompt_lines.append(f"You are a financial analysis assistant. Today is {self._current_date}. Below you will analyze the stock {self.ticker}. You will see the historical stock prices and financial sentiment tweets from the past {self.window_size} days. Based on this information, please predict whether the stock price will rise (1) or fall (0) on the next day. ")
        # prompt_lines.append("以下是历史股价和当天的金融情绪推文：\n")
        prompt_lines.append("Here are the historical stock prices and financial sentiment tweets for the day:")
        
        start_date = self.price_date[self._current_index - self.window_size]
        end_date = self.price_date[self._current_index]
        mask = (self.price_merge['Date'] >= start_date) & (self.price_merge['Date'] <= end_date)
        for _, row in self.price_merge.loc[mask].iterrows():
            if not pd.isna(row['Open']) and not pd.isna(row['Close']) and not pd.isna(row['High']) and not pd.isna(row['Low']) and not pd.isna(row['Volume']):
                line = f"Date: {row['Date']} | Open: {row['Open']:.2f}, Close: {row['Close']:.2f}, High: {row['High']:.2f}, Low: {row['Low']:.2f}, Volume: {int(row['Volume'])}"
            else:
                line = f"Date: {row['Date']} | No transactions today"
            if isinstance(row['senti_label'], list) and isinstance(row['processed'], list):
                senti_all = " | Today's Financial Sentiment Tweets:"
                for i, (text, senti) in enumerate(zip(row['processed'], row['senti_label'])):
                    senti_text = f"Tweet: {text}, sentiment prediction: {senti}"
                    senti_all += f"（{i+1}）"
                    senti_all += senti_text
            else:
                senti_all = " | No financial sentiment tweets today"
            
            line += senti_all

            prompt_lines.append(line)

        prompt_lines.append("Based on the above historical stock prices and financial sentiment tweets, please predict the stock market trend for the next day (up=1, down=0) and provide a brief reasoning. Please note that financial sentiment tweets are for reference only. Please consider the next day's rise and fall in combination with historical stock prices and tweets.")

        # 输出格式规范（关键：让 LLM 输出可解析）
        prompt_lines.append("\nIMPORTANT: Output MUST be EXACTLY in the following format (no extra commentary):")
        prompt_lines.append("LINE1: TRENDS: <0 or 1>   (0 = Fall, 1 = Rise)")
        prompt_lines.append("LINE2: EXPLANATION: <a one-line concise explanation of why you chose those days and this action>")
        prompt_lines.append("Remember: ONLY output these 2 lines, nothing else.")

        return "\n".join(prompt_lines)
                    
    
    # -------------------------
    # 解析 LLM 输出（容错）：返回 (action_int, explanation)
    # -------------------------
    @staticmethod
    def parse_llm_response(response_text: str) -> Tuple[int, List[int], str]:
        """
        解析 LLM 输出，预期格式：
          LINE1: ACTION: 1
          LINE2: EXPLANATION: reason...
        容错解析策略：
          - 忽略大小写与多余空白
          - 尝试从任意位置提取第一个出现的 0/1 作为 action（若未找到，默认 1=Buy）
          - 尝试提取 SELECTED_DAYS 中的整数列表，若失败返回 []
          - EXPLANATION 为剩余文本拼接
        """
        if not isinstance(response_text, str):
            response_text = str(response_text)

        lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
        action = None
        explanation = ""

        # 合并到单字符串方便正则抽取
        whole = "\n".join(lines)

        # 找 ACTION 行或第一个 0/1
        mact = re.search(r"TRENDS\s*[:\-]\s*([01])", whole, flags=re.IGNORECASE)
        if mact:
            action = int(mact.group(1))
        else:
            # 退路：抽取第一个单独的 '0' 或 '1' token
            toks = re.findall(r"(?<!\d)(0|1)(?!\d)", whole)
            if toks:
                action = int(toks[0])

        # 找 EXPLANATION（取 ACTION 之后的文本）
        mexp = re.search(r"EXPLANATION\s*[:\-]\s*(.+)", whole, flags=re.IGNORECASE | re.DOTALL)
        if mexp:
            explanation = mexp.group(1).strip()
        else:
            # 退路：把非已识别行拼成 explanation
            # 去掉包含 ACTION 的行，剩余的当 explanation
            rem_lines = []
            for ln in lines:
                if re.search(r"ACTION", ln, flags=re.IGNORECASE):
                    continue
                rem_lines.append(ln)
            explanation = " ".join(rem_lines).strip()

        # 最后保障 action 有值（默认 Buy = 1 更为保守/激进可改）
        if action is None:
            action = 1

        return int(action), explanation
    
    def _update_position(self, action):
        trade = False
        if self._position is None:
            if action == Actions.Buy.value:
                self._position = Positions.Long
            else:
                self._position = Positions.Short
            self._last_trade_index = self._current_index
            print(f"Predicted action is {self.action_dict[action]}, the current position is {self.position_dict[self._position.value]}")
            return trade
        else:
            if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
                trade = True
            else:
                trade = False
            print(f"Predicted action is {self.action_dict[action]}, the current position is {self.position_dict[self._position.value]}")
            return trade
    
    def _calculate_reward(self, action):
        step_reward = 0
        current_price = self.price_df.iloc[self._current_index]['Close']
        self.current_price = current_price
        next_price = self.price_df.iloc[self._current_index + 1]['Close']
        if action == Actions.Buy.value:
            step_reward = next_price - current_price
        elif action == Actions.Sell.value:
            step_reward = current_price - next_price
        return step_reward

    def _update_profit(self):
        current_price = self.price_df.iloc[self._current_index]['Close']
        last_trade_price = self.price_df.iloc[self._last_trade_index]['Close']
        if self._position == Positions.Long:
            shares = (self._total_profit * (1 - 0.0001)) / last_trade_price
            self._total_profit = (shares * (1 - 0.0005)) * current_price
        elif self._position == Positions.Short:
            shares = (self._total_profit * (1 - 0.0005)) / last_trade_price
            self._total_profit = (shares * (1 - 0.0001)) * current_price

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.history[key].append(value)

    # 渲染/plot 等保持原样（简略复制）
    def _render_frame(self):
        self.render()

    def render(self):
        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()
        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)
        plt.suptitle("Total Reward: %.6f" % self._total_reward + ' ~ ' + "Total Profit: %.6f" % self._total_profit)
        end_time = time()
        process_time = end_time - start_time
        pause_time = (1 / self.metadata['render_fps']) - process_time
        if pause_time > 0:
            plt.pause(pause_time)

    def plot_all(self, save_path=None):
        days = np.arange(len(self.price_date[:-1]))
        prices = np.array(self.price_df['Close'].tolist()[:-1])
        rewards = np.array(self._step_reward_history)
        total_rewards = np.array(self._total_reward_history)
        profits = np.array(self._total_profit_history)

        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

        # 子图1：价格 + 持仓
        axes[0].plot(days, prices, label="Close Price", color="black")

        long_days = [i for i, pos in enumerate(self._position_history) if pos == Positions.Long]
        short_days = [i for i, pos in enumerate(self._position_history) if pos == Positions.Short]

        axes[0].scatter(long_days, prices[long_days], marker="^", color="green", label="Long")
        axes[0].scatter(short_days, prices[short_days], marker="v", color="red", label="Short")

        axes[0].set_ylabel("Price")
        axes[0].set_title("Price and Positions")
        axes[0].legend()

        # 子图2：每日 Reward
        axes[1].bar(days, rewards, color=["green" if r >= 0 else "red" for r in rewards])
        axes[1].set_ylabel("Step Reward")
        axes[1].set_title("Daily Reward")
        axes[1].grid(True)

        # 子图3：累积 Reward
        axes[2].plot(days, total_rewards, color="blue", label="Total Reward")
        axes[2].set_ylabel("Reward")
        axes[2].set_title("Total Reward")
        axes[2].legend()
        axes[2].grid(True)

        # 子图4：累积 Profit
        axes[3].plot(days, profits, color="blue", label="Profit")
        axes[3].set_ylabel("Profit")
        axes[3].set_title("Cumulative Profit")
        axes[3].legend()
        axes[3].grid(True)

        plt.xlabel("Day")
        fig.suptitle("Total Reward: %.6f" % self._total_reward + ' ~ ' + "Total Profit: %.6f" % self._total_profit)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)
