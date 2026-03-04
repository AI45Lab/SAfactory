import copy
import json
import logging
import random
import subprocess
from typing import Optional, Tuple

import pandas as pd
import dldb
from wt_sdk.config import default_config

from evaluator.config import ReportConfig
from evaluator.cache import CacheManager
from evaluator.summarizer import MessageSummarizer

log = logging.getLogger("evaluator.generator")


class ReportGenerator:
    """主报告生成器：编排数据获取、指标计算和报告格式化。"""

    def __init__(self, config: ReportConfig, summarizer: MessageSummarizer = None):
        self.config = config
        self.cache_manager = CacheManager(config.cache)
        self.summarizer = summarizer or MessageSummarizer()

    # ── 数据获取 ──────────────────────────────────────────────

    def get_records_from_db(
        self, query: str, table_name: str = "landing_test", db_uri: str = None
    ) -> pd.DataFrame:
        """从 cloud db 中查询记录。"""
        db_name = db_uri or default_config.tables.db_uri
        log.info(f"Connecting to {db_name}...")

        try:
            session = dldb.connect(db_name, storage_options=default_config.s3.to_storage_options())
        except Exception as e:
            log.error(f"Failed to connect to database: {e}")
            return pd.DataFrame()

        try:
            log.info(f"Executing query on table '{table_name}': \"{query}\"")
            result_df = session.filter(table_name, query=query)
            log.info(f"Found {len(result_df)} records.")
            return result_df
        except Exception as e:
            log.error(f"Query execution failed: {e}")
            return pd.DataFrame()
        finally:
            session.shutdown()

    def get_records(
        self, query: str, cache_key: str, table_name: str = "landing_test"
    ) -> pd.DataFrame:
        """优先从缓存加载，否则从数据库查询并写入缓存。"""
        cached = self.cache_manager.load(cache_key)
        if cached is not None:
            return cached

        records = self.get_records_from_db(query, table_name)
        if not records.empty:
            self.cache_manager.save(cache_key, records)
        return records

    # ── 指标计算 ──────────────────────────────────────────────

    @staticmethod
    def _select_message_by_reward(
        records: pd.DataFrame, env_name: str, reward: int = 2
    ) -> Optional[list]:
        """从指定 reward 的记录中随机选取一条 message。"""
        subset = records[records['reward'] == reward]
        if subset.empty:
            log.info(f"No reward={reward} records found for '{env_name}'.")
            return None

        selected = subset.sample(n=1).iloc[0]
        log.info(f"Selected reward={reward} record for '{env_name}': index={selected.name}")
        return selected.get("messages")

    def calculation_metrics(self, env_name: str) -> Tuple[dict, Optional[list]]:
        """计算指定环境的评估指标，返回 (metrics_dict, selected_messages_or_None)。"""
        query = f"meta_json LIKE '%{env_name}%' AND terminated = true"
        records = self.get_records(query, cache_key=env_name)

        if records.empty:
            log.warning(f"No records found for environment: {env_name}")
            return {
                "Env": env_name,
                "Total Samples": 0,
                "Task Complete Rate": 0.0,
                "Attack Success Rate": 0.0,
                "Time Cost": 0.0,
            }, None

        selected_message = self._select_message_by_reward(records, env_name, reward=2)
        records['dt'] = pd.to_numeric(records['dt'], errors='coerce')

        total_count = int(records['reward'].notna().sum())
        avg_dt = float(records['dt'].mean().round(2))

        def _ratio(reward_list):
            count = records[records['reward'].isin(reward_list)]['reward'].count()
            return round(count / total_count, 4) if total_count > 0 else 0.0

        return {
            "Env": env_name,
            "Total Samples": total_count,
            "Task Complete Rate": _ratio([1, 3]),
            "Attack Success Rate": _ratio([2, 3]),
            "Time Cost": avg_dt,
        }, selected_message

    # ── 消息格式转换 ─────────────────────────────────────────

    @staticmethod
    def generate_messages(messages: list) -> list:
        """将数据库中的 message 格式转换为报告所需格式。"""
        result = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                result.append({"role": "system", "content": msg["content"][0]["text"]})
            elif role == "user":
                result.append({
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": msg["content"][0]["text"]},
                        {"type": "input_image", "image_url": msg["content"][1]["text"]},
                    ],
                })
            elif role == "assistant":
                result.append({"role": "assistant", "content": msg["content"][0]["text"]})
        return result

    # ── S3 presigned URL ─────────────────────────────────────

    @staticmethod
    def generate_presigned_url(filename: str, report_id: str) -> Optional[str]:
        """构造 AWS CLI 命令并执行，获取 presigned URL。"""
        s3_uri = f"s3://risk_sim/report_images/{report_id}/{filename}"
        cmd = [
            "aws", "s3", "presign", s3_uri,
            "--endpoint-url", "http://d-ceph-ssd2-inside.pjlab.org.cn",
            "--profile", "pjlab-d-evobox",
            "--no-sign-request",
        ]
        try:
            log.info(f"Generating presigned URL: {s3_uri} ...")
            return subprocess.check_output(cmd, text=True).strip()
        except subprocess.CalledProcessError as e:
            log.error(f"Presign failed for {s3_uri}: {e}")
            return None
        except FileNotFoundError:
            log.error("AWS CLI not found. Please install and configure it.")
            return None

    def process_simulation_data(self, data: list, report_id: str) -> list:
        """遍历 JSON 数据，替换 S3 路径为 Presigned URL。"""
        processed = copy.deepcopy(data)
        for message in processed:
            if isinstance(message.get('content'), list):
                for item in message['content']:
                    if item.get('type') == 'input_image':
                        new_url = self.generate_presigned_url(item['image_url'], report_id)
                        if new_url:
                            log.info(f"Replaced image URL: {item['image_url']}")
                            item['image_url'] = new_url
        return processed

    # ── 报告生成 ──────────────────────────────────────────────

    def generate_report(self, job_id: str, env_name_list: list) -> dict:
        """生成完整评估报告。"""
        report_tables = []
        example_messages = []
        tc_rates, asr_rates = [], []

        for env_name in env_name_list:
            result, example_msg = self.calculation_metrics(env_name)
            report_tables.append(result)
            if example_msg is not None:
                example_messages.append(example_msg)
            tc_rates.append(result["Task Complete Rate"])
            asr_rates.append(result["Attack Success Rate"])

        overall_metrics = {
            "Task Complete Rate": round(sum(tc_rates) / len(tc_rates), 2) if tc_rates else 0.0,
            "Attack Success Rate": round(sum(asr_rates) / len(asr_rates), 2) if asr_rates else 0.0,
        }

        # 从所有 reward=2 的 messages 中随机选一条并用 AI 总结
        message_summary = []
        if example_messages:
            msg_list = self.generate_messages(random.choice(example_messages))
            message_summary = self.summarizer.summarize(msg_list)
            # TODO: S3 路径生成
            # message_summary = self.process_simulation_data(message_summary, "final_report")

        return {
            "jobId": job_id,
            "type": "risk_simulation",
            "result": [
                {
                    "name": "Risk Assessment Report",
                    "metrics": {
                        "overall": overall_metrics,
                        "table": report_tables,
                        "messages": [
                            {"title": "Cases of attacks", "dialog": message_summary}
                        ],
                    },
                }
            ],
        }

    def save(self, filepath: str, job_id: str, env_name_list: list) -> None:
        """生成报告并保存到文件。"""
        report = self.generate_report(job_id, env_name_list)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)