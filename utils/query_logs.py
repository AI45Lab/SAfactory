import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import argparse
from db import SessionLocal, InteractionSession, InteractionStep

def print_step_details(step):
    """打印单步交互的Prompt、Response和Reward信息"""
    print(f"\n===== 步骤 {step.step_number} =====")
    print(f"时间: {step.timestamp}")
    print(f"奖励: {step.reward}")
    print(f"是否完成: {step.done}")
    
    print("\n----- Prompt -----")
    print(step.llm_prompt[:500])  # 显示前500字符，避免输出过长
    if len(step.llm_prompt) > 500:
        print("...（内容过长，已截断）")
    
    print("\n----- Response -----")
    print(step.llm_response[:500])
    if len(step.llm_response) > 500:
        print("...（内容过长，已截断）")

def query_by_session(session_id):
    """查询指定会话ID的详细交互数据"""
    db = SessionLocal()
    try:
        # 获取会话信息
        session = db.query(InteractionSession).filter(
            InteractionSession.id == session_id
        ).first()
        
        if not session:
            print(f"未找到ID为 {session_id} 的会话")
            return
        
        # 打印会话基本信息
        print(f"===== 会话 {session_id} 详情 =====")
        print(f"环境: {session.env_name}")
        print(f"模型: {session.model_name}")
        print(f"时间: {session.start_time} ~ {session.end_time}")
        print(f"总奖励: {session.total_reward}")
        print(f"状态: {'已完成' if session.completed else '未完成'}")
        print(f"总步骤数: {len(session.steps)}\n")
        
        # 打印每步的详细信息（包含Prompt、Response、Reward）
        for step in session.steps:
            print_step_details(step)
            
    finally:
        db.close()

def list_recent_sessions(limit=5):
    """列出最近的会话，用于获取会话ID"""
    db = SessionLocal()
    try:
        sessions = db.query(InteractionSession).order_by(
            InteractionSession.start_time.desc()
        ).limit(limit).all()
        
        print(f"最近 {limit} 个会话：")
        for session in sessions:
            print(f"ID: {session.id}, 环境: {session.env_name}, "
                  f"模型: {session.model_name}, 总奖励: {session.total_reward}, "
                  f"时间: {session.start_time.strftime('%Y-%m-%d %H:%M')}")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查询LLM与环境交互的日志数据")
    parser.add_argument("--list", action="store_true", help="列出最近的会话")
    parser.add_argument("--session-id", type=int, help="指定会话ID，查看详细步骤")
    args = parser.parse_args()
    
    if args.list:
        list_recent_sessions()
    elif args.session_id:
        query_by_session(args.session_id)
    else:
        print("请使用 --list 查看会话列表，或使用 --session-id 指定会话ID查看详情")
