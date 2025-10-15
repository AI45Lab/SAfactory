from .config import engine, Base, SessionLocal, get_db
from .models import InteractionSession, InteractionStep

# 初始化数据库（创建所有表）
def init_db():
    # 创建所有表（如果不存在）
    Base.metadata.create_all(bind=engine)
    print("数据库表初始化完成")

# 对外暴露的接口
__all__ = [
    "init_db", 
    "SessionLocal", 
    "get_db",
    "InteractionSession",
    "InteractionStep"
]