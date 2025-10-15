from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

# 声明基类（所有模型都继承这个基类）
Base = declarative_base()

# 数据库连接配置
# 可以从环境变量读取配置，方便不同环境（开发/测试/生产）切换
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///llm_env_interactions.db"  # 默认使用SQLite
)

# 创建数据库引擎
# 对于SQLite，需要添加check_same_thread=False参数（仅用于开发）
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}  # SQLite特定配置
    )
else:
    # 其他数据库（如PostgreSQL/MySQL）的配置
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,          # 连接池大小
        max_overflow=20,       # 连接池溢出大小
        pool_recycle=3600      # 连接回收时间（秒）
    )

# 创建会话工厂（用于创建数据库会话）
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

# 依赖函数（用于FastAPI中获取数据库会话）
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()