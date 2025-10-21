import argparse
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.registry import env_registry, init_env_registry

def main():
    parser = argparse.ArgumentParser(description="环境注册管理工具")
    subparsers = parser.add_subparsers(dest="command", required=True, help="命令类型")

    # 列出所有注册环境
    list_parser = subparsers.add_parser("list", help="列出所有已注册环境")
    list_parser.add_argument("--details", action="store_true", help="显示详细信息")

    # 验证环境注册
    verify_parser = subparsers.add_parser("verify", help="验证环境注册是否正确")
    verify_parser.add_argument("env_name", help="要验证的环境名称")

    # 重新加载环境配置
    reload_parser = subparsers.add_parser("reload", help="重新加载所有环境配置")

    args = parser.parse_args()

    # 初始化注册器
    init_env_registry()

    if args.command == "list":
        envs = env_registry.list_envs()
        print("已注册环境:")
        for name, desc in envs.items():
            print(f"- {name}: {desc}")
            if args.details:
                try:
                    config = env_registry.get_env_config(name)
                    print(f"  端口: {config.get('port')}")
                    print(f"  启动命令: {config.get('launch_command')}")
                    print(f"  客户端类: {config.get('client_class')}")
                except Exception as e:
                    print(f"  无法获取详细信息: {str(e)}")

    elif args.command == "verify":
        try:
            config = env_registry.get_env_config(args.env_name)
            client_class = env_registry.get_client_class(args.env_name)
            print(f"环境 {args.env_name} 注册验证成功!")
            print(f"配置信息: {config}")
            print(f"客户端类: {client_class.__name__}")
        except Exception as e:
            print(f"环境 {args.env_name} 注册验证失败: {str(e)}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "reload":
        try:
            env_registry.load_env_configs()
            print(f"成功重新加载环境配置，当前注册环境: {', '.join(env_registry.list_envs().keys())}")
        except Exception as e:
            print(f"重新加载环境配置失败: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()