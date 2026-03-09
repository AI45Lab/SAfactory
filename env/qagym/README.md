# 环境部署说明
提供两种环境部署方式, 包括:
- 本地部署
- Docker 部署

## 本地部署
```bash
# 1. 创建 conda 环境
conda create -n qa-gym python=3.10 -y
conda activate qa-gym

# 2. 项目需要固定 `starlette` 版本以兼容 FastAPI：
echo "starlette>=0.40.0,<0.49.0" > constraints.txt

# 3. 依赖下载
pip install -r requirements.txt -c constraints.txt
pip install rayjob_sdk-0.3.11-py3-none-any.whl -c constraints.txt
pip install "protobuf>=6,<7" openrt -c constraints.txt

# 4. 配置环境变量
export OPENAI_API_KEY=你的key
export OPENAI_BASE_URL=你的url
```


## Docker 部署
- 使用 Dockerfile 构建镜像(需配置 OPENAI 环境变量)
```bash
docker build -f ./env/qagym/Dockerfile -t qa-gym:1.3 .
```
- 直接拉取镜像
```bash
docker pull registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/qa-gym:1.3
```