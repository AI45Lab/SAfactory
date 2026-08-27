# S3 + LanceDB 存储

Safactory 可以通过 `wt-data-platform-sdk` 将轨迹和环境数据持久化到以 S3 为对象存储、LanceDB 为数据引擎的存储平台。SQLite 仍是默认的本地存储策略；云存储相关依赖单独维护在 `requirements-cloud.txt` 中。

安装可选依赖：

```bash
pip install -r requirements-cloud.txt
```

创建本地 `.env` 文件并填写数据平台连接参数（请勿提交包含凭证的文件）：

```bash
# 可选值：production 或 test
WT_SDK_PROFILE=test
WT_SDK_DB_URI=s3://YOUR_DATA_DATABASE
WT_SDK_ENV_CONFIG_DB_URI=s3://YOUR_ENV_CONFIG_DATABASE
WT_SDK_S3_ENDPOINT=https://YOUR_S3_ENDPOINT
WT_SDK_S3_ALLOW_HTTP=true
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_EC2_METADATA_DISABLED=true
```

启动 Safactory 前，将配置加载到进程环境：

```bash
set -a
source .env
set +a
```

然后将 gateway 的 `storage_type` 设置为 `cloud`，并使用 `--storage-type cloud` 启动 Safactory。SDK profile 只负责选择 Safactory 的 Cloud landing 目标；Safactory 不访问 serving 表。完整配置和表说明请参阅 [AI45Lab/wt-data-platform-sdk](https://github.com/AI45Lab/wt-data-platform-sdk)。
