# Geo3K-VL

Geo3K-VL 是仓库中的几何视觉语言推理环境，对应运行时注册名 `geo3k_vl_test`。每条任务通常包含：

- `problem` 或 `question`：题目文本
- `answer`：标准答案
- `images`：图像 URL 列表

环境支持多轮交互，也允许模型在回答过程中调用内置评分工具后再给出最终答案。

## 配置文件

推荐从 `env/geo3k_vl_test/geo3k_vl_test_env_configs.yaml` 开始：

| 字段 | 说明 |
|------|------|
| `env_name: geo3k_vl_test` | 环境注册名 |
| `dataset` | parquet 数据集路径 |
| `env_params.config_path` | 可选，指向运行时配置文件 |

运行时配置默认从 `env/geo3k_vl_test/geo3k_vl_test_env_runtime.yaml` 读取，主要包括：

| 参数 | 说明 |
|------|------|
| `max_turns` | 最大交互轮数 |
| `max_images` | 注入到 prompt 的图片数上限 |
| `echo_images_on_feedback` | 工具反馈时是否重复回显图片 |

## 运行参考

```bash
python launcher.py \
  --mode local \
  --env-config env/geo3k_vl_test/geo3k_vl_test_env_configs.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_LLM_API_KEY \
  --llm-model YOUR_MODEL_NAME \
  --pool-size 1
```

## 说明

- 公开文档中使用 `Geo3K-VL` 指代该环境，仓库里的注册名仍然是 `geo3k_vl_test`。
- 运行前请把数据集路径替换成你本地可访问的 parquet 文件路径。
