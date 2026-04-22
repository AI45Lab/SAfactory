# DeepEyes

DeepEyes 是 AIEvoBox 中的多轮视觉工具使用环境。每条任务通常包含：

- `prompt`：初始消息
- `images`：图像路径、URL 或 data URL
- `reward_model.ground_truth`：参考答案
- `extra_info.question`：原始问题文本

环境会根据模型输出解析工具调用，支持裁剪等视觉操作，并在结束时结合 judge 配置计算奖励。

## 配置文件

建议从 `env/deepeyes/deepeyes_env.yaml` 开始。常见字段：

| 字段 | 说明 |
|------|------|
| `env_name: deepeyes_env` | 环境注册名 |
| `dataset` | parquet 数据集路径 |
| `dataset_load_mode: parquet_row_ref` | 按 parquet 行引用延迟加载 |
| `env_params.config_path` | 运行时配置文件，默认指向 `deepeyes_env_runtime.yaml` |

可选的其他示例配置：

- `env/deepeyes/deepeyes_visual_08_env.yaml`
- `env/deepeyes/deepeyes_visual_012_env.yaml`
- `env/deepeyes/deepeyes_thinklite_env.yaml`
- `env/deepeyes/deepeyes_mixed_v1v8_thinklite_env.yaml`

## 运行时配置

`env/deepeyes/deepeyes_env_runtime.yaml` 主要控制：

| 参数 | 说明 |
|------|------|
| `max_turns` | 单个样本允许的最大轮数 |
| `allow_rotate_tool` | 是否允许旋转类工具 |
| `http_timeout_s` | 图像 / 网络请求超时 |
| `judge_base_url` / `judge_model` / `judge_api_key` | 判分模型配置 |
| `cleanup_temp_dir_on_close` | 结束时是否删除临时裁剪目录 |
| `crop_root` | 临时裁剪结果保存目录 |

## 运行参考

```bash
python launcher.py \
  --mode local \
  --env-config env/deepeyes/deepeyes_env.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_LLM_API_KEY \
  --llm-model YOUR_MODEL_NAME \
  --pool-size 1
```

## 说明

- 运行前请把数据集路径、judge 模型地址和输出目录替换为本地可用配置。
- 如果不需要额外 judge，可以将运行时配置中的 `judge_model` 留空，或改为你自己的评分端点。
