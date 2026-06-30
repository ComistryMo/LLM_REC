# 环境初始化

## 默认环境

- Conda env：`onereason-rec`
- Python：默认 3.11，可用 `PYTHON_VERSION=3.10` 覆盖。
- CUDA/PyTorch：默认安装 CUDA 12.1 wheel，可用 `CUDA_INDEX` 覆盖。
- Transformers：优先 `5.3.0`；若安装或导入失败，回退到 `4.56.2`。
- ms-swift：`4.3.2`
- LLaMA-Factory：`0.9.5`

## 安装命令

```bash
cd /data/hz/llmrec_competition
bash scripts/setup/check_env.sh
bash scripts/setup/create_conda_env.sh
conda activate onereason-rec
```

## 模型下载

```bash
bash scripts/setup/download_model.sh
ls -lh /data/hz/models/OneReason-0.8B-pretrain-competition
python scripts/sanity/load_model.py --model /data/hz/models/OneReason-0.8B-pretrain-competition
```

如果 HuggingFace 访问较慢：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/setup/download_model.sh
```

## 版本记录命令

```bash
python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
try:
    import swift
    print("swift", getattr(swift, "__version__", "unknown"))
except Exception as exc:
    print("swift import failed", exc)
try:
    import llamafactory
    print("llamafactory", getattr(llamafactory, "__version__", "unknown"))
except Exception as exc:
    print("llamafactory import failed", exc)
PY
```

## 已知兼容性问题

- FAQ、开发机指南和模型卡对 Transformers 版本的表述不完全一致，因此必须以实际模型加载结果为准。
- Swift CLI 参数名可能随版本变化，训练脚本会读取 `swift sft --help` 后动态添加支持的参数。
- LLaMA-Factory 的 `template: default` 只是保守默认；如果官方模型卡或 tokenizer 提供更准确 chat template，应优先使用 tokenizer 自带模板。
- 当前自动化不处理 SSH 密码登录。请使用 SSH key 或平台终端执行命令。
- 远端 CPU 被虚拟成较老的 Core2 级别，缺少 x86-v2/AVX 指令；已将 `numpy` 固定到 `1.26.4`，并使用 `polars-lts-cpu==1.33.1` 代替默认 Polars runtime。

## 远端实测记录

时间：2026-06-30

- 仓库：`/data/hz/llmrec_competition`
- Conda env：`onereason-rec`
- Python：`3.11.15`
- GPU：6 x NVIDIA A100-SXM4-80GB
- Driver / CUDA：NVIDIA Driver `570.169`，系统 CUDA `12.8`
- PyTorch：`2.5.1+cu121`
- torch CUDA：`12.1`
- Transformers：`5.3.0`
- ms-swift：`4.3.2`
- LLaMA-Factory：`0.9.5`
- datasets：`4.0.0`
- accelerate：`1.11.0`
- peft：`0.18.1`
- pandas：`2.3.3`
- polars import version：`1.33.1` via `polars-lts-cpu`
- scikit-learn：`1.9.0`
- Model path：`/data/hz/models/OneReason-0.8B-pretrain-competition`
- Model load test：passed, `Qwen3ForCausalLM` + `Qwen2Tokenizer`, chat template present.

## 环境检查项

`scripts/setup/check_env.sh` 会记录：

```text
nvidia-smi
df -h
free -h
conda --version
python --version
git --version
```

日志写入 `logs/env_check_*.log`，默认不提交 git。
