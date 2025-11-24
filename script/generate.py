import gc
import time

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines import FlowMatchEulerDiscreteScheduler

# 清理 GPU 缓存
torch.cuda.empty_cache()
gc.collect()

print(f"CUDA 版本: {torch.version.cuda}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"可用 GPU 数量: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    property = torch.cuda.get_device_properties(i)
    print(f"GPU {i} 名称: {torch.cuda.get_device_name(i)}")
    print(f"GPU {i} 内存: {property.total_memory / 1024 / 1024 / 1024:.2f} GB")
    print(f"GPU {i} 架构: {property.major}")
    print(f"GPU {i} 小版本: {property.minor}")
    print(f"GPU {i} 最大线程数: {property.max_threads_per_block}")
    print(f"GPU {i} 最大线程数: {property.max_threads_per_multiprocessor}")
    print(f"GPU {i} 最大线程数: {property.max_threads_per_block}")
    print(f"GPU {i} 最大线程数: {property.max_threads_per_block}")

# 根据 GPU 数量选择策略
if torch.cuda.device_count() > 1:
    # 多 GPU 平均分配
    device_map = "balanced"
    max_memory = {0: "12GB", 1: "12GB"}
else:
    # 单 GPU
    device_map = "cuda"
    max_memory = {0: "12GB"}

pipe = DiffusionPipeline.from_pretrained(
    # 可以是本地模型路径, 也可以是在线模型
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print("✅ 模型加载成功")


"""
加载 lora, 可以是本地模型路径, 也可以是在线模型
"""
try:
    pipe.load_lora_weights("flymy-ai/qwen-image-realism-lora")
    print("lora 加载成功")
except Exception as e:
    print(f"lora 加载失败: {e}")

# 设置 Eular 采样器
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, shift=0.7, use_dynamic_shifting=True
)

# 启用内存优化
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

# 检查内存分配
print("\n=== GPU内存使用情况===")
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"GPU {i}:{allocated:.1f}GB /{total:.1f}GB ({allocated / total * 100:.1f}%)")

# 生成图片
prompt = ""
negative_prompt = (
    "ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2),"
)


start_time = time.time()
result = pipe(
    prompt=prompt,
    # 负向提示词
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    # 推理步数
    num_inference_steps=40,
    # 提示词引导强度
    true_cfg_scale=5,
    # 随机种子
    generator=torch.Generator().manual_seed(123456),
)
result.images[0].save("image.png")
print(f"图片已保存！用时：{time.time() - start_time:.1f}秒")
