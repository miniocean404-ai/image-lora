import gc
import time

import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler


def cleanup_gpu():
    """清理GPU缓存和垃圾回收"""
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU 缓存已清理")


def get_device_config(device_arg):
    """
    根据参数返回设备配置
    返回字典: {"type": "single"|"balanced", "device": str, "device_map": str|dict, "max_memory": dict}
    """
    num_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {num_gpus}")

    if num_gpus == 0:
        print("未检测到GPU，将使用CPU")
        return {"type": "single", "device": "cpu"}

    # 1. 指定了 CPU
    if device_arg == "cpu":
        return {"type": "single", "device": "cpu"}

    # 2. 指定了特定 GPU ID
    if str(device_arg).isdigit():
        gpu_id = int(device_arg)
        if gpu_id < num_gpus:
            print(f"✅ 用户指定使用 GPU {gpu_id}")
            # 设置默认设备
            torch.cuda.set_device(gpu_id)
            return {"type": "single", "device": f"cuda:{gpu_id}"}
        else:
            raise ValueError(f"指定的 GPU {gpu_id} 不存在 (可用: {num_gpus})")

    # 3. 双卡/多卡平衡模式 (balanced)
    if device_arg == "balanced":
        print("✅ 检测到多GPU，启用平衡分配模式 (balanced)")
        # 动态构建 max_memory
        max_memory = {}
        for i in range(num_gpus):
            _, total_bytes = torch.cuda.mem_get_info(i)
            total_gb = int(total_bytes / (1024**3))
            # 直接使用显存大小，diffusers会自动处理预留
            max_memory[i] = f"{total_gb}GB"

        print(f"   显存配置: {max_memory}")
        return {"type": "balanced", "device_map": "balanced", "max_memory": max_memory}

    # 4. 无法识别的参数
    raise ValueError(f"不支持的设备参数: {device_arg}. 请使用 'cpu', 'balanced', 或 GPU ID (如 '0', '1')")


def load_pipeline(model_path, device_config):
    """加载基础模型"""
    print(f"正在从 {model_path} 加载模型...")

    # 根据设备选择数据类型
    if device_config["type"] == "single" and device_config["device"] == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    load_args = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if device_config["type"] == "balanced":
        load_args["device_map"] = device_config["device_map"]
        load_args["max_memory"] = device_config["max_memory"]

    pipe = DiffusionPipeline.from_pretrained(**load_args)

    # 手动将模型移动到目标设备（仅单卡模式需要）
    if device_config["type"] == "single":
        pipe = pipe.to(device_config["device"])
        print(f"✅ 基础模型加载成功，已移至: {device_config['device']}")
    else:
        print("✅ 基础模型加载成功，使用多卡平衡模式")

    return pipe


def load_lora_to_pipeline(pipe, lora_path):
    """加载LoRA权重"""
    try:
        print(f"正在尝试加载LoRA: {lora_path}...")
        pipe.load_lora_weights(f"{lora_path}")
        print(f"✅ LoRA从 {lora_path} 加载成功")
    except Exception as e:
        print(f"⚠️ LoRA加载失败, 继续使用基础模型...: {e}")
    return pipe


def setup_pipeline_config(pipe):
    """设置调度器和内存优化"""
    # 设置调度器
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=7.0, use_dynamic_shifting=True)
    # 启用内存优化
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # 打印当前进程内存分配情况
    print("\n=== 当前进程GPU内存分配 ===")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: 已分配 {allocated:.1f}GB / 已保留 {reserved:.1f}GB")

    return pipe


def run_inference(pipe, prompt, negative_prompt, output_path="generated_image.png"):
    """执行推理并保存图片"""
    print(f"🎨 开始生成图片...\nPrompt: {prompt}")
    start_time = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=50,
        true_cfg_scale=5,
        generator=torch.Generator().manual_seed(346346),
    )

    result.images[0].save(output_path)
    elapsed = time.time() - start_time
    print(f"✅ 图片已保存至 {output_path}！用时: {elapsed:.1f}秒")


def main():
    # 配置设备: "balanced" (多卡), "0" (单卡0), "1" (单卡1), "cpu"
    # 不支持 "auto"，必须明确指定
    DEVICE_CONFIG = "balanced"

    # 1. 初始清理
    cleanup_gpu()

    # 2. 获取硬件配置
    device_config = get_device_config(DEVICE_CONFIG)

    # 3. 加载基础模型
    model_path = "./models/Qwen-Image-2512"
    pipe = load_pipeline(model_path, device_config)

    # 4. 加载 LoRA (按需修改路径)
    lora_path = "./train/qwen2512-makeup-lora/checkpoint-2500/pytorch_lora_weights.safetensors"
    pipe = load_lora_to_pipeline(pipe, lora_path)

    # 5. 配置与优化
    pipe = setup_pipeline_config(pipe)

    # 6. 推理
    prompt = "小红书的调性风格广告，美妆类，美女画着精致妆容，拼接为一张图 1girl, beauty, fashion, portrait, detailed"
    negative_prompt = "blurry, low quality, distorted, deformed"
    run_inference(pipe, prompt, negative_prompt, "./dist/generated_image.png")


if __name__ == "__main__":
    main()
