# 源项目地址

https://github.com/FlyMyAI/flymyai-lora-trainer

---

### 安装 torch:
检测不到 cuda 可用及 GPU 属相大概率是下载的是 cpu 版本或者 cuda 版本与 torch 版本不对、或者源不对，[可上官网查看对应版本](https://pytorch.org/get-started/previous-versions/)

* cuda 12.6: `pip3 install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126`
* 安装 torch cpu 版本: `uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`
  或者
* uv 安装 torch 处理: https://uv.oaix.tech/guides/integration/pytorch/#pytorch_1
  ```toml
  [tool.uv.sources]
  torch = [
    { index = "pytorch-cu126", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
  ]
  torchvision = [ 
    { index = "pytorch-cu126", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
  ]

  [[tool.uv.index]]
  name = "pytorch-cu126"
  url = "https://download.pytorch.org/whl/cu126"
  # 当设置 explicit = true 时，该索引将 不会 被自动用于解析未明确指定来源的依赖包。只有那些在 [tool.uv.sources] 中显式声明使用该索引的包，才会从这里查找
  explicit = true
  ```


# 问题
1. Windows 不支持 deepspeed==0.17.4, Windows 暂未安装



