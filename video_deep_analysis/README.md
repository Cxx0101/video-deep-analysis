# 视频深度分析工具

运行 `setup.ps1`（只需一次）后，双击 `start.bat`，浏览器将打开 `http://127.0.0.1:8000`。首次处理含深度图的任务时，会下载 MiDaS Small 官方权重到 `models/midas_v21_small_256.pt`，需要联网且可能花数分钟；之后离线可用。

程序会自动使用 NVIDIA CUDA（若 PyTorch 检测到可用），否则使用 CPU。输入视频限定 500 MB，输出始终为 MP4。

独立 venv 位于 `%LOCALAPPDATA%\\VideoDeepAnalysis\\venv`。这是为了绕开 Windows 在深层目录安装 PyTorch 时的 260 字符路径限制；项目代码、上传文件、结果、模型和模型缓存都仍保留在本项目目录中。
