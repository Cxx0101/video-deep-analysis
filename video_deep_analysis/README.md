# 视频深度分析工具

本工具可在浏览器中上传短视频，并生成 MP4 格式的分析结果。支持灰度深度图、人体姿态骨架、面部 478 点网格，以及三者的组合叠加。

## 运行环境

- Windows 10 / Windows 11 64 位
- Python 3.11（64 位）
- 网络连接：首次安装依赖、首次使用深度图时需要
- NVIDIA CUDA 为可选项；检测到可用 GPU 时自动使用，否则使用 CPU

## 从 Git 克隆后运行

先安装 Python 3.11（64 位）。随后在 PowerShell 中执行：

```powershell
git clone https://github.com/Cxx0101/video-deep-analysis.git
cd video-deep-analysis\video_deep_analysis

py -3.11 -m venv "$env:LOCALAPPDATA\VideoDeepAnalysis\venv"
& "$env:LOCALAPPDATA\VideoDeepAnalysis\venv\Scripts\python.exe" -m pip install --upgrade pip
& "$env:LOCALAPPDATA\VideoDeepAnalysis\venv\Scripts\python.exe" -m pip install -r requirements.txt

.\start.bat
```

浏览器会打开 `http://127.0.0.1:8000`。也可以手动访问该地址。

> 虚拟环境安装在 `%LOCALAPPDATA%\VideoDeepAnalysis\venv`，避免 Windows 深层目录的路径长度限制。模型、上传视频和生成结果仍保存在项目目录内。

## 使用方法

1. 上传不超过 500 MB 的 MP4、MOV、AVI、MKV、WebM 或 M4V 视频。
2. 选择分析模式。
3. 点击“开始分析”，等待任务完成。
4. 在页面中预览并下载生成的 MP4。

也可以在“视频链接”输入框中粘贴有权下载的公开视频 URL。工具会使用 yt-dlp 在本机下载单条视频、自动开始所选分析模式，并在任务完成后清理下载临时文件。链接下载默认限制为 500 MB，播放列表不会被下载。

首次处理包含深度图的任务时，程序会自动下载 MiDaS Small 模型至 `models/`。之后可离线使用该模型。

## 分析模式

1. 灰度深度图
2. 人体姿态骨架叠加
3. 深度图 + 人体姿态骨架
4. 面部 478 点网格
5. 深度图 + 人体姿态 + 面部网格全部叠加

## Git 说明

模型、缓存、虚拟环境、上传视频、处理结果、PyInstaller 构建目录和桌面版发布文件均由 `.gitignore` 排除，不会上传到仓库。克隆后请按上方步骤重新安装依赖；模型会在首次需要时自动下载。
