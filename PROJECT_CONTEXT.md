# 项目维护上下文

## 维护规则

1. 开始修改前先阅读本文件和 `README.md`。
2. 每完成一个用户可见的新功能、功能调整或运行方式变更，必须同步更新 `README.md`；涉及架构、接口、目录或维护流程时，也更新本文件。
3. 实现后至少完成与改动范围相匹配的实际验证；不要只做静态检查。
4. 不删除用户素材、`data/cookies.txt`、模型或既有结果，除非用户明确要求。测试产生的临时文件应仅删除已确认的测试文件。
5. 代码、模型和缓存优先保留在项目目录内；Windows 中文和空格路径必须继续可用。

## 技术概览

- 后端：Python 3.11 + FastAPI，入口为 `app.py`。
- 前端：单页静态页面 `static/index.html`，不使用前端构建工具。
- 视频处理：OpenCV、MediaPipe、FFmpeg（优先系统 FFmpeg，否则使用 imageio-ffmpeg）。
- 深度模型：MiDaS Small ONNX，存放在 `models/midas_v21_small_256.onnx`，CPU 推理。
- 网络下载：`yt-dlp`。
- 服务地址：仅 `http://127.0.0.1:8000`。

## 主要模块

### 可复用视频来源

视频来源与处理任务分离：

- 本地上传：`POST /api/sources/upload`
- 视频预览：`GET /api/sources/{source_id}/preview`
- 对来源创建工具任务：`POST /api/sources/{source_id}/jobs`

来源在当前服务进程内保存，可被多个工具任务复用。前端先导入和预览来源，再在左侧工作台选择工具。

网络链接仍通过 `POST /api/downloads` 创建下载预览任务；下载完成后会自动注册为可复用来源，来源 ID 与下载任务 ID 相同。

### 工具任务

`POST /api/sources/{source_id}/jobs` 的表单参数：

- `tool=analysis`，并传入 `mode`：`depth`、`pose`、`depth_pose`、`face`、`all`。
- `tool=separate`：导出无声 MP4 和 MP3 音频。

任务进度：`GET /api/jobs/{job_id}`。

- 分析任务完成后返回 `download_url`。
- 分离任务完成后返回 `downloads.video` 与 `downloads.audio`。

保留了旧接口（`/api/jobs`、`/api/downloads/{job_id}/process` 等），以维持原有自动化回归测试兼容性；新页面应优先使用 `/api/sources/` 接口。

### Cookie

- 保存：`POST /api/cookies`
- 状态：`GET /api/cookies/status`
- 清除：`DELETE /api/cookies`
- 路径：`data/cookies.txt`（Netscape cookies.txt 格式，Git 忽略）

下载任务只复制 Cookie 的临时副本并在结束后删除，持久 Cookie 保留供后续下载使用。

## 前端工作区约定

左侧菜单存在两个独立工作区：

- `#analysisNav` ↔ `#analysisPanel`：五种视频智能分析方式。
- `#separateNav` ↔ `#separatePanel`：音视频分离。

切换统一由 `selectWorkspace()` 处理，必须同时更新菜单和工作区的 `active` 类，避免出现多个菜单同时高亮或不匹配的工作区。

“导入视频”和“源视频预览”是两个工作区共用区域。预览标题旁的“已选择”标签应保持紧凑，不要放到卡片最右侧。

## 验证建议

基础检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py
.\.venv\Scripts\python.exe smoke_test.py
```

`smoke_test.py` 会验证五种旧版分析接口。若修改可复用来源、音视频分离或前端流程，还应额外验证：

1. 本地上传 → 预览 → 音视频分离 → 下载 MP4 和 MP3。
2. 使用同一个 `source_id` 再提交一次分析任务，无需再次上传。
3. 左侧两个菜单互斥高亮，且只显示对应工作区。
4. 服务保持在 8000 端口可访问：`GET /api/health`。

## 运行和清理注意事项

- Windows 从项目目录用 `.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000` 启动服务；启动脚本是用户优先使用的入口。
- macOS 使用 Python 3.11；不要将依赖升级为只支持 Python 3.12+ 的版本。
- `data/`、`.venv/`、`models/`、日志和媒体结果不应提交 Git。
- 若重启服务，内存中的来源和任务 ID 会失效；磁盘文件仍按忽略规则保留。不要在没有用户授权时批量清理这些文件。
