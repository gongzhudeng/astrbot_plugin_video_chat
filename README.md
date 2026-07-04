# 灵犀 · 视频链接理解

AstrBot 插件，让 AI 自动理解视频链接内容，支持抖音（视频/图文）、B 站（含字幕提取）。

## 支持平台

| 平台 | 类型 | 说明 |
|------|------|------|
| 抖音 | 视频 | `v.douyin.com` 短链 / `douyin.com/video/` 完整链接 |
| 抖音 | 图文帖 | `/share/note/` 链接，提取所有图片发给视觉模型 |
| B 站 | 视频 | BV号、av号、`b23.tv` 短链、完整链接，支持字幕提取 |

## 功能特性

- **无需命令**：基于 LLM Tool 模式运行，AI 自动判断何时解析视频，无需手动输入指令
- **抖音图文支持**：自动识别图文帖，提取全部图片交给视觉模型理解
- **B 站字幕优先**：配置 SESSDATA 后，B 站视频优先提取字幕，比抽帧更准确、更省资源
- **智能抽帧**：按秒率抽帧，支持只分析前 N 秒 + 帧数上限，适配本地模型

## 安装

在 AstrBot 管理后台的**插件市场**搜索 `astrbot_plugin_video_chat` 安装，或手动克隆：

```bash
cd data/plugins
git clone https://github.com/gongzhudeng/astrbot_plugin_video_chat.git
```

重启 AstrBot 后即可生效。

### 依赖

插件会自动安装所需 Python 包（`yt-dlp`、`bilibili-api-python`、`Pillow` 等）。

抽帧功能需要系统安装 `ffmpeg`：

- **Windows**：从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载，加入系统 PATH
- **Linux/macOS**：`apt install ffmpeg` / `brew install ffmpeg`

## 配置说明

安装后在 AstrBot 后台 → 插件 → 灵犀 · 视频链接理解 中配置。

### 基础配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 视频转述模型供应商 | 下拉选择已配置的视觉模型供应商，留空使用当前会话模型 | 空（当前模型） |
| 转述提示词 | 发给视觉模型的提示词，可自定义字数和风格 | 内置简洁风格 |
| 允许临时下载视频 | 开启后支持抽帧路径；关闭时只走 video_url 直传 | 关闭 |

### 抽帧配置

仅在模型不支持 `video_url`（需开启"允许临时下载"）时生效。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 抽帧速率（帧/秒） | 每秒抽几帧 | 1.0 |
| 最大帧数上限 | 超出时自动降频，防止长视频超载 | 30 |
| 只分析前 N 秒 | 0 表示分析全程，默认前 2 分钟 | 120 |

### B 站字幕配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| B站 SESSDATA Cookie | 填写后 B 站视频优先提取字幕 | 空（不启用） |
| 字幕同时抽帧 | 有字幕时是否同时抽帧一起发给模型 | 关闭 |

#### 获取 SESSDATA

1. 在**电脑浏览器**登录 [bilibili.com](https://www.bilibili.com)
2. 按 `F12` 打开开发者工具
3. 切换到「应用」(Application) → 存储 → Cookie → `https://www.bilibili.com`
4. 找到 `SESSDATA` 一行，复制「值」列的完整内容
5. 粘贴到插件配置中保存

> SESSDATA 有效期约数月，失效后需重新获取。

### 其他配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 最大视频大小 (MB) | 临时下载时的大小上限 | 200 |
| yt-dlp Cookies 文件路径 | Netscape 格式的 cookies.txt 绝对路径，用于抖音需登录时的抓取 | 空 |

## 使用方法

配置完成后，直接发送包含链接的消息即可：

```
你帮我看看这个 https://www.bilibili.com/video/BV1xxxx
这个视频说了什么 https://v.douyin.com/xxxx/
```

AI 会自动识别链接并调用视频理解工具，无需任何特殊指令。

## 工作流程

```
抖音图文帖（/share/note/）
  → 原生 HTTP 提取图片列表
    → 下载图片 → 发给视觉模型

抖音视频
  → 原生 HTTP 提取直链
    ├─ 成功 → 直传 URL 或下载抽帧
    └─ 失败 → yt-dlp 兜底

B 站链接 + SESSDATA 非空
  → 尝试获取字幕
    ├─ 有字幕（字幕同时抽帧=关）→ 字幕文本直传 AI
    ├─ 有字幕（字幕同时抽帧=开）→ 字幕 + 抽帧一起发给 AI
    └─ 无字幕 → 走下方普通流程

B 站链接（无 SESSDATA）
  → bilibili-api-python 原生下载
    ├─ 成功 → ffmpeg 抽帧
    └─ 失败 → yt-dlp 兜底
```

## 常见问题

**Q：抖音视频提示需要 Cookies？**  
A：用浏览器扩展「Get cookies.txt LOCALLY」在登录状态下导出 douyin.com 的 cookies.txt，在配置中填写文件绝对路径。

**Q：发了链接但 AI 没有解析视频？**  
A：确认当前使用的主模型支持 Function Calling（Tool Use）。部分小模型不具备工具调用能力。

**Q：B 站视频无法获取字幕？**  
A：并非所有 B 站视频都有字幕。UP 主未上传、且 B 站未生成 AI 字幕的视频无法提取，此时插件自动降级到抽帧模式。

**Q：抽帧路径报错找不到 ffmpeg？**  
A：安装 ffmpeg 并加入系统 PATH。

## 开源协议

MIT