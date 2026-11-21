# 安全说明

[English](SECURITY.md) · [**中文**](SECURITY.zh.md) · [日本語](SECURITY.ja.md)

## 报告漏洞

请通过 GitHub 的 **Security → Report a vulnerability** 私下提交，或直接联系提交历史里的维护者。
可能暴露某个部署的问题请不要开公开 issue。一般几天内会有初步回复。

## 仓库里刻意不含哪些东西

项目在私有机器上开发，公开树是由打包脚本生成的，会剥掉：

- LLM 的 Bearer token 及其他任何凭据（真值在 `/etc/meeting-copilot/env`，权限 0640）
- 与本机绑定的绝对路径（目录一律走 `paths.py` + `MC_DATA`）
- 开发机的主机名、用户名、邮箱、内网地址、设备序列号
- 调试截图与运行时产物

CI（`.github/workflows/ci.yml`）每次 push 都会重跑同样的扫描。若你发现漏网的具体信息，
请按漏洞私下报告。

## 部署加固

两个 Web 服务**按设计没有鉴权**——它们面向可信内网。若要暴露到 `localhost` 之外：

1. 只绑定内网网卡；如果需要跨网访问，把两个端口放到带 TLS 与鉴权的反向代理（或 VPN）后面
2. `/etc/meeting-copilot/env` 保持 `0640`，属主为服务运行用户
3. 注意字幕页的 `POST /api/control` 与 `/api/look` 对任何能连到该端口的客户端开放：
   同一内网的人可以启动/停止采集、让设备去看屏幕。这在会议室里是预期行为，在公网上不是
4. 服务以普通用户运行，不需要 root；只有 `install-units.sh` 和基于 docker 的 LLM 部署需要 sudo

## 隐私特性

全部推理都在本地。音频、屏幕帧、译文都不出设备；唯一的网络流量是安装时那一次有记录的模型下载。
