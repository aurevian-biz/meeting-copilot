# Security policy

[English](SECURITY.md) · [中文](SECURITY.zh.md) · [日本語](SECURITY.ja.md)

## Reporting a vulnerability

Open a private report through GitHub's **Security → Report a vulnerability** tab, or e-mail
the maintainers listed in the commit history. Please do not open a public issue for
anything that could expose a deployment. Expect an initial reply within a few days.

## What is deliberately not in this repository

The project is developed on a private machine, and the public tree is generated from it by
a packaging script that strips:

- LLM bearer tokens and any other credential (they live in `/etc/meeting-copilot/env`, mode 0640)
- machine-specific absolute paths (everything goes through `paths.py` + `MC_DATA`)
- the developer's hostname, username, e-mail address, LAN addresses and device serials
- debug screenshots and runtime artefacts

CI (`.github/workflows/ci.yml`) re-runs the same scan on every push. If you find something
that slipped through, treat it as a vulnerability and report it privately.

## Deployment hardening

The two web services have **no authentication** by design — they are meant for a trusted
LAN. Before exposing them beyond `localhost`:

1. Bind to the LAN interface only, and put both ports behind a reverse proxy with TLS and
   authentication (or a VPN) if they must be reachable from other networks.
2. Keep `/etc/meeting-copilot/env` at `0640`, owned by the service user.
3. Note that the caption page accepts `POST /api/control` and `/api/look` from any client
   that can reach the port: anyone on the LAN can start/stop capture or make the device
   look at the screen. That is the intended behaviour for a meeting room, not for a
   public network.
4. The services run as a normal user; they do not need root. Only `install-units.sh` and the
   docker-based LLM setup require sudo.

## Privacy properties

All inference is local. Audio, screen frames and transcripts never leave the device; the only
network traffic is the one documented model download at install time.
