## Local Control Environment

- OS: Windows 11
- Shell: **PowerShell**（用户使用 PowerShell，不是 bash；本地命令应使用 PowerShell 语法）
- Project root: E:\PycharmProjects\pythonProject\sam2_pre_new
- Local ffmpeg: C:\ffmpeg\bin\ffmpeg.exe

## Remote Server

### V100 Server (primary, 2025M_LvShaoting)
- SSH alias: `lvshaoting-gpu` (port 6100, 1-GPU entry), `lvshaoting-gpu-4x` (port 4100, 4-GPU entry)
- HostName: 183.175.157.242
- User: `2025M_LvShaoting`
- Key: C:/Users/glitterrr/.ssh/aris_ed25519
- Connect: `ssh lvshaoting-gpu` (no password needed, key auth)
- Home directory: /IMBR_Data/Student-home/2025M_LvShaoting
- GPUs: Tesla V100
- Use screen or tmux for long-running jobs
- Assume Linux shell on the remote server

### Pro 6000 Server (separate host, different account)
- SSH alias: `lvshaoting-pro6000`
- HostName: 183.175.157.243, Port 6000 (different box from V100, not same IP)
- Internal host: amax-Rack-Server (10.10.10.201)
- User: `2025Lv_Zhaoting`
- Home: /datanas01/nas01/Student-home/2025Lv_Zhaoting (NAS-mounted)
- Key: C:/Users/glitterrr/.ssh/aris_ed25519 (same key as V100 account)
- Connect: `ssh lvshaoting-pro6000`
- GPUs: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (verified 2026-04-21)

## Paths

- DAVIS_ROOT: /path/to/DAVIS
- CHECKPOINT_ROOT: /path/to/checkpoints
- Preferred checkpoint: /path/to/checkpoints/sam2.1_hiera_tiny.pt
- FFMPEG_PATH: ffmpeg

## Resource Policy

- Do not assume A100 or H100
- First run a 1-GPU profiling pilot
- Measure peak VRAM, step time, dataloader speed, ffmpeg overhead, and estimated wall-clock
- Only recommend 2 or 4 V100s if profiling shows clear speedup or parallel benefit
- Prefer fewer GPUs if run dependencies are sequential
- Distinguish between:
  - data parallelism
  - multi-experiment parallelism
- Recommend max_parallel_runs explicitly after profiling

## Git & GitHub

- Remote: `git@github.com:reversebutterfly/sam2-pre-new.git`
- GitHub SSH key (shared across V100 and Pro 6000): `~/.ssh/github_ed25519`
  - Fingerprint: `SHA256:t+yfeeNvmUAFjLE/HgbpWqFZK8A5qD+6SPwiKGvMJB8`
  - Same key file on both remotes (copied V100 → Pro 6000 on 2026-04-21 via base64 pipe; never touched local disk)
  - V100 reaches github.com via mihomo `127.0.0.1:7890`; Pro 6000 has direct access (no proxy needed)
  - If this key is ever rotated, replace on BOTH machines and update the single GitHub entry
  - 本地 Windows 没有此密钥，push 操作需要通过远程服务器中转
  - 或配置本地 SSH agent forwarding / 本地添加同一密钥
- **每次编写新代码前，必须先将当前代码同步到 git 仓库**（git add + git commit）
- Commit message 用英文，简要描述当前状态
- 确保工作区干净后再开始新的代码修改
- 不要在有未提交更改的情况下开始写新功能