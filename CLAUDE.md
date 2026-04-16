## Local Control Environment

- OS: Windows 11
- Shell: **PowerShell**（用户使用 PowerShell，不是 bash；本地命令应使用 PowerShell 语法）
- Project root: E:\PycharmProjects\pythonProject\sam2_pre_new
- Local ffmpeg: C:\ffmpeg\bin\ffmpeg.exe

## Remote Server

- SSH alias: `lvshaoting-gpu` (configured in C:/Users/glitterrr/.ssh/config with key aris_ed25519)
- Connect: `ssh lvshaoting-gpu` (no password needed, key auth)
- Home directory: /IMBR_Data/Student-home/2025M_LvShaoting
- GPUs: unknown (check with nvidia-smi after login)
- Primary target GPU type: Tesla V100
- Use screen or tmux for long-running jobs
- Assume Linux shell on the remote server

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
- GitHub SSH key 在**远程服务器**上: `~/.ssh/github_ed25519`
  - 本地 Windows 没有此密钥，push 操作需要通过远程服务器中转
  - 或配置本地 SSH agent forwarding / 本地添加同一密钥
- **每次编写新代码前，必须先将当前代码同步到 git 仓库**（git add + git commit）
- Commit message 用英文，简要描述当前状态
- 确保工作区干净后再开始新的代码修改
- 不要在有未提交更改的情况下开始写新功能