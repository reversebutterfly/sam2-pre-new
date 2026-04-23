## Local Control Environment

- OS: Windows 11
- Shell: **PowerShell**（用户使用 PowerShell，不是 bash；本地命令应使用 PowerShell 语法）
- Project root: E:\PycharmProjects\pythonProject\sam2_pre_new
- Local ffmpeg: C:\ffmpeg\bin\ffmpeg.exe

## Remote Server

### Connection Policy (IMPORTANT)
- **All remote SSH must be direct** — do NOT use ProxyJump, jump hosts, or SOCKS proxies for the SSH channel itself. The public SSH ports (6100 / 4100 / 6000) expose the target boxes directly.
- Symptom when direct connection is bypassed / misrouted: `kex_exchange_identification: Connection closed by remote host` repeating in bursts. Observed 2026-04-21 during Pro 6000 work — resolved by reconnecting directly after a short wait.
- If SSH drops with `kex_exchange_identification`, wait ~1-2 min and retry directly. Do NOT add `-o ProxyCommand=...` or tunnel through V100 / any other host.
- HTTP(S) proxies (for GitHub/pip on V100) are fine and orthogonal to this rule — they apply inside the session, not to the SSH transport.

### V100 Server (primary, 2025M_LvShaoting)
- SSH alias: `lvshaoting-gpu` (port 6100, 1-GPU entry), `lvshaoting-gpu-4x` (port 4100, 4-GPU entry)
- HostName: 183.175.157.242 — **direct connection only**
- User: `2025M_LvShaoting`
- Key: C:/Users/glitterrr/.ssh/aris_ed25519
- Connect: `ssh lvshaoting-gpu` (no password needed, key auth)
- Home directory: /IMBR_Data/Student-home/2025M_LvShaoting
- GPUs: Tesla V100
- Use screen or tmux for long-running jobs
- Assume Linux shell on the remote server

### Pro 6000 Server (separate host, different account)
- SSH alias: `lvshaoting-pro6000`
- HostName: 183.175.157.243, Port 6000 — **direct connection only, different box from V100 (not same IP)**
- Web UI (file manager): <http://183.175.157.243:6000>
- Internal host: amax-Rack-Server (10.10.10.201)
- User: `2025Lv_Zhaoting`
- Home: /datanas01/nas01/Student-home/2025Lv_Zhaoting (NAS-mounted, 21 TB on volume1/nas01)
- Key: C:/Users/glitterrr/.ssh/aris_ed25519 (same key as V100 account)
- Connect: `ssh lvshaoting-pro6000`
- CPU: AMD EPYC 9654 96-core
- OS: Ubuntu 24.04.4 LTS
- GPUs: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each (verified 2026-04-21)
- Conda: `~/miniconda3` with env `memshield` (torch 2.8.0 cu128, required for Blackwell sm_120)
- LD hook: `$CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh` prepends pip-installed nvidia libs (avoids system CUDA 12.4 / 13.0 cuBLASLt conflict). Always `conda activate memshield` before running SAM2 — do not rely on system CUDA.

#### Pro 6000 first-login initialization (already done for 2025Lv_Zhaoting)
- After first password change, must run `sh /datanas01/nas01/Student-home/init_user.sh` once. Do not re-run on existing account.
- Default `passwd` command changes the account password.
- `apt` / `apt-get` are allowed with `sudo` (prompts for the account's own password). Common `sudo apt-get install pkg` works.
- Non-admin user has NO other sudo privileges — contact admin for anything else.

#### Pro 6000 canonical paths (from lab manual)
| purpose | path |
|---|---|
| user home / code / data | `/datanas01/nas01/Student-home/2025Lv_Zhaoting` |
| user conda envs (auto-placed here) | `/LAI_Data/Anaconda_envs/2025Lv_Zhaoting/` |
| lab public datasets (**read-only, use this for DAVIS etc. if present**) | `/datanas02/public_data` |
| lab private datasets (read-only) | `/datanas02/private_data` |
| lab shared assets | `/datanas02` |
| wifi re-auth scripts | `/datanas01/LAB_Data/Share_code/wifi/` |
| system anaconda (do not modify) | `/usr/local/anaconda3` |
| multi-CUDA switcher | `/home/amax/switch-cuda.sh` |
| user init script | `/etc/skel/user_init_setup.sh` |

**Check `/datanas02/public_data` first before downloading any standard dataset** — may already contain DAVIS/YouTube-VOS/etc. with read-only access (much faster than NAS-mounted home dir for I/O).

#### Pro 6000 CUDA version switching
- Installed: CUDA 13.0 (default), CUDA 12.4 also available.
- Switch via: `source /home/amax/switch-cuda.sh 12.4` then verify with `nvcc -V`.
- **Our memshield env uses pip-installed CUDA 12.8 libs via the LD hook** — does NOT depend on system CUDA. Do not call switch-cuda.sh in our workflow.

#### Pro 6000 wifi re-auth (after reboot only)
Campus network requires periodic auth. Normally not our concern, but if network drops:
```bash
screen -S wifi
cd /datanas01/LAB_Data/Share_code/wifi
bash wifi.sh       # or: bash wifi-run
# Ctrl-A+D to detach
```

### Lab Resource Policy (IMPORTANT — applies to both V100 and Pro 6000)
- **1 GPU per user by default**; up to 2 GPU per job with admin approval; up to 3 concurrent jobs max.
- **Large runs should execute at night / weekends.** Avoid peak daytime slots.
- **All long-running jobs must background** via `nohup` / `screen` / `tmux`. Our convention: `nohup bash -c '...' > log 2>&1 < /dev/null &`
- **No interactive GPU sessions.** Do not leave PyCharm/VSCode connected with a GPU-holding kernel — treated as a violation.
- **Do NOT run compute on the login node.** All work via screen + background.
- **Re-check `nvidia-smi` immediately before launching** — other users' processes can spawn within seconds of a free slot; double-check to avoid bumping someone.
- **Admin contact required for**: new account, quota issue, sudo beyond apt, unusual crash. Do NOT attempt to reboot or modify system config.

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
## Method Design Constraints (project-specific, 2026-04-23)

用户在 research-refine round-ception 后明确定下的攻击方法方向：

- **必须使用 "插帧 + 修改原有帧" 策略**（decoy-style insertion + δ perturbation on originals）。**不使用 pure suppression**。
- **插入位置必须是"最优插入点"** — 利用对原视频的完整访问，分析 SAM2 在原视频上的 **vulnerability windows**（置信度波动、运动不连续、scene change 等），在那些位置插帧效果最强。不再用 FIFO canonical schedule（{6,12,14}）— 那是 v2/v4 基于"defeat FIFO self-healing"叙事选的，但 B2 已证伪该叙事。
- **Insert 的作用不是毒化 memory bank**（B2 证明 bank 架构上不关键），而是 **让 SAM2 在处理 insert 那一帧时，其当前帧 Hiera 特征通路受扰动**。Insert 本质是"current-frame pathway attack delivered as a fake frame"。
- **Eval 设定**：数据集保护 threat model — eval 在处理过的整个视频（包含 insert 帧和 δ 帧），不是 clean-suffix。
- **保真约束**：f0（mask prompt 帧）两-tier 特保真（ε=2/255, SSIM≥0.98）；其它帧 ε=4/255 + 每帧 LPIPS ≤ F_lpips（floor-grounded，~0.20）。
- **监督**：GT-free，使用 clean-SAM2 pseudo-labels + 置信加权。
- **任何 refinement / experiment 都必须遵守**：保留 insert 为核心机制，不能被 reviewer / Codex 建议"删掉 insert 简化为 pure δ"。如果 reviewer 这么建议，pushback 并引用此约束。

