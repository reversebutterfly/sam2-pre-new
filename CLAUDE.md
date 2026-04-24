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

## Design Philosophy (2026-04-23)

**仅在正确性瓶颈场景下**：方法有效性 > 代码复用。

- **"正确性瓶颈"** = 指当前方法/实验设计在**科学 / 正确性**层面有问题（例如：eval 不对应论文 claim、attack 机制的理论假设已被证伪、数据监督 target 有歧义、loss function 与希望优化的目标不一致等）。
- 正确性瓶颈时：**不要默认选择"最小改动、最大复用现有代码"的方案**。先问："如果从零设计，什么方案最直接解决这个问题？" 再评估需要付出多大改动成本。
- **敢于提出与前面失败方案本质不同的方法**。如果 v1/v2/v3 都失败了，v4 不应该只是 v3 的 "加 epsilon" 微调 —— 应该考虑：是不是某个底层假设要换？是不是整个数据流要重构？是不是该换个攻击面？
- 前面失败方案的代码"沉没成本"不是保留它们的理由。sunk cost is not a technical argument.

**不适用**（就打补丁/最小改动即可）：
- **OOM / 显存瓶颈** — 这是工程问题，gradient checkpointing / early-stop / 截断等补丁都是合理工具，不需要为此推翻方法。
- **速度 / 吞吐瓶颈** — 同上，工程优化不是方法重设计的触发条件。
- **代码维护性 / 技术债** — 跟方法正确性无关。

这条写进 CLAUDE.md 是因为 VADI 项目里有过 v1→v2→v3→v4 的方法迭代路径，每次重设计都是因为**正确性/有效性**问题（v2 bank-poisoning 被 B2 证伪、v3 pure-δ 用户拒绝）。这种场景下不要因为"改动小"就保留失败方案的残骸。

## Paper Direction Constraint (2026-04-24，**硬约束**)

**永远保持 decoy 方向的正向方法论文。不写 audit / 负面 / falsification-only 论文。**

触发背景：2026-04-24 decisive round 结果（10 clips × 4 configs）把 VADI 原 3 条主张全部证伪 —— placement top 输给 random（2/10 wins）、δ 是净伤害（joint < insert-only by 0.156 pp mean）、insert-only 是最强 config（mean 0.537 J-drop）。Codex 两轮审查都推荐 audit paper pivot。用户在 14:45 明确拒绝此方向，定下以下约束：

- **方向锁定**：paper 方向必须是 **decoy-based frame insertion attack on SAM2** 的正向方法论文。即便实证结果挑战某个具体组件（placement 启发式、δ 对原始帧的扰动、joint optimization 公式），**也要在 decoy 框架内精炼方法**，而不是转向 "SAM2 attack-surface audit" 或 "negative results" 论文。
- **允许的调整**：在 decoy 大方向内可以：
  - 调整优化目标（例如按 codex 建议把 δ 和 ν 解耦：insert-only 正是一种合法 decoy 攻击）
  - 调整 placement 策略（random vs vulnerability vs 新 heuristic）
  - 调整 loss 函数（contrastive decoy margin 的不同实现）
  - 调整 threat model 细节（ε/LPIPS 预算等）
- **禁止的调整**：
  - **禁止**把项目重新叙述为 "analysis / audit / falsification" 论文
  - **禁止**把负面结果（placement 反相关、δ 负贡献）作为 paper 主卖点
  - **禁止**提议"放弃 decoy 转做纯 suppression"或"放弃 insert 转做纯 δ"（老约束 §Method Design Constraints）
- **如何调和 decisive 的证伪结果**：
  - 负面信号（placement inversion, δ 反贡献）作为**方法内部的校准选择理由**，不是"整个项目走错了"的证据。示例："基于 10-clip paired ablation，我们采用 vulnerability-aware top-K 的**松弛变体**（例如 top-K 混合 random-K，或避开前 N 帧后再从 top-K 选）"。
  - 最强 config（K3_insert_only_top，mean 0.537）可以作为**论文主 claim 的基础**：它是一个 decoy attack（插入合成 decoy 帧诱导 SAM2 偏离真实 mask），只是不在原始帧上加 δ。这完全兼容 decoy 方向。
- **这条约束也适用于任何后续 research-review / codex pushback**：reviewer 若建议 pivot 到 audit paper，pushback 并引用此约束。可以承认"reviewer 的观点在学术上自洽"，但说明项目约束优先。

## Code Review Protocol (2026-04-24，**硬规则**)

**编写/修改代码后，部署实验之前必须先让 Codex 审查一次代码正确性**。不论改动大小，只要涉及 method 行为（不只是打字错误 / 注释修复），都必须过 codex reviewer 审一轮，列明潜在 bug / 边界条件 / 正确性风险，收到 verdict 后再决定是否 commit+push+run。

触发背景：本项目有过多次"自测通过就直接上 GPU 跑"的经历，浪费了 GPU 小时。典型案例：
- Phase 1 ablation 跑出 J_attacked ≈ 0.96 全部塌陷 —— 事后定位是 v5 driver 的 `delta_support_mode="off"` 把 loss query window 一并清空，ν 失去梯度信号。如果 push 前走一次 codex review，`loss_query window` 与 `delta support window` 耦合问题**是能被查出来的**；几十分钟 GPU 被浪费。
- v5 first-draft 也靠 codex review 抓出了 4 个问题（insert-frame mask supervision、γ 分 insert/post、ν clamp 缺失、W_clean_override 范围验证）。那次的流程是对的。

**Mandatory 流程**（针对 method/experiment 代码）：

1. 本地自测（`python -m <module>` / 脚本 `--dry-run`）必须先 PASS。
2. **然后**用 `mcp__codex__codex` 或 `mcp__codex__codex-reply`（续线 thread）让 Codex 过一遍：
   - 列出核心改动（diff 的关键部分 + 语义说明）
   - 具体让 Codex 找：correctness bugs / 边界条件 / 会不会把某个 metric 搞坏 / 上下文中是否有 silent failure
3. 收到 Codex 反馈：
   - **有 High-severity 问题** → 必须修掉后重新 review
   - **有 Medium/Low** → 权衡是否阻塞
   - **全 clear** → 继续 commit+push+run
4. 完成实验后的结果解读时，**结果反常 / 指标塌陷，第一反应不是"方法问题"**，而是先回头自查 + 让 Codex review 一遍 code path 是否有实现 bug（**v5 Phase 1 塌陷就是这种情况，误判为"方法问题"**）。

**不需要 Codex review 的情况**（打补丁级别）：
- 文档 / 注释 / typo 修正
- 打印日志格式微调
- 非 method 行为的纯重命名
- Dependency 升级（除非牵扯 API 行为变化）
- 打包 / launch 脚本 / CLI 参数默认值调整（只要不改 algorithm）
- OOM 补丁（gradient checkpointing / 截断）—— 工程补丁不是方法改动

**违反本规则**（已知"工程优化 + 算法调整"混在一起） → 视为 correctness-risk 改动，**必须** review。

被 Codex 审出的 bug，即使修掉了，也要在 commit message 或 review doc 中记录一笔（"codex R1 Fix: xxx"），便于后续回溯。

此约束下的默认新方向（待用户进一步确认）：
**VADI-lite** = "vulnerability-informed decoy-frame insertion"，freeze δ=0（或 δ 只加在 prompt 帧周围做防御性 budget），主攻 ν on inserted decoy content，placement 用 top-K 的鲁棒性变体（例如 top-3 丢弃命中前 5 帧的，或与 random 混合）。
