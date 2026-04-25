# Bundle B — SD-Inpaint 相关方案备份（不实施，存档供未来回溯）

**创建时间**：2026-04-25 night
**触发**：用户在 Bundle B sub-session 3 提出 "是否选用 SD 做 inpaint" 的问题，gpt-5.4 xhigh 审稿人给出 "SD 在当前 Stage 14 数学下是 dead weight" 的结论，用户决定备份这条线供未来需要时回溯。
**关联文档**：
- 主决策：`BUNDLE_B_INPAINTER_REVIEW.md`（最终选择 Option B，无 inpainter）
- gpt-5.4 thread: `019dc51a-c71a-7971-bece-116a592de2f5`
- gpt-5.2 thread: `019dc511-5154-7a82-9083-525ffb078442`

## 当前不采用 SD 的核心原因（一句话）

Stage 14 的混合公式 `x_edited = (1 - α·soft_decoy) · x_warped + (α·soft_decoy) · duplicate`，其中 `α_max = 0.35`，导致 SD 生成的"原物体被抹掉的背景"位于 soft_decoy ≈ 0 的区域，整块被丢弃，**不会进入 SAM2 的输入**。在不重叠的常规情况下，B/C/D 三种 compositor 对 SAM2 来说像素完全相同。

## 触发未来重启 SD 方案的条件

**只有同时满足以下两条**，才需要回到这份备份：

1. Bundle B Option B 的 pilot 落入 [+0.02, +0.05) 灰色区间（既未达接受门槛，也未触发永久 cut δ）
2. 实证测量发现 `sum(soft_decoy × true_mask) > 0.1` 在显著比例（>20%）的 bridge 帧上发生 —— 即重叠 regime 不是罕见情况

如果 Bundle B 通过了 pilot（≥ +0.05），不需要回到此文档。如果 pilot 失败到 < +0.02，也不需要 —— 那是 cut δ 的信号，不是切换 inpainter 的信号。

## 备份方案 1：α_max → 1.0 硬替换 + SD inpaint

**思路**：把 `apply_continuation_overlay` 的 `α_max` 从 0.35 抬到 1.0，在 soft_decoy 区域内**完全替换** x_warped 为 duplicate。这样 duplicate 的全部像素（包括非诱饵区域的 SD 幻觉背景）都会进入 x_edited。

**当前实现的影响范围**：
- `memshield/decoy_continuation.py:init_bridge_edit_params(alpha_max=0.35)` 默认值需改
- `scripts/run_vadi_v5.py:VADIv5Config` 加 `oracle_traj_alpha_max: float = 0.35`，pilot 时设 1.0
- `apply_continuation_overlay` 数学不变，只是 α 上限改

**为什么 SD 在这条路径下有意义**：
- α=1 时 duplicate 全部进入 x_edited，SD 生成的背景（在 soft_decoy 边界附近的渐变区域）会真实进入 SAM2 输入
- 但 LaMa 也同样进入 → SD 相对 LaMa 的优势仍是"语义合理性"，不是"是否进入"

**风险**：
- LPIPS 在诱饵区域立即超标（α=1 等于把 35% 强度的扰动放大到 100%），fidelity gate 会全部 revert
- 需要同时降低 lpips_orig_cap 的执行严格度，或者把诱饵区域的 LPIPS 单独豁免 —— 这是一个有审稿风险的设计选择
- 重设计了 Stage 14 的核心机制，需要走 codex 完整 review + 自测 + pilot 三步

**预估效果**：未知。审稿人原话 "amplifier knob, not SD-specific" —— α=1 同时放大 B 和 SD，SD 不一定比 B 更好。

**触发优先级**：Bundle B 失败 + 重叠 regime 显著 → 先试 LaMa 的 α=1 版本，再决定是否升级到 SD。

## 备份方案 2：background-first 合成范式（"重定位攻击"）

**思路**：彻底放弃 "duplicate 是 x_warped 的扰动" 的设计，改成 "duplicate 直接是 x_warped 的替代品"：

```python
# 现在（duplicate 模式）:
duplicate[c_t] = paste(x_clean[c_t], object_at_decoy)            # 原物体 + 诱饵物体都可见
x_edited = (1 - α·soft_decoy) · x_warped + (α·soft_decoy) · duplicate

# 备份方案 2（move 模式）:
background[c_t] = SD_inpaint(x_clean[c_t], true_mask_c)          # 抹掉原物体
duplicate[c_t] = paste(background[c_t], object_at_decoy)         # 只有诱饵物体可见
x_edited = duplicate[c_t]                                         # 直接替换，不再混合
```

**为什么 SD 在这条路径下真正必要**：
- background 的整帧像素都进入 SAM2 输入，包括原物体被抹掉的位置
- 在那个位置，B（保留原物体）和 D（SD 抹掉）的像素**完全不同**，SAM2 的 memory_attention 会编码完全不同的特征
- 攻击语义从"在诱饵处加一个幽灵副本"变成"把物体真的搬到诱饵处"，**信号强度数量级提升**
- LaMa 通常能力不够（处理大区域、复杂背景容易失真）；SD 的 prompt 引导（"clean background scene"）能给出更合理的整体场景

**实现复杂度**：
- 不仅是 compositor 替换，要重写 Stage 14 的 `_run_oracle_trajectory_pgd` 主循环
- LPIPS 计算口径变化（整帧扰动而非局部混合）
- α / warp 等参数的意义改变（不再是混合权重，而是诱饵物体的位置/姿态参数）
- 大约 800-1200 LOC 改动 + 完整 codex review + 重新 pilot

**审稿人评价**：
> "This is the first redesign where inpainting could matter. It is a **new relocated-object attack**, not a Bundle B compositor swap."

**预估效果**：审稿人未给出具体数字，但承认这是 SD 真正能发挥价值的路径。如果 Round 5 的 +0.05 失败，这是值得探索的方向。

**触发优先级**：
- 不在 Round 5 范围内（user override 是"修改δ"而非"重设计 Stage 14 范式"）
- 如果未来开第 7 轮 cut-δ override（用户已用完 5 次，规则上不允许），这是新方向的备选
- 也可能作为**独立论文**的范式（attack 类型从 "perturbation" 升级到 "relocation"）

## 备份方案 3：全帧低 α 覆盖

**思路**：把 soft_decoy 的支持区域扩展到全帧（dilate_px → ∞ 或 feather_sigma → ∞），但 α 保持极低（0.05-0.1），让 SD 的全帧幻觉以低强度渗入整张图。

**预估效果**：审稿人评价 **"Bad trade. You spend LPIPS on diffuse background changes and weaken the localized signal."**

**触发优先级**：不推荐，仅作完整性记录。

## 关于 "method effectiveness > efficiency" 原则

CLAUDE.md 明确：correctness 瓶颈下，方法上限优先于代码复用 / 工程开销。

但本案不是 correctness 瓶颈：
- Stage 14 的梯度流是正确的（`shift_mask_torch` 可微）
- duplicate 内容用 detached integer offset 是设计选择，不是 bug
- B 在数学上和 C/D 等价（不重叠 regime），不是"为了省 LOC 妥协"

因此本案适用的是更前一条：**"Don't add features beyond what the task requires."** SD 在当前数学下不带来攻击效果提升，只增加部署成本，属于 over-engineering。

如果 Stage 14 数学发生变化（备份方案 1/2），SD 的 cost-benefit 重新评估时，"method effectiveness > efficiency" 会重新触发。

## 重启此备份的工作量估算

| 路径 | LOC 改动 | GPU 验证成本 | 触发概率（Bundle B 假设失败时） |
|---|---|---|---|
| 备份 1（α_max=1） | ~50 LOC + 重新调参 | ~4 GPU-小时（pilot） | 30% |
| 备份 2（重定位攻击） | ~1000 LOC + 重新设计 | ~24 GPU-小时（pilot + ablation） | 10% |
| 备份 3（全帧低 α） | ~30 LOC | ~4 GPU-小时 | < 5% |

如果 Bundle B 通过（≥ +0.05），三条路径触发概率都接近 0。

## 重要的元决策

**这次没有走 SD，主要是因为 gpt-5.4 xhigh 审稿人确认在当前 Stage 14 数学下 SD 是 dead weight，不是因为成本顾虑。** 如果未来回到此文档，第一件事是检查 Stage 14 数学是否还是 `α_max ≤ 0.35 + soft_decoy 门控` 的形式 —— 如果是，dead weight 结论仍然成立；如果数学已变（α 调高 / 范式改成 move），需要重新走 codex review + 设计选型，不能直接套用本备份。

## 文件交叉引用

- `BUNDLE_B_INPAINTER_REVIEW.md`：主决策文档，记录选择 Option B 的完整论证
- `HANDOFF_NEXT_SESSION.md`：sub-session 3 起点 handoff（其中"Inpainting model selection"一节本可能引导走 SD 路径，但被 gpt-5.4 审稿驳回）
- `CLAUDE.md` §"No-Proxy Implementation"：不允许悄悄降级，但允许显式分析后等价替代（本案）
- `CLAUDE.md` §"Design Philosophy"：correctness 瓶颈下方法优先，本案不是 correctness 瓶颈
- `memshield/decoy_continuation.py:apply_continuation_overlay`：决定 SD dead weight 结论的核心数学
- `memshield/oracle_trajectory.py:shift_mask_torch`：trajectory 梯度的真正承载者
