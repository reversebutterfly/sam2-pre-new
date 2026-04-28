# MemoryShield 实现思路：Suppression 与 Decoy

> 记录两个 regime 的实现想法，不贴代码。重点是"为什么这么做"和"做的时候踩过的坑"。

---

## 0. 共享框架（两 regime 共用）

两个 regime 共享**同一个 PGD 优化器**，只有 loss 不同。这是 fair comparison 的前提。

### 攻击面与窗口

- 原视频 $T$ 帧，第 0 帧有 DAVIS 标注作为 SAM2 prompt。
- 攻击者能做两件事：
  - **扰动前缀**：对 $f_0..f_{L-1}$（$L\!=\!15$）加 $\ell_\infty$ 扰动。预算 $\varepsilon_0\!=\!2/255$（conditioning frame 更谨慎），其余 $\varepsilon\!=\!4/255$。
  - **插帧**：在前缀内插入 $K\!=\!3$ 帧，预算 $\varepsilon_{\text{ins}}\!=\!8/255$。插入位置选在 **FIFO 共振点**（SAM2 memory bank 大小为 7，一般选在 $f_3$、$f_7$、$f_{11}$ 后，恰好在旧 memory 要被挤出的时机把对抗 memory 塞进去）。
- 评估窗口 $f_L..f_{T-1}$ **与攻击窗口严格不相交**。任何掉分都必须经过 SAM2 的 memory bank 才能影响评估帧 —— 这是 "memory poisoning" 叙事能成立的关键。

### 3-stage PGD

| Stage | 优化对象 | 目的 |
|---|---|---|
| 1. Perturb-only warmup (~15 步) | 只优化 $\delta_t$（$t<L$），insert 基帧冻结 | 让扰动先按 loss 塑形，避免 insert 和扰动互相拖累 |
| 2. Insert-only warmup (~10 步) | 冻结 $\delta_t$，只优化 insert 残差 | insert 的起点是合成帧，要先把它"拉"到攻击方向 |
| 3. Joint (~25 步) | 所有对抗参数联合 | 加上 read-path 监督（让 poison 在 rollout 时还能持续欺骗）|

每 10 步做一次 fake-quant（round 到 8-bit），避免 PGD 收敛到浮点空间的小数点扰动里（那在实际视频编码后会被丢掉）。

---

## 1. Suppression：让对象"消失"

### 目标

让 SAM2 在评估窗口里认为"场景里根本没这个对象"。量化目标：object_score logit 跌到负值 + 在 GT 区域上的 mask logit 都变成深负。

### Loss 思路

对每一张攻击前缀内的原始帧：

1. **在 GT 区域内压低 mask logit**：用 `softplus(ℓ · g)` —— masked softplus，把 GT 掩码内的正 logit 都往下拖。归一化除以 mask 面积，保证小目标和大目标权重一致。
2. **把 object_score 钉在一个带状区间**：
   - 不是一味往负无穷怼。太负的 score 会让 SAM2 对整帧做"完全不可见"判断，然后它自己会做一些兜底处理（例如 hole-filling），反而把攻击给"修复"了。
   - 所以 loss 是 `ReLU(s - z_hi) + α·ReLU(z_lo - s)` ，只在 $s$ 超出 $[z_{\text{lo}}, z_{\text{hi}}]$ 时罚，让 $s$ 停在"刚刚好让对象消失"的临界。
3. **conditioning 帧 $f_0$ 权重减半**：$f_0$ 是 SAM2 的特权 memory（单独一个槽位，不走 FIFO），对它做重手扰动容易被发现；而且只要后续几帧把 FIFO 填满，$f_0$ 的贡献会被稀释，所以不值得在 $f_0$ 上花预算。

### 为什么 Suppression "强"

- Loss 目标简单、方向明确：logit 只需要往一个方向推，不需要"安排"任何替代物。
- 不需要插帧也能达到 0.7+ drop，插帧只是锦上添花。
- 在 10/10 DAVIS 测试 clip 上都能把 $\mathcal{J}\&\mathcal{F}$ 打到 0（均值 drop 0.744）。

### 缺点

- 故事性弱：reviewer 会问"这只是降了 object score，算不上真正的对抗攻击"。
- 容易被 defender 检测：输出突然没对象是很异常的信号。
- 不能体现 SAM2 memory 的复杂性，纯输出层就能解决。

---

## 2. Decoy：让对象"移位"

### 目标

让 SAM2 继续报告"对象存在"（$s\!>\!0$），但把 mask 预测到**错误的空间位置** —— 一个和 GT 有位移 $(\Delta y, \Delta x)$ 的 decoy 区域。这是真正的 "memory-mediated mislocalization"，比 Suppression 更难，也更有 story。

### 空间区域分解（核心设计）

给定 GT 掩码 $g_t$ 和 decoy 区域 $d_t = \text{shift}(g_t, \Delta y, \Delta x)$，把帧上的每个像素分到 4 类：

| 区域 | 定义 | loss 方向 |
|---|---|---|
| **core** $c_t$ | GT 的 erosion（内圈）| logit 推**下**（弱压，已是 object 的"深处"，自然难激活 decoy）|
| **ring** $r_t$ | GT 减 core 减 decoy | logit 推**下**（强压，防止 optimizer 把 decoy "涂抹"到 GT 边界上来骗 loss）|
| **decoy** $d_t$ | shift 后的目标区域 | logit 推**上**（主要目标）|
| **bridge** $b_t$ | 连接 core 和 decoy 的窄带 | logit 轻推**上**（让 mask 视觉上连贯，不是两个割裂的斑）|

ring 比 core 压得重，是因为 **ring 是 optimizer 最容易"作弊"的地方** —— 如果不专门压 ring，PGD 会把 decoy 向 GT 边缘"蔓延"，既占了 decoy 位置（骗正激活），又占了 GT 边缘（骗掉原激活），$\mathcal{J}\&\mathcal{F}$ 看着下降了，但其实 mask 只是涨了一圈没挪位，这不是 mislocalization，是 blur。

### Rank loss：让 decoy 赢过 ring

光是 "decoy 上推 + ring 下压" 还不够，因为两者是**独立**优化的。真正的"欺骗"需要 **decoy logit 的顶端 > ring logit 的顶端**（否则 SAM2 的 mask 还是会选 ring 上那一撮高响应）。

所以加一项 rank：`softplus(A(ℓ, ring) − A(ℓ, decoy) + margin)`，其中 $A(\cdot)$ 是一个"峰值聚合器"。这里**有两个版本**：

- **Round 1（hard top-20%）**：取 mask 内 top 20% logit 的均值。想法是"让 decoy 的最高 20% 像素高过 ring 的最高 20% 像素"。
- **Round 2（soft CVaR_{0.5}）**：sigmoid gate 围绕 **detached 中位数** 做软选择，等价于"top 50% 的平滑均值"。梯度对所有 in-mask 像素都有流。

**为什么 Round 1 的 hard top-k 失败了**（Stage 2 regression 的核心诊断）：

- hard top-k 选的像素集合 **frame-by-frame、step-by-step 在变**。
- 在纹理不均匀的 clip（cows 的毛、dog 的纹理）上，ring 里哪些像素是"top 20%"会跳来跳去。
- PGD 就变成追着"这一步 top 20% 是谁"跑，每次塑造出来的 "尖峰" 都是**瞬时的**，过两帧就被 memory refresh 抹掉。
- 结果：cows `-0.91`、dog `-0.55` 的掉分。

Round 2 的 soft CVaR 修法：用 detached quantile（不回传梯度，保持 rank 稳定）+ sigmoid gate（梯度对所有像素流动）。**"尖但软" 替换 "尖但脆"**。

### Insert 帧的设计

- **基帧来源**：从 $f_0$ 切出标注对象 → 位移到 decoy 位置 → Poisson 融合（消除 seam）。这是一个**合成帧**，它的存在本身不是为了"看起来和真实场景一样"，而是"给 SAM2 memory encoder 一个看起来合理但把 attention 引到错位置的锚"。
- **SSIM 约束**：对 insert 的扰动做 SSIM 约束，不是对"原帧"（没有原帧），而是对**Poisson 基帧**。这保证 PGD 不会把 insert 扰成高频噪声。
- **3 个 insert 的位置**：固定在 $f_3$、$f_7$、$f_{11}$ 之后（FIFO 共振点）。SAM2 memory bank 容量 7，每次 FIFO 滑一帧挤一帧；3 个 insert 正好在每次旧 memory 快要满但还没溢出时"塞进去"，让 poison memory 保持在 bank 里占比高。

### Read-path loss：让 poison 在 rollout 时持续生效

单纯 "insert 帧 loss 做好" 不够 —— PGD 可能找到一个"只在 insert 帧上骗过 SAM2，下一帧就回正"的局部解。

所以加 read-path 监督：在评估窗口 $f_L..f_{L+H-1}$ 的 SAM2 rollout 输出上，要求 `ℓ(decoy) > ℓ(true) + margin`。梯度会通过 SAM2 的 memory attention 一路回传到 perturbation 和 insert 参数，强制 poison 必须"在 memory 里活着"。

权重随 rollout 距离递减（远处的帧权重低），但设了**权重下限** 0.5（Round 2），避免远处帧完全被忽略 —— 否则 optimizer 只会关心 $f_L..f_{L+5}$，拿不到长视野信号。

### Round 2 (still pending eval) 的三个关键调整

1. **EVAL_HORIZON 5 → 7**（后来想拉到 10）：rollout 监督窗口拉长，防止 "5 帧内骗过就不管后面" 的短视解。
2. **hard top-k → soft CVaR_{0.5}**：上面详述。
3. **w_ring 1.5 → 1.0**：ring 压得太重，会让 decoy 的 positive loss 被它盖住；降下来之后 decoy 自己能长。

---

## 3. 关键差异总结

| 维度 | Suppression | Decoy |
|---|---|---|
| 目标 | object_score → 负 | object_score > 0 且 mask 在错位置 |
| Loss 结构 | 单目标：压 GT 内 logit + 钉 score 带 | 四分区 + rank + read-path + score 保正 |
| Teacher memory | 不需要 | v3 曾用（合成教师视频的 memory features）；v4 之后移除，改为纯输出监督 |
| Insert 必要性 | 可选（加分但不必要）| 必要（insert 是 poison 的载体）|
| 预算敏感度 | 低（扰动就够）| 高（insert 8/255 是可见上限）|
| DAVIS-10 均值 drop | **0.744** | 0.547 (v4) / 0.415 (Stage 2) |
| 可解释性 | 低（就是降 score） | 高（mislocalization 有清晰的几何结构）|
| Story 强度 | 弱（"降 score 不是攻击"）| 强（"骗 memory 往错位置"）|
| Reviewer 风险 | "这不算对抗攻击" | "为什么不直接 Suppression"（因为 dominated）|
| SAM2Long 迁移 | 未测试 | 均值 retention 0.52（bmx-trees 0.76 强迁移，其他弱）|

---

## 4. Fidelity 维度的实现思路

### 测量

- 攻击原始帧 vs DAVIS clean 帧：1:1 配对，pixel-level SSIM / PSNR / LPIPS(alex)。
- insert 帧没有 clean 对照，用"时间上最近的原始帧"做近似对照（衡量"这个 insert 和周围真实场景有多不搭")。

### 观察

- 攻击原始帧：LPIPS **0.016**（远低于 publication-grade 0.05 阈值）—— **已经很隐蔽**。
- insert 帧：LPIPS **0.16** —— 明显可见，就是 reviewer 说的 "pasted ghost"。

### 含义

用户"高保真"的 ask 实际上映射到 **insert 帧**，不是扰动本身。**所有 fidelity 问题都在 3 张合成 insert 上**。

### 下一步方向（尚未实现）

1. **感知 loss + edit mask**：在 insert 的 edit 区域内对齐 Poisson base；edit 区域外匹配邻近 clean 帧；seam 上加 CIELAB $\Delta E$ + 轻 TV 正则。
2. **运动一致初始化**：decoy offset 搜索要避开"和对象运动方向同向"和"与未来轨迹重叠"的方向，否则 insert 会和真实运动冲突，更刺眼。
3. **频域残差参数化**：把 insert 的可训练残差限制到 DCT/wavelet 的低频段，高频扰动被压住 —— 既降 LPIPS 又更能扛 codec 压缩。

---

## 5. Round 2 行动项（按期望收益排序）

1. **Soft CVaR ablation on DAVIS-10**（当前 running）：确认 cows / dog 是否回来。
2. **Perceptual insert loss**：把 insert 的 LPIPS 从 0.16 压到 0.05。
3. **MI-FGSM + TI-FGSM + DIM**：在 Stage 3 joint 优化里换 PGD 更新规则，提高对 SAM2Long 的迁移率（目标 retention ≥ 0.5）。
4. **DCT/wavelet 残差重参化**：补上频域 hiding。
5. **运动一致的 decoy offset 搜索**：现在只在 $f_0$ 上选一次 offset，应该在 $f_0..f_{14}$ 整个前缀上搜索，同时拒绝 motion-aligned / 未来轨迹重叠 / 需要 alpha blend 的候选。

---

## 参考代码位置（方便定位，不贴实际代码）

- `run_two_regimes.py`
  - `_supp_loss` / `_supp_read`：Suppression 的 write + read loss
  - `_decoy_write` / `_decoy_read`：Decoy 的四分区 + rank + read loss
  - `_soft_cvar_mean` / `_topk_mean`：Round 2 软聚合器 / Round 1 硬 top-k（已标 DEPRECATED）
  - `optimize_unified`：3-stage PGD 的主循环
- `memshield/decoy.py`
  - `find_decoy_region` / `create_decoy_base_frame`：Poisson 基帧构造
- `scripts/sam2long_eval.py`：SAM2Long cross-evaluator
- `scripts/measure_fidelity.py`：SSIM / PSNR / LPIPS 测量
