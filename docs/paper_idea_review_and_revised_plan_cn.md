# DAGC 论文 Idea 评述与新版研究规划（v2）

> 项目：Dynamics-Aware Gradient Clipping（DAGC）
> 评述日期：2026-09-03
> 目标：把现有“大而全”的研究提纲收敛成可证伪、可执行、与最新直接相关工作有清楚边界的顶会论文计划。
> 重要说明：本文中的“建议”“评分”和“新假设”是评述意见，不是输入材料或所调研论文已经证明的结论。

## 0. 结论先行

这个方向值得继续，但不建议按现有 A0-A9 全线平推，也不建议把“一个新的自适应阈值算法”作为第一贡献。

最有机会形成顶会论文的部分不是“根据历史梯度调整阈值”，因为 AutoClip、AGC、AdaGC、StableAdamW 等工作已经覆盖了历史分位数、参数相对尺度、逐张量 EMA 阈值和更新裁剪。真正仍有辨识度的主线是：

> **在控制裁剪增益的边际分布、累计更新预算和即时梯度尺度后，裁剪事件的时间组织，以及裁剪发生在有状态优化器内部的具体位置，是否仍会改变未来不稳定风险与最终函数解？**

建议把论文重新定位为一篇“机制与因果识别优先、控制器次之”的工作：

1. 先建立稳健的裁剪事件定义和可重复的事件级动力学现象；
2. 再证明时序信息相对梯度范数、学习率和更新范数具有增量预测价值；
3. 用预注册的分支实验识别事件时序与优化器状态污染的因果效应；
4. 只有前三项通过门槛，才设计 DAGC-v2；否则论文停留在机制发现，不强行制造算法。

当前成熟度评估：

| 维度 | 评分 | 评述 |
|---|---:|---|
| 问题重要性 | 8/10 | 训练稳定性、loss spike、鲁棒学习率与裁剪调参均有现实价值。 |
| 原始观察的新颖性 | 6/10 | “切换、暴露、事件时序”有潜力，但必须证明不是阈值附近噪声和学习率缩放的重述。 |
| 当前算法新颖性 | 3/10 | 现有控制器仍主要依赖历史范数/EMA/方向对齐，与已有自适应裁剪重叠较大。 |
| 可证伪性 | 8/10 | 可以设计清晰的负结果门槛、事件预测和受控分支实验。 |
| 当前实验可执行性 | 4/10 | A0-A9、10-20 seeds、稠密相图和大规模验证同时推进，范围过大。 |
| 当前投稿就绪度 | 3/10 | 核心对照存在数学同一性问题，直接相关工作和统计识别尚未补齐。 |

**总评：Conditional Go。** 先用 4-6 周完成机制筛选与因果识别；只有出现稳定、跨配置且超越强基线的信号，才进入算法与规模化阶段。

## 1. 评述依据与材料关系

本次阅读的三份主要材料是：

| 材料 | 作用 | 本次判断 |
|---|---|---|
| `ai_research_outline_cn0.pdf` | 完整研究库存：动力学量、A0-A9、代码结构、风险路径 | 适合作为实验仓库，不适合作为线性执行计划。 |
| `ai_research_outline_cn.pdf` | 面向 8 页顶会主文的压缩版本：主张、Related Work、主图和阶段门槛 | 叙事更成熟，但沿用了若干需要修正的因果对照和稳定性表述。 |
| `paper_research_2605_optimization.md` | 五篇随机优化/裁剪论文的理论边界与证据强度 | 说明“裁剪”在隐私、重尾、局部光滑、无界梯度中的目标不同，不能把这些工作直接拼接成神经网络训练动力学结论。 |

两个 PDF 都是 24 页。新版 `ai_research_outline_cn.pdf` 已经比 `cn0` 更接近投稿叙事：它明确提出 `mechanism -> measurement -> prediction -> control`，要求负结果条款，并把主文压缩为 8 页。然而，新旧两版共同承担了太多主张：局部稳定性、切换、振荡、轨迹选择、预测、控制器和大规模泛化几乎都被列为必要贡献。这会让任一薄弱环节拖垮全文。

## 2. 现有 Idea 最强的部分

### 2.1 问题不是“裁剪有没有用”，而是“何时、为何、在哪里起作用”

这比单纯比较固定阈值的 accuracy 更有研究价值。把裁剪系数

$$
\alpha_t=\min\left(1,\frac{c_t}{\lVert g_t\rVert}\right)
$$

视为沿轨迹产生的内生增益序列，可以研究其时间结构、与优化器内部状态的交互，以及在不稳定转折点附近的作用。

### 2.2 已经具有可证伪的研究链

现有提纲没有把成功定义成“新算法平均准确率高一点”，而是要求：

$$
\text{机制} \rightarrow \text{测量} \rightarrow \text{预测} \rightarrow \text{干预} \rightarrow \text{训练收益}.
$$

这个逻辑是正确的。尤其是“若 A1-A7 不支持动力学信号，就报告负结果并停止复杂控制器”这一条应保留。

### 2.3 轨迹选择使用了功能空间终点

提纲已经意识到参数距离不足以证明不同 basin，提出了 held-out disagreement、CKA、calibration、linear mode connectivity 等指标。这比仅展示参数距离更可信。

### 2.4 工程骨架与论文问题基本对齐

仓库已有 `clipping`、`dynamics`、`hessian`、`switching`、`oscillation`、`trajectory` 和相应分析模块，说明研究问题可以被记录和复现。新的规划不需要推倒重来，重点是重写“测量契约、干预语义和门槛”。

## 3. 必须优先修正的关键问题

### 3.1 A7 的 norm-matched control 与全局裁剪在数学上相同

当前 A7 写道：把原始更新缩放到裁剪更新的范数，以“匹配步长但破坏状态依赖”。对无动量 SGD 的全局范数裁剪，

$$
\Delta\theta_t^{\text{clip}}
=-\eta_t\alpha_t g_t.
$$

若“范数匹配”仍沿 $g_t$ 方向，并把范数设为 $\lVert\Delta\theta_t^{\text{clip}}\rVert$，得到的仍是同一个向量 $-\eta_t\alpha_t g_t$。它既没有破坏状态依赖，也不是独立对照。

因此，论文不能笼统声称“gradient clipping 不只是 step-size reduction”。更准确的分层表述是：

- 对无动量 SGD，全局范数裁剪在每一步就是一个由当前梯度决定的标量学习率；可检验的新问题是**增益序列的时序是否超越其边际分布与累计预算**。
- 对 momentum/AdamW，裁剪发生在动量/矩估计之前还是更新之后，会改变内部状态；这里才可能识别出超越当前步更新范数的机制。

### 3.2 “切换事件”可能只是阈值附近抖动

当前定义 $s_t=\lVert g_t\rVert/c_t-1$，并以 $s_ts_{t+1}<0$ 作为 switching event。若 $q_t$ 在 1 附近受 minibatch 噪声影响，切换数会被任意放大；改变日志频率也可能改变结论。

需要引入：

1. 进入/退出不同阈值的 hysteresis；
2. 最小驻留时间；
3. 对事件检测参数的敏感性分析；
4. 保持梯度范数序列但随机化阈值交叉的负对照；
5. 在固定 probe batch 上重复计算关键方向与曲率量，区分轨迹变化和 batch 噪声。

### 3.3 $\eta_t\lambda_{\max}(H_t)\approx2$ 不能无条件外推

经典边界 2 来自局部二次模型下的确定性梯度下降。momentum、AdamW、随机 minibatch、权重衰减、预条件和时变学习率都会改变稳定性边界。

新版实验应采用两层口径：

- 在全批量或大批量、无动量 SGD 的诊断实验中，保留 $S_t=\eta_t\lambda_{\max}(H_t)$ 与 2 的比较；
- 在有状态优化器中，把它称为诊断 proxy，并优先使用优化器感知量，例如预条件曲率的 Rayleigh quotient、更新 Jacobian 的谱半径近似，或 moment mismatch 指标；不得把 2 当成普适阈值。

### 3.4 近二周期指标容易被随机 batch 和学习率衰减混淆

$C_1(t)<0$、$C_2(t)>0$ 与两步回返比 $R_2(t)$ 可以描述 alternating dynamics，但不同 minibatch 会自然改变梯度方向；学习率变小时，参数回返比也会被缩放效应干扰。

必须增加：

- 固定 probe batch 的梯度对齐序列；
- 同 batch 的相邻方向比较；
- 去趋势后的自相关/功率谱；
- 与 batch-shuffled、time-shuffled 序列比较；
- 使用“period-2-like”或“alternating”措辞，除非存在严格周期证据。

### 3.5 参数轨迹分离不是轨迹选择的充分证据

在非凸网络中，两条训练轨迹在参数空间迅速分离是常态，且置换/缩放对称性会夸大距离。主结论必须以功能空间和干预后的长期效应为主：

- held-out prediction disagreement；
- calibration 与 subgroup 指标；
- CKA/representation similarity；
- linear mode connectivity barrier；
- 在相同后续训练策略下，差异是否持续而不是短暂偏移。

### 3.6 当前算法与已有自适应裁剪工作的差异不足

直接相关工作已经覆盖：

- AutoClip：用历史梯度范数分位数自动选阈值；
- AGC：按参数/梯度相对尺度做 unit-wise clipping；
- AdaGC：按张量维护梯度范数 EMA，处理时间与空间异质性，并在 LLM/VLM 规模比较 GlobalGC、AGC、Clippy、ZClip 等；
- StableAdamW：根据 AdamW 二阶矩失配做 update clipping；
- 高维 clipped-SGD 动力学：用确定性近似分析裁剪对稳定性和优化速度的影响，并讨论调度阈值。

当前 `DynamicsAwareClipping` 使用 exposure EMA、gradient alignment、滑动中位数边界和多个平滑/强度参数。若没有事件时序的独立因果证据，它容易被评价为“AutoClip/AdaGC 加一个 alignment heuristic”。

此外，现实现含 `gamma`、`beta`、`relax`、`osc_weight`、`beta_a`、上下界比例、初始化阈值和窗口长度等多个选择，与提纲“最多 1-2 个重要超参数”的目标不一致。

### 3.7 预测必须证明增量价值，而不是相关性

梯度范数、loss、学习率、训练阶段本身就可能预测不稳定。切换/暴露/振荡指标若只与这些量相关，不构成新发现。

预测实验必须比较：

$$
\text{Base features}
=\{t,\eta_t,L_t,\lVert g_t\rVert,\lVert\Delta\theta_t\rVert\}
$$

与

$$
\text{Base} + \{\text{exposure history, burst, switching, alignment, moment mismatch}\}.
$$

评估应按 seed、模型或配置分组做 out-of-distribution 验证，不能随机拆分同一训练轨迹上的 step。

## 4. 新的论文定位

### 4.1 推荐标题方向

首选工作标题：

> **Timing Matters in Gradient Clipping: Predictive Events, Causal Interventions, and Optimizer-State Effects**

中文：

> **梯度裁剪中的时序效应：预测性事件、因果干预与优化器状态机制**

若最终控制器结果足够强，再使用：

> **Dynamics-Aware Gradient Clipping via Event-Timed Control**

不建议在机制尚未成立前把 DAGC 放在标题中心。

### 4.2 两层核心主张

**主张 A：时序效应。** 在控制裁剪增益的边际分布、累计缩放和训练阶段后，增益的时间排列仍对未来不稳定风险、恢复时间和最终函数解有可重复影响。

**主张 B：状态注入效应。** 对 momentum/AdamW 等有状态优化器，pre-state gradient clipping 与 post-update clipping 即使匹配当前应用更新范数，也会因为内部状态不同而产生不同的后续动力学。

**派生主张 C：事件控制。** 若 A/B 成立，可用低成本、具有滞回的事件控制器减少有害状态进入，同时避免持续过度裁剪。

### 4.3 明确不主张什么

- 不声称神经网络中普遍存在严格 period-2 orbit；
- 不声称 $\eta\lambda_{\max}=2$ 对所有优化器都是精确边界；
- 不声称所有裁剪收益都独立于学习率缩放；
- 不把参数距离等价为不同 basin；
- 不把重尾、DP、局部光滑和无界梯度论文的定理直接外推到深度网络；
- 不以一个新的阈值 EMA 规则作为充分创新。

## 5. 新研究问题与预注册假设

### RQ1：事件定义是否稳健？

在检测分辨率、hysteresis 宽度、minibatch 噪声和阈值扰动下，clipping episode 是否仍有一致的进入、驻留和退出结构？

**H1：** 经过 hysteresis 和最小驻留过滤后，事件率在相同配置的独立 seeds 间具有可接受重现度，且明显区别于 time-shuffled surrogate。

### RQ2：事件历史是否具有增量预测价值？

在控制当前梯度范数、loss、学习率、更新范数和训练阶段后，历史暴露、burst、切换密度、方向反转与 moment mismatch 是否能提前预测未来 $H$ 步的不稳定？

**H2：** 加入事件历史后，grouped hold-out 的 AUPRC、Brier score 或对数似然稳定改善；这种改善在至少三个配置家族中同号。

### RQ3：时序本身是否有因果作用？

保持裁剪增益集合、累计增益或 duty cycle 相同，仅改变其时间顺序，是否改变后续不稳定风险与恢复时间？

**H3：** 原始事件对齐序列与 time-shuffled/block-shuffled 序列产生显著不同的未来状态转移概率；差异不能由总更新预算解释。

### RQ4：裁剪位置是否通过优化器状态起作用？

对 momentum/AdamW，pre-moment gradient clipping 与 post-update clipping 在匹配当前应用更新范数时，是否产生不同的 moment 污染、loss spike 和恢复过程？

**H4：** 两种干预在当前 step 可匹配，但后续 moment mismatch、更新方向和 spike hazard 出现系统差异。

### RQ5：事件控制器是否形成 Pareto 改善？

**H5：** DAGC-v2 在不牺牲正常配置性能的前提下，降低失败率/恢复时间或扩大稳定学习率区间，并优于 AutoClip、AGC、AdaGC 与 update clipping。

## 6. 可操作的事件与结果定义

### 6.1 滞回裁剪状态

定义 $q_t=\lVert g_t\rVert/c_t$。使用两个阈值：

$$
z_t=
\begin{cases}
1, & q_t>1+\delta_{\mathrm{on}},\\
0, & q_t<1-\delta_{\mathrm{off}},\\
z_{t-1}, & \text{otherwise}.
\end{cases}
$$

再要求最小驻留 $d_{\min}$，从而形成 episode，而不是逐 step 的符号翻转。

主要时序特征：

- time-to-first-exposure；
- rolling duty cycle；
- episode length 与间隔；
- burst intensity；
- transition rate；
- 自上次进入/退出的时间；
- 与 gradient alignment、moment mismatch 同步的 event-aligned 特征。

### 6.2 不稳定结果

预先选定一个主结果和若干次结果：

- 主结果：未来 $H$ 步是否出现 loss spike/发散/非有限值；
- 次结果：最大 loss overshoot、恢复时间、失败运行率、稳定学习率区间；
- 诊断结果：probe-batch 曲率、方向反转、二阶矩失配；
- 长期结果：测试性能、校准、函数分歧、表示相似度与 mode-connectivity barrier。

禁止在看完所有曲线后再修改 spike 定义。阈值需由 pilot 数据冻结，并在 confirmatory seeds 上使用。

## 7. 新实验计划：五阶段、四个门槛

### 阶段 P0：测量与语义验收（第 1-2 周）

目标：证明日志、裁剪位置、checkpoint 分支和统计单位正确。

任务：

1. 记录 raw gradient、clipped gradient、optimizer preconditioned update、实际参数更新与 moment state；
2. 明确 global/per-tensor、pre-moment/post-update 的实现语义；
3. 为 hysteretic episode、gain replay、time shuffle、checkpoint branching 增加单元测试；
4. 使用固定 probe batch 验证梯度对齐与 Hessian-vector product 重复性；
5. 做 deterministic replay：相同 checkpoint、数据顺序和 RNG 必须产生相同轨迹；
6. 修复运行环境后执行完整测试。当前本机测试在收集阶段被 `torch` 缺失和包导入路径阻断，这不是科学结果，但意味着 P0 尚未完成。

**G0 门槛：** 数值恒等式、replay 和干预语义全部 PASS；否则不得运行大 sweep。

### 阶段 P1：低成本现象筛选（第 2-4 周）

建议最小矩阵：

| 轴 | 首轮选择 |
|---|---|
| 数据集 | CIFAR-10、CIFAR-100 |
| 模型 | ResNet-18、ViT-Tiny/Small 中选一个资源可承受版本 |
| 优化器 | SGD+momentum、AdamW |
| 学习率 | stable / transition / unstable 三段，而非一开始稠密扫描 |
| 裁剪 | none、固定 global、tuned global |
| seeds | screening 2-3；只对候选区域扩展到 5-10 |

首轮只回答三件事：

1. episode 是否在不同 seeds 中重现；
2. event-aligned signature 是否超越 norm-only baseline；
3. 哪些模型/优化器真正产生值得研究的不稳定状态。

**G1 门槛：** 至少一个事件历史特征对未来不稳定具有跨 seed、跨配置的稳定增量预测价值；否则停止“switching mechanism”主线，转为阈值鲁棒性或负结果论文。

### 阶段 P2：因果分支实验（第 4-8 周）

从 pilot 运行选取进入 transition regime 前的 checkpoint。每个 checkpoint 复制相同模型、优化器、数据顺序与 RNG 状态，只改变一个干预。

核心干预：

1. **Endogenous clipping：** 原状态依赖裁剪；
2. **Frozen gain replay：** 使用参考运行预先冻结的 $\alpha_t$ 序列，避免分支后由新轨迹重新计算；
3. **Time-shuffled replay：** 保持同一窗口内 $\alpha_t$ 多重集合，随机改变顺序；
4. **Block-shuffled replay：** 近似保留短程相关和 burst 长度，但改变与危险状态的对齐；
5. **Duty-cycle matched random gating：** 匹配裁剪频率，不匹配事件时间；
6. **Pre-state vs post-update clipping：** 对 momentum/AdamW 匹配当前更新范数，但允许内部状态路径不同；
7. **No clipping / tuned fixed clipping：** 基准端点。

注意：同一条分支轨迹中的 training steps 不是独立样本。统计单位是 seed-checkpoint branch，使用 common random numbers 和 paired analysis。

**G2 门槛：** 至少一个时序/位置干预对预注册主结果产生稳定、可重复、非微小的效应，且 95% CI 不跨越预设无效区间。若只有参数距离变化而无功能或稳定性变化，不算通过。

### 阶段 P3：预测器与 DAGC-v2（第 8-11 周）

先做预测，不先拍脑袋写控制器。

基线预测器：

- 当前梯度范数；
- loss 与 loss slope；
- learning rate；
- update norm；
- training progress。

候选增量特征：

- hysteretic exposure state 与 burst age；
- fixed-batch gradient alignment；
- AdamW moment mismatch 或 SGD momentum mismatch；
- 相对历史尺度的 robust z-score；
- 低成本曲率 proxy。

验证协议：leave-one-seed/config/model-out；报告 AUPRC、AUROC、Brier、校准曲线和提前量。只保留在分组外推中稳定的 1-2 个信号。

若 G1/G2 通过，构造具有两态滞回的 DAGC-v2：

$$
\text{NORMAL}\xrightleftharpoons[h_t<h_{\mathrm{off}}]{h_t>h_{\mathrm{on}}}\text{PROTECT},
\qquad h_{\mathrm{on}}>h_{\mathrm{off}}.
$$

控制器要求：

- 仅一个主要 aggressiveness 参数；
- 阈值尺度来自 robust running quantile，而不是新增问题相关常数；
- 有 minimum dwell time，避免 controller chatter；
- 明确作用位置：pre-moment、post-moment 或 post-update；
- 当预测信号失效时退化为固定/分位数裁剪；
- 正常区域近似恒等，避免为了防极端情况持续减速。

**G3 门槛：** 相比 tuned fixed clipping 与最强自适应基线，至少满足一项稳定性收益，并保持其余指标在非劣区间：失败率显著下降、恢复时间缩短、稳定学习率范围扩大或调参敏感性降低。单次 accuracy 小幅上涨不够。

### 阶段 P4：确认性与规模实验（第 11-14 周）

只有 G3 通过才运行：

- 选定 phase cells 上 10 个独立 seeds；
- 一个真正更大规模视觉任务，例如 ImageNet-1k + ResNet-50 或 ViT-S；
- 若已有语言模型算力和成熟训练栈，可增加一个中等规模 LM 作为外部验证，但不应同时新建两套大规模系统；
- 报告 wall-clock、显存、通信、额外归约与失败重跑成本。

### 阶段 P5：冻结与写作（第 14-16 周）

1. 冻结主结果、图表脚本和配置；
2. 负结果和失败 runs 保留在结果注册表；
3. 主文只保留 3 个贡献；
4. 匿名代码和复现包使用不可修改的提交/分支；
5. 所有表格从聚合文件自动生成，不手抄数值。

截至本评述日期，ICML 2027 官方论文截止日期尚未在官网公开；不要把上一届日期当成已确认日程。建议把 2026-12-15 设为内部结果冻结线，待官方日期公布后倒排。

## 8. Baseline 体系重排

### 8.1 必须 baseline

| 类型 | 方法 | 目的 |
|---|---|---|
| 无裁剪 | SGD/AdamW | 绝对基线 |
| 固定裁剪 | default fixed、tuned fixed | 区分算法收益与调参收益 |
| 历史统计 | AutoClip | 对照历史分位数阈值 |
| 参数相对尺度 | AGC | 对照 unit-wise norm-relative 规则 |
| 局部时间自适应 | AdaGC | 最直接的现代强基线 |
| 更新裁剪 | StableAdamW/Adafactor-style update clipping | 对照 optimizer-state mismatch 机制 |
| 时序负对照 | replay、time-shuffle、block-shuffle、random gate | 识别时序因果效应 |

### 8.2 条件 baseline

- SAM 只有在论文声称改善 sharpness/flatness 本身时才纳入主表，否则放 Related Work 或附录；
- ZClip、SPAM 等 loss-spike 方法仅在 LLM 实验成为主任务时纳入；
- DP-SGD-RC、RSC-ZO、CGTVR、clipped AdamW 作为理论边界和机制参照，不应因都含 clipping 就放进同一神经网络 benchmark。

公平性要求：

- 相同 tuning budget；
- 同 seed 与数据顺序；
- 区分默认、有限调参和 oracle/tuned 结果；
- 同时报告失败率，不能删除 diverged runs 后只平均成功运行；
- 比较总 GPU 时间，包括失败重跑成本。

## 9. 统计与因果分析方案

### 9.1 实验单位

- 训练 step 用于构造时间序列特征；
- seed 或 checkpoint branch 才是推断单位；
- 不得把一个 run 中数千个 step 当成数千个独立样本。

### 9.2 主要分析

1. 分层/混合效应模型：模型、数据集、优化器为固定效应或分层因素，seed/config 为随机效应；
2. paired bootstrap 或 paired permutation test：用于 checkpoint branches；
3. 生存/风险模型：预测首次 spike 或失败时间；
4. grouped cross-validation：验证预测器跨 seed/config/model 的外推；
5. 多重比较校正：仅用于预注册的次要结果族；主结果保持单一。

### 9.3 必报内容

- mean/median、95% CI、effect size；
- 所有 seeds 与失败 runs；
- 数据/配置层面的异质性；
- 事件定义与窗口长度敏感性；
- negative-control 结果；
- 预测器校准，而不仅是分类准确率。

## 10. 代码库改造清单

### 10.1 `src/dynamics.py`

- 新增 hysteretic episode state machine；
- 区分 raw-batch 与 fixed-probe 诊断；
- 为 SGD、momentum、AdamW 分别定义稳定性/状态失配 proxy；
- 所有指标注明单位、采样频率和前视/滞后关系，防止信息泄漏。

### 10.2 `src/clipping.py`

- 把“测量”“增益生成”“裁剪应用位置”拆成独立接口；
- 实现 frozen gain、time-shuffle、block-shuffle、random gate；
- 实现 pre-moment 与 post-update 干预；
- 暂停把当前多参数控制器作为最终 DAGC；保留为 heuristic baseline；
- DAGC-v2 在 G1/G2 后再定型，并把关键自由参数压缩到 1-2 个。

### 10.3 `src/train.py`

- checkpoint 必须保存模型、optimizer、scheduler、data sampler 和所有 RNG 状态；
- 日志同时包含 raw grad norm、clipped grad norm、moment norm、preconditioned update norm、actual parameter update；
- 所有因果分支共享数据顺序和随机数流；
- 记录干预协议版本和配置哈希。

### 10.4 `analysis/`

- `event_detection.py`：事件检测与 surrogate；
- `predictive_increment.py`：base vs base+history 的 grouped validation；
- `causal_branches.py`：paired estimands 与 CI；
- `optimizer_state.py`：moment mismatch、pre/post placement 分析；
- 图表脚本只读取冻结后的 tidy result tables。

### 10.5 测试与结果注册

新增测试：

1. 无动量 SGD 中 global clipping 与标量增益更新的严格等价；
2. time shuffle 保持增益多重集合与累计量；
3. pre/post clipping 在当前步匹配时，优化器 state 按预期不同；
4. hysteresis 消除阈值附近单步 chatter；
5. checkpoint 分支 deterministic replay；
6. 指标不读取未来 step；
7. failed runs 进入结果注册表。

## 11. 主文贡献、图表与结构

### 11.1 最多三个贡献

1. **可识别的事件框架：** 稳健定义 clipping episode，并证明其历史对未来不稳定具有增量预测价值；
2. **因果证据：** 通过增益重排和优化器内外位置干预，区分即时尺度、时序与状态注入效应；
3. **可选控制器：** 若前两项成立，给出低成本、低参数的事件触发 DAGC-v2，并展示稳定性 Pareto 改善。

### 11.2 主图

| 图 | 内容 | 必须支持的结论 |
|---|---|---|
| Figure 1 | 识别图：即时尺度、时序、优化器状态位置三条路径 | 说明为什么旧对照不可识别，为什么新干预可识别 |
| Figure 2 | hysteretic clipping episodes 与 event-aligned dynamics | 现象可重复，不是阈值 chatter |
| Figure 3 | Base vs Base+History 的 grouped predictive increment | 时序指标具有前瞻增量价值 |
| Figure 4 | replay/shuffle/pre-vs-post 分支的因果结果 | 时序或状态位置产生因果效应 |
| Figure 5 | DAGC-v2 或机制跨模型复现 | 算法收益或机制外部有效性 |

### 11.3 主表

| 表 | 内容 |
|---|---|
| Table 1 | 事件定义重现度、预测增量与跨配置泛化 |
| Table 2 | 因果干预的 paired effect、CI 与失败率 |
| Table 3 | DAGC-v2 与 fixed/AutoClip/AGC/AdaGC/StableAdamW 的性能-稳定性-开销 |

### 11.4 8 页结构

1. Introduction：精确提出时序与位置识别问题；
2. Related Work and Identifiability：直接回应 AutoClip、AGC、AdaGC、StableAdamW 与高维 clipped-SGD；
3. Event Definitions and Measurement Protocol；
4. Predictive Dynamics；
5. Causal Interventions；
6. Event-Timed Controller（若 G3 未过，则改为 Cross-Setting Validation）；
7. Experiments and Limitations；
8. Conclusion。

## 12. 16 周执行时间线

| 周 | 目标 | 交付物 | 决策 |
|---|---|---|---|
| 1-2 | P0 测量、环境、replay、干预语义 | 测量规范、测试报告、配置注册 | G0 |
| 2-4 | P1 粗粒度筛选 | 候选状态区、事件稳定性、预测基线 | G1 |
| 4-6 | 精炼事件与 confirmatory runs | 冻结事件定义、固定 probe 协议 | 继续/收缩 |
| 6-8 | P2 因果分支 | replay/shuffle/pre-post paired results | G2 |
| 8-10 | P3 预测器冻结与增量复核 | grouped CV、校准与提前量 | 特征冻结 |
| 10-11 | DAGC-v2 | 单参数/双参数控制器、消融 | G3 |
| 11-14 | 强 baseline 与一次规模验证 | 主结果表、开销表、失败率 | G3 通过后扩展 |
| 14-15 | 写作与复核 | 8 页主稿、附录、结果注册 | 内部审稿 |
| 16 | 冻结 | 匿名代码、配置、图表与复现包 | 投稿准备 |

资源原则：先用不超过总预算约 15% 完成 G0/G1；未通过门槛时不得用大规模算力“寻找显著性”。稠密相图只在被筛出的 transition region 细化。

## 13. 风险与备选论文路径

### 风险 A：事件预测不超过 norm-only baseline

结论：switching/exposure 不是独立机制指标。停止 DAGC-v2，把工作转为“哪些裁剪动力学指标没有增量信息”的系统负结果，或回到更清晰的阈值鲁棒性问题。

### 风险 B：预测成立但因果重排无效

结论：事件是状态标记而非控制杠杆。论文可改为训练不稳定性的监测/预警工作，不应声称 trajectory control。

### 风险 C：pre/post 位置有效，但时序无效

结论：聚焦 optimizer-state contamination，形成与 StableAdamW/AdaGC 的机制比较；删除 switching controller 叙事。

### 风险 D：机制成立但 DAGC-v2 不优于 AdaGC/StableAdamW

结论：保留机制论文，把新算法放附录或删除。顶会论文可以靠强机制与因果证据成立，不必强行赢 benchmark。

### 风险 E：只在 CIFAR 成立

结论：不能声称广泛训练规律。要么完成一个 ImageNet/中型 LM 外部验证，要么把论文明确定位为受控机制研究并弱化实用算法主张。

## 14. 模拟审稿意见与预先回答

| 潜在审稿质疑 | 必须准备的回答 |
|---|---|
| “裁剪就是自适应学习率。” | 对 SGD 承认逐步等价；研究的是增益时序与边际分布的区别。对有状态优化器用 pre/post placement 识别内部状态效应。 |
| “switching 只是阈值 chatter。” | hysteresis、minimum dwell、surrogate、固定 probe batch、检测敏感性。 |
| “AdaGC 已经做了时间自适应。” | 直接把 AdaGC 设为强基线；核心新意是事件时序的增量预测与因果重排，而非 EMA 阈值。 |
| “trajectory distance 没有意义。” | 功能分歧、CKA、校准、mode connectivity 与长期持续性为主，参数距离仅作诊断。 |
| “Hessian 边界不适用于 AdamW。” | 经典边界仅用于受控 SGD；AdamW 使用 optimizer-aware proxy，不把 2 当普适常数。 |
| “从同一 run 取很多 step 造成伪显著。” | 以 seed/checkpoint branch 为推断单位，grouped split 与 paired analysis。 |
| “算法只是更多超参数。” | 控制器由已验证信号派生，采用 hysteresis 和 robust scale，主要自由参数不超过 1-2 个，并报告敏感性。 |

## 15. 立即执行的七日清单

### Day 1-2：修正科学契约

- [x] 把 A7 从 norm-matched scaling 改为 gain replay/time shuffle/pre-post placement；
- [x] 把 EoS=2 的使用范围限定到受控 SGD；
- [x] 把“period-2”统一改为“period-2-like/alternating”，直到严格证据出现；
- [x] 在 README 中把算法贡献标为 contingent。

### Day 2-4：修复测量与测试

- [ ] 建立可复现 Python 环境并安装项目依赖（按要求留待 Linux）；
- [ ] 使用 editable install 后执行 `python -m pytest`（按要求尚未运行）；
- [x] 完成 raw/preconditioned/applied update 的分层日志代码；
- [x] 实现 hysteretic episode、完整 checkpoint 与 replay 测试代码（待 Linux 验证）。

### Day 4-7：跑最小 pilot

- [ ] CIFAR-10 + ResNet-18 + SGD-momentum，选 3 个学习率区间；
- [ ] 每格 2 seeds，仅用于筛选；
- [ ] 绘制事件重现度、event-aligned signature 和 base-vs-history 预测；
- [ ] 在第 7 天做第一次 Go/No-Go，不根据单张好看的曲线扩大算力。

## 16. 最终建议

保留现有项目名称与工程骨架，但改变论文成功标准：

$$
\boxed{
\text{稳健事件定义}
\rightarrow
\text{增量预测}
\rightarrow
\text{时序/位置因果识别}
\rightarrow
\text{可选控制器}
}
$$

最重要的收缩是：**DAGC 不再是研究起点，而是机制成立后的结果。**

如果因果门槛通过，这将比“再提出一个自适应 clipping threshold”更有辨识度；如果门槛未通过，也能及时停止昂贵的大规模实验，并得到可信的负结论或更窄的论文方向。

## 参考资料

### 输入材料

- `ai_research_outline_cn.pdf`，SHA256: `9A692AA071326D21B791A0812DF1CEC66462F438B9ED2669D5A689D86FDFBE01`
- `ai_research_outline_cn0.pdf`，SHA256: `7E1BCFDBAC257C0E18CF8F8A29992E7F7322078975252E9233E212D6AA1796D3`
- `paper_research_2605_optimization.md`，SHA256: `E3476CC6B2E2FC35B6AA333266F05977CC5755BE4E3D7582F188669FA28AB14B`

### 本次新颖性核对使用的直接相关工作

1. Zhang et al., [Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity](https://arxiv.org/abs/1905.11881).
2. Seetharaman et al., [AutoClip: Adaptive Gradient Clipping for Source Separation Networks](https://arxiv.org/abs/2007.14469).
3. Brock et al., [High-Performance Large-Scale Image Recognition Without Normalization](https://proceedings.mlr.press/v139/brock21a.html)（AGC）.
4. Cohen et al., [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065).
5. Li et al., [Analyzing Sharpness along GD Trajectory: Progressive Sharpening and Edge of Stability](https://arxiv.org/abs/2207.12678).
6. Marshall et al., [To Clip or not to Clip: the Dynamics of SGD with Gradient Clipping in High-Dimensions](https://arxiv.org/abs/2406.11733).
7. Wortsman et al., [Stable and Low-Precision Training for Large-Scale Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20bd42d82998bc61732c00452228e814-Abstract-Conference.html)（StableAdamW/update clipping）.
8. Wang et al., [AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](https://arxiv.org/abs/2502.11034).

### 投稿日程核对

- [ICML future meetings](https://icml.cc/Conferences/FutureMeetings)：截至 2026-09-03 只确认 2027 举办区域，未见正式论文截止日期。
- [ICML 2026 dates](https://icml.cc/Conferences/2026/Dates)：仅可作为往年节奏参考，不是 ICML 2027 的已确认日程。
