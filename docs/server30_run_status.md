# 30 端口服务器：执行状态与查看入口

状态快照：2026-10-10 00:38（Asia/Shanghai，2026-10-09 16:38 UTC）。

**本批已完成全部 156 组筛选，在 G1 按冻结门槛停止。后台已退出，自动监督已暂停。**
完整分析见 [筛选结果与 G1 停止报告](server30_results.md)。

- 本地分支：`develop/lyx`，从原 `origin/lyx` 研究路线创建。
- 正式实验源码：`488032c`；冻结摘要与本地实验源码一致。
- G0 已通过：服务器测试套件 35 项通过，包含 CUDA 严格恢复与反向传播检查；
  6 个真实模型检查全部通过，最大范数关系相对误差 `1.17343e-7`。
- P1 已结束 **156/156** 组：154 组完成完整训练划分的 20 epochs，2 组按失稳规则终止。
  Fashion-MNIST / SmallCNN / SGD 为 48/48，CIFAR-10 / ResNet-18 / SGD 为 60/60，
  AdamW 为 48/48。两次终止均为 Fashion-MNIST 的 loss explosion，结果已保留。
- 进程错误计数为 0，退出码为 0；已确认无运行中实验，GPU 计算进程列表为空。
  维修后 8 个续跑运行均完成，最终逐步记录连续且无重复。维修暂停不计为科学失败。
- **G1 未通过**：只有 2 个含双类标签的可评估运行，未达到至少 10 个的要求；
  历史预测增量未达门槛，家族留出无有效双类比较；事件稳健性为 27/63（42.86%），
  低于 75%。P2/P3/P4 未运行，不形成对应因果或算法主张。
- 自动监督（名称：DAGC 实验每20分钟监督，ID：`dagc`）状态为 **PAUSED**。
  本批已到达明确科学停止条件，不再自动启动后续训练。

## 服务器文件

任务根目录：`/mnt/data/users/yuxin/workspace/DAGC-develop-lyx`。

| 内容 | 根目录下的路径 |
|---|---|
| 正式实验目录 | `experiments/lyx_full_20261009_r2/` |
| 实时状态 | `experiments/lyx_full_20261009_r2/status.json` |
| 最终门槛报告（完成后生成） | `experiments/lyx_full_20261009_r2/campaign_report.json` |
| 完整筛选分析 | `experiments/lyx_full_20261009_r2/p1/analysis/` |
| 后台执行日志 | `operations/campaign-r2.log` |
| 启动脚本 | `operations/launch_campaign_r2.sh` |
| 只读状态检查 | `operations/monitor_status.py` |
| 依赖版本 | `operations/environment.lock.txt` |
| 冻结源码归档 | `operations/source-488032c.tar.gz` |
| 维修暂停与检查点清单 | `operations/maintenance_pause.json` |
| 维修后恢复预检 | `operations/resume_preflight_20261010.json` |
| 维修后恢复记录 | `operations/maintenance_resume_20261010.json` |
| 续跑日志连续性核验 | `operations/resumed_log_audit_20261010.json` |

实际运行环境与数据缓存位于 `/tmp/dagc-yuxin-runtime/`。原始数据包、配置、逐步
记录及检查点保存在任务根目录。环境归档于 2026-10-09 12:08:54 UTC 完成，
文件为 `operations/environment-py312-cu128.tar.gz`（约 4.0 GiB），另有 SHA-256 清单。

查看当前进度：

```bash
ssh -p 30 -i ~/.ssh/id_yuxin yuxin@47.116.73.163 \
  '/usr/bin/python3 /mnt/data/users/yuxin/workspace/DAGC-develop-lyx/operations/monitor_status.py'
```

训练具备检查点恢复能力，但基础设施错误可能让后台程序停止；上述状态检查会显示
`infrastructure_error`。本批实际状态为 `stopped_at_scientific_gate`，不是基础设施失败；
不应重启同一批次来绕过 G1。新科学方案必须另建协议和输出目录。
本次维修后 `/tmp` 缓存仍在，已核验后直接复用。如果后续维修清空缓存，需从持久归档
恢复相同路径的源码、环境和数据，核验依赖及冻结摘要，再恢复训练和自动监督。

已取回本地的 G0 报告、测试日志、依赖清单和冻结协议位于 `results/server30/`；
完整筛选的配置、摘要、manifest 和分析证据位于 `results/server30/final/`。
首次 G0 的确定性池化错误及其日志保留在服务器旧实验目录中，未混入正式 P1。
