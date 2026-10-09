# 30 端口服务器：执行状态与查看入口

状态快照：2026-10-10 00:15（Asia/Shanghai，2026-10-09 16:15 UTC）。

**用户已要求维修后开始，节点 30 的本任务实验及每 20 分钟自动监督均已恢复。**

- 本地分支：`develop/lyx`，从原 `origin/lyx` 研究路线创建。
- 正式实验源码：`488032c`；冻结摘要与本地实验源码一致。
- G0 已通过：服务器测试套件 35 项通过，包含 CUDA 严格恢复与反向传播检查；
  6 个真实模型检查全部通过，最大范数关系相对误差 `1.17343e-7`。
- P1 已结束 **128/156** 组（82.1%）；8 组中断训练已恢复，另有 20 组排队。
  Fashion-MNIST / SmallCNN / SGD 已结束 48/48 组，其中 2 组触发预声明失稳条件，
  已保留为科学失败结果；CIFAR-10 / ResNet-18 / SGD 已结束 60/60 组；
  AdamW 已结束 20/48 组。维修暂停不计为科学失败。
  恢复前确认无其他 GPU 任务；现有 8 个训练进程正常运行，进程错误计数为 0。
  维修后核验源码、配置、数据路径和全部依赖与冻结记录完全一致，8 张 GPU 运算检查均通过。
  8 个中断检查点均已完整读取验证：7 个最近轮次检查点、1 个初始检查点。
  已完成的 128 组直接复用；恢复运行重做最后检查点之后的训练，初始检查点对应运行从头训练。
  每组使用完整声明训练划分，最多训练 20 轮。
- G1 尚无结果。程序完成 P1 后自动分析，并按 G1/G2/G3 门槛决定后续训练；
  门槛不通过会保留负结果并停止相应主张。
- 自动监督（名称：DAGC 实验每20分钟监督，ID：`dagc`）状态为 **ACTIVE**，每 20 分钟检查。
  正常推进保持安静，在阶段完成、门槛结论、异常或需要用户行动时通知。

## 服务器文件

任务根目录：`/mnt/data/users/yuxin/workspace/DAGC-develop-lyx`。

| 内容 | 根目录下的路径 |
|---|---|
| 正式实验目录 | `experiments/lyx_full_20261009_r2/` |
| 实时状态 | `experiments/lyx_full_20261009_r2/status.json` |
| 最终门槛报告（完成后生成） | `experiments/lyx_full_20261009_r2/campaign_report.json` |
| 后台执行日志 | `operations/campaign-r2.log` |
| 启动脚本 | `operations/launch_campaign_r2.sh` |
| 只读状态检查 | `operations/monitor_status.py` |
| 依赖版本 | `operations/environment.lock.txt` |
| 冻结源码归档 | `operations/source-488032c.tar.gz` |
| 维修暂停与检查点清单 | `operations/maintenance_pause.json` |
| 维修后恢复预检 | `operations/resume_preflight_20261010.json` |
| 维修后恢复记录 | `operations/maintenance_resume_20261010.json` |

实际运行环境与数据缓存位于 `/tmp/dagc-yuxin-runtime/`。原始数据包、配置、逐步
记录及检查点保存在任务根目录。环境归档于 2026-10-09 12:08:54 UTC 完成，
文件为 `operations/environment-py312-cu128.tar.gz`（约 4.0 GiB），另有 SHA-256 清单。

查看当前进度：

```bash
ssh -p 30 -i ~/.ssh/id_yuxin yuxin@47.116.73.163 \
  '/usr/bin/python3 /mnt/data/users/yuxin/workspace/DAGC-develop-lyx/operations/monitor_status.py'
```

训练具备检查点恢复能力，但基础设施错误可能让后台程序停止；上述状态检查会显示
`infrastructure_error`。本次维修暂停已经解除，当前恢复至 `p1`。再次恢复时须使用同一冻结源码、
配置和数据，并先确认原进程已退出及 GPU 无他人占用。不要修改门槛以跳过错误。
本次维修后 `/tmp` 缓存仍在，已核验后直接复用。如果后续维修清空缓存，需从持久归档
恢复相同路径的源码、环境和数据，核验依赖及冻结摘要，再恢复训练和自动监督。

已取回本地的 G0 报告、测试日志、依赖清单和冻结协议位于 `results/server30/`。
首次 G0 的确定性池化错误及其日志保留在服务器旧实验目录中，未混入正式 P1。
