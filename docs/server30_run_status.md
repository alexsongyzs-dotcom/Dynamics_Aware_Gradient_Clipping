# 30 端口服务器：执行状态与查看入口

状态快照：2026-10-09 23:04（Asia/Shanghai，15:04 UTC）。

**因用户要求维修服务器，节点 30 的本任务实验和自动监督已暂停。须等待用户明确要求恢复。**

- 本地分支：`develop/lyx`，从原 `origin/lyx` 研究路线创建。
- 正式实验源码：`488032c`；冻结摘要与本地实验源码一致。
- G0 已通过：服务器测试套件 35 项通过，包含 CUDA 严格恢复与反向传播检查；
  6 个真实模型检查全部通过，最大范数关系相对误差 `1.17343e-7`。
- 暂停前调度器记录 P1 已结束 **128/156** 组（82.1%）；8 组训练已中断，另有 20 组未启动。
  Fashion-MNIST / SmallCNN / SGD 已结束 48/48 组，其中 2 组触发预声明失稳条件，
  已保留为科学失败结果；CIFAR-10 / ResNet-18 / SGD 已结束 60/60 组；
  AdamW 已结束 20/48 组。维修暂停不计为科学失败。
  已确认本任务调度器及全部训练进程退出，GPU 计算进程列表为空。暂停前进程错误计数为 0。
  8 个中断运行均有持久检查点：7 个最近轮次检查点、1 个初始检查点；已核验文件存在及归档格式。
  恢复时会重做最后检查点之后的训练，初始化检查点对应的运行需要从头训练。
  每组使用完整声明训练划分，最多训练 20 轮。
- G1 尚无结果。程序完成 P1 后自动分析，并按 G1/G2/G3 门槛决定后续训练；
  门槛不通过会保留负结果并停止相应主张。
- 自动监督（名称：DAGC 实验每20分钟监督，ID：`dagc`）状态为 **PAUSED**；
  保留每 20 分钟的频率设置，但维修期间不执行，也不会自动重启训练。

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

实际运行环境与数据缓存位于 `/tmp/dagc-yuxin-runtime/`。原始数据包、配置、逐步
记录及检查点保存在任务根目录。环境归档于 2026-10-09 12:08:54 UTC 完成，
文件为 `operations/environment-py312-cu128.tar.gz`（约 4.0 GiB），另有 SHA-256 清单。

查看当前进度：

```bash
ssh -p 30 -i ~/.ssh/id_yuxin yuxin@47.116.73.163 \
  '/usr/bin/python3 /mnt/data/users/yuxin/workspace/DAGC-develop-lyx/operations/monitor_status.py'
```

训练具备检查点恢复能力，但基础设施错误可能让后台程序停止；上述状态检查会显示
`infrastructure_error`。本次用户要求暂停，状态为 `paused_maintenance`。恢复时须使用同一冻结源码、
配置和数据，并先确认原进程已退出及 GPU 无他人占用。不要修改门槛以跳过错误。
维修后 `/tmp` 缓存可能已清空，需从持久归档恢复相同路径的源码、环境和数据，
核验依赖及冻结摘要，再恢复训练和自动监督。

已取回本地的 G0 报告、测试日志、依赖清单和冻结协议位于 `results/server30/`。
首次 G0 的确定性池化错误及其日志保留在服务器旧实验目录中，未混入正式 P1。
