# 30 端口服务器：执行状态与查看入口

状态快照：2026-10-09 19:45（Asia/Shanghai，11:45 UTC）。

- 本地分支：`develop/lyx`，从原 `origin/lyx` 研究路线创建。
- 正式实验源码：`488032c`；冻结摘要与本地实验源码一致。
- G0 已通过：服务器测试套件 35 项通过，包含 CUDA 严格恢复与反向传播检查；
  6 个真实模型检查全部通过，最大范数关系相对误差 `1.17343e-7`。
- P1 已完成 **24/156** 组；另有 8 组运行中，当前第 17–20/20 轮。
  当前进程错误计数为 0。每组使用完整声明训练划分，最多训练 20 轮；
  达到预声明失稳条件的运行保留为失败结果。
- G1 尚无结果。程序完成 P1 后自动分析，并按 G1/G2/G3 门槛决定后续训练；
  门槛不通过会保留负结果并停止相应主张。
- 按用户最新要求，已设置当前任务每 2 小时自动跟进（名称：DAGC 实验每两小时跟进，
  ID：`dagc`），检查训练、处理可恢复故障，并按冻结门槛安排后续实验。
  正常推进时保持安静，仅在阶段完成、门槛结论、异常或需要用户行动时通知。
  全部完成或达到明确科学停止条件后汇总结果并暂停自动跟进。服务器后台程序持续执行。

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

实际运行环境与数据缓存位于 `/tmp/dagc-yuxin-runtime/`。原始数据包、配置、逐步
记录及检查点保存在任务根目录。环境归档在此快照时仍在后台制作，不能视为已完成。

查看当前进度：

```bash
ssh -p 30 -i ~/.ssh/id_yuxin yuxin@47.116.73.163 \
  '/usr/bin/python3 /mnt/data/users/yuxin/workspace/DAGC-develop-lyx/operations/monitor_status.py'
```

训练具备检查点恢复能力，但基础设施错误可能让后台程序停止；上述状态检查会显示
`infrastructure_error`。每 2 小时跟进时会排查可恢复故障。恢复时须使用同一冻结源码、
配置和数据，并先确认原进程已退出及 GPU 无他人占用。不要修改门槛以跳过错误。

已取回本地的 G0 报告、测试日志、依赖清单和冻结协议位于 `results/server30/`。
首次 G0 的确定性池化错误及其日志保留在服务器旧实验目录中，未混入正式 P1。
