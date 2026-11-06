# E2 标注工作包（annotation kit）

日期：2026-11-05 | 打包内容：待标注文件 + 标注所需全部材料 + 当前项目文档与代码
（对应上一包 `e1_v3_deliverables_20261104.tar.gz` 为 E1 阶段交付，本包面向 E2 标注阶段。）

---

## 0. 一句话说明

你（论文作者）需要在本地打开本包中的图片/视频，**人工填写 3 个 CSV**
与 1 份复核意见——这是 v4-experimental 消融实验与论文性能数字的
**唯一真值来源**。在标注完成前，系统不计算任何准确率/召回率。

## 1. 三项任务（按顺序，总计约 3–4 小时）

### 任务 A：填 20 行人工真值（约 2.5h，最优先）

- 文件：`datasets/annotations/e1_manual_labels.csv`（15 列 schema，全部空白）
- 指南：`docs/annotation_guidelines.md`（**必读**——七问题→字段映射、
  每个重点样本该看哪张图、unknown 兜底规则）
- 重点样本与材料：
  - **0016**（目标身份切换）：`runs/manual_review/0016_contact_sheet.jpg`
    + `0016_gen_f055/062/063_boxes.jpg`
  - **0024**（生成协议是否要求行人）：`runs/manual_review/0024_contact_sheet.jpg`
    + gen 全片（本地 `repos/SynPeDAS/generated_video/`，不在本包）
  - **0003 / 0044**（Region Presence ≠ Crossing）：
    `runs/synpedas_smoke_v3_1/<样本名>__gen/keyframes/` 与 `annotated.mp4`

### 任务 B：协议复核（至少 val 子集，约 0.5–1h）

- 文件：`datasets/annotations/event_protocol.csv`（211 行，
  `event_required` 全部 unknown）
- 方法：按 `auto_prompt_*` 证据列排序，逐例读 prompt 原文
  （`video_generation_prompts/*.txt` 已随包附带 211 份），
  填 true / false / unknown + `protocol_review_status=reviewed`
- 优先：val split 中的样本（见 `e2_split_proposal.csv` 的 split 列）

### 任务 C：划分确认（约 15 分钟）

- 文件：`datasets/annotations/e2_split_proposal.csv`
- 检查点：① 同源样本是否整组同侧（脚本已保证，抽查即可）；
  ② **把 1–2 个 no-person-prompt 样本从 test 调入 val**
  （当前 val 缺该 stratum）；
  ③ 确认后把 `review_status` 改为 confirmed

## 2. 文件地图

```
README.md                        本文件
docs/                            15 份文档（标注指南、事件协议设计、
                                 横穿语义、E2 实验计划、文献调研等）
src/                             wmeval 代码 + E2 新模块
                                 （actor_binding / crossing_evidence）
datasets/annotations/
  e1_manual_labels.csv           ★ 任务 A：人工真值表（20 行待填）
  event_protocol.csv             ★ 任务 B：事件协议（211 行待复核）
  e2_split_proposal.csv          ★ 任务 C：划分提案（待确认）
  manifest.csv                   211 组四元组索引
runs/
  manual_review/                 24 个标注用图（2 张 contact sheet +
                                 20 张带框关键帧 + 2 份对齐记录）
  inspect_*.jpg                  3 张早期抽查帧
  synpedas_smoke_v3_1/           20 个视频目录——含 ★keyframes/
                                 （每人轨首/中/尾帧）与 ★annotated.mp4
                                 （带框视频，标注主要材料）+ 全部数值证据
  synpedas_smoke{,_v2,_v3}/      历史版本 summary（谱系对照）
video_generation_prompts/        211 份生成 prompt 原文（任务 B 用）
generated_video/*.json           211 份生成参数元数据（seed/guidance/
                                 negative_prompt，任务 B 用）
```

**不在包内**（体积原因，本机路径仍可用）：
原始 JAAD 片段与生成视频本体（`repos/SynPeDAS/JAAD_crossing_moments/`、
`repos/SynPeDAS/generated_video/*.mp4`，共 2.6GB——任务 A 的 0024 gen
全片目视需要它们）；YOLO 权重（`ckpt/yolov10x.pt`）。若在别处解包使用，
需另克隆 SynPeDAS 仓库（git clone https://github.com/levilovearch/SynPeDAS.git）。

## 3. 标注纪律速记（详见 annotation_guidelines.md §0）

1. 只填人工目视确认的结果——**禁止**抄 v3.1 自动判定；
2. `unknown` 是合法且鼓励的答案；
3. 真实视频不自动标 Valid（real 侧是参照不是被评对象）；
4. `event_required=unknown/no` 的样本，`event_satisfied` 不得填 yes/no，
   `human_verdict` 不得为 invalid（协议未确认不算违约）。

## 4. 完成后触发（预注册，标注期间不回头改规则）

1. 20 行非空（unknown 计为已填）→ 启动 v4-experimental 消融 B0–B3
   （`docs/e2_experiment_plan.md` §4，含防 Uncertain 刷指标的硬约束）；
2. 0003/0044 的 Q5 答案 → 校准 L3 报告阈值；
3. 0016 Q3 答案 → 决定 redetection 归因保留或回滚；
4. 0024 Q1 答案 → 决定 Invalid 判定的协议前置条件。

## 5. 包完整性

- 生成命令与校验值见包外 `dist/SHA256SUMS.txt`；
- 解包校验：`tar tzf e2_annotation_kit_20261105.tar.gz | wc -l`
  （应与打包记录一致）；
- v3.1 实验输出在本包中为**冻结快照**（打包时刻与 runs/ 一致）。
