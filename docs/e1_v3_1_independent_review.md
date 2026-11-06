# E1 v3.1 独立复核报告：已确认结果与局限

日期：2026-11-05 | 性质：对已完成实验的独立核查（本轮无重跑、无改动 v3.1 输出）
结论等级：区分【已确认】【未确认】【不成立】三档。

---

## 1. 已确认的结果（有可复核证据）

| # | 结果 | 证据 |
|---|---|---|
| C1 | 20/20 视频处理成功，判定分布 real 3/7/0、gen 3/6/1 | `runs/synpedas_smoke_v3_1/summary.csv` |
| C2 | 检测层跨轮完全复现：v3 与 v3.1 的 num_detections/num_tracks 20/20 相同 | 对比脚本输出（error_analysis §2.3） |
| C3 | v3→v3.1 判定变化恰 2 处，均为已证实 bug 的直接因果（0024 real 假 failed order；0016 gen 假 INVALID） | 同上 + selftest 回归 |
| C4 | Bug A（order 精度）与 Bug B（redetection 盲区）修复正确 | selftest 27 项（当时）全过，含 4 项回归 |
| C5 | 0016 gen 的 track 链空间-时间-速度连续（t64→t30→t64，dx≈0.002/帧） | detections.jsonl 数值分析 |
| C6 | 0024 gen 全部 82 检测为 car、person=0；其 prompt 通篇无行人 | detections.jsonl + prompt 原文 |
| C7 | 0003/0044 gen 行人 dx=0.073/0.108、无区域两侧证据；real 侧 0.345/0.518 贯通 | tracks.json 数值分析 |
| C8 | 本轮（E2）新修复：check_order None 传播（F-07）、target_fps 三处一致性（F-08），自测 35+20 全过 | selftest_metrics.py / selftest_e2.py |

注意 C4 与 C8 的区别：C4 是 E1 轮修复；C8 是 E2 轮对**边界/潜在**
问题的修复——F-07 不影响 v3.1 输出（两事件序列路径行为不变），
F-08 亦不影响（E1 全程 target_fps=None）。**v3.1 结果在两轮修复下
均无需重跑**，其证据链保持冻结。

## 2. 未确认的视觉现象（PENDING_HUMAN_VISUAL_REVIEW）

| # | 现象 | 等待的回答 |
|---|---|---|
| U1 | 0016 gen f62→f63 的"换手"是否同一行人 | 标注 Q3（0016） |
| U2 | 0024 gen 夜景中是否存在小/模糊行人 | 标注 Q2（0024） |
| U3 | 0003/0044 gen 行人位移不足是否目视可辨 | 标注 Q5（region≠crossing） |
| U4 | 0016 real t75 / 0030 real t37 的弱证据归因目视成立性 | 标注 weak-evidence 抽查 |
| U5 | **0044 PROMPT_SOURCE_CONFLICT**（2026-11-05 第二轮修正定级）：prompt "from left to right" vs 源/生成视频均 image_right→image_left（real dx=−0.518, gen dx=−0.108，**生成与源同向**） | 数据集构建层缺陷候选；已隔离出干净 prompt-adherence 统计（`prompt_source_generation_audit.csv`）；待人工确认源/生成 actor 同一性与生成器条件输入范围 |
| U6 | **0035 像素位移符号不同**（real image_left dx=−0.523 vs gen image_right dx=+0.126；**语义反转未证实**——prompt 为 toward_camera、双侧 bbox 高度比 3.58/3.35 均为接近证据、相机运动未排除） | 第二轮修正：第一轮"方向反转"表述降级为像素层观察；待人工确认接近语义与自车运动 |

核查材料包：`dist/e2_prompt_direction_review_20261105.tar.gz`
（8 样本：0002/0003/0016/0024/0030/0035/0037/0044，含 real/gen 视频、
prompt、v3.1 证据与 README 数值表）。
**第二轮修正**（外部独立审计 + `docs/direction_audit_report.md`）：
主审计表 `datasets/annotations/prompt_source_generation_audit.csv`
（五类分类：conflict 1 / image_motion_difference 1 / insufficient 6，
语义反转 0——全部 unknown）；修正后回归测试 14/14。

**本轮模型未查看任何图像**（纯文本 API 纪律），上述全部依赖人工。

## 3. 尚未成立的研究结论（禁止提前宣称）

| # | 不得宣称 | 原因 |
|---|---|---|
| N1 | "误放行率/误拦截率下降" | 无人工真值（20 行标注表全空） |
| N2 | "三态判定优于二元判定" | 消融 B0-B3 未运行；且需证明不是用高 Uncertain Rate 换低 IAR |
| N3 | "验证器召回率达标" | 真实 invalid 样本仅 1 个（0024）且协议依据待确认 |
| N4 | "规则泛化到未见场景" | 20 条样本全部参与过规则开发（v1-v3.1 六轮迭代） |
| N5 | "事件条件遵循率 X%" | event_protocol.csv 全部 unknown，未复核 |

## 4. v3.1 的结构性局限（E2 设计的出发点）

1. **无样本级条件**：一套固定规则（行人+固定区域）套所有视频；
2. **无 Actor 身份**：person 证据跨个体混用（D2，详见
   event_protocol_design.md §1）；
3. **region_presence 语义过宽**（C7 的机制根源）；
4. **Invalid 侧样本枯竭**：修复两个假阳性后只剩 1 个真候选——
   召回评估需要外部异常源（DoTA 仅作候选辅助，见 dataset_survey 修正）；
5. **20 样本阈值过拟合风险**：六轮规则迭代都看同一批样本
   （e2_experiment_plan.md 的划分即为此设计）。

## 5. 进入下一轮的条件（预注册）

- [ ] 20 行人工标注完成（unknown 计为完成，空白不算）；
- [ ] event_protocol.csv 至少 Val 集子集复核（event_required 非 unknown）；
- [ ] split proposal 人工确认（含 val 补 no-person stratum 的调配）；
- [ ] 三份设计文档（event_protocol / crossing_semantics /
      annotation_guidelines）作者通读并认可语义。

满足后启动 v4-experimental（消融 B0-B3），期间**禁止**根据测试集
结果回调阈值（单变量推进纪律，回退需登记 failure_log）。
