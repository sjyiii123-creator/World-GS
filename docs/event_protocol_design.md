# 事件协议设计：条件、事件、Actor 与状态判定

日期：2026-11-05（E2 第二轮）
状态：设计文档 + 可运行原型（`src/wmeval/actor_binding.py`、
`src/wmeval/crossing_evidence.py`，自测 `src/selftest_e2.py` 20/20）。
**v4-experimental 的一部分；未接入 v3.1 判定链**。

---

## 1. 动机（v3.1 实测暴露的三个结构缺陷）

| # | 缺陷 | v3.1 实测证据 |
|---|---|---|
| D1 | **无样本级条件**：系统默认"所有视频都必须有行人横穿固定道路区域" | 0024 的 prompt 通篇无行人，"行人缺失=Invalid" 缺协议依据 |
| D2 | **Actor 无身份**：所有 person 轨迹共享事件证据 | 理论上行人 A 可满足 presence、行人 B 满足 region_presence，系统误判"完成横穿" |
| D3 | **检查结果 conflated**：`event_times` 把"未决(None)"与"判定 False"都记 None | check_order 无法区分（F-07 修复时记录的局限） |

## 2. 三层协议结构

```
生成条件 (condition)      —— "模型被要求做什么"      [event_protocol.csv, 人工确认]
    ↓ 满足与否
事件状态 (event state)    —— "视频中实际发生了什么"   [crossing_evidence.py, 自动证据]
    ↓ 归因
最终判定 (verdict)        —— Valid / Invalid / Uncertain [v4 adjudicate, 待设计]
```

**核心纪律**：condition 层永不自动填写（`event_required=unknown` 直到人工
复核 prompt + video2world 任务定义）；state 层只输出证据与状态枚举，
不输出 Valid/Invalid；verdict 层是唯一做三态判定的地方。

## 3. 条件层：`datasets/annotations/event_protocol.csv`

字段（人工确认字段 + 机械证据字段两段）：

| 字段 | 填写方 | 语义 |
|---|---|---|
| sample_id / prompt / generation_type | 脚本 | 索引与条件来源（video2world, seed, guidance） |
| **event_required** | **人工** | true / false / **unknown**（三值；无法确认不许默认 true） |
| event_type / actor_category / actor_identity / required_behavior | 人工 | 事件类型、目标类别与身份、要求行为 |
| temporal_constraint / spatial_constraint | 人工 | 时窗/空间约束（prompt 或任务定义中**明确**出现的才填） |
| condition_source | 脚本 | prompt 路径 + metadata 路径 + 生成参数（可追溯） |
| constraint_confidence / protocol_review_status | 人工 | 约束置信度、复核状态 |
| auto_prompt_* / auto_negative_prompt / auto_declared_frames | 脚本 | **机械事实**（regex 匹配、元数据值），非真值 |

现状：211 行已生成，`event_required` 全部 `unknown`、`review_status=pending`。
机械证据分布：204/211 prompt 提及行人、164 含横穿动作短语、7 个无行人
（含 0024）。**注意 regex 保守**（0003 的 "crossing the asphalt" 未被
横穿短语表匹配）——auto 列只用于人工复核排序，不用于判定。

video2world 条件的特殊语义（人工复核需知）：input_path 指向原始 JAAD
片段，task=video2world（首帧条件 + 文本描述生成延续）。因此 prompt 中
的行人横穿描述是**强条件证据**，但"描述了场景内容"≠"全部内容都是必须
满足的硬约束"——这正是需要逐例人工确认的原因。

## 4. Actor 绑定（条件层→状态层的桥梁）

### 4.1 问题

事件证据必须绑定到**身份链（identity chain）**而非"任意 person"。
v3.1 的教训：0016 gen 同一行人被 ByteTrack 切成
t64(f45-54)→t30(f55-62)→t64(f63-92) 三段；若不关联，"行人消失"假阳性
（v3 误判 Invalid）。

### 4.2 输入/输出协议（已实现）

```
输入:  tracks: list[Track]           # group_tracks 输出
       max_gap=15                    # 接管窗口（帧），对齐 uncertainty.py
       max_pred_err=0.15             # 外推位置容差（norm）
       ambiguity_ratio=1.5           # 歧义判定比
输出:  list[TrackChain]
       TrackChain = {
         chain_id, track_ids[], category,
         links: [{from_track, to_track, gap_frames, pred_err_norm,
                  velocity_norm_per_frame,
                  ambiguous_with?, second_err_norm?}],
         confidence: high|medium|low|ambiguous,
         first_frame, last_frame }
```

### 4.3 关联算法（纯几何，无 ReID）

1. 同类别、按 first_frame 排序，贪心正向生长；
2. **候选语义 = v3.1 redetection case-(b) 的泛化**：候选 track 在
   `(last_frame, last_frame+max_gap]` 窗口内有检测、外推位置（末中心 +
   尾部中位速度×gap）误差 ≤ max_pred_err、且**在 last_frame 当帧无检测**
   （共存保护：站在旁边的另一行人是不同的人）；
   候选不必是新 track——**交替接管**（0016 拓扑：t70 与 t71 无同帧共存、
   时间交错）正是要链接的模式；
3. 多候选：最小 err 者胜出；若次优 err < 1.5×最优 err 且两者都在门限内
   → **歧义显性化**（chain.confidence=ambiguous，rival 记入证据，
   不静默解决）；
4. 每条 track 只进一条链（remaining 消费）。

### 4.4 已知局限（记录在案）

- 贪心 seed 顺序依赖：全局最优关联（候选图 DP/匈牙利）留给 v4 需要时；
- 两个候选都完美命中外推点时（合成数据病态）tie-break 取先出现者，
  不判歧义；
- 链内 track 自身的检测空洞（如 t64 的 f55-62 缺检）天然保留在
  chain_detections 里，不另行修补；
- 无外观证据（衣着/体型）——几何之外的下一步可选项，明确**不引入
  大型 ReID**（任务纪律）。

### 4.5 测试用例（selftest_e2.py，已通过 20/20）

| 用例 | 断言 |
|---|---|
| 0016 三段碎片（t10/t11/t12） | 合并为一条链，conf ≥ medium |
| **0016 精确交替拓扑**（t70: f12-37,f55-62; t71: f45-54,f63-92） | 一条链 {70,71}，全部原始帧保留、有序无重复 |
| 远距两行人 | 不合并（两条链） |
| 双候选后继 | 链 confidence=ambiguous，rival 被点名且**留在链外** |
| 链证据 | 每 link 记录 gap + 外推误差 + 速度 |

## 5. 状态层：分层横穿证据（详见 crossing_event_semantics.md）

Actor 绑定的输出直接喂给 `evaluate_crossing(tracks, num_frames,
region, chains=...)`：**每条链独立评估 L1-L4**，`summarize()` 只允许
**单条链**满足全部层级才算 completed——D2 的结构性防御。

## 6. 下游判定（v4 adjudicate，待人工真值后设计）

规则草案（不在本轮实现）：
- condition.event_required=false → 不评价该事件（verdict 不受该事件影响）；
- condition.event_required=unknown → 事件失败**不产生 Invalid**，只产生
  Uncertain（协议未确认不计入违约率）；
- state 层 ambiguous 链 / partial_clip_cut / in_progress → Uncertain
  （与 v3.1 归因哲学一致：证据不足宁可 Uncertain）；
- 只有 condition=true 且 state 层有明确 completed/失败的证据才可能
  Valid/Invalid。

## 7. 与 v3.1 的关系

v3.1 输出**保持不变**（本轮零改动其判定链）；本设计的全部模块为并行
证据层。v4-experimental 将在人工标注完成后基于同一 20 条样本 +
Val 集运行对照（见 e2_experiment_plan.md 消融设计）。
