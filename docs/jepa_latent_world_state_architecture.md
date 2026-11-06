# JEPA 潜在世界状态模块架构（第五/六/七优先级）

日期：2026-11-05 | 代码：`src/wmeval/latent_world_state/`
（六文件 + 12/12 单元测试；**零新训练**，Adapter 复用既有实现）

## 1. 模块清单与职责

| 文件 | 职责 | 复用/新增 |
|---|---|---|
| `encoder_adapter.py` | 冻结视觉特征 + **模型溯源**（provenance：model_id/映射版本/裁剪配置/frozen 标志） | **委托** `latent_features.LatentFeatureExtractor`（单一实现、单一缓存） |
| `actor_tokenizer.py` | Actor 因果观测组织（帧/时间戳/bbox/可见性/特征，identity=(source,actor)） | 新增数据类 |
| `temporal_predictor.py` | 由（因果历史 + 指定 Δ）输出未来潜在特征；`predict_is_causal` 守卫 | **包装**既有 GRU checkpoint |
| `association_evidence.py` | 身份延续证据：identity mask（含重复 actor）、**每候选 delta**、evidence quality、kin 主排序 + 歧义 | 新增纯函数 |
| `event_state_evidence.py` | **接口only**：waiting/entering/crossing/completed 状态契约；`labels_available()=False` 如实报告 | 新增 no-op |
| `interfaces.py` | 模块接线 + **off-by-default 保证**：`LatentEvidenceSwitch(enabled=False)` 时链 links 原样返回、`can_affect_event_verdict()` 恒 False | 新增开关 |

## 2. 两类状态的分离（任务 §七）

| | identity_state | event_state |
|---|---|---|
| 问题 | 是否同一交通参与者 | 等待/进入/横穿进度/完成 |
| 证据 | 观测特征 + 预测特征比对 | **需可靠人工/官方行为标签**（未具备） |
| 指标 | 区分间距 / 竞争集 top-1 | 事件 P/R（未运行） |
| 本轮状态 | 已实现（模块化） | **接口与协议 only** |

**不混淆禁令的代码化**：普通余弦相似度只出现在
association_evidence 的证据列——event_state 模块没有任何余弦
判定路径（no-op 返回 unknown）。

## 3. 四条禁止混淆的落位（任务 §六）

1. 预测目标存在 ≠ 观察到目标——TemporalPredictor 输出仅作
   CandidateEvidence.predicted_similarity（证据列），EventVerifier
   仍只消费真实观测；
2. 潜在状态连续 ≠ 身份确认——association 决策含 ambiguity 门
   （top-2 差 < margin → uncertain）；
3. 预测误差大 ≠ 生成视频不安全——无任何路径将预测误差映射为
   安全结论；
4. 身份正确 ≠ 完成横穿——crossing_evidence 的 L1-L4 判定不读
   身份证据。

## 4. 生成式接口核查（第八优先级，引用既有审计）

- Generator（Wan/Cosmos/LTX/RayNova）**零接触**（未修改任何生成
  器/Checkpoint）；
- SynPeDAS 生成条件范围：**已由 E2.1 条件审计确立**
  （`docs/synpedas_conditioning_audit.md`：
  `likely_first_~16_frames_replayed_then_77_generated`）——
  生成合同不含"复制源视频后段"；
- **conditioning_scope 已确认**（上述结论），故"严格的未来预测
  实验"表述成立的前提是：预测目标是**生成 roll-out 内的观测**，
  而非源视频后段——J 系列全部实验在真实侧（JAAD 观测）构建，
  迁移到生成侧的前置条件（独立 actor/事件标签）未满足，接口
  已预留、未宣称运行。

## 5. 开关一致性回归（验收问题 6/7）

- `LatentEvidenceSwitch()` 默认 off：链 links **原样返回**
  （identity 相等断言）+ `can_affect_event_verdict()=False`；
- legacy 套件（selftest_metrics 35 + selftest_e2 20）全绿——
  **旧事件验证器在关闭 J2 时输出完全一致**（模块零侵入）。
