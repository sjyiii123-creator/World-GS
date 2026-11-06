# J2.3.2 候选特征完整性（P0-A 修复）

日期：2026-11-05 | 独立确认：Workflow 四 agent 并行复算
（3,011/555/17/3 与审计 EVIDENCE 完全一致）+ 修复后 skeptic 验证

## 1. 缺陷确认（数值，agent 独立复现）

`run_j2_3_1_baselines.py` 的框检索（只扫 anchor_ped_id==ped 的行、
混扫 anchor+cand frames、缺失静默 continue）：
- **3,011 正确** / **555 缺失**（15.5%——其中 208 个 ped 在该视频
  从未担任 anchor、347 个 anchor 身份在其他帧） / **17 错框**
  （全部来自跨身份行的 cand_frames——含 3 个真主候选：
  video_0214 val / video_0310 val / video_0305 expl——**监督特征
  取错了行人的图像区域**）；至少 291 个 anchor 的候选集被污染。

## 2. 修复（build_fixed_query_v3.py + run_j2_3_2_baselines.py）

- **规范化索引** (source, actor, frame)→(bbox, occ)：从 samples.csv
  **双侧正确归属**构建（anchor→anchor_ped、cand→cand_ped）；
  94,748 唯一观测、0 bbox 冲突、276,690 重复折叠；
- 基线直接读 manifest 的 bbox 列（来源即索引）——**无行扫描、
  无静默跳过**：缺失→feature_coverage 计数 + 示例记录；
- **特征缓存安全**：缓存键含 bbox 哈希——17 个错误框的旧特征
  处于不可达键下（自然失效）；正确框获得全新提取。

## 3. 覆盖率（修复后实测）

**3,597/3,597 候选特征全覆盖（0 缺失）**——按 Split 与 Tier 的
覆盖率 100%（原 555 缺失全部恢复）。

## 4. 回归

两 Actor 同帧共存测试（selftest_j2_1 身份键组）+ 索引 bbox 与
manifest 抽样比对（skeptic agent ≥50 行验证）。
