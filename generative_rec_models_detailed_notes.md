# 生成式推荐模型细节梳理

本文档整理当前已学习的五个模型：

- TIGER
- OneRec
- RankMixer
- UniRec
- JD GenRec

整理口径统一为：

```text
1. 这个模型解决什么问题
2. item / feature / SID 表示怎么来
3. 训练样本怎么构造
4. 模型结构怎么跑
5. 线上推理怎么做
6. 关键模块和容易混淆的点
7. 和其他模型的区别
```

参考论文：

- TIGER: [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- OneRec: [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965)
- RankMixer: [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)
- UniRec: [UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute](https://arxiv.org/abs/2604.12234)
- JD GenRec: [GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation](https://arxiv.org/abs/2604.14878)

---

## 0. 总览

| 模型 | 所在链路 | 核心输入 | 核心输出 | 关键机制 |
| --- | --- | --- | --- | --- |
| TIGER | retrieval / candidate generation | 用户历史 item 的 SID tokens | 下一个 item 的 SID | RQ-VAE Semantic ID + Encoder-Decoder Transformer |
| OneRec | retrieve + rank/session generation | 用户行为序列 SID | high-value session SID 序列 | Encoder-Decoder + MoE + Reward Model + DPO |
| RankMixer | ranking | 用户、候选、序列、交叉特征 tokens | 多目标 ranking scores | Token Mixing + Per-token FFN + Sparse-MoE |
| UniRec | generative retrieval / ranking bridge | 用户序列 + attribute/SID prefix | attribute tokens + SID tokens | CoA + CDC + Gated-CrossAttn + RFT/DPO |
| JD GenRec | industrial generative retrieval | 用户历史 SID prompt | 候选 item SID | Page-wise NTP + Token Merger + GRPO-SR |

最简单的横向理解：

```text
TIGER:
  把召回改成生成 next item SID。

OneRec:
  直接生成一组 high-value session，试图统一 retrieve 和 rank。

RankMixer:
  不生成 item，而是把 ranking 特征交互大模型化。

UniRec:
  先生成 item attributes，再生成 SID，补足 item-side feature crossing。

JD GenRec:
  面向工业线上 generative retrieval，重点解决分页训练、长历史效率和偏好对齐。
```

---

## 1. TIGER

论文：[`Recommender Systems with Generative Retrieval`](https://arxiv.org/abs/2305.05065)

### 1.1 模型定位

TIGER 的目标是把传统推荐召回：

```text
user embedding
  ↓
ANN / nearest neighbor search
  ↓
top-K item embeddings
```

改成：

```text
用户历史 item 序列
  ↓
item 转 Semantic ID token 序列
  ↓
Encoder-Decoder Transformer
  ↓
生成下一个 item 的 Semantic ID
  ↓
Semantic ID 映射回真实 item
```

所以 TIGER 是一个 **生成式召回模型**。它不是完整推荐系统，也不是精排模型。

一句话：

```text
TIGER = Semantic ID tokenizer + Seq2Seq Transformer generative retrieval
```

### 1.2 Content Embedding 怎么来

TIGER 不直接对原始 item id 建模，而是先把 item 内容编码成连续向量。

论文在 Amazon 数据上使用的 item 内容包括：

```text
title
price
brand
category
```

这些字段会被拼成文本，然后输入预训练文本 encoder。论文实验里使用的是 **Sentence-T5**，得到 768 维 content embedding。

流程：

```text
item metadata
  title / price / brand / category
        ↓
拼接成文本
        ↓
Sentence-T5
        ↓
768-d item content embedding
```

这个 embedding 来自 item 内容，不来自用户点击行为。

如果业务里有图片，也可以换成 image encoder 或 multimodal encoder；但 TIGER 论文主实验使用的是文本内容。

### 1.3 RQ-VAE 怎么生成 Semantic ID

拿到 item content embedding 后，TIGER 用 RQ-VAE 把连续向量变成离散 token。

论文配置大致是：

```text
输入: 768-d content embedding
DNN Encoder: 768 → 512 → 256 → 128 → 32
Residual Quantizer: 3 层 codebook
每层 codebook size = 256
code vector dim = 32
DNN Decoder: 重建原 content embedding
```

残差量化过程可以理解为：

```text
z = DNN Encoder(content_embedding)

第 1 层:
  从 codebook_1 找最接近 z 的 code，得到 c1
  residual_1 = z - code_vector(c1)

第 2 层:
  从 codebook_2 找最接近 residual_1 的 code，得到 c2
  residual_2 = residual_1 - code_vector(c2)

第 3 层:
  从 codebook_3 找最接近 residual_2 的 code，得到 c3
```

于是 item 得到：

```text
Semantic ID = (c1, c2, c3)
```

直觉上：

```text
c1 更粗粒度
c2 / c3 更细粒度
```

论文的可视化也观察到：第一层 token 更接近粗类别，后面的 token 更偏细分类。

### 1.4 Collision 怎么处理

多个 item 可能被映射到同一个 `(c1, c2, c3)`。

TIGER 的处理方式是追加第 4 个 token 做区分：

```text
item A → (7, 1, 4)
item B → (7, 1, 4)

变成:

item A → (7, 1, 4, 0)
item B → (7, 1, 4, 1)
```

没有冲突的 item 也补一个 `0`：

```text
item C → (9, 2, 8, 0)
```

所以最终每个 item 是长度为 4 的 Semantic ID：

```text
item_id → (c1, c2, c3, collision_token)
```

离线需要维护两张表：

```text
item_id → semantic_id
semantic_id → item_id
```

这两张表在 RQ-VAE tokenizer 训练完成后生成，并在推荐模型训练和线上推理中使用。

### 1.5 训练样本怎么构造

用户行为先按时间排序：

```text
user_5:
item_515 → item_233 → item_64
```

假设：

```text
item_515 → (5, 25, 78, 0)
item_233 → (5, 23, 55, 0)
item_64  → (5, 25, 55, 0)
```

训练输入是用户历史 item 的 Semantic ID 展平：

```text
[user_token_5,
 5,25,78,0,
 5,23,55,0]
```

训练目标是下一个 item 的 Semantic ID：

```text
[5,25,55,0]
```

论文还加入 user token。为了控制词表大小，原始 user id 会 hash 到一组 user-specific tokens 中。论文实验里使用 2000 个 user tokens。

训练目标是标准自回归 cross-entropy：

```text
P(next_item_sid | user_token + history_sid_tokens)

= P(c1 | context)
  × P(c2 | context, c1)
  × P(c3 | context, c1, c2)
  × P(c4 | context, c1, c2, c3)
```

也就是说，decoder 像语言模型一样一个 token 一个 token 生成目标 item 的 SID。

### 1.6 模型结构

TIGER 使用 T5-style Encoder-Decoder Transformer。

```text
Encoder:
  输入 user token + 历史 SID tokens
  用双向 self-attention 编码用户历史

Decoder:
  从 BOS 开始
  自回归生成下一个 item 的 SID tokens
```

论文实验配置包括：

```text
Encoder layers: 4
Decoder layers: 4
Attention heads: 6
Head dim: 64
Input embedding dim: 128
MLP dim: 1024
Dropout: 0.1
Total params: 约 13M
Batch size: 256
```

词表大致是：

```text
4 个 SID 位置 × 每个位置 256 个 codeword
```

注意：第 1 位的 `5` 和第 2 位的 `5` 通常属于不同位置/不同 codebook 的 token 空间。

### 1.7 线上推理怎么做

线上来了一个用户请求：

```text
用户最近点击:
item_A, item_B, item_C
```

先查表转成 SID：

```text
item_A → (a1,a2,a3,a4)
item_B → (b1,b2,b3,b4)
item_C → (c1,c2,c3,c4)
```

构造 encoder 输入：

```text
[user_hash_token,
 a1,a2,a3,a4,
 b1,b2,b3,b4,
 c1,c2,c3,c4]
```

然后：

```text
Encoder 编码用户历史
Decoder 从 BOS 开始生成
```

生成过程：

```text
step 1: 生成第 1 个 code
step 2: 基于第 1 个 code 生成第 2 个 code
step 3: 基于前 2 个 code 生成第 3 个 code
step 4: 基于前 3 个 code 生成第 4 个 code
```

最后得到候选 Semantic ID，再查表：

```text
semantic_id → item_id
```

得到召回候选。

### 1.8 Beam Search 怎么用

推理时通常不用单步 greedy，而是 beam search。

假设 beam size = 5：

```text
第 1 步:
  保留概率最高的 5 个 c1

第 2 步:
  每个 c1 展开若干 c2
  从所有组合里保留累计 log-prob 最高的 5 个

第 3 步:
  展开 c3

第 4 步:
  展开 c4

得到 top beam 个完整 Semantic ID
```

每条路径分数：

```text
score = log P(c1)
      + log P(c2 | c1)
      + log P(c3 | c1,c2)
      + log P(c4 | c1,c2,c3)
```

SID 长度固定，所以长度归一化不是重点。

生成后做：

```text
完整 SID
  ↓
过滤 invalid SID
  ↓
过滤重复 item
  ↓
sid_to_item 查表
  ↓
top-K candidate items
```

### 1.9 Invalid ID 怎么处理

生成式推荐会遇到非法 SID：

```text
(12, 88, 31, 0)
```

这串 ID 可能在 `semantic_id → item_id` 表里不存在。

原因是 SID 组合空间很大：

```text
256^4
```

但真实 item 只占其中很少一部分。

工程处理：

```text
1. beam size 设大一点
2. 多生成一些 SID
3. 过滤 invalid SID
4. 直到拿够 top-K valid items
```

更强的工程方案是 constrained decoding：把合法 SID 建成 trie，每一步只允许生成当前 prefix 下存在的下一个 token。

### 1.10 Cold-start 怎么做

TIGER 的冷启动能力来自 content embedding。

新 item 没有点击历史也没关系，只要有内容字段：

```text
title / brand / category / price
  ↓
Sentence-T5
  ↓
RQ-VAE
  ↓
Semantic ID
```

如果新 item 和已有 item 内容相似，它们可能共享部分 SID prefix。模型虽然没见过这个具体 item，但可能学过相近 SID 路径。

### 1.11 关键点

```text
TIGER = 用内容语义把 item 编成离散 SID
      + 用 Seq2Seq Transformer 生成下一个 item 的 SID
      + 用 SID 查表还原成 item
```

最重要的不是 Transformer，而是 item 表示从 atomic ID / dense embedding 变成了：

```text
可生成
可共享语义
可层级展开
可冷启动扩展
```

---

## 2. OneRec

论文：[`OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment`](https://arxiv.org/abs/2502.18965)

### 2.1 模型定位

OneRec 比 TIGER 更激进。

TIGER 主要做：

```text
用户历史 → 生成下一个 item SID → 作为召回候选
```

OneRec 想做：

```text
用户历史行为
  ↓
OneRec Encoder-Decoder
  ↓
直接生成一组推荐 session
```

它希望把传统推荐链路：

```text
Recall → Pre-rank → Rank → Re-rank
```

压成：

```text
User behavior sequence
  ↓
OneRec
  ↓
High-value session
```

所以 OneRec 不是生成一个 next item，而是生成一组 item，也就是一个推荐 session。

### 2.2 Video Semantic ID 怎么来

OneRec 也不直接生成原始 video id，而是生成 video 的 Semantic ID。

每个视频先有一个多模态 embedding：

```text
video content / multimodal features
  ↓
pretrained multimodal representation
  ↓
video embedding e_i
```

然后 OneRec 使用 **multi-level balanced residual K-means quantization**。

和 TIGER 的区别：

```text
TIGER:
  Sentence-T5 content embedding + RQ-VAE

OneRec:
  multimodal embedding + balanced residual K-means
```

残差量化过程：

```text
第 1 层:
  r_i^1 = e_i
  s_i^1 = 找离 r_i^1 最近的 cluster centroid

第 2 层:
  r_i^2 = r_i^1 - centroid(s_i^1)
  s_i^2 = 找离 r_i^2 最近的 cluster centroid

第 3 层:
  继续对 residual 做聚类
```

最终：

```text
video_i → (s_i^1, s_i^2, s_i^3)
```

论文实验配置：

```text
codebook layers L = 3
每层 cluster 数 K = 8192
```

图里的：

```text
<a_6><b_1><c_5>
```

就可以理解为某个视频的 3 层 Semantic ID。

### 2.3 Balanced K-means 是什么

普通 K-means 只管距离近，不管每个 cluster 分到多少 item。

OneRec 用 balanced K-means，让每个 cluster 的 item 数接近：

```text
总视频数 |V|
cluster 数 K
每个 cluster 容量 w = |V| / K
```

每轮迭代时，每个 centroid 只接收容量范围内的样本。

目的：

```text
1. 让 code 使用更均匀
2. 避免部分 code 过载
3. 让生成模型更容易覆盖不同语义区域
```

离线也需要维护：

```text
video_id → semantic_id
semantic_id → video_id / video candidates
```

### 2.4 训练样本怎么构造

OneRec 的输入是用户历史行为序列：

```text
H_u = 用户有效观看 / 点赞 / 关注 / 分享过的视频序列
```

论文实验里历史长度：

```text
n = 256
```

每个历史视频转成 SID：

```text
video_A → <a_6><b_1><c_5>
video_B → <a_2><b_1><c_7>
```

Encoder 输入类似：

```text
SEP <a_6><b_1><c_5> SEP <a_2><b_1><c_7> ...
```

训练目标不是一个下一个视频，而是一个 high-value session：

```text
target session:
video_1, video_2, video_3, video_4, video_5

转成:
BOS <a_9><b_7><c_1>
BOS <a_4><b_5><c_4>
...
```

论文实验里 target session size 通常设为：

```text
m = 5
```

这些 high-value session 来自线上日志，例如：

```text
真实观看视频数不少于阈值
session 总观看时长超过阈值
用户有点赞 / 收藏 / 分享 / 关注等互动
```

### 2.5 主模型结构

OneRec 是 T5-like Encoder-Decoder。

```text
Encoder:
用户历史 SID 序列
  ↓
Fully-visible self-attention
  ↓
FFN
  ↓
用户兴趣表示 H

Decoder:
target session SID tokens
  ↓
causal self-attention
  ↓
cross-attention 读取 Encoder 输出 H
  ↓
MoE FFN
  ↓
预测下一个 SID token
```

Encoder 负责理解用户历史兴趣。

Decoder 一边看已经生成的 session prefix，一边通过 cross-attention 读取用户兴趣，然后继续生成后面的推荐视频。

训练 loss 是 next-token prediction：

```text
L_NTP = - log P(target session SID tokens | user history)
```

也就是 teacher forcing：

```text
给 decoder 看前面的 token
让它预测下一个 SID token
```

### 2.6 MoE 在 OneRec 里做什么

OneRec 在 decoder 的 FFN 部分使用 Sparse MoE。

普通 FFN：

```text
每个 token 都过同一个 MLP
```

MoE：

```text
Router 看当前 token hidden state
  ↓
选择少数几个 expert
  ↓
只激活这些 expert
```

论文实验配置中：

```text
experts = 24
top experts = 2
```

作用：

```text
1. 扩大模型总参数量
2. 每次推理只激活部分参数
3. 让不同用户兴趣、视频类型、session 模式走不同 expert
```

直觉：

```text
不同短视频类型和用户偏好模式差异很大，
单一 FFN 容量可能不够。
```

### 2.7 第一阶段训练：NTP Seed Model

第一阶段是监督学习。

流程：

```text
线上日志
  ↓
筛出 high-value session
  ↓
视频转 SID
  ↓
训练 OneRec 预测整个 session
```

目标：

```text
max P(S | H_u)
```

其中：

```text
H_u = 用户历史 SID 序列
S = 高价值 session SID 序列
```

训练完得到 seed model：

```text
M_t
```

这个 seed model 已经能根据用户历史生成“看起来像高价值 session”的推荐列表。

### 2.8 Reward Model 怎么来

OneRec 要做 preference alignment，需要知道候选 session 哪个更好。

所以它训练一个 Reward Model：

```text
R(u, S) → session reward score
```

输入：

```text
用户 u
候选 session S = {v_1, v_2, ..., v_m}
```

Reward Model 会建模 user-session 相关性和 session 内部 item 之间的关系。

大致流程：

```text
user representation
video/session representation
  ↓
target-aware interaction
  ↓
session self-attention
  ↓
multi-tower prediction
```

它预测多个业务目标，例如：

```text
swt: session watch time
vtr: view probability
wtr: follow probability
ltr: like probability
```

可以把 Reward Model 理解成一个 session-level ranking simulator。

### 2.9 IPA / DPO 怎么训练

有了 seed model 和 Reward Model 后，OneRec 做 Iterative Preference Alignment。

对某个用户历史：

```text
当前模型 M_t
  ↓
beam search 生成 N 个候选 session
  ↓
Reward Model 给每个 session 打分
  ↓
最高分 session = chosen
最低分 session = rejected
  ↓
用 DPO 训练 M_{t+1}
```

论文实验里：

```text
N = 128
DPO sample ratio = 1%
```

DPO 的直觉：

```text
让新模型 M_{t+1} 相比旧模型 M_t
更偏向 chosen session
更远离 rejected session
```

通常不会只用 DPO，还会和 NTP/SFT 混合，防止模型偏离真实日志分布。

### 2.10 线上推理怎么做

线上请求：

```text
用户最近 256 条有效行为
  ↓
video_id → SID
  ↓
拼成 Encoder 输入
  ↓
Encoder 得到用户兴趣表示 H
```

Decoder 生成推荐 session：

```text
BOS
  ↓
生成 video_1 的 SID: <a><b><c>
  ↓
生成 video_2 的 SID: <a><b><c>
  ↓
...
  ↓
生成 m 个视频
```

论文线上相关优化包括：

```text
beam search
KV cache decoding
fp16
Sparse MoE
```

生成后：

```text
SID → video_id
去重
合法性过滤
业务过滤
返回推荐 session
```

### 2.11 OneRec 和 TIGER 的区别

```text
TIGER:
  生成 next item SID
  主要替代召回
  point-wise generation

OneRec:
  生成整个 session 的多个 video SID
  目标是统一 retrieve + rank
  list-wise / session-wise generation
  加 Reward Model + DPO 做偏好对齐
```

最重要的理解抓手：

```text
OneRec = SID tokenizer
       + Encoder-Decoder session generator
       + Sparse MoE 扩模型容量
       + Reward Model 选偏好样本
       + DPO / IPA 做生成结果对齐
```

---

## 3. RankMixer

论文：[`RankMixer: Scaling Up Ranking Models in Industrial Recommenders`](https://arxiv.org/abs/2507.15551)

### 3.1 模型定位

RankMixer 不是生成式召回模型。

它是 **ranking stage 的大模型 backbone**。

推荐链路里它大概在这里：

```text
Recall / Retrieve
  ↓
Pre-rank
  ↓
RankMixer ranking model
  ↓
rerank / business rules
  ↓
最终展示
```

输入不是用户历史 SID，输出也不是生成 item。

输入是：

```text
用户特征
候选视频特征
用户行为序列特征
交叉特征
```

输出是：

```text
这个用户对这个候选视频的多目标打分
```

例如：

```text
finish score
skip score
like score
comment score
duration score
ad value score
```

RankMixer 的核心问题：

```text
ranking 里有大量异质特征，
如何做高效、可扩展、GPU 友好的 feature interaction？
```

### 3.2 输入特征怎么来

RankMixer 的一个训练样本通常是线上日志里的一个 impression：

```text
(user, candidate item, context, label)
```

特征大致分为：

```text
User features:
  user id, user profile, user statistics

Candidate / video features:
  video id, author id, category, content embedding, video statistics

Sequence features:
  用户最近看过 / 点过 / 互动过的视频序列
  先经过 sequence module 得到用户兴趣向量

Cross features:
  user 和 candidate 之间的交叉特征
```

这些原始特征先变成 embedding：

```text
ID feature → embedding lookup
numerical feature → normalize / projection
sequence feature → sequence module → sequence embedding
cross feature → embedding / dense feature
```

然后进入 RankMixer 的第一步：feature tokenization。

### 3.3 Feature Tokenization 怎么做

RankMixer 不把每个特征都当一个 token。

原因：

```text
ranking 里可能有几百个特征。
如果一个 feature 一个 token，会有太多小 token，
GPU 利用率差，每个 token 分到的建模容量也太少。
```

它也不把所有特征拼成一个大向量。

原因：

```text
这样会退化成普通 DNN，
不同语义空间被混在一起，
强势特征容易压过弱势特征。
```

RankMixer 采用折中方案：

```text
先按语义把特征分组
  ↓
每组 embedding concat
  ↓
切成 T 个 feature tokens
  ↓
每个 token 投影到 D 维
```

所以输入变成：

```text
X0 ∈ R^{T × D}
```

这里的 token 不是文本 token，也不是 item SID token，而是 **feature group token**。

比如可以理解为：

```text
token 1: user profile / user stats
token 2: candidate video ID / author / category
token 3: sequence interest representation
token 4: user-candidate cross features
...
```

真实系统里具体怎么分组依赖业务特征工程。

### 3.4 RankMixer Block

一个 RankMixer Block：

```text
Input feature tokens X
  ↓
Multi-head Token Mixing
  ↓
Residual + LayerNorm
  ↓
Per-token FFN / Sparse-MoE PFFN
  ↓
Residual + LayerNorm
  ↓
Output feature tokens
```

公式化：

```text
S = LN(TokenMixing(X) + X)

X_next = LN(PFFN(S) + S)
```

堆叠 L 层后：

```text
X_L
  ↓
mean pooling
  ↓
output representation
  ↓
multi-task prediction heads
```

论文里的配置示例：

```text
RankMixer-100M:
  D = 768
  T = 16
  L = 2

RankMixer-1B:
  D = 1536
  T = 32
  L = 2
```

它不是靠堆很多层，而是靠更宽的 token 表示和 PFFN 参数来扩容量。

### 3.5 Token Mixing 怎么混

假设：

```text
T = 4 个 feature tokens
每个 token D = 8 维
H = 4 个 heads
```

每个 token 先切成 4 份：

```text
x1 = [x1_h1, x1_h2, x1_h3, x1_h4]
x2 = [x2_h1, x2_h2, x2_h3, x2_h4]
x3 = [x3_h1, x3_h2, x3_h3, x3_h4]
x4 = [x4_h1, x4_h2, x4_h3, x4_h4]
```

然后按 head 重新拼：

```text
new_token_1 = [x1_h1, x2_h1, x3_h1, x4_h1]
new_token_2 = [x1_h2, x2_h2, x3_h2, x4_h2]
new_token_3 = [x1_h3, x2_h3, x3_h3, x4_h3]
new_token_4 = [x1_h4, x2_h4, x3_h4, x4_h4]
```

每个新 token 都拿到了所有原始 feature token 的一部分信息。

这一步像 reshape / shuffle：

```text
没有 attention score
没有 QK inner product
没有注意力矩阵
```

为什么不用 self-attention？

```text
NLP token 通常在同一个语义空间里，
attention 用内积衡量相似度比较合理。

推荐特征非常异质：
  user id embedding
  video id embedding
  author id embedding
  category feature
  numerical statistics

这些空间之间不一定存在稳定的内积相似度。
```

论文 ablation 中，self-attention 的效果略差，而且 FLOPs 明显更高。

### 3.6 Per-token FFN 是什么

普通 Transformer FFN：

```text
所有 token 共享同一套 FFN 参数
```

RankMixer：

```text
token 1 用 FFN_1
token 2 用 FFN_2
token 3 用 FFN_3
...
```

也就是 Per-token FFN。

对第 t 个 token：

```text
v_t = W2_t · GELU(W1_t · s_t + b1_t) + b2_t
```

每个 token 有独立的 `W1_t / W2_t`。

这么做的原因：

```text
推荐特征的语义差异很大。
user profile、candidate video、sequence interest、cross feature
不应该全部塞进同一个共享 MLP。
```

它和 MMoE 也不一样：

```text
MMoE:
  多个 expert 看的是同一个 input

RankMixer PFFN:
  不同 token 看的是不同 feature subspace
  不同 token 还有自己独立的 FFN 参数
```

直觉：

```text
输入也切开，参数也切开。
```

Token Mixing 负责跨 token 交换信息，PFFN 负责在各自语义空间内做非线性变换。

### 3.7 Sparse-MoE 版本怎么扩容

基础版 RankMixer 是 dense PFFN。

为了继续扩到更大参数，论文把每个 token 的 PFFN 换成 Sparse-MoE。

对第 i 个 token：

```text
s_i
  ↓
router
  ↓
选择若干 expert
  ↓
加权求和得到 v_i
```

普通 Sparse-MoE 常用 top-k routing：

```text
每个 token 固定选 k 个 expert
```

RankMixer 认为这样不适合 ranking。

原因：

```text
不同 token 信息量不同。

高信息 token:
  user / history / cross token
  可能需要更多 expert

低信息 token:
  不需要浪费同样多 expert
```

所以它使用 ReLU Routing：

```text
G_ij = ReLU(router(s_i))
v_i = Σ_j G_ij · expert_ij(s_i)
```

ReLU gate 的好处：

```text
不是固定 top-k，
而是让 token 动态决定激活多少 expert。
```

再加 L1-style regularization 控制平均激活比例。

### 3.8 DTSI-MoE

Sparse-MoE 还有一个问题：expert 可能训练不充分。

有些 expert 很少被路由到，最后变成 under-trained expert。

RankMixer 用：

```text
Dense-training / Sparse-inference
```

即：

```text
训练时:
  尽量让 expert 获得充分梯度更新

推理时:
  只激活少量 expert 控制成本
```

论文里用了两个 router：

```text
h_train
h_infer
```

训练时两个 router 都更新，稀疏正则主要作用在 inference router 上；线上只用 inference router。

直觉：

```text
训练阶段别让 expert 饿死，
推理阶段再省计算。
```

### 3.9 训练时怎么做

RankMixer 是标准 supervised ranking。

训练样本：

```text
用户 u
候选视频 v
上下文 context
用户行为序列 history
反馈 label
```

label 可以是：

```text
finish
skip
like
comment
duration
conversion
```

训练流程：

```text
raw features
  ↓
embedding lookup / numerical projection / sequence module
  ↓
feature tokenization 得到 T × D tokens
  ↓
RankMixer Block × L
  ↓
mean pooling
  ↓
multi-task heads
  ↓
finish / skip / like / duration 等预测
  ↓
supervised loss 更新参数
```

loss 可以理解为工业多目标 ranking loss，例如 BCE / regression loss 的组合。

论文训练系统：

```text
sparse embedding 参数异步更新
dense RankMixer 参数同步更新
dense optimizer: RMSProp, lr = 0.01
sparse optimizer: Adagrad
```

### 3.10 线上推理怎么做

线上没有生成过程。

RankMixer 做 candidate scoring：

```text
召回 / 粗排给出候选视频集合
  ↓
对每个 candidate 构造 user-item-context features
  ↓
查 embedding / 计算 sequence features / cross features
  ↓
tokenization 成 T × D feature tokens
  ↓
RankMixer forward
  ↓
multi-task heads 输出多个分数
  ↓
业务融合成最终 ranking score
  ↓
按 score 排序
```

RankMixer 线上不会做：

```text
beam search
SID decoding
invalid ID filtering
semantic_id → item_id mapping
```

这些是 TIGER / OneRec / GenRec 的问题。

RankMixer 的核心问题是：

```text
如何在低延迟下给大量候选 item 打分。
```

### 3.11 为什么 1B 参数延迟还能稳住

论文线上对比：

```text
OnlineBase:
  15.8M 参数
  107G FLOPs
  MFU 4.47%
  Latency 14.5ms

RankMixer-1B:
  1.1B 参数
  2106G FLOPs
  MFU 44.57%
  Latency 14.3ms
```

参数涨了约 70 倍，但延迟基本不涨。

原因：

```text
1. FLOPs / Param 下降
   参数涨很多，但计算没有同比例上涨。

2. MFU 提升
   RankMixer 主要是大矩阵乘，适合 GPU。
   per-token FFN 可以 fuse 成更大的并行 kernel。

3. fp16 推理
   半精度让理论硬件吞吐提高。
```

RankMixer 的设计把模型从 memory-bound 往 compute-bound 推，让 GPU 利用率显著提升。

### 3.12 关键点

```text
RankMixer = feature tokenization
          + parameter-free Token Mixing
          + Per-token FFN
          + optional Sparse-MoE scaling
          + multi-task ranking heads
```

它不是生成模型，而是 ranking backbone。

---

## 4. UniRec

论文：[`UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute`](https://arxiv.org/abs/2604.12234)

### 4.1 模型定位

UniRec 关注的是生成式推荐和传统判别式 ranker 之间的表达差距。

传统 ranker 可以直接看很多 item-side features：

```text
category
seller
brand
price
shop
historical stats
```

然后做 user-item feature crossing。

普通生成式推荐通常只做：

```text
p(s0, s1, s2 | user)
```

也就是只生成 SID token。

问题：

```text
SID 是对 item 语义的压缩，
传统 ranker 能直接使用的一些 item attributes
可能在生成过程中变成隐变量。
```

UniRec 的解决方案：

```text
先生成 item attributes
再生成 item Semantic ID
```

也就是 **Chain-of-Attribute, CoA**。

目标从：

```text
s0, s1, s2
```

变成：

```text
a1, a2, ..., am, s0, s1, s2
```

### 4.2 Capacity-constrained SID 怎么构造

UniRec 仍然先把 item 映射成 SID。

输入：

```text
item text / image / multimodal content
  ↓
multimodal embedding
  ↓
residual quantization
  ↓
SID = (s0, s1, s2)
```

但 UniRec 不用普通 RQ-KMeans，而是 **Capacity-constrained SID**。

普通 RQ-KMeans 可能做到 item count 均衡，但推荐系统里流量是长尾的：

```text
cluster A:
  1000 个 item，全是高曝光爆品

cluster B:
  1000 个 item，全是长尾商品
```

item 数一样，但训练曝光量完全不一样。

所以 UniRec 平衡的是 exposure load。

给每个 item 一个曝光权重：

```text
w_i = item_i 的历史曝光量 / traffic weight
```

每个 cluster 的负载：

```text
V_k = sum(w_i), for items assigned to cluster k
```

约束：

```text
V_k ≤ τ * C_cap
```

其中：

```text
C_cap = 所有曝光量 / cluster 数
τ = 容忍系数
```

论文实验配置：

```text
SID 层数 L = 3
每层 codebook size K = 4000
τ = 1.05
```

构造流程：

```text
第 0 层:
  对 item embedding 聚类，得到 s0

第 1 层:
  residual = embedding - centroid(s0)
  对 residual 聚类，得到 s1

第 2 层:
  继续对 residual 聚类，得到 s2

每层聚类时:
  先按最近 centroid 分配
  如果某个 cluster 曝光负载超 cap
  就把部分 item 修复到最近的未满 cluster
```

目的：

```text
避免一小批热门 SID path 吃掉大部分训练曝光，
缓解 token path collapse 和 Matthew effect。
```

### 4.3 Attribute token 和 attribute embedding 怎么来

UniRec 的 attribute 不是从 RQ-KMeans 量化出来的。

它来自 item 本身的结构化属性表，例如：

```text
category L2
category L3
seller
brand
shop
```

论文实验主要使用商品类目层级：

```text
L2 category → L3 category
```

比如：

```text
item_123:
  L2 category = Beauty
  L3 category = Face Cream
  SID = (s0, s1, s2)
```

训练 target：

```text
[BOS, L2_Beauty, L3_Face_Cream, s0, s1, s2]
```

Attribute token ID 来自 metadata vocab：

```text
L2 category vocab:
  Beauty → 31
  Electronics → 52

L3 category vocab:
  Face Cream → 712
  Lipstick → 845
```

Attribute embedding 是普通 embedding lookup：

```text
attribute token id
  ↓
embedding table lookup
  ↓
attribute embedding
```

这些 embedding 是模型训练时学出来的参数。

区分一下：

```text
SID:
  从 item multimodal embedding 量化来

Attribute token:
  从 item metadata / structured feature 来

Attribute embedding:
  attribute token id 查 embedding table 得到
```

### 4.4 CoA 为什么有效

普通生成式推荐：

```text
p(s0, s1, s2 | user)
```

UniRec：

```text
p(a1, a2, ..., am | user)
× p(s0, s1, s2 | user, a1, a2, ..., am)
```

直觉：

```text
先确定要生成什么属性的商品，
再在这个属性条件下生成具体 SID。
```

好处：

```text
1. 补回 item-side feature crossing
   让生成模型更接近传统 ranker 的能力。

2. 缩小 beam search 空间
   category / brand / seller 确定后，
   后续 SID 的不确定性下降。

3. 生成路径更有语义
   不再只是抽象 code，而是先走可解释属性链。
```

论文实验中：

```text
Direct SID:
  s0, s1, s2

CoA:
  L2 → L3 → s0 → s1 → s2
```

CoA 明显提高了 token hit ratio 和 beam search hit ratio。

### 4.5 CDC: Conditional Decoding Context

CDC 有两部分：

```text
1. Task-Conditioned BOS
2. Content Summary
```

#### 4.5.1 Task-Conditioned BOS

普通生成模型使用固定：

```text
<BOS>
```

UniRec 把 BOS 变成和任务/场景相关的 embedding：

```text
<BOS_click_mainfeed>
<BOS_purchase_search>
<BOS_cart_similar_items>
```

电商推荐里不同目标差异很大：

```text
click:
  关注即时兴趣

purchase:
  关注转化意图

cart:
  关注购买准备

main feed:
  偏探索

search:
  偏 query relevance
```

Task-Conditioned BOS 的作用：

```text
告诉模型这次是在什么场景下，
为哪个业务目标生成 item。
```

#### 4.5.2 Content Summary

生成到某一步时，模型已经有 prefix：

```text
L2, L3, s0, s1
```

单个 token embedding 只能表示单个 token，但很多语义来自组合：

```text
(L2, s0)
(L3, s1)
(s0, s1)
```

如果显式建所有组合表，参数会爆炸：

```text
4000 × 4000 = 16M
```

所以 UniRec 用 hash trick：

```text
已有 prefix path
  ↓
多个 hash 函数
  ↓
查共享 hash embedding table
  ↓
拼成 content summary
```

论文实验里用了：

```text
M = 3 个 hash 函数
d_hash = 64
```

组合包括：

```text
(L2, s0)
(L2, s1)
(L3, s0)
(L3, s1)
(s0, s1)
```

Content Summary 是给 decoder 一个“当前生成路径的组合语义摘要”。

### 4.6 模型主干

UniRec 使用：

```text
Decoder-Only backbone + Cross-Attention to user behavior sequence
```

模型输入有三类：

```text
1. User static profile
   user id, demographics, context features

2. Behavior sequence
   用户按时间排序的点击 / 浏览行为
   每个行为包含 item, shop, category 等特征

3. SID-level multimodal features
   由多层 SID 对应的多模态特征组成
```

用户行为序列处理成：

```text
H_seq = [h1, h2, ..., hT]
```

论文实验配置：

```text
behavior sequence length T = 200
multimodal SID sequence length L_mm = 100
model dim = 256
cross-attention layers = 3
attention heads = 8
```

Decoder 侧 query 是：

```text
Task-BOS, attributes, s0, s1, s2
```

Cross-attention：

```text
Query = 当前 decoding path
Key / Value = 用户行为序列 H_seq
```

也就是每一步生成 attribute / SID token 时，都可以读取用户历史行为。

### 4.7 Gated-CrossAttn

普通 cross-attention：

```text
Attention(Q, K, V)
```

UniRec 加一个可学习 gate：

```text
GatedCrossAttn = γ * Attention(Q, K, V)
```

`γ` 控制用户行为上下文注入 decoder 的强度。

直觉：

```text
有些场景强依赖用户最近行为，
有些场景更依赖当前任务或商品属性。
gate 可以让模型自己调节。
```

Cross-attention 后接：

```text
MMoE-FFN
```

论文实验里：

```text
MMoE experts = 4
activation = SwiGLU
hidden dim = 4 * d_model
```

### 4.8 Hierarchical Rank Head

UniRec 不用一个统一 LM head 预测所有 token。

它为每个 decoding step 配一个独立 Rank Head。

第 t 步输入包括：

```text
1. 当前 cross-attention 输出 q_t
2. 已生成 prefix 的 embedding
3. Content Summary c_t
4. 用户聚合表示 h_agg
```

拼起来：

```text
x_t = [q_t, prefix_embedding, content_summary, h_agg]
```

然后过 SENet / MaskNet 风格的 rank head，输出当前 step 的 softmax：

```text
p(token_t | prefix, user, task)
```

每一步词表不同：

```text
预测 attribute 时:
  词表是 category / seller / brand 等属性域

预测 SID 时:
  词表是当前 SID layer 的 codebook
```

所以叫 Hierarchical Rank Head。

### 4.9 训练时怎么做

基础训练是 teacher forcing 的 next-token prediction。

目标 item 表示成：

```text
target = (a1, ..., am, s0, s1, s2)
```

训练时：

```text
输入 prefix: BOS
预测: a1

输入 prefix: BOS, a1
预测: a2

输入 prefix: BOS, a1, a2
预测: s0

输入 prefix: BOS, a1, a2, s0
预测: s1

输入 prefix: BOS, a1, a2, s0, s1
预测: s2
```

loss：

```text
L_NTP = - Σ log p(target_token_t | target_prefix, user, task)
```

论文中还用 engagement 权重，例如 click / conversion 样本权重更高。

优化器：

```text
AdamW
learning rate = 3e-4
```

### 4.10 RFT: Reward-Driven Fine-tuning

NTP 学的是曝光日志分布，但业务更关心：

```text
GMV
purchase
watch time
conversion
```

所以 UniRec 加 RFT。

对每个训练样本算业务 reward：

```text
R(u_i, x_i)
```

论文实验中用 GMV 作为业务 reward 信号。

然后在 batch 内做 advantage normalization：

```text
reward 高于 batch 平均 → 权重变大
reward 低于 batch 平均 → 权重变小
```

训练目标变成：

```text
高价值样本的 NTP loss 权重大
低价值样本的 NTP loss 权重小
```

直觉：

```text
不只是学用户看过什么，
而是更用力地学哪些曝光带来了更高业务价值。
```

### 4.11 DPO 偏好对齐

UniRec 还用 DPO 做偏好优化。

同一个 request 下，可能曝光了多个 item。根据用户行为定义偏好等级：

```text
purchase = 2
click = 1
exposure only = 0
```

构造偏好对：

```text
purchased item > clicked item
clicked item > exposure-only item
```

DPO 目标：

```text
让当前模型相对 reference model
提高 preferred item 的生成概率
降低 rejected item 的生成概率
```

UniRec 的 DPO 更偏 item-level / request-level preference pair。

它和 OneRec 的区别：

```text
OneRec:
  生成多个 session
  用 reward model 选 chosen / rejected session

UniRec:
  在同一个 request 的曝光 item 中
  根据 purchase / click / exposure 构造 item-level preference pair
```

### 4.12 Layer-wise Stop Gradient

UniRec 的 DPO 还有一个重要技巧：Layer-wise Stop Gradient。

原因：

```text
前面的 attribute / s0 / s1 是后续生成的条件。
如果 DPO 把前缀层改乱，整个解码路径会不稳定。
```

所以 UniRec 让 DPO 主要更新最后一层 SID rank head：

```text
attribute / early SID prefix:
  stop gradient

final SID layer:
  allow gradient
```

直觉：

```text
别让偏好优化把前面的属性和粗 SID 生成搞崩；
只在更细粒度的最后 SID 层做偏好拉开。
```

最终 alignment loss：

```text
L = L_RFT + λ_DPO * L_DPO
```

论文实验配置：

```text
λ_RFT : λ_DPO = 20 : 3
DPO β = 0.1
```

### 4.13 线上推理怎么做

线上请求包括：

```text
用户静态特征
用户最近行为序列
当前场景 / 任务目标
```

先构造：

```text
H_seq = 用户行为序列表示
h_agg = 用户聚合表示
Task-Conditioned BOS = 根据场景和目标选 BOS embedding
```

然后自回归生成：

```text
step 1: 生成 L2 category
step 2: 生成 L3 category
step 3: 生成 s0
step 4: 生成 s1
step 5: 生成 s2
```

每一步都用：

```text
Gated-CrossAttn 读取用户行为
Content Summary 注入 prefix 组合语义
Hierarchical Rank Head 输出当前层 token 概率
```

beam search：

```text
BOS
  ↓
保留 top beam 个 L2
  ↓
展开 L3，保留 top beam 条路径
  ↓
展开 s0
  ↓
展开 s1
  ↓
展开 s2
  ↓
得到 top-K attribute + SID paths
```

最后：

```text
SID path
  ↓
sid_to_item
  ↓
合法性过滤 / 去重 / 业务规则
  ↓
推荐候选
```

### 4.14 和 TIGER / OneRec 的区别

```text
TIGER:
  生成 next item SID
  重点是 Semantic ID + Seq2Seq retrieval

OneRec:
  生成 high-value session
  重点是 session-wise generation + MoE + DPO

UniRec:
  先生成 item attributes，再生成 SID
  重点是补足生成式推荐缺少 item-side feature crossing 的问题
```

核心抓手：

```text
UniRec = Capacity-constrained SID
       + Chain-of-Attribute
       + Task-Conditioned BOS
       + Content Summary
       + Gated-CrossAttn decoder
       + Hierarchical Rank Head
       + RFT / DPO alignment
```

---

## 5. JD GenRec

论文：[`GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation`](https://arxiv.org/abs/2604.14878)

### 5.1 模型定位

JD GenRec 仍然是 **generative retrieval**，不是 RankMixer 那种精排模型。

推荐链路中大概是：

```text
用户历史
  ↓
GenRec 生成候选 item SID
  ↓
SID 映射回 item
  ↓
进入后续 rank / rerank
```

它和 TIGER 很像：

```text
用户历史 SID → 生成目标 item SID
```

但 GenRec 更关注工业线上落地问题：

```text
1. Page-wise NTP
   分页推荐里，同一个用户历史可能对应多个正反馈 item。

2. Token Merger
   用户历史很长，每个 item 又是多个 SID token，prefill 成本太高。

3. GRPO-SR
   用 RL 做偏好对齐，同时防止 reward hacking。
```

一句话：

```text
GenRec = decoder-only generative retrieval
       + Page-wise NTP
       + prompt-side Token Merger
       + GRPO-SR preference alignment
```

### 5.2 Semantic ID 怎么来

GenRec 不生成原始 item id，而是生成 item Semantic ID。

离线流程：

```text
item image + title + description
  ↓
multimodal encoder
  ↓
item dense embedding
  ↓
recommendation-oriented fine-tuning
  ↓
RQ K-means
  ↓
SID(item) = (s1, s2, s3)
```

论文主文提到使用 Qwen2.5-VL 这类多模态模型，把商品图片和文本描述编码成连续向量。

因为通用多模态 embedding 只知道视觉/文本相似，不一定知道推荐相关性，GenRec 会用推荐领域的 collaborative pairs 继续 fine-tune embedding model，让表示更贴近：

```text
共点
共购
相似偏好
推荐语义
```

然后用 RQ K-means 做残差量化：

```text
第 1 层:
  对 item embedding 聚类，得到 s1

第 2 层:
  对 residual 再聚类，得到 s2

第 3 层:
  继续对 residual 聚类，得到 s3
```

最终：

```text
item_i → SID(item_i) = [s_i^1, s_i^2, s_i^3]
```

### 5.3 模型主干：Decoder-only Transformer

TIGER 是 encoder-decoder，GenRec 使用 decoder-only Transformer。

这让它更容易复用 LLM 推理优化：

```text
KV cache
prefill / decode 分离
batch decoding
large language model serving stack
```

用户历史：

```text
H = [v1, v2, ..., vn]
```

转成 prompt：

```text
S_u =
SID(v1), <sep>,
SID(v2), <sep>,
...
SID(vn)
```

如果每个 item 有 3 个 SID token：

```text
item_1 = [s1, s2, s3]
```

那么 100 个历史 item 就变成 300+ 个 token。

这正是 Token Merger 要解决的问题。

### 5.4 Page-wise NTP

普通 point-wise NTP：

```text
Input:
用户历史 H

Target:
下一个 item
```

也就是：

```text
H → item_a
```

但工业推荐是分页请求。用户一次看到一页商品，在同一页里可能：

```text
曝光 item_1
点击 item_2
下单 item_3
点击 item_4
```

如果做 point-wise 训练，会变成多个样本：

```text
H → item_2
H → item_3
H → item_4
```

问题：

```text
同一个输入 H，
对应多个正确答案。
```

这会导致 one-to-many ambiguity：

```text
模型不知道应该把概率集中给谁，
最后概率质量被摊薄。
```

GenRec 把 target 从单个 item 改成整页 item 序列：

```text
Input:
用户历史 H

Target:
Y_page = ordered items + clicked items + exposed items
```

并按交互强度排序：

```text
下单 item
  ↓
点击 item
  ↓
曝光 item
```

公式化：

```text
Y_page = [SID(v): v ∈ O ∪ C ∪ E] 按交互强度排序
```

训练目标仍然是标准 next-token prediction：

```text
L_SFT = - Σ_t log Pθ(y_t | S_u, y_<t)
```

区别在于：

```text
y_t 来自整页 target sequence，
而不是单个 next item。
```

好处：

```text
1. 解决同一个输入对应多个正反馈 item 的冲突
2. 一个 forward pass 监督多个 item，梯度更密集
3. 保留 page 内部多个 item 的相对关系
4. 更贴近工业分页推荐场景
```

### 5.5 Token Merger

SID 的缺点：

```text
一个 item 不再是一个 token，
而是多个 token。
```

例如：

```text
item_i → [s_i^1, s_i^2, s_i^3]
```

如果用户历史有 100 个 item：

```text
原始 item 序列长度:
100

SID token 序列长度:
300 + separators
```

Decoder-only Transformer 的 prefill 成本会显著增加。

GenRec 在 prompt side 引入 Linear Token Merger。

对一个 item 的三个 SID token embedding：

```text
e(s_i^1), e(s_i^2), e(s_i^3)
```

先 concat：

```text
Concat(e(s_i^1), e(s_i^2), e(s_i^3))
```

再过 linear layer：

```text
h_{v_i} = Linear(Concat(e(s_i^1), e(s_i^2), e(s_i^3)))
```

于是：

```text
[s_i^1, s_i^2, s_i^3]
  ↓
compressed item_i
```

输入侧从：

```text
s1, s2, s3, <sep>, s1, s2, s3, <sep>
```

变成：

```text
compressed_item_1, <sep>, compressed_item_2, <sep>
```

论文说这样能把 prompt 长度降低约 2 倍。

### 5.6 为什么叫 asymmetric representation

Token Merger 只压缩 prompt / prefill side，不压缩输出侧。

```text
Prompt / prefill:
  [s1, s2, s3] → compressed item vector

Training target / decoding:
  仍然生成 [s1, s2, s3]
```

为什么不能输出也压缩？

```text
如果输出是 compressed vector，
模型就没法通过 LM Head 做 SID token softmax，
也没法自然 beam search 生成合法 item ID。
```

所以 GenRec 保留输出侧 full-resolution SID decoding：

```text
输入侧为了效率压缩，
输出侧为了可生成性不压缩。
```

### 5.7 线上推理怎么做

训练是 page-wise：

```text
用户历史 → 生成整页 target sequence
```

线上是 point-wise beam search：

```text
用户历史
  ↓
历史 item 转 SID
  ↓
Token Merger 压缩 prompt
  ↓
decoder-only Transformer
  ↓
beam search 生成若干 item SID
  ↓
SID → item_id
  ↓
过滤 invalid / duplicate / 业务规则
  ↓
召回候选
```

如果每个 item SID 长度是 3，beam search 过程：

```text
step 1:
  生成 top-K 个 s1

step 2:
  展开每个 s1，生成 s2，保留 top-K paths

step 3:
  展开 s3，得到完整 SID

最后:
  查 sid_to_item
```

生成的 SID 如果不存在，就是 hallucination / invalid SID。

论文评估中使用 HaR：

```text
Hallucination Rate = invalid SID 占比
```

### 5.8 GRPO-SR 为什么需要

Page-wise SFT 本质还是监督学习。

它学的是：

```text
历史日志里用户和页面怎么互动。
```

线上目标更复杂：

```text
点击
下单
满意度
长期价值
相关性
```

如果直接用 RL 根据 reward model 优化，容易 reward hacking。

例如模型生成：

```text
SID 合法
reward model 给分不低
但和用户真实兴趣无关
```

GenRec 提出：

```text
GRPO-SR = Group Relative Policy Optimization
        + Supervised Regularization
```

### 5.9 Hybrid Reward

对同一个用户 prompt，当前 policy 生成一组候选：

```text
o_1, o_2, ..., o_G
```

每个候选 item 会被 reward model 打分。

GenRec 不直接使用稀疏点击/下单信号，而是用 SIM-based dense preference model：

```text
r_pref_i ∈ [0, 1]
```

但 dense reward model 可能被钻空子，所以加 relevance gate：

```text
G_i = 1 if relevance_score_i > τ
G_i = 0 otherwise
```

最终 hybrid reward：

```text
r_i = G_i * r_pref_i
```

如果候选和用户语义不相关，即使 reward model 给了分，也会被 gate 压掉。

还有 reward calibration。

如果生成候选命中真实页面里的正反馈 item：

```text
D+ = ordered items ∪ clicked items
```

它的 reward 会被锚定到组内最高 reward：

```text
positive hit → r_max
```

这样可以避免 reward model 低估真实正反馈 item。

### 5.10 GRPO-SR 怎么训练

GRPO 的思路：

```text
不要看单个候选的绝对 reward，
而是在同一组候选里做相对比较。
```

流程：

```text
用户 prompt S_u
  ↓
当前模型 rollout 生成 G 个候选 item
  ↓
每个候选算 hybrid reward
  ↓
组内标准化，得到 relative advantage
  ↓
提高高 advantage 候选的概率
  ↓
降低低 advantage 候选的概率
```

GenRec 的 RL 阶段和线上保持一致：

```text
point-wise rollout
```

也就是一次生成一个 item SID，而不是生成整页。

因此有一个不对称：

```text
SFT:
  page-wise target，训练更密集

RL:
  point-wise rollout，贴近线上 beam search serving
```

### 5.11 NLL regularization

纯 RL 容易把模型推偏。

尤其 reward model 有噪声时，模型可能越来越不像真实用户行为分布。

所以 GRPO-SR 加 NLL regularization：

```text
对真实正反馈 item D+
继续做 NLL 约束
```

直觉：

```text
GRPO:
  往 reward 更高的候选方向推

NLL regularization:
  别忘了真实用户点过 / 买过什么
```

它和普通 RLHF 的 KL penalty 不完全一样。

GenRec 用 NLL 直接锚定真实用户正反馈轨迹，而不是只约束别离 reference model 太远。

### 5.12 训练配置和实验信息

论文数据来自 JD.com 大规模推荐平台：

```text
约 560M 用户交互序列
一个月数据
最后一天测试，其余训练
```

模型 backbone：

```text
Qwen2.5 decoder-only variants
1.5B / 3B / 7B
```

训练：

```text
8 × NVIDIA H100
AdamW
前 1% steps linear warmup
之后 cosine decay
```

论文发现：

```text
1.5B → 3B 提升明显
3B → 7B 收益变小
```

线上 A/B：

```text
Base SFT:
  click count +8.5%
  transaction count +7.3%

GRPO-SR alignment:
  click count +9.5%
  transaction count +8.7%
```

论文称 GenRec with GRPO-SR 已部署到 JD 首页信息流生产流量。

### 5.13 和 TIGER / OneRec / UniRec 的区别

```text
TIGER:
  encoder-decoder
  point-wise next item SID
  重点是 Semantic ID + generative retrieval

OneRec:
  encoder-decoder / session generator
  生成 high-value session
  用 DPO 做偏好对齐

UniRec:
  先生成 attribute，再生成 SID
  重点补 item-side feature crossing

GenRec:
  decoder-only
  Page-wise NTP 训练
  Token Merger 降低长历史 prefill 成本
  GRPO-SR 做偏好对齐
  线上仍保持 point-wise beam search retrieval
```

核心抓手：

```text
GenRec = multimodal SID
       + decoder-only generative retrieval
       + Page-wise NTP
       + prompt-side Token Merger
       + point-wise beam search serving
       + GRPO-SR preference alignment
```

---

## 6. 横向理解

### 6.1 它们分别改写推荐链路的哪里

```text
TIGER:
  改写 retrieval。
  从 ANN 检索变成生成 next item SID。

OneRec:
  改写 retrieve + rank/session。
  从用户历史直接生成 high-value session。

RankMixer:
  改写 ranking backbone。
  把 ranking 特征 token 化，用 Token Mixing + PFFN 做大规模特征交互。

UniRec:
  改写 generative retrieval 的 item-side 表达。
  用 CoA 先生成属性，再生成 SID，补足 feature crossing。

JD GenRec:
  改写工业 generative retrieval 的训练与对齐。
  用 Page-wise NTP、Token Merger、GRPO-SR 处理生产问题。
```

### 6.2 训练和推理的不对称

几个模型里有很多“不对称”设计。

TIGER：

```text
训练:
  teacher forcing 生成 next item SID

推理:
  beam search 生成多个 SID candidates
```

OneRec：

```text
训练:
  先 NTP 训练 high-value session generator
  再 DPO 做偏好对齐

推理:
  beam search 生成 session
```

RankMixer：

```text
训练:
  supervised multi-task ranking loss

推理:
  对候选 item 打分

没有生成式 decoding。
```

UniRec：

```text
训练:
  teacher forcing 生成 attribute + SID
  RFT / DPO 对齐业务目标

推理:
  beam search 先生成 attributes，再生成 SID
```

JD GenRec：

```text
SFT 训练:
  page-wise target

RL 训练:
  point-wise rollout

线上推理:
  point-wise beam search
```

### 6.3 它们的 SID 设计差异

```text
TIGER:
  Sentence-T5 content embedding
  RQ-VAE
  3 层 code + collision token

OneRec:
  multimodal embedding
  balanced residual K-means
  3 层 SID

UniRec:
  multimodal embedding
  capacity-constrained residual quantization
  平衡 exposure load，缓解 Matthew effect

JD GenRec:
  multimodal encoder + recommendation-oriented fine-tuning
  RQ K-means
  3 层 SID
```

### 6.4 Beam Search 在不同模型中的角色

```text
TIGER:
  beam search 生成 next item SID candidates。

OneRec:
  beam search 生成多个 candidate sessions，
  也用于 IPA 阶段给 reward model 选 chosen/rejected。

UniRec:
  beam search 生成 attribute + SID path，
  CoA 用来让搜索路径更稳定。

JD GenRec:
  线上 point-wise beam search 生成 item SID candidates，
  RL 阶段也对齐这个 point-wise serving protocol。

RankMixer:
  不用 beam search。
```

### 6.5 Preference Alignment 对比

```text
OneRec:
  Reward Model 给生成的 session 打分
  选择 chosen / rejected
  用 DPO 对齐 session generator

UniRec:
  RFT 用业务价值重加权 NTP
  DPO 用同一 request 内 purchase/click/exposure 构造偏好对
  Layer-wise stop gradient 稳定 prefix

JD GenRec:
  GRPO-SR
  组内相对 reward 优化
  relevance gate 防 reward hacking
  NLL regularization 锚定真实正反馈

TIGER:
  原始模型主要是监督式 generative retrieval，
  没有复杂 preference alignment。

RankMixer:
  本身是 ranking supervised learning，
  不属于生成式偏好对齐。
```

### 6.6 最适合的学习顺序

```text
1. TIGER
   先理解 Semantic ID + generative retrieval 的基本范式。

2. OneRec
   看从 next-item generation 走向 session-wise generation。

3. UniRec
   看生成 SID 时如何补回 item-side attributes 和 feature crossing。

4. JD GenRec
   看工业线上如何处理 page-wise training、prompt compression 和 RL alignment。

5. RankMixer
   反过来看即使有生成式召回，ranking stage 仍然需要可扩展的大模型 backbone。
```

### 6.7 一句话收束

```text
TIGER 解决“item 怎么生成”。
OneRec 解决“session 能不能直接生成”。
UniRec 解决“生成 SID 会不会丢 item attributes”。
JD GenRec 解决“生成式召回怎么工业化上线”。
RankMixer 解决“候选来了之后怎么高效精排”。
```

