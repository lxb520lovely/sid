# 生成式推荐模型学习笔记

这份笔记围绕五个模型/框架展开：

- TIGER
- HSTU / Generative Recommenders
- OneRec
- RankMixer
- UniRec


---

## 1. TIGER

![TIGER 模型图](tiger.png)

### 1.1 模型定位

TIGER 的全称可以理解为 **Transformer Index for Generative Recommenders**。它主要解决的是推荐系统里的 **召回 / candidate generation** 问题。

传统召回通常是：

```text
user embedding
    ↓
ANN / nearest neighbor search
    ↓
top-K item embeddings
```

TIGER 换了一种思路：

```text
用户历史 item 序列
    ↓
转成 Semantic ID token 序列
    ↓
Encoder-Decoder Transformer
    ↓
生成下一个 item 的 Semantic ID
    ↓
Semantic ID 映射回真实 item
```

所以 TIGER 的核心不是做排序，而是把召回改写成一个 **生成 item Semantic ID** 的问题。

一句话：

> TIGER = Semantic ID tokenizer + Seq2Seq Transformer generative retrieval。

### 1.2 模型结构

TIGER 可以拆成两个模块：

```text
1. Semantic ID Generation
   item content → content encoder → embedding → quantization → Semantic ID

2. Generative Retrieval Model
   user history Semantic IDs → Transformer Encoder-Decoder → next item Semantic ID
```

第一个模块负责给 item 建立可生成的离散 ID。

第二个模块负责根据用户历史生成下一个 item 的 ID。

### 1.3 输入输出

训练阶段输入：

```text
用户历史交互序列:
u, item_1, item_2, ..., item_t

每个 item 的 Semantic ID:
item_i → (c1, c2, c3)
```

训练目标：

```text
预测下一个 item 的 Semantic ID:
item_{t+1} → (c1, c2, c3)
```

推理阶段输入：

```text
用户当前历史行为序列
```

推理阶段输出：

```text
候选 Semantic ID
    ↓
映射回真实 item
    ↓
召回候选列表
```

### 1.4 数据流

完整数据流可以分成三段。

第一段，离线构造 Semantic ID：

```text
Item metadata / content
        ↓
Content Encoder
        ↓
Item embedding
        ↓
Quantization
        ↓
Semantic ID table
        ↓
item_id ↔ semantic_id
```

第二段，训练生成式召回模型：

```text
User behavior sequence
        ↓
Replace each item with Semantic ID tokens
        ↓
Bidirectional Transformer Encoder
        ↓
Encoded context
        ↓
Transformer Decoder
        ↓
Predict next item Semantic ID
        ↓
Cross-entropy loss over generated code tokens
```

第三段，线上召回：

```text
User recent behavior sequence
        ↓
Convert history items to Semantic ID tokens
        ↓
Encoder encodes user history
        ↓
Decoder generates candidate Semantic IDs
        ↓
Beam search / constrained decoding
        ↓
Map Semantic IDs back to item IDs
        ↓
Candidate recall list
```

### 1.5 对着图怎么讲

你可以按论文图的左右两部分讲。

左半边是 **Semantic ID generation**：

```text
Item Content Information
    ↓
Content Encoder
    ↓
Embedding
    ↓
Quantization
    ↓
Semantic ID
```

这里要强调：TIGER 不直接生成原始 item ID，而是先把 item 变成有语义结构的离散 token 序列。

右半边是 **Transformer Encoder-Decoder**：

```text
用户历史 item
    ↓
替换成 Semantic ID tokens
    ↓
Bidirectional Transformer Encoder
    ↓
Encoded Context
    ↓
Transformer Decoder
    ↓
生成 Next Item 的 Semantic ID
```

比如图里：

```text
Item 233 → Sem.ID = (5, 23, 55)
Item 515 → Sem.ID = (5, 25, 78)

Decoder 生成:
Item 64 → Sem.ID = (5, 25, 55)
```

### 1.6 关键点

- Semantic ID 是 TIGER 的核心，不是普通 item ID。
- Semantic ID 由内容 embedding 量化得到，所以相似 item 可能共享部分 code。
- Transformer Decoder 生成的是一个 ID token 序列，而不是直接输出 item embedding。
- TIGER 更像生成式召回模型，而不是完整推荐系统。

### 1.7 局限

- 主要解决 retrieval，不直接解决 rank / rerank。
- Semantic ID 质量依赖 content encoder 和 quantization。
- 如果多个 item 共享同一个 Semantic ID，需要额外 disambiguation。
- 生成式 beam search 的线上成本需要考虑。


---

## 3. OneRec

![OneRec 模型图](onerec.png)

### 3.1 模型定位

OneRec 是快手提出的端到端生成式推荐框架。它的目标比 TIGER 更激进。

TIGER 主要做：

```text
生成式召回
```

OneRec 想做：

```text
用户历史 + 上下文
        ↓
生成式模型
        ↓
直接生成一个推荐 session
```

也就是说，OneRec 想替代传统推荐系统里的多阶段 cascade：

```text
Recall → Pre-rank → Rank → Re-rank
```

改成：

```text
User behavior sequence
        ↓
OneRec
        ↓
High-value session
```

### 3.2 模型结构

OneRec 图分成两部分：

```text
(a) The Architecture of OneRec
(b) Iterative Preference Alignment
```

上半部分是生成模型本体：

```text
OneRec Encoder
        ↓
Hidden representation H
        ↓
OneRec Decoder
        ↓
High-value session tokens
```

下半部分是偏好对齐：

```text
Training Data
        ↓
OneRec_t
        ↓
Beam Search
        ↓
Reward Model
        ↓
chosen / rejected
        ↓
DPO Training
        ↓
OneRec_{t+1}
```

### 3.3 OneRec Encoder

图左上是 **OneRec Encoder**。

输入是用户行为序列：

```text
User Behavior Sequences H_u
```

在图中表现为：

```text
SEP, <a_6>, <b_1>, <c_5>, SEP, <a_2>, <b_1>, <c_7>
```

这里 `<a_6> <b_1> <c_5>` 可以理解为视频 item 的 Semantic ID token。

Encoder 内部结构：

```text
Fully Visible Self-Attention
        ↓
Add & RMS Norm
        ↓
Feed Forward
```

Encoder 的输出：

```text
H
```

也就是用户兴趣表示。

### 3.4 OneRec Decoder

图右上是 **OneRec Decoder**。

Decoder 输入是 high-value sessions：

```text
BOS, <a_9>, <b_7>, <c_1>, ..., BOS, <a_4>, <b_5>, <c_4>
```

Decoder 内部结构：

```text
Causal Self-Attention
        ↓
Fully Visible Cross-Attention
        ↓
Add & RMS Norm
        ↓
MoE Layer
        ↓
Output tokens
```

其中 cross-attention 的 key/value 来自 Encoder 输出的 `H`。

所以数据流是：

```text
用户历史 → Encoder → H
                       ↓
Decoder cross-attention 读取 H
                       ↓
生成 session token
```

### 3.5 MoE Layer

Decoder 里有 MoE Layer：

```text
Router
    ↓
Expert 1 / Expert 2 / Expert 3 / Expert 4 / ...
    ↓
combine
```

MoE 的作用是扩大模型容量，让不同 token / 样本路由到不同 experts。

在推荐场景中，不同用户兴趣、不同视频类型、不同 session 模式可能需要不同子网络建模。

### 3.6 Iterative Preference Alignment

OneRec 下半图是偏好对齐。

流程是：

```text
Training Data
        ↓
OneRec_t
        ↓
Beam Search 生成多个候选 session
        ↓
Reward Model 给每个 session 打分
        ↓
Select chosen / rejected
        ↓
DPO Training
        ↓
OneRec_{t+1}
```

这个过程类似 LLM 的偏好对齐，只是对象从文本回答变成了推荐 session。

### 3.7 输入输出

输入：

```text
用户行为序列
用户上下文
历史观看 / 点击 / 互动 item 的 Semantic ID
```

输出：

```text
一个 high-value session
```

也就是多个推荐视频的 SID 序列。

例如：

```text
Input:
用户历史行为 token

Output:
<a_9><b_7><c_1>, <a_4><b_5><c_4>, ...
```

这些 token 再映射回真实视频 item。

### 3.8 OneRec 数据流

训练主流程：

```text
User behavior sequences
        ↓
Video Semantic ID tokens
        ↓
OneRec Encoder
        ↓
User interest representation H
        ↓
OneRec Decoder
        ↓
Generate high-value session tokens
        ↓
NTP loss
```

偏好对齐流程：

```text
Training data
        ↓
Current OneRec model
        ↓
Beam search candidate sessions
        ↓
Reward model scoring
        ↓
chosen / rejected sessions
        ↓
DPO training
        ↓
Updated OneRec model
```

线上推理：

```text
User history
        ↓
Encode user interest
        ↓
Decoder autoregressively generates session SID
        ↓
Map SID to videos
        ↓
Recommended session
```

### 3.9 对着图怎么讲

你可以这样讲：

1. 左上角是 Encoder，负责读用户行为序列。
2. Encoder 输出 `H`，作为用户兴趣表示。
3. 右上角是 Decoder，先做 causal self-attention，再通过 cross-attention 读取 `H`。
4. Decoder 后面接 MoE Layer，提高模型容量。
5. 最上方输出 high-value session 的 token，并用 NTP loss 训练。
6. 下方是偏好对齐：beam search 生成多个 session，reward model 选 preferred pair，DPO 训练下一轮模型。

### 3.10 关键点

- OneRec 是 session-wise generation，不只是 next-item prediction。
- Encoder 建模用户历史，Decoder 生成推荐 session。
- Cross-attention 是用户兴趣和生成 session 之间的连接。
- MoE 用来扩展模型容量。
- IPA / DPO 把生成结果和用户偏好对齐。

### 3.11 局限

- 系统复杂，训练和 serving 成本较高。
- 需要高质量 session 数据和 reward model。
- 生成式 session 的可控性和线上延迟需要工程优化。
- SID tokenizer 的质量仍然非常关键。

---

## 4. RankMixer

![RankMixer 模型图](rankmixer.png)

### 4.1 模型定位

RankMixer 严格来说不是生成式召回模型。它更像是推荐系统 **ranking stage 的大模型化 backbone**。

为什么放在生成式推荐调研里？

因为即使 TIGER / OneRec 可以生成候选 item 或 session，工业推荐系统通常仍然需要 ranking 模型来做精细排序。

RankMixer 解决的问题是：

```text
ranking 里有大量异质特征
user profiles
video features
sequence features
interacted features

如何高效做大规模 feature interaction？
```

### 4.2 模型结构

图左边是 RankMixer 主干：

```text
User Profiles
Video Features
Sequence Features
Interacted Features
        ↓
Tokenization
        ↓
T × D feature tokens
        ↓
RankMixer Block × L
        ↓
Mean pooling
        ↓
RankMixer output
        ↓
finish / skip / like / ...
```

图右边展开了两个模块：

```text
1. Token Mixing
2. SMoE variant of PFFN
```

### 4.3 输入输出

输入：

```text
User profile features
Item / video features
Behavior sequence features
Cross / interacted features
```

经过 tokenization 后变成：

```text
T feature tokens
每个 token 是 D 维
所以输入矩阵是 T × D
```

输出：

```text
ranking prediction targets
```

比如：

```text
finish
skip
like
click
duration
conversion
```

### 4.4 RankMixer Block

RankMixer Block 的主流程：

```text
Feature tokens
        ↓
Token Mixing
        ↓
Add & Norm
        ↓
Per-token FFN
        ↓
Add & Norm
        ↓
Output feature tokens
```

这和 Transformer 有点像，但它不用标准 self-attention 来做 token 交互。

原因是推荐特征非常异质：

```text
user id
item category
video author
historical behavior
statistical feature
context feature
```

这些 token 之间不一定适合用自然语言里的 attention 相似度来建模。

RankMixer 用 **Token Mixing** 做更硬件友好、更适合推荐特征的交互。

### 4.5 Token Mixing

图右下角是 Token Mixing。

它大概做的是：

```text
T tokens × D dim
        ↓
Split into H heads
        ↓
在 token/head 维度重组
        ↓
Merge back
        ↓
H tokens / mixed tokens
```

你可以把它理解为：

> 不通过 attention score，而是通过结构化的维度切分和重排，让不同 feature tokens 发生信息混合。

### 4.6 Per-token FFN / Sparse-MoE PFFN

图左中间是 Per-token FFN。

普通 Transformer 的 FFN 通常对所有 token 共享参数：

```text
same FFN for all tokens
```

RankMixer 的想法是：

```text
不同 feature token 用不同 FFN / 专家
```

因为推荐特征的语义差异很大：

```text
user token
item token
sequence token
context token
```

共享一个 FFN 可能表达力不足。

图右上角是 Sparse-MoE variant of PFFN：

```text
ReLU Routing
        ↓
Sparse-MoE
        ↓
PFFN experts
        ↓
output
```

### 4.7 数据流

```text
Raw ranking features
        ↓
Feature embedding
        ↓
Tokenization
        ↓
T × D feature token matrix
        ↓
Token Mixing
        ↓
Add & Norm
        ↓
Per-token FFN / Sparse-MoE PFFN
        ↓
Add & Norm
        ↓
Repeat L layers
        ↓
Mean pooling
        ↓
Prediction heads
        ↓
ranking scores
```

### 4.8 对着图怎么讲

你可以按三块讲：

第一块，底部输入：

```text
User Profiles / Video Features / Sequence Features / Interacted Features
```

这些先 tokenization 成 `T × D` 的 feature tokens。

第二块，左侧主干：

```text
Token Mixing
    ↓
Per-token FFN
    ↓
RankMixer output
```

第三块，右侧展开：

```text
Token Mixing 如何 split / merge
Sparse-MoE PFFN 如何通过 routing 选择 expert
```

### 4.9 关键点

- RankMixer 是 ranking backbone，不是生成式召回模型。
- 它把推荐 ranking 特征组织成 tokens。
- 用 Token Mixing 替代或弱化 self-attention 的角色。
- 用 Per-token FFN / Sparse-MoE 提升异质特征表达能力。
- 目标是让 ranking model 也能 scale up。

### 4.10 局限

- 它不直接生成推荐列表。
- 需要依赖已有候选集。
- 模型结构偏工业 ranking，和 SID/tokenizer 关系不如 TIGER/OneRec/UniRec 直接。
- 对特征工程和线上 serving 系统依赖较强。

---

## 5. UniRec

![UniRec 模型图](unirec.png)

### 5.1 模型定位

UniRec 关注的是生成式推荐的一个关键问题：

```text
生成式模型只生成 SID，是否会丢失 item-side feature crossing 能力？
```

传统 discriminative ranker 可以直接看很多 item 特征：

```text
category
brand
seller
price
content
historical stats
```

然后和 user features 做 crossing。

但是普通 generative recommender 可能只是：

```text
p(s0, s1, s2 | user)
```

这样生成的是 SID token，item-side features 显式参与得不够。

UniRec 的解决方案是：

```text
先生成 item attributes
再生成 item Semantic ID
```

也就是 **Chain-of-Attribute, CoA**。

### 5.2 模型结构

UniRec 图可以分成三层：

```text
1. Tokenization: Capacity-constrained Semantic ID
2. UniRec Architecture
3. Alignment: RFT / DPO
```

### 5.3 第一层：Capacity-constrained Semantic ID

图最上面是 tokenizer。

流程：

```text
Multimodal Embedding
        ↓
RQ-KMeans
        ↓
codebook0 / codebook1 / codebook2
        ↓
Semantic ID
```

普通 RQ-KMeans 可能出现一个问题：

```text
热门 item 太多挤到相同或相近 code path
```

UniRec 加了 capacity constraint：

```text
V_k ≤ τ C_cap
```

直观理解：

> 每个 code path 的容量不能无限拥挤，要避免热门 item 导致 token path collapse。

如果某些路径 overlap 太严重，就 repair。

### 5.4 第二层：UniRec Architecture

图中间是主模型。

左边输入：

```text
User Sequence
SID Sequence
```

它们作为：

```text
K & V
```

进入 **Gated-CrossAttn**。

主干结构：

```text
Query prefix
        ↓
RMSNorm
        ↓
Gated-CrossAttn
        ↓
RMSNorm
        ↓
MMoE-FFN
```

右边是 **Hierarchical Rank Head**。

这里是 UniRec 的重点。

它不是直接生成：

```text
s0, s1, s2
```

而是生成：

```text
bos, a1, a2, a3, s0, s1, s2
```

其中：

```text
a1, a2, a3
```

可以理解为 item attributes，比如 category、seller、brand 等。

### 5.5 Chain-of-Attribute

普通生成式推荐：

```text
p(s0, s1, s2 | user)
```

UniRec：

```text
p(a1, a2, a3 | user)
× p(s0, s1, s2 | user, a1, a2, a3)
```

也就是：

```text
先生成属性链
再生成 Semantic ID
```

这样做的好处是：

- 生成路径更有语义；
- 属性先验能缩小候选空间；
- item-side features 能显式进入生成过程；
- 更接近 discriminative ranker 的 feature crossing 能力。

### 5.6 Content Summary / CDC

图右侧有 content summary。

它用：

```text
s0
s1
(s0, s1)
hash0 / hash1 / hash2
shared table
```

来构造条件解码上下文。

作用是：

```text
让模型不仅知道当前生成到哪个 token，
还知道这个 token path 对应的内容语义摘要。
```

这有助于后续生成更稳定。

### 5.7 第三层：Alignment

图最下面是训练对齐：

```text
NTP Loss
Preference Pair
        ↓
Reweight
layer-wise Stop Gradient
        ↓
RFT Loss
DPO Loss
        ↓
L = L_RFT + λ_DPO L_DPO
```

可以理解为三类目标：

```text
NTP:
学习生成 attribute + SID 路径

RFT:
按照业务价值对训练样本加权

DPO:
用偏好对做直接偏好优化
```

### 5.8 输入输出

输入：

```text
User behavior sequence
SID sequence
Static profile
Behavior sequence features
SID-level multimodal features
```

输出：

```text
attribute tokens:
a1, a2, a3

Semantic ID tokens:
s0, s1, s2

最终 item candidates
```

### 5.9 数据流

离线 tokenizer：

```text
Item multimodal embedding
        ↓
RQ-KMeans
        ↓
Capacity-constrained codebooks
        ↓
Semantic ID
```

模型训练：

```text
User sequence + SID sequence
        ↓
Gated-CrossAttn backbone
        ↓
Hierarchical Rank Head
        ↓
Generate attributes
        ↓
Generate SID tokens
        ↓
NTP / RFT / DPO losses
```

推理：

```text
User context
        ↓
Generate attribute chain
        ↓
Generate Semantic ID path
        ↓
Map SID to item
        ↓
Candidate items / recommendation list
```

### 5.10 对着图怎么讲

你可以按三层讲。

第一层，上方 tokenizer：

```text
多模态 item embedding
    ↓
RQ-KMeans
    ↓
capacity-constrained Semantic ID
```

第二层，中间 architecture：

```text
User Sequence / SID Sequence
    ↓
Gated-CrossAttn
    ↓
MMoE-FFN
    ↓
Hierarchical Rank Head
    ↓
attribute + SID generation
```

第三层，下方 alignment：

```text
NTP loss
RFT loss
DPO loss
```

最后强调：

> UniRec 的核心不是单纯生成 SID，而是用 Chain-of-Attribute 先生成属性，再生成 SID，从而补足生成式推荐对 item-side features 的表达能力。

### 5.11 关键点

- UniRec 关注生成式推荐和判别式 ranking 的表达差距。
- CoA 是核心：先生成 attributes，再生成 SID。
- Capacity-constrained SID 用来缓解 token path 拥挤。
- Gated-CrossAttn 连接用户序列和生成 prefix。
- RFT + DPO 用来做业务价值和偏好对齐。

### 5.12 局限

- 结构复杂，训练目标多。
- 需要高质量 item attribute 和 multimodal embedding。
- CoA 的 attribute 设计会影响性能。
- 仍然依赖 SID tokenizer 质量。

---

## 6. GenRec: Preference-Oriented Generative Retrieval

论文链接：[GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation](https://arxiv.org/abs/2604.14878)

![GenRec 模型图](genrec.svg)

### 6.1 模型定位

GenRec 是 JD App 上线的工业级生成式推荐框架。它和 TIGER、OneRec 都很像，因为它们都基于：

```text
item → Semantic ID
用户历史 → SID token sequence
生成模型 → 生成目标 item / item list 的 SID
```

但 GenRec 的重点不是重新提出 Semantic ID，而是解决 **generative retrieval 在工业线上落地时的三个问题**：

```text
1. Page-wise NTP:
   分页请求下，同一个用户历史可能对应多个正反馈 item。

2. Token Merger:
   用户历史很长，而每个 item 又是多 token SID，prefill 成本高。

3. GRPO-SR:
   生成模型需要对齐用户偏好，但普通 RL 容易 reward hacking。
```

一句话：

> GenRec = decoder-only generative retrieval + Page-wise NTP + prompt-side Token Merger + GRPO-SR preference alignment。

它更像 TIGER 的工业增强版：仍然偏 **召回 / retrieval**，但训练目标、输入压缩和偏好对齐都更贴近线上生产系统。

### 6.2 模型结构

GenRec 使用 **decoder-only Transformer**，而不是 TIGER 那种 encoder-decoder。

整体结构可以拆成三块：

```text
User history prompt
        ↓
Token Merger 压缩 prompt 里的 item SID
        ↓
Decoder Layers
        ↓
LM Head
        ↓
Generate predicted item SID sequence
```

论文图里的关键点是：

```text
输入侧:
item_1 = [s1, s2, s3]
item_2 = [s1, s2, s3]

经过 Merger:
[s1, s2, s3] → compressed item vector

Decoder 输入:
compressed item_1, <sep>, compressed item_2, ...

输出侧:
不压缩，仍然生成完整 SID token:
predicted item_1: [s1, s2, s3]
<sep>
predicted item_2: [s1, s2, s3]
```

所以 GenRec 是一种 **asymmetric representation architecture**：

```text
prompt / prefill side:
多 token SID 被压缩

training / decoding side:
不压缩，仍然保持完整 Semantic ID token
```

这样既减少了用户历史输入长度，又保留了生成 item SID 时的细粒度能力。

### 6.3 输入输出

输入是用户历史行为序列：

```text
H = {v_1, v_2, ..., v_n}
```

其中每个 item 都被映射成 Semantic ID：

```text
SID(v_i) = {s_i^1, s_i^2, s_i^3}
```

因此用户历史 prompt 可以写成：

```text
S_u = [SID(v_1), <sep>, SID(v_2), <sep>, ..., SID(v_n)]
```

训练阶段输出是 page-wise target：

```text
Y_page = [SID(v) : v ∈ O ∪ C ∪ E]
```

其中：

```text
O = ordered items
C = clicked items
E = exposed items
```

并且这些 item 会按交互强度排序。

推理阶段输出是：

```text
beam search 生成多个 candidate item SID
        ↓
SID 映射回真实 item
        ↓
召回候选
```

所以 GenRec 有一个很重要的不对称：

```text
训练:
page-wise list target

推理:
point-wise beam search candidates
```

### 6.4 对着图怎么讲

可以按图从左下到右上讲：

第一步，看输入：

```text
item_1 的 SID tokens
<sep>
item_2 的 SID tokens
...
```

每个 item 原本有多个 SID token。

第二步，看 Merger：

```text
item 的多个 SID embedding
        ↓
concat
        ↓
linear projection
        ↓
compressed item vector
```

也就是图里的：

```text
compressed item_1
<sep>
compressed item_2
```

第三步，看 Decoder Layers：

```text
compressed prompt
        ↓
decoder-only Transformer
        ↓
LM Head
```

第四步，看输出：

```text
predicted item_1 <sep> predicted item_2 ...
```

输出侧不使用 compressed item，而是完整生成每个 item 的 SID token。

第五步，看虚线框：

```text
No Compressed Items for Training & Decoding
```

这句话非常关键，表示：

> 压缩只发生在 prompt/prefill 侧，训练 target 和 decoder 生成侧仍然是完整 SID token。

---

### 6.5 重点一：Page-wise NTP

#### 6.5.1 它解决什么问题

普通 generative retrieval 通常是 point-wise NTP：

```text
Input:
用户历史 H

Target:
下一个 item 的 SID
```

也就是：

```text
H → item_a
```

但工业推荐经常是分页请求。用户看到的是一页商品/内容，而不是一个单点 item。

在同一个 page request 里，用户可能：

```text
点击 item_1
购买 item_2
曝光 item_3
点击 item_4
```

于是会出现：

```text
同一个用户历史 H
    → item_1 是合理 label
    → item_2 也是合理 label
    → item_4 也是合理 label
```

如果还是 point-wise 训练，就会得到多个样本：

```text
H → item_1
H → item_2
H → item_4
```

这会导致 **one-to-many ambiguity**：  
同一个输入 prefix 对应多个互相竞争的输出，模型被迫把概率分散到多个 item 上。

#### 6.5.2 GenRec 怎么做

GenRec 把 target 从单个 item 改成整页交互 item 序列：

```text
Input:
用户历史 H

Target:
Y_page = [ordered items, clicked items, exposed items]
```

也就是：

```text
H → item_1, item_2, item_4, ...
```

训练目标仍然是 autoregressive next-token prediction：

```text
L_SFT = - Σ_t log Pθ(y_t | S_u, y_<t)
```

区别在于：

```text
y_t 来自整页 target sequence
而不是单个 next item
```

#### 6.5.3 为什么有用

Page-wise NTP 的好处是：

```text
1. 解决同一个输入对应多个正反馈 item 的冲突
2. 一个 forward pass 里监督多个 item，梯度更密集
3. 保留 page 内部多个 item 的相对关系
4. 更贴近工业分页推荐场景
```

和 OneRec 的关系：

```text
OneRec:
生成 high-value session，偏端到端 session recommendation。

GenRec:
训练时生成 page-wise target，但推理仍保持 point-wise beam search，
更方便接入现有 retrieval pipeline。
```

---

### 6.6 重点二：Token Merger

#### 6.6.1 它解决什么问题

SID 的优点是 item 可生成、可泛化，但缺点是：

```text
一个 item 不再是一个 token
而是多个 token
```

比如：

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

这会显著增加 decoder-only Transformer 的 prefill 成本。

#### 6.6.2 GenRec 怎么做

GenRec 在 prompt side 引入 **linear Token Merger**。

对于一个 item 的三个 SID token embedding：

```text
e(s_i^1), e(s_i^2), e(s_i^3)
```

先 concat：

```text
Concat(e(s_i^1), e(s_i^2), e(s_i^3))
```

再过一个 linear layer：

```text
h_{v_i} = Linear(Concat(e(s_i^1), e(s_i^2), e(s_i^3)))
```

于是：

```text
[s_i^1, s_i^2, s_i^3]
        ↓
compressed item_i
```

图里就是：

```text
item_1 的多个圆点 token
        ↓
Merger
        ↓
compressed item_1
```

#### 6.6.3 为什么叫 asymmetric

因为它只压缩输入 prompt，不压缩输出。

```text
Prompt / prefill:
[s1, s2, s3] → compressed item vector

Training target / decoding:
仍然生成 [s1, s2, s3]
```

这点非常重要。

如果输出也压缩成一个 vector，模型就没法用 LM Head 按 SID token 自回归生成 item 了。  
GenRec 保持输出侧 full-resolution decoding，保证生成结果还是合法的 SID token sequence。

#### 6.6.4 为什么有用

Token Merger 的价值：

```text
1. 降低 prompt length
2. 降低 prefill latency
3. 允许使用更长用户历史
4. 仍然保留输出侧细粒度 SID generation
```

你可以把它理解成：

> 输入侧为了效率，把历史 item 的 SID 压成一个 item-level latent token；输出侧为了可生成性，仍然逐个 SID token 解码。

---

### 6.7 重点三：GRPO-SR Preference Alignment

#### 6.7.1 它解决什么问题

Page-wise NTP 只是监督学习。

它学到的是：

```text
历史日志里用户点过/买过/曝光过什么
```

但线上推荐还需要优化用户满意度，比如：

```text
点击
购买
长期偏好
相关性
不被 reward hacking 诱导
```

如果直接用 RL 优化 reward，又可能出现一个问题：

```text
模型生成语法合法的 SID
但这些 SID 对用户并不相关
却骗过了 reward model
```

这就是 reward hacking。

#### 6.7.2 GenRec 怎么做

GenRec 提出 **GRPO-SR**：

```text
Group Relative Policy Optimization
        +
Supervised Regularization
```

流程：

```text
User prompt S_u
        ↓
当前 policy 生成一组候选 item
        ↓
Reward model 给每个候选打分
        ↓
组内相对比较，计算 advantage
        ↓
GRPO 更新 policy
        ↓
NLL regularization 拉住真实用户正反馈轨迹
```

这里的 “group relative” 意思是：  
不是看单个候选的绝对 reward，而是在同一组候选里做相对比较。

#### 6.7.3 Hybrid Reward

GenRec 的 reward 不是单一 reward model 分数，而是：

```text
hybrid reward = relevance gate × dense preference reward
```

dense preference reward：

```text
用 reward model 估计用户对候选 item 的偏好分数
```

relevance gate：

```text
判断候选 item 是否和用户语义相关
不相关则 reward 置低或置零
```

这样可以缓解：

```text
生成合法但无关 SID 组合
却获得较高 reward
```

#### 6.7.4 NLL regularization

GRPO-SR 还加了 NLL regularization。

原因是：  
纯 RL 容易把模型推离真实用户行为分布。

NLL regularization 的作用是：

```text
让模型在优化 reward 的同时
仍然保持对真实正反馈 item 的生成概率
```

可以理解为：

```text
GRPO:
往 reward 更高的方向推

NLL regularization:
别偏离真实用户行为太远
```

#### 6.7.5 为什么有用

GRPO-SR 的价值：

```text
1. 比纯 SFT 更直接优化用户偏好
2. 组内相对 reward 比绝对 reward 更稳定
3. relevance gate 缓解 reward hacking
4. NLL regularization 保持真实行为分布
```

和 OneRec 的 DPO/IPA 类似，GenRec 也在做生成式推荐的偏好对齐。  
区别是：

```text
OneRec:
用 reward model 选择 chosen/rejected，再 DPO 训练 session generator。

GenRec:
用 GRPO-SR 在 point-wise rollout 候选上做 group-relative policy optimization，
并用 NLL regularization 稳住真实正反馈。
```

---

### 6.8 完整数据流

离线 item 表示：

```text
Item image / text
        ↓
Multimodal encoder
        ↓
Recommendation-oriented embedding
        ↓
RQ K-means
        ↓
SID(v) = [s1, s2, s3]
```

Page-wise SFT：

```text
User history H
        ↓
Convert each item to SID triplet
        ↓
Prompt-side Token Merger
        ↓
Decoder-only Transformer
        ↓
Generate Y_page
        ↓
Page-wise NTP loss
```

Preference alignment：

```text
User prompt S_u
        ↓
Generate rollout candidates
        ↓
Dense reward model
        ↓
Relevance gate
        ↓
Hybrid reward
        ↓
GRPO update
        ↓
NLL regularization
```

Online serving：

```text
User history
        ↓
SID prompt + Token Merger
        ↓
Decoder-only Transformer
        ↓
Point-wise beam search
        ↓
Candidate SID
        ↓
Map SID to item
        ↓
Retrieval candidates
```

### 6.9 和 TIGER / OneRec / UniRec 的区别

和 TIGER：

```text
TIGER:
encoder-decoder，point-wise next item SID generation。

GenRec:
decoder-only，page-wise NTP 训练，Token Merger 压缩长历史，
再用 GRPO-SR 做偏好对齐。
```

和 OneRec：

```text
OneRec:
目标是生成 high-value session，更像端到端 session generator。

GenRec:
训练时用 page-wise target，但线上仍保持 point-wise beam search，
更像工业 generative retrieval 框架。
```

和 UniRec：

```text
UniRec:
核心是 Chain-of-Attribute，先生成 category/seller/brand 等属性，再生成 SID。

GenRec:
核心不是 attribute chain，而是 page-wise supervision、Token Merger 和 GRPO-SR。
```

### 6.10 关键点

- GenRec 仍然是 generative retrieval，不是完整排序模型。
- 使用 decoder-only Transformer，方便复用 LLM 推理优化。
- Page-wise NTP 解决分页请求下的一对多 label ambiguity。
- Token Merger 只压缩 prompt side，不压缩 decoding side。
- GRPO-SR 用 group-relative RL 做偏好对齐。
- Hybrid reward 用 dense reward + relevance gate 缓解 reward hacking。

### 6.11 局限

- 仍然依赖 Semantic ID 质量。
- Page-wise target 的构造依赖具体分页机制和日志定义。
- Reward model / relevance gate 的质量会影响 RL 对齐结果。
- 虽然 Token Merger 降低了 prefill 成本，但 beam search 仍有线上开销。
- 更偏 JD 电商场景，迁移到短视频/内容流需要重新设计 page-wise target。

---

## 7. 六个模型的横向理解

### 7.1 它们分别改写推荐链路的哪里

```text
TIGER
    改写 retrieval：用 Transformer 生成 Semantic ID 召回候选。

HSTU
    改写 user behavior modeling：把推荐数据序列化，做工业级 sequence backbone。

OneRec
    改写 cascade pipeline：从用户历史直接生成 high-value session。

RankMixer
    改写 ranking backbone：把 ranking 特征 token 化，用 mixing block 做特征交互。

UniRec
    改写 generative ranking / retrieval 表达：先生成 attributes，再生成 SID。

GenRec
    改写工业 generative retrieval 训练与对齐：Page-wise NTP、Token Merger、GRPO-SR。
```

### 7.2 重点对比

| 模型 | 核心结构 | 输入 | 输出 | 主要作用 |
|---|---|---|---|---|
| TIGER | Semantic ID + Seq2Seq Transformer | 用户历史 SID token | next item SID | 生成式召回 |
| HSTU | Sequentialized features + HSTU layers | 用户行为流 | future action / hidden state | 工业序列 backbone |
| OneRec | Encoder-Decoder + MoE + DPO | 用户行为序列 | high-value session SID | 端到端生成推荐 |
| RankMixer | Token Mixing + Per-token FFN | ranking feature tokens | ranking scores | 排序特征交互 |
| UniRec | CoA + Gated-CrossAttn + RFT/DPO | 用户序列 + SID 序列 | attributes + SID | 补足生成式推荐表达 |
| GenRec | Decoder-only + Page-wise NTP + Token Merger + GRPO-SR | 用户历史 SID prompt | page-wise target / beam candidates | 工业生成式召回与偏好对齐 |

### 7.3 推荐学习顺序

建议按这个顺序学：

```text
1. TIGER
   先理解 Semantic ID 和生成式召回。

2. OneRec
   看生成式推荐如何从 next item 走向 session generation。

3. UniRec
   看生成式推荐如何补 item attribute / feature crossing。

4. HSTU
   看工业推荐如何把用户行为做成 sequence backbone。

5. RankMixer
   看 ranking stage 如何大模型化和高效做 feature interaction。

6. GenRec
   看工业 generative retrieval 如何处理分页、一对多 label、长历史输入和 RL 偏好对齐。
```

