> [!NOTE]
> **移植说明（lwd-cpu-offload-min）**:本文档随 host-staging 实现自
> `prefill_only_cpu_offload` 分支移植。路径映射:文中的
> `vllm_ascend/worker/lwd_cloud/` → 本仓 `vllm_ascend/worker/lwd_cloud_model/`,
> `vllm_ascend/distributed/lwd_comm/` → `vllm_ascend/distributed/lwd_data_plane/`,
> `LwdCommRequest/LwdCommFuture` → `LwdTransferRequest/LwdTransferFuture`,
> `LwdCloudMixedScheduler._schedule_impl` → `LwdCloudScheduler._lwd_schedule_step`。
> 机制描述与实现一致;示例代码中的类型名按上表对读。

# 云侧 host 暂存 + 按请求拆分存储 / 按请求粒度调度 设计文档

> 前置:`lwd_pre_recv.md`(提前收)。本方案在它的 DONE 点之后插一层 host
> 暂存,并把"批"这个概念从云侧调度里**彻底去掉**:载荷按请求拆开存,调度按请求
> 粒度走原生组批。
> 状态:**v7 已实现**(代码见 §12)。本文件记录方案、不变量与验证方式。
> 已验证前提(提问方确认):本环境 **NPU→HOST 与 HOST→NPU 拷贝均可行**。
> 目标模型:**Qwen3.6-27B 与 DSv4-flash 都要支持**。
> 离线自检脚本(`_pre_recv_*_check.py`)当前**不在本仓**,见 §8.1 说明。

---

## 1. 问题:批把"存储"和"调度"两件事捆死了

### 1.1 现状(提前收之后)

一条 UP 批到云侧之后,它的**载荷是一整块连续 buffer**,而且这块 buffer 同时承担
两个角色:

| 角色 | 现状 |
|---|---|
| **落点** | `torch.empty(numel)` 一整块,`numel = Σ_i n_i·(H+4LK)`,按批分配 |
| **记账单位** | 云侧台账(`prefill_notify_queue` 的队首 / `lwd_batch.seqno`)是**批** |
| **调度驱动** | 批内所有 item 必须**同一步一起排**(`_lwd_verify_full_admission`:排程量 == 预告量,不等即 fail-fast) |

于是三件事被迫绑在一起:**一批数据落在哪块内存、什么时候能被算、算多少**。
由此长出两类代价:

1. **整批才能算**:某个请求因为 KV 不够算不动,整批都留队首;队首卡住,后面
   所有预告一起等(head-of-line)。这是"无法下发"的直接来源。
2. **一个请求拖住一整批的内存**:批内一个请求被 abort 或长期算不完,那块
   buffer(几十 MB 级)就一直占着,即使其余请求早就算完了。

### 1.2 根因

**"批"是边侧组批策略的产物,却被云侧当成了存储与调度的单位。**

边侧组批是有意义的(见 §3);但云侧完全没必要继承这个单位 —— 载荷一旦落到
host,它就可以被切成**互相独立的按请求存储单元**,而调度也就可以回到原生的
"按请求"粒度。

### 1.3 本方案要什么

> 云侧收完边侧发来的 embedding 后,从 NPU 搬到 host,并且**按请求拆开存**;
> 云侧调度器把边侧发来的 notify 也**拆成单个请求**,放进 waiting 队列,按原生
> 的组批方式去组;下发到 worker 后,再从**该请求自己的** host 内存把 embedding
> 取回 NPU 拼上去算。

由此得到的性质:

* 请求之间**不再互相牵连**:一个请求算不动、被 abort、走得慢,都不影响别人;
* 存储按请求记账:某请求算完就可以标记释放,不需要等它的"批友";
* 调度回到原生:批只是**传输容器**,不再是调度单位。

### 1.4 非目标与两条总原则

**总原则一:不做侵入式修改。** 原生调度器(`vllm/v1/core/sched/scheduler.py` 及其
配套)尽量**不动**;需要改的行为用**继承**写在自己的子类里。只有当改动量极小、
且不这样改就没法表达时,才碰原生。本方案的所有行为都应落在
`lwd_control/control_cloud_scheduler/` 与 `vllm_ascend/worker/lwd_cloud/` 里。

**总原则二:只为这一条路服务。** 本方案就是**唯一方案**。改造前那些与新语义
不适配的代码(阈值复现、全准入核验、按批记账、`consume` 后即释放……)
**一律删掉,不保留兼容分支、不做"两条路并存"**。留着没有用,还会让后来人
分不清哪条是活的。

**非目标**:

* 不改边侧的组批与发送策略(§3 会论证为什么继续保持组批);
* 不改 seqno 配对协议、不改尺寸公式;
* 不改相位调度器(`prefill_first` / `decode_first`)逃生通道;
* 不改 DOWN 通道;
* **不放大边侧在途窗口**(`_LWD_MAX_PENDING_CHUNKS = 2` 保持不动);
* **只走 PD_mix 一条路径**,PD_mix 下用不到的判断直接删。

---

## 2. 现状里已经具备的条件(改前先认清)

改动量比想象小,因为**"按请求粒度"这条线的骨架已经在代码里了**:

| 条件 | 位置 | 说明 |
|---|---|---|
| 云上每个请求已是**独立的原生 `Request`** | `lwd_cloud_engine._lwd_build_request` | `prompt_ids=None` + `prompt_embeds=torch.zeros(prompt_len, H)` 占位 + `lwd_embeds_placeholder=True` |
| 请求**逐个**走原生 ADD 进 waiting | `_lwd_promote` → `input_queue.put_nowait((ADD, (request, 0)))` | 与边侧何时组批无关,云侧本来就是逐个入队的 |
| 每个请求**独立准入** | `LwdRequestNotify` 逐请求过 `_lwd_gate_pending` | 有独立的门,与 RangeNotify 解耦 |
| RangeNotify **逐 item 带请求身份** | `LwdRangeItem(request_id, offset, num_tokens, has_mrope, mrope_positions)` | 云侧已在逐 item 登记 seqno,拆分信息本来就够 |
| 占位 embeds **不随 MQ 运输** | `NewRequestData` 只带 shape/dtype,worker 本地分配 | 35MB 零 buffer 不上慢通道 |
| 注入点已是"按请求写窗口" | `_lwd_inject_remote_embeds` 逐请求写 `inputs_embeds.gpu[out:out+n]` | 只要给它"该请求的 host 源"即可 |

**结论**:本方案不是从零搭"按请求调度",而是把**已经按请求入队**的骨架,接上
**按请求的存储**与**按请求的取数**,并删掉那层"批必须整批一起算"的约束。

---

## 3. 关键决策:边侧组批传,云侧按请求拆【已定】

这是提问方点名要分析的问题。**结论:继续组批传,云侧在 host 上按请求拆开。**

### 3.1 为什么"逐请求发"的动机已经消失

逐请求发送唯一的原始动机是:**批粒度逼着云侧整批消费**。而新方案**本来就要在
host 上按请求拆开存** —— 拆开之后云侧看到的就是逐请求粒度,batch 在云侧已经
不可见。既然拆分无论如何都要做,"在线上就拆开"只剩下一堆代价,没有收益。

### 3.2 逐请求发要付出的代价

| 代价 | 说明 |
|---|---|
| **HCCL op 数 × N** | 组批是**一批一条 op**;逐请求要每请求一条 `isend`/`irecv`。而 **HCCL P2P 无 tag**,两侧必须严格同序投递,配错一位就是数据串台(现协议靠"一个 seqno = 一条 op"这条纪律兜住,拆成 N 条就是 N 倍的配对风险) |
| **控制面流量 × N** | 提示(hint)、水位上报都是**一批一条**。逐请求 = 每条消息都要走旁路 MQ,而旁路 MQ 是正确性依赖(丢一条水位停摆) |
| **打爆边侧 pending 闸门** | `_LWD_MAX_PENDING_CHUNKS = 2` 数的是"未消费 chunk 数"。批内每请求各占一个名额 ⇒ 两个小请求就把窗口占满,边侧组批能力掉一半(而组批正是它维持吞吐的手段) |
| **尺寸公式调用 × N** | `hash_payload_numel` 每请求算一次,虽然便宜,但收端要为每条单独分配/记账 |
| **验收面 × N** | 内存/水位/abort 的账从"每批"变成"每请求 op",排查面变大 |

### 3.3 组批传没有损失什么

* **字节数不变**:载荷是 `n·(H+4LK)`,**线性可加**,怎么切分总量一样;
* **粒度可通过拆分拿到**:云侧拆完即按请求粒度,与线上是否拆开无关;
* **拆分成本是一次 H2H**:接收落点必然是一整块连续 pinned(D2H 是"一次 recv →
  一次拷贝"),拆分时再按请求拷到各自的 pinned(§4.1)。

### 3.4 唯一的反方论据,以及为什么先不做

**尾延迟**:同批里如果有一个长请求,短请求要等它一起发。组批按"批内最大"
对齐,长请求会拖住短请求的发送时刻。

这条是真实的,但:

* 边侧的 chunk 单元本来就把长请求切成小块(`_lwd_chunk_unit`,默认退化为整个
  预算),所以"长请求"在批里的尺寸已被 chunk 化压过一轮;
* 要量化它必须实测(拆开发之后,短请求的首 token 延迟到底改善多少),而现在
  没有真机数据;
* 代价是 §3.2 那一整张表。

**建议**:首版维持组批;把"逐请求发送"记成 §9 的可评估项,等有真机数据再定。

### 3.5 决策

| 选项 | 结论 |
|---|---|
| 边侧**逐请求**发送 | **不采纳**(代价见 §3.2,收益已消失) |
| 边侧**组批**发送 + 云侧按请求拆 | **采纳** |

---

## 4. 按请求拆分存储

### 4.1 拆分的物理形态:每请求独立 pinned【v6 已定,与 v5 相反】

D2H 是"一次 recv → 一次拷贝",所以**接收落点必然是一整块连续 pinned**。但
"落点"与"存储"是两件事:收到之后还要不要搬到每请求自己的 buffer?

| 方案 | 做法 | 额外拷贝 | 释放粒度 |
|---|---|---|---|
| 甲(v5 选的) | 同一块接收 pinned 上按请求切**视图**,各自记账 | 零 | 批级(最后一个请求结算完才还) |
| **乙(v6 采纳)** | 拆分时把每请求的切片 **H2H 拷到自己的 pinned** | 每请求一次 H2H | **请求级(请求结束才还)** |

**为什么 v6 反过来选乙**:因为**生命周期要求变了**。

* v5 的语义是"按步消费",批内请求差不多同时结算,批级释放够用;
* v6 的语义是 **"全部存下来,请求结束才释放"**(提问方明确要求)。共享一块载体
  做不到"某请求结束就还它那部分" —— 一块 pinned 只能整体释放。若沿用甲,那么
  **一个长请求会把整批的内存钉到它自己结束**,而这正是要消除的耦合。
* 乙的代价是拆分时一次 H2H(host 内存带宽,几十 MB ≈ 1~2ms/批,与计算流并行),
  换来的是**内存账目与请求生命周期严格对齐**,这是本方案的核心诉求。

**这也让 host 占用的语义变清楚了**:占用 = Σ 在途请求字节,而不是"在途批数 ×
批大小"。前者可预测(§4.4),后者会被"批里最慢那个"拖住。

### 4.2 每请求存储单元

```python
@dataclass
class LwdReqEmbed:
    """一个请求在 host 上的 embedding 存储单元(拆分后的最小访问粒度)。

    **生命周期 = 请求的生命周期**(§4.1):收到就存下,一直留到该请求结束,
    不是"访问完就释放"。
    """
    req_id: str
    # 该请求自己的 pinned 载体(乙方案):embeds 区 + 可选专家 ID 区。
    # 独立分配、独立释放,所以"请求结束才释放"不需要等任何批友。
    host: "torch.Tensor"          # 1-D bf16 pinned,长度 = numel + tail_numel
    numel: int                    # embeds 元素数(整个 prompt)
    prompt_rows: int              # prompt 总行数(该请求要算的 embed 行数)
    hidden_size: int
    # 专家 ID 尾段(DSv4;Qwen 恒 0)
    tail_off: int = -1
    tail_numel: int = 0
    num_hash_layers: int = 0
    num_experts_per_tok: int = 0
    # ---- 进度(两个独立指针,别混) ----
    recv_rows: int = 0            # 已收到的行数(累积上界,由水位上报驱动,§5.2)
    consumed: int = 0             # 已算的行数(权威 = num_computed,§5.7)
    aborted: bool = False
```

* `recv_rows` 与 `consumed` 是**两个不同的指针**:
  * `recv_rows` 是**能算到哪**(数据面进度);
  * `consumed` 是**已经算到哪**(原生水位);
  * 不变式:`0 ≤ consumed ≤ recv_rows ≤ prompt_rows`。
* 访问:`host[consumed*H : (consumed+n)*H]` —— 一次零拷贝切片,
  **起点就在该请求自己的 buffer 里**(不需要 `prompt_start` 参与);
* 释放判据:请求结束(§6)。**不是** `consumed == prompt_rows` 就释放 ——
  `_lwd_preprocess_prompt_embeds` 与 draft proposer 在读 `inputs_embeds.gpu`,
  host 那份留着不碍事,而请求结束后才真正没人要。

> 实现注:一个请求可能跨多个 batch 到达(边侧把它切成多段发)。此时按它自己
> 声明的 `offset` 落进同一块 buffer,必要时**扩容并保留已写内容**;写入偏移
> 必须用"相对该 buffer 第 0 行"的 `offset − row_base`,用 prompt 绝对行号会
> 写到自己范围之外(实现期踩过,见 §11)。

### 4.3 载荷布局与按请求拆分

```
第 1 步(通道口):recv 落进**轮换的接收 pinned**(池化,长度 = 本批 numel)
                 D2H 一次拷完整批 → 归还设备槽
        │
        │  第 2 步(拆分,H2H 一次/请求)
        ▼
每请求自己的 pinned:
   req0.host = [ embeds(prompt0 × H) | tail(prompt0 × 4LK) ]   ← 留到 req0 结束
   req1.host = [ embeds(prompt1 × H) | tail(prompt1 × 4LK) ]   ← 留到 req1 结束
   ...
   (Qwen:L=0 ⇒ 无 tail 区)
```

* **拆分是 H2H 一次**(host 内存带宽,几十 MB ≈ 1~2ms/批),换来的是
  **请求级的独立生命周期** —— 这正是本方案要的;
* 接收 pinned 是**池化轮换**的(深度 = 在飞批数),拆分完立刻归还;
* 接收 pinned 与请求 pinned **职责分离**:前者只活一个 D2H 周期,后者活整个
  请求周期。这样"留到请求结束"不会把接收缓冲也一起钉住。
* 拆分前有一道**几何校验**:`numel` 必须等于"行表 × (H + 4LK)"。行表与载荷
  不符说明边云版本或组批发散 —— 此时 fail-fast,而不是照错的行表切下去
  (切错就是静默串数据)。

> 为什么拆成独立 buffer 而不是"同一块上切视图":见 §4.1 的取舍 —— 因为
> **生命周期要跟请求走**,共享载体做不到"某请求结束就还它那部分"。

### 4.4 host 占用:全存,不设上限【已定】

* **占用 = Σ_{在途请求} (prompt_rows × H + tail) × 2 bytes**,按请求累加、
  按请求释放(§6);
* **不设 host 硬上限**(提问方已核算,认为内存足够):收到就存、留到请求结束,
  不做"超限只排队不暂存"那一套 —— 那套会引入"数据不进暂存 ⇒ `recv_rows` 永远
  0 ⇒ 永不调度"的活锁,是不必要的复杂度;
* 量级参考(DSv4 示例:H=7168、L=32、K=8 ⇒ 每行 8192 元素 = 16 KiB):

  | 并发在途 prompt 行数 | pinned 占用 |
  |---|---|
  | 10k 行(约 2 个 5k prompt) | ~160 MiB |
  | 50k 行 | ~800 MiB |
  | 100k 行 | ~1.6 GiB |

* **仍然保留观测,但不做闸门**:按请求打印 `recv_rows / consumed / prompt_rows`
  与 pinned 字节数,便于看到"是谁占着、占了多少";
* **后续(不在本次实现)**:若真出现内存不足,**用 LRU 淘汰某些请求的 host
  embedding**;淘汰后若该请求又被排到,就**让边侧重发一次这段 embedding**
  (边侧本来就有重发能力)。这是后话,先不做,当前假设内存足够。

### 4.5 设备乒乓槽(worker 侧)

* 槽数 = 在飞上限(与边侧在途窗口同源);
* 槽的生命周期:**D2H 完成即归还**(不是等消费)—— 这是"NPU 占用与在途深度
  解耦"的来源,与 v4 一致;
* 每槽一块 `torch.empty(max_numel, bf16, device="npu")`,终身复用;
  小批取 `slot.dev[:numel]` 视图;
* 因为拆分只在**端点 rank** 做,非端点 rank 没有 host 池,所以"本步搬哪些行"
  必须由**每个 rank 各自**从 `num_scheduled_tokens` 算出(§5.7,W1)。

---

## 5. 按请求粒度调度

### 5.1 调度主线:notify 拆成请求,走原生组批

```
边侧组批发送(一条 notify = 一个 seqno = 一次 UP send)
        │
        ▼
云引擎 IO 线程收到 notify
  ├─ 逐 item 登记 seqno(既有:_lwd_seqno_registry)
  └─ 不再把"整条 notify"当成调度单位
        │
        ▼
云调度器:把 notify 拆成**逐请求**的 embedding 可用性台账
  每个请求一条 LwdReqEmbed(§4.2),独立记账、独立推进
        │
        ▼
原生组批:waiting/running 里的请求按原生方式组(§2 已具备:逐请求 ADD)
  本步排多少行 = 原生按 token 预算切(原生 chunked prefill 语义)
        │
        ▼
SO 下发(带逐请求的"本步要算多少行")
        │
        ▼
worker/runner:按请求从它自己的 host 视图取 [consumed, consumed+n) 行
              → H2D → 广播 → 注入
```

**与 v4 的关键差别**:v4 是"云侧算出一个窗口、把排程量**精确约束**成窗口";
v7 是"云侧只保证**数据可用**,排多少交给原生"。前者要求预测与权威逐请求相等
(fail-fast),后者让原生当权威,我们只做**可用性闸门 + 上界核验**。

### 5.2 已收 token 上报:每个请求"收到多长"必须让 engine 知道

**这是本方案的关键机制**(提问方指定)。云上维护一个**按请求的已收行数台账**:

```
recv_rows[req] = 该请求 prompt 里**已经收到并暂存**的行数   (0 ≤ recv_rows ≤ prompt_rows)
```

**上报路径**(复用既有旁路 MQ,不动数据面):

```
worker 通信线程:D2H 完成(§4.4 的三段式第 ③ 段)
  ├─ 既有:done MQ 上报 (channel, seqno)              → 引擎推进水位(批粒度就绪)
  └─ 新增:**同一条消息捎带该批逐请求的行数增量**
        [(req_id, delta), ...]                       → 引擎累加 recv_rows
```

* 引擎侧累加用 **`max()` 而不是 `+=`**:`recv_rows[req] = max(recv_rows[req], 已收)`
  —— 幂等、天然抗重复上报与乱序(边侧队满重试会产生重复预告,既有协议已经
  按"单调性"去重,这里再兜一层);
* 上报的是**增量**:worker 侧记住"已经上报到哪",只报新到的部分,否则同一批
  重复内容会把 done MQ 刷满;
* 行数从哪来:`LwdRangeNotify.items` 逐 item 带 `offset`/`num_tokens`,
  **worker 侧在挂 recv 时已经拿到**(提示里的 `req_ids`/`rows`/`offsets` 字段);
* **不需要改数据面协议**,也不需要边侧多带字段 —— 边侧已经在预告里说了
  "这个请求的这段 offset/长度";
* 兼容:老边侧不带逐请求行数时,退化为"整批算一个请求"(只影响调度粒度,
  不影响正确性)。

**为什么必须上报到 engine**:调度器(engine 进程)才做组批决定,它必须知道
"每个请求现在能算多少行"。worker 收完不上报,engine 就只知道"批到了",不知道
"哪个请求到哪了"。

### 5.3 下发规则:按"已收多少"与"预算"取小【提问方指定】

组批时,对每个候选请求:

```
avail  = recv_rows[req] − consumed[req]        # 这个请求现在还能算多少行(数据面)
n      = min(avail, budget_left)               # 再看预算:谁小听谁的
if n < 1: 本步不排它(留 waiting)
else:     排 n 行,  budget_left −= n
```

* **`avail < 预算`** ⇒ 下发 `avail`(收到多少算多少);
* **`预算 < avail`** ⇒ 下发 `预算`(被预算限制);
* **`avail == 0`** ⇒ **这条请求本步不允许进 batch**(提问方明确要求)。

这正是"数据面未到就不允许加入 batch 计算"的实现:`avail == 0` 直接把它挡在
批外,而不是排上去再读脏数据。**长序列同样按这条规则下发** —— 一个 10 万 token
的 prompt 收到 500 行就下发 500 行,收到 2048 行而预算只有 1024 就下发 1024 行。

#### 5.3.1 "数据未到不许组批"怎么加才不侵入原生【已定:加在可见集上】

**(a) 现状是真的会发生 —— 已查实**

* `LwdRequestNotify`(请求元数据)由边侧在 `add_request` 时发布(**请求一入队就发**);
* `LwdRangeNotify`(chunk 数据预告)要等该 chunk 被边侧排程才发布;
* 云侧 `_lwd_promote` 只在收到 `LwdRequestNotify` 时触发,**与 embedding 是否
  到达无关**(全仓唯一调用点)。

⇒ **元数据必然早于 embedding 到达,所以"ADD 早于数据"是常态而非竞态。**
请求进了 waiting,原生组批时可能把它排上,而它的 `prompt_embeds` 还是
`torch.zeros` 占位(§2)⇒ **算到全零 embedding,静默错算**。

**(b) 加在哪:可见集,不改原生一行**

| 方案 | 侵入性 | 结论 |
|---|---|---|
| 改原生 `_schedule_prefills` 加"数据就绪"判定 | **高**(动核心 scheduler) | 否决 |
| 复用原生的 `WAITING_FOR_REMOTE_KVS` 状态阻塞 | 中(要动核心 + executor) | 否决 |
| **延迟 ADD(数据到了再 promote)** | 低,但改变入队时刻与 FCFS 次序 | 备选 |
| **可见集过滤:`recv_rows == 0` 的请求不进本步可见集** | **零新增原生改动** | **采纳** |

**采纳可见集过滤的理由**:它落在**已有的**机制上 ——
`_lwd_schedule_for_visible_reqs(req_ids)` 本来就是"临时把一批请求从三个队列里
挑出来单独调度",只需在**构造 `req_ids` 时**加一条条件。原生一行不动,
继承层加一个谓词。

**实现落点(已编码,`lwd_cloud_mixed_scheduler._schedule_impl`)**:谓词写在
"分类完 items 之后、求窗口之前",条件是
`recv_rows - request.num_computed_tokens <= 0`(覆盖"完全没到"与"到了但已算完"
两种情形),并打一条 `GATE seqno=… visible=N/M` 日志。

> **非侵入性已有硬证据**:W2 专项自检用 `git status` 实测
> **原生 `vllm/v1/core/sched/scheduler.py` 一行未改**。这条断言会随代码自动跑,
> 谁改了就红。

* **只需要这一条**:"数据完全没到"的请求不进 batch。至于"到了但不够预算",
  由 §5.3 的下发规则(`min(avail, budget)`)处理,**不需要**再过滤;
* 被过滤的请求留在 waiting,下步 `recv_rows` 变了再进来 —— 与既有
  "闸门未开就本步只跑 decode"同款节流,不引入新状态;
* **失败方向是安全的**:多挡一步只损失一点调度机会,不会算错。

**(c) 再加一道数值核验(W2)**

可见集过滤保证"完全没数据的不进批",但它不保证"原生排的行数 ≤ 已收行数"。
所以下发前必须核验:

```
out.num_scheduled_tokens[req] ≤ recv_rows[req] − consumed[req]      (W2)
```

违反了不会报错,只会算出一批零 embedding —— **这是本方案唯一"静默错算"型的
风险**,必须显式拦住。实现里 `_lwd_verify_w2` 还会单独挡"`recv_rows == 0`
却被排"这一档(那说明可见集谓词失效,是最严重的一档),并把
`recv_rows`/`consumed`/`window`/`offset` 全部报出来便于定位。

**(d) 原生单步排多少**不**天然受"已收"限制 —— 但用总预算就能表达**

已核实:原生 `num_new_tokens` 的推导(`scheduler.py`)是

```
num_new_tokens = request.num_tokens − num_computed_tokens        # 想让多少
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold                # 只有设了才切
num_new_tokens = min(num_new_tokens, token_budget)               # 再看预算
```

原生**没有**逐请求上界钩子。所以逐请求上界靠**总预算**表达:

```
self.max_num_scheduled_tokens = decode_exact + Σ_i min(avail_i, room)
```

这个式子与原生 FIFO 前缀序组合出的结果**精确**等于"逐请求取小":

* 原生按 waiting 序逐个吃预算,吃到谁就是谁;
* `avail_i <= 0` 的请求**不占预算**(跳过),所以在没有数据时不会把预算
  浪费掉、也不会堵住后面的请求;
* 一旦原生"跳过前面的请求去排后面的"(序不一致),总量仍 ≤ `Σavail`,
  而 W2 会在下发后当场炸 —— 不是静默错算。

**因此不再需要"块对齐"这一层**(v4 的 `budget // BS * BS` 与
`min(remaining // BS * BS, budget)` 已删):原生自己会按 `block_size` 切块,
我们只负责"不许超过已收"。

`long_prefill_token_threshold` 的担忧也随之降级:它只影响原生"单步愿意排多少",
而预算已经先封住了上界;若它让原生排得**少于** `Σwindows`,步后核验
`native_prefill != sum(windows)` 会报出来(响亮失败,不是错算)。
**部署前仍建议确认它的取值**。

#### 5.3.2 边云 chunk 边界从此无关

提问方明确指出:**原生 chunk 切分与边侧 chunk 完全解耦**。

* 云侧只知道"每个请求收到了多少行",**不知道也不关心边侧是怎么切的**;
* 边侧怎么发、切多大、几个请求合批,都不影响云侧的下发决定;
* 云侧下发多少 = `min(已收 − 已算, 预算)`,与边侧 chunk 边界没有任何关系;
* 因此 v4 里那套"复现边侧 chunk 边界"的机制(阈值复现、`offset` 对账的严格
  语义)**整体失去意义**。

> 一个直接推论:边侧 chunk 化对云侧**退化为纯粹的流量整形手段** —— 它控制
> "数据多大块地到",不控制"云侧一次算多少"。这是两侧解耦后最干净的分工。

### 5.4 非侵入式实现准则(本节所有闸门的落点)

为了让 §5.1~§5.3 的机制**不侵入原生调度器**,本方案的行为只落在三处:

| 落点 | 性质 | 用到的既有机制 |
|---|---|---|
| `lwd_cloud_engine`(engine 进程) | 继承 `EngineCoreProc` | 收控制面消息、维护 `recv_rows` 台账、逐请求 ADD |
| `lwd_cloud_mixed_scheduler`(engine 进程) | 继承原生 Scheduler | `_lwd_schedule_for_visible_reqs`(既有可见集机制)、可见集谓词、预算覆写、W2 核验 |
| `vllm_ascend/worker/lwd_cloud/*`(worker 进程) | 继承 NPUWorker / V1 runner | 三段式上报、暂存池、取数注入 |

**唯一被"新增谓词"的地方**是可见集的构造(§5.3.1b)。
**原生 `vllm/v1/core/sched/scheduler.py` 一行不改。**
唯一的原生文件改动是 `vllm/v1/core/sched/output.py` 里删掉两个字段
(`LwdEmbedBatch.windows` / `chunk_offsets`,见 §5.8)。

### 5.5 为什么不让 engine 直接访问那份 host embedding【提问方的问题】

提问方问:**为什么不跟原生一样,把数据搬到 host 之后让 engine 进程直接访问?
是因为没拿到内存地址吗?**

**答案是"进程隔离 + engine 根本不需要读它",不是地址问题。**

**(1) 进程布局(已核实本仓)**

* 调度器在 **EngineCore 进程**(`LwdCloudEngineCore(EngineCoreProc)`,引擎主进程);
* worker 是**独立进程**:`WorkerProc.make_worker_process(...)` 逐个创建
  (`multiproc_executor.py`),engine 侧只持有 `self.workers` 句柄与两条 MQ。

⇒ worker 进程里 `malloc`/`pin` 出来的地址,**engine 进程默认看不到**。地址是
进程私有的(虚拟地址空间),不是"没拿到地址",而是"这个地址在对面进程里才有意义"。

**(2) 想让 engine 访问,有三条路,都不划算**

| 做法 | 代价 |
|---|---|
| 共享内存(`/dev/shm`、`torch.multiprocessing` shared tensor) | 双方都要 map;生命周期、读写同步、崩溃残留都要管。**这正是旁路 MQ 已经用了的机制** —— 但 MQ 传的是**几百字节的元数据**,embedding 是**几十 MB**,量级完全不同 |
| engine 侧只读 map 同一块 | 需要 engine 参与 map 与解映射;还要保证"worker 写、engine 读"的次序 |
| 走 IPC 把 tensor 送过去 | **本仓已经明确否决过同类做法**:`NewRequestData` 的注释写着"35MB 级零 buffer 不上 MQ(overflow 慢通道实测单程 240ms+)" —— 所以占位 buffer 只传形状。几十 MB 的 embedding 走 MQ 同理不可行 |

**(3) 更关键:engine 不需要读 embedding 的内容**

* **调度只需要数值**:`recv_rows`、`consumed`、`prompt_rows`、`budget` ——
  全是标量。这些本来就走旁路 MQ(§5.2),几百字节;
* **数据不需要经过 engine**:路径是
  `边侧 --HCCL--> worker 接收槽 --D2H--> worker 的 host pinned --H2D--> NPU`,
  全程在 worker 内。engine 只是"下决定",不搬数据;
* 让 engine 参与搬运只会多一次拷贝、多一个故障点、多一层同步,没有任何收益。

**(4) UVA / recv 直落 host —— 【已排除,不再回头】**

同一个问题还点出了"能不能把接收直接落进 host、省掉 D2H"这个方向。
**已查实并确认不可行**:

> `vllm_ascend/patch/__init__.py`(patch 清单第 12 条)明文写着:
> **"ASCEND NPUs do not support UVA yet, so we need to wrap it in vLLM."**
> 做法是把 vLLM 原生的 `UvaBuffer`(设备直接寻址 host pinned 的那个东西)
> 做成 dummy class;Future Plan 是"Remove this patch when NPU support UVA"。
>
> 旁证:`simple_kv_offload/worker.py` 注释说明 `cudaHostRegister` 是 CUDA-only,
> Ascend 侧只能用 `torch.zeros(pin_memory=True)`;该模块的 pinned 只用于 **H2D
> 拷贝**,不是设备直接寻址。`patch_uva.py` 里还检查 triton-ascend 3.2.1
> "disables the UVA feature"。

**⇒ 提问方已到官网确认:Ascend 侧不支持数据直接传到 host**(HCCL recv 落
host pinned 这条路也走不通)。因此本方案**按 v7 定稿实现**:

```
既定(两跳):  HCCL recv → NPU 乒乓槽 --D2H--> host 请求缓冲 --H2D--> NPU 计算
```

**两跳拷贝是本方案的既定成本,不再作为"待优化项"挂着。** 若将来 CANN 支持了,
回看这一节。分层诊断的正确姿势是**把"HCCL 真不行"和"torch_npu 这层拦住了"
分开**:① `pin_memory=True` 能否造出 pinned;② pinned host tensor 能否作为
HCCL operand;③ P2P `irecv` 能否直接落 host("expects NPU tensor"类 = 绑定
限制,值得再用 CANN C API 试;"not supported/invalid"类 = HCCL 级限制);
④ H2D/D2H 带宽实测(可用于给 §10 定量)。

> **注意区分层次**:UVA 解决"设备计算单元能否直接寻址 host";
> "HCCL 能否收进 host"是 DMA 落点问题;"engine 进程能否访问 worker 的地址"是
> 进程隔离问题(要共享内存)。三个是不同层次,别混。

### 5.6 budget 与 KV 口径(照抄原生,沿用 v4 结论)

```
budget(step) = min( self.max_num_scheduled_tokens − decode_exact ,
                    (free_blocks − decode_blocks) × BS )
```

* **预算基数是实例属性 `self.max_num_scheduled_tokens`**(`= config.max_num_scheduled_tokens`,
  缺省才回退 `max_num_batched_tokens`),原生 `token_budget` 用的就是它;
* **decode 优先**:`free_blocks` 先扣 decode 本步新增(与原生 `_schedule_running`
  先于 `_schedule_prefills` 的次序一致);
* **KV 块数用原生函数**:`coordinator.get_num_blocks_to_allocate(...)` 做**只读
  探针**(`apply_admission_cap=False`,`new_computed_blocks` 首阶段传空)——
  **不**接着调 `allocate_new_blocks`。原生是"尝试-失败"式(`allocate_slots` 返回
  `None`),而它会真的分配,所以不能当探针;
* **对齐交给原生**:v7 不再自己把预算对齐到 `block_size`,只负责"不许超过
  已收";原生自身按 `block_size` 切块。

> 预算与 KV 口径与"按批还是按请求"无关,所以这一节与 v4 一致。

### 5.7 取数指针:必须与原生水位一致

第 i 个请求本步要算 `[consumed, consumed + n)`(`consumed` 是**该请求自己**的
已算行数):

```
consumed = request.num_computed_tokens − prompt_start   # 由每个 rank 各自算出
n        = out.num_scheduled_tokens[req_id]             # 原生本步给它的行数
取数区间  = req.host[consumed * H : (consumed + n) * H]  # 该请求自己的 buffer
```

* `consumed` **必须由每个 rank 各自从 Request 状态算出**,不能从 worker 的
  host 池台账取 —— 那个池只有端点 rank 有,非端点会算成 0,各 TP rank 就会
  注入到不同位置(这是 v4 实现期踩过的坑,见 §11);
* **下发量上限**由调度器按 §5.3 决定并核验(W2),worker 侧再独立核验一次
  (读到"已收 − 已算"之外就抛错),两道防线都在;
* 非端点 rank 没有 host 池,它按同样的**行数**分配接收缓冲等端点广播 ——
  行数来自同一份 `num_scheduled_tokens`,所以各 rank 必然一致(W4)。

### 5.8 从"批队首"到"请求独立":删掉的东西

| v4(窗口化) | v7(按请求) |
|---|---|
| `prefill_notify_queue` 队首 = 调度单位,卡住即 head-of-line | **拆成逐请求台账**,请求之间无队首关系 |
| `_lwd_verify_full_admission`(排程量==预告量) | **删除**(原生当权威,只留 W2 上界核验) |
| `_lwd_windowed_admission` 逐 item 求窗口 | **删除**,换成 §5.3 的"按已收/预算取小" |
| `LwdEmbedBatch.windows` / `chunk_offsets` | **删除**(下发量由原生 `num_scheduled_tokens` 表达) |
| `long_prefill_token_threshold` **阈值复现** | **删除**(不再复现边侧边界) |
| 调度器侧"已消费行数台账" | **删除**:与原生水位是同一件事的第二份拷贝,两份可能分歧(§5.7 唯一权威 = 原生水位) |
| 块对齐(`budget // BS * BS`) | **删除**(原生自己按 block 切;我们只受"已收"约束) |
| `consume` 后即释放 | **请求结束才释放**(§6) |
| 窗口起点从 worker 台账取 | **每个 rank 各自算**(§5.7 W1) |

**一个直接好处**:v4 的"chunk 内已消费必须是前缀、预算跨 item 拼满时不能有
空洞"这套保序纪律,在这里**整块消失** —— 请求独立推进,谁先算完谁先释放,
不需要任何顺序假设。

---

## 6. 生命周期与释放

### 6.1 一个请求的存储单元:存到请求结束【提问方指定】

```
   边侧发来一段 → D2H 到接收 pinned → 拆到该请求自己的 pinned(累积)
        │                                     │
        │                                     └─ recv_rows 增长(§5.2 上报)
        ▼
   ACTIVE ──本步排它 consumed..consumed+n──> consumed += n
        │                                        │
        │                                        └─ consumed == prompt_rows
        │                                              (prompt 算完,进入 decode)
        │
        └── 请求结束(finish / abort)──> RELEASE:还 pinned + 清台账
```

**关键语义:embedding 不随"用完"释放,而是留到"请求结束"。**

* **不是** `consumed == prompt_rows` 就释放 —— prompt 算完只是进入 decode,
  这时 host 那份**留着也不碍事**,而提前释放反而要引入"谁还在读"的判定;
* **释放点 = 请求结束**:`finish_requests`(自然结束 / 长度截断 / abort)时还;
* 已算的行不回滚(历史事实),abort 只要求剩余行不再被算;
* **请求级释放**(§4.1 乙):某请求结束就还它自己那块,不需要等批友 ——
  这是"全存下来"与"不互相牵连"这两个要求的交汇点。

### 6.2 释放的三条路径

| 路径 | 触发 | 动作 |
|---|---|---|
| **正常结束** | `finish_requests`(EOS / max_tokens / 长度截断) | 还该请求的 pinned + 清台账 |
| **abort** | `finish_requests(ABORTED)`(可能延迟,§6.3) | 同上;剩余行冻结 |
| **启动/关停兜底** | 引擎收尾、请求已被别的路径终结 | 兜底清账(避免关停期挂在半路) |

**实现落点**:worker 在 `execute_model` 的 `finished_req_ids` 分支上调
`_lwd_release_finished(...)` —— 那是引擎给 worker 的存活信号,不需要新协议。

> 注意:这与 v4 的"`live_rows == 0` 才释放"不同。v4 的判据是"没人再要这些行了";
> v7 的判据是"这个请求结束了"。后者更简单,也更符合"全存下来"的语义 ——
> 因为**留着不释放不是问题,提前释放才是问题**。

### 6.3 abort:两条线

1. **调度器侧**:请求从逐请求台账摘除、冻结剩余行;`finish` 的延迟判据要认
   "已冻结"的请求不再等它消费(否则请求迟迟不终结 —— v4 修过同款问题);
2. **跨进程**:调度器的台账只在调度器进程里,**worker 的暂存池看不到**。必须
   把 abort 名单捎带给 worker,worker 才能冻结它那份(释放仍走 §6.1 的请求结束)。

### 6.4 abort 名单跨进程:随批捎带

* `LwdEmbedBatch.aborted_req_ids` 承载"已摘除、剩余行永不再算"的请求 id;
* 摘除即排队(`_lwd_abort_pending`),每步 `schedule()` 出口随批捎带;
* **本步没有批**(纯 decode / 空步)时,挂一个**只带名单的退化批**
  (`req_ids=[]`),否则"被 abort 且当时没批要下发"就没人通知;
* 已有**非 EMBED 批**时名单**放回队列**等下一步,**绝不丢弃**;
* worker 在 `if not meta.req_ids: return` **之前**处理名单,逐请求冻结。

> 这一节沿用 v4 已实现并已验证的机制(§11)。

---

## 7. 对账不变量

| # | 不变量 | 位置 | 违反后果 |
|---|---|---|---|
| **W1** | `consumed == request.num_computed_tokens − prompt_start`(**每个 rank 各自算**) | 调度器 + runner | 取数指针漂移 ⇒ 注入错位,静默乱码 |
| **W2** | `out.num_scheduled_tokens[req] ≤ recv_rows[req] − consumed`(**下发量不得超过已收量**) | 调度器(下发前)+ runner(注入前) | **算到 `torch.zeros` 占位 ⇒ 静默错算**(§5.3.1c) |
| **W3** | 该请求的 host buffer 里 `[consumed, consumed+n)` 必须已被写入 | stager(`read_range`) | 读到未写入的零值区 |
| **W4** | `win_dev` 布局 == "本步被排请求序 + 段内偏移",所有 TP rank 一致 | 调度器 ↔ runner ↔ 各 rank | 广播后各 rank 取错行 |
| **W5** | `0 ≤ consumed ≤ recv_rows ≤ prompt_rows` | worker + 调度器 | 读到未到达的数据 / 重复注入 |
| **W6** | 请求结束后其 pinned 必须已还,且不再被排 | worker | host 泄漏 / 悬浮引用 |
| **W7** | 各请求的 host buffer 互相独立,释放一个不影响另一个 | stager | 提前释放别人的数据 |

**W1 与 W2 是本方案的两根支柱**:

* **W1** 保证"取数指针"与"原生水位"一致 —— 指针由 `num_computed` 推出,而
  `num_computed` 是原生推进的权威水位;
* **W2** 保证"算的行数不超过收到的行数" —— 这是"数据面未到不允许进 batch"
  在数值上的落地。**它是最不能省的一条**:违反了不会报错,只会算出一批
  零 embedding,表现为"输出莫名其妙",而不是崩溃。

### 7.1 mrope positions 的切片(最容易错的一处)

控制面载荷是 `[3, chunk_rows]` int64 的行主序连续字节。请求内取 `[k, k+n)`:

```
payload[r*chunk_rows*8 + k*8 : r*chunk_rows*8 + (k+n)*8]   r = 0,1,2
```

**不是**一段连续字节(按 `k*8` 起 `(k+n)*24` 长去切会把 3 行混进一段)。
已实现并有专门的离线自检(§8)。

> 后续可做的简化:每个请求的 mrope positions 可以随 embedding 一起**按请求存
> 下来**(与 embeds 同生命周期),取数时只切一次连续段。当前实现仍从批级
> 控制面载荷里切,因为 positions 走控制面、不进 host 暂存。

---

## 8. 验证策略

### 8.1 离线(无 NPU,可跑)

| 脚本 | 覆盖 |
|---|---|
| `_pre_recv_mrope_check.py` | mrope 切片:**对着独立写的 `[3,n]` 列切片模型逐位比**(21 个窗口 + 首末边界 + "真实实现 ≠ 朴素连续切片"回归护栏) |
| `_pre_recv_host_stage_check.py` | 暂存池状态机:槽归还时机、**按请求独立存储**、`recv_rows`/`consumed` 双指针、**读不能超过已收(W2/W3)**、abort 只冻结不释放、**请求结束才释放**、多 chunk 扩容保留已有行、专家 ID 尾段、几何校验 |
| `_pre_recv_worker_check.py` | worker 拆分 + **已收行数增量上报** + **abort 冻结(不释放)** + **请求结束释放**(真实 handler + 真实 stager) |
| `_pre_recv_e2e_check.py` | 端到端时间线 + **按请求独立推进(乱序)** + 分配规则(已收 500/预算 1024 ⇒ 500 等) |
| `_pre_recv_w2_check.py` | **W2 专项**:从真实调度器源码 AST 抽出判定表达式执行,与独立参考模型对拍;含"真实实现 ≠ 朴素实现"护栏、核验函数在越界时抛错、边界不误杀、以及 `git status` 实测原生 `scheduler.py` 未被修改 |
| `_pre_recv_path_check.py` / `_pre_recv_link_check.py` | 跨文件接线、被删除机制的残留、提示/水位形状、日志可 grep 性 |

> **这些脚本当前不在本仓** —— 它们是开发期放在工作区根目录的独立脚本
> (无 torch/NPU 依赖,`python <script>.py` 直接跑,退出码 0 = 全过)。
> 需要连同改动一起提交时,把它们放进 `tests/` 下的相应位置即可。

### 8.2 待真机验证(开发机无 NPU)

1. **mrope 切片**:离线已证实,但真机 `frombuffer().view(3,n)` 路径未跑;
2. **KV 探针只读性**:`get_num_blocks_to_allocate` 的预测值必须等于随后
   `allocate_slots` 的实际分配块数;
3. **D2H/H2H/H2D 与 `staged_event.query()` 时序**;
4. **性能读数**:`d2h` / `h2h` / `h2d` / 每步 H2D 字节数,对比 §10 预估;
5. **开关两态数值对拍**:`VLLM_ASCEND_LWD_HOST_STAGE=0/1` 同 prompt token 逐位一致,
   **两个模型各跑一遍**;
6. **`ABORT-SYNC` / `RELEASED` 日志**:abort 后、请求结束后 pinned 是否真的降下来;
7. **`long_prefill_token_threshold` 的部署取值**(§5.3.1d)。

---

## 9. 风险与待定

1. **W2 是"静默错算"型风险,已优先实现**:`num_scheduled_tokens > recv_rows − consumed`
   时不会报错,只会拿 `torch.zeros` 占位算出一批错结果。调度器下发前与 runner
   注入前各有一道核验(§5.3.1c / §7)。
2. **`long_prefill_token_threshold` 的部署取值**(§5.3.1d):它只影响原生"单步
   愿意排多少",预算已先封住上界;排得比预算少时步后核验会报出来。
   **部署前确认取值更稳**。
3. **H2H 拆分开销**(提问方:真机再看):每请求一次 host 拷贝,在 D2H 之后、
   同一条流上。批内请求多、每请求小时拷贝次数多但每次短 —— 要观察它是否成为
   通道流上的长尾。**暂不优化,等真机数据**。
4. **`recv_rows` 的边界**:上报用 `max()` 累加是为了**幂等**;若对端给的
   `offset + rows` 超过它自己声明的 prompt 长度,说明**边云版本或组批发散**,
   此时 stager 会 fail-fast 而不是静默截断。**属校验项,不是性能风险**。
5. **逐请求发送(尾延迟)**:见 §3.4,首版不做,列为可评估项。
6. **`_lwd_dropped` 集合的清理**:目前只增不减(字符串级,实际无碍),但要留意。
7. ~~**UVA / recv 直落 host**~~ —— **【已排除】** 官网确认 + 仓库明文:Ascend 不支持
   数据直接传到 host。本方案按 v7 实现,**两跳拷贝是既定成本**,后续不再作为
   待优化项跟踪(§5.5 留了记录)。
8. **LRU 淘汰 + 边侧重发**:仅在内存真不够时启用,**本次不实现**(§4.4)。

---

## 10. 性能预估

* **每步 H2D = 本步被排请求的行数 × H × 2 bytes**,与边侧 chunk 大小无关
  (1024 行、H=7168 ⇒ **14 MiB ≈ 0.7 ms**);
* **D2H 每批一次**,与批大小成正比(5120 行 86 MiB ≈ 4.3 ms),在通道流上与
  计算流并行;
* **H2H 拆分**:每批一次(整批字节),host 内存带宽,1~2 ms/批量级,也在通道流上;
  **待真机确认**(§9-3);
* **TP 广播一次**(端点 → 各 rank),大小 = 本步 Σrows;
* **多段 H2D**:段数 = 本步被排请求数(每请求一块独立 pinned,天然不连续)。
  每段都很长,P 次 `copy_` 的发射开销可忽略;
* **host 占用**:按请求累加、请求结束释放(§4.4 表)。**不设上限**,假设内存足够;
  不足时的 LRU 淘汰 + 边侧重发是后话(§4.4)。

---

## 11. 修订记录

### v7 实现期修正(编码轮次记录)

| 发现 | 处理 |
|---|---|
| **写入偏移用了 prompt 绝对行号** | 每个请求的 buffer 都从自己的第 0 行开始,必须用 `offset − row_base`;否则多请求批里第二个请求的 embedding 会写到自己范围之外(离线视图假件逐位比对抓出) |
| **续传会覆盖丢数据** | 一个请求跨多个 seqno 到达时,原写法新建 buffer 丢前面的行;改成"按 offset 落位 + 必要时扩容并保留已写内容" |
| **接收缓冲用 `id()` 追踪却返回视图** | 归还时找不到 root,缓冲永不释放;改为按需精确分配并返回 root |
| **逐请求上界可以不用块对齐** | 原生没有 per-request 钩子,但"总预算 = decode_exact + Σmin(avail,room)"配合 FIFO 前缀序**精确**等于逐请求取小 ⇒ v4 的块对齐整层删除(§5.3.1d) |

### v6 → v7(非侵入式闸门 + 取消 host 上限 + 只此一条路)

| 反馈 | 落点 |
|---|---|
| **尽量不侵入原生调度器,否则用继承;把"数据没到不允许调度"加上去限制它组就好** | **§1.4 总原则一**;§5.3.1 重写为方案对比表,结论:**加在可见集过滤上** —— `_lwd_schedule_for_visible_reqs` 是既有机制,只需在构造 `req_ids` 时加一条谓词,**原生一行不动**;并说明这**只需要这一条** |
| **本方案就是唯一方案;不适配的旧代码改掉,不要保留** | **§1.4 总原则二**:阈值复现、全准入核验、按批记账、consume 后即释放等**一律删除,不做兼容分支** |
| **长序列下发也按"已收多少 + 预算"** | **§5.3.1d**:下发量 = `min(recv_rows − consumed, budget)`,并给出用总预算表达逐请求上界的实现结论 |
| **host 侧不设上限,全部存着;LRU 淘汰是后话** | **§4.4 重写**:删掉 `MAX_MB` 硬上限与"超限只排队不暂存"(那会引入活锁);保留观测、不做闸门;LRU + 边侧重发记为后续(本次不实现) |
| H2H 时间真机再看 | §9-3、§10 标注"待真机确认",暂不优化 |
| **"这一点风险我没看懂"** | **§9-4 重写**为一句人话 + 明确它是**校验项不是性能风险** |
| 逐请求发送先不管 | §3.4 / §9-5 保持"可评估项,首版不做" |
| **为什么不让 engine 直接访问 host embedding?是没拿到地址吗?** | **§5.5**:答案是**进程隔离 + engine 不需要读它**,不是地址问题(已核实 worker 是独立进程 `make_worker_process`);并把 UVA / recv 直落 host 记为**已排除**并留档 |

### v5 → v6(全量留存 + 按已收量下发 + 边云解耦)

| 反馈 | 落点 |
|---|---|
| **所有 embedding 都存下来,请求结束才释放**(不是访问完就释放) | **§4.1 反转取舍**:从"同一块 pinned 上切片(批级释放)"改为 **每请求独立 pinned(请求级释放)**——因为共享载体做不到"某请求结束就还它那部分"。**§6.1 重写**释放语义:释放点 = 请求结束。§4.4 重算 host 占用 |
| **数据面未到不允许组进 batch** | **§5.2 新增按请求已收行数台账** `recv_rows[req]`,由 worker 在 D2H 完成时随既有 done MQ **捎带**上报;**§5.3 下发规则** = `n = min(recv_rows − consumed, budget_left)`,`avail == 0` 直接挡在批外 |
| **下发量取"已收"与"预算"的较小值** | **§5.3** 写死该规则;并查实两个前置:**(a)** `_lwd_promote` 不看数据面,ADD 与数据到达无关 ⇒ 必须加可见集闸门 + **W2 核验**;**(b)** 原生 `num_new_tokens` 不天然受"已收"限制 |
| **原生 chunk 与边侧 chunk 完全无关** | **§5.3.2** 明确写死:云侧只看"每个请求收到多少行",边侧 chunk 化退化为**纯流量整形**;v4 那套"复现边侧 chunk 边界"的机制**整体失去意义**,§5.8 列为删除 |

**另外新增**:W2 提升为两根支柱之一(§7,静默错算型风险);§8.1 增加 W2 专项
离线自检;§9 风险重排;§10 补 H2H 拆分开销。

### v4 → v5(按请求拆分存储 + 按请求粒度调度)

| 维度 | v4(窗口化消费) | **v5(按请求拆分)** |
|---|---|---|
| 存储粒度 | 批(一整块 + 行表) | **按请求拆记账与访问** |
| 调度粒度 | 批(notify 队首) | **请求**(notify 拆开,走原生组批) |
| 排多少谁定 | 云侧算窗口,**精确约束**原生 | **原生定**,云侧只做数据可用性闸门 |
| 批的耦合 | 批内必须保序、前缀消费 | **无顺序假设,允许乱序** |
| 队首阻塞 | 有(head-of-line) | **无**(请求之间独立) |
| 删掉的机制 | 全准入三检、阈值复现 | 再加:`windows`/`chunk_offsets`、窗口推导、保序纪律 |
| 边侧发送 | 组批 | **组批(§3 论证后不变)** |
| 跨进程 abort | 随批捎带 | **沿用** |
| KV/budget 口径 | 原生只读探针 + decode 优先 | **沿用**(与粒度无关) |

### v1 → v4 的关键结论(仍然有效,直接继承)

* **v1**:按"放大在途窗口换 NPU 显存"写,前提被否定;
* **v2**:改为"云侧按实时 KV 决定算多大";
* **v3**:补齐 budget 推导、双模型支持、abort 语义;
* **v4**:窗口对齐、prefix cache 首阶段不做、decode 优先、KV 口径照抄原生。

**v1~v4 里继续成立、后续版本原样继承的部分**:
乒乓设备槽与"D2H 完成即归还";D2H 挂通道流借 FIFO 保证"数据已完整"且水位仍是
前缀;`staged_event` 与 DONE 语义升级;专家 ID 留 host 按进度切片;budget 与 KV
的**原生只读探针**口径;decode 优先;abort 名单跨进程捎带;mrope 两级切片。

### 实现进展

* **v4 阶段**:13 个文件、~1300 行;6 个离线自检 367 条断言;771 文件编译 0 错误。
* **v7 阶段(当前)**:按 §5.8 完成结构性重写。两仓共 13 个改动文件 +
  1 个新增文件(`lwd_host_stage.py` 重写为按请求存储);
  离线自检 **7 个脚本 432 条断言全绿**;**1058 文件编译 0 错误**;
  W2 专项自检用 `git status` **实测原生 `scheduler.py` 未被修改**。
  真机验证项见 §8.2。

---

## 12. 附:相关代码位置

| 关注点 | 位置 |
|---|---|
| 通道 FIFO / 尺寸 / 流桥接 | `vllm_ascend/distributed/lwd_comm/channel.py` |
| 提前收协议(提示 / 水位) | `vllm_ascend/distributed/lwd_comm/pre_recv.py` |
| **暂存池(按请求存储 + 双指针)** | `vllm_ascend/worker/lwd_cloud/lwd_host_stage.py` |
| 拆分 / 已收行数上报 / abort / 释放 | `vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py` |
| 取数(按请求 H2D)/ 广播 / 注入 | `vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py` |
| Hash 载荷与专家 ID 记账 | `vllm_ascend/worker/lwd_hash/lwd_hash_routing.py` |
| 请求构建(逐请求 ADD)+ **已收行数台账** | `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py` |
| **调度主体(可见集谓词 / 预算 / W2)** | `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_mixed_scheduler.py` |
| 批元数据定义 | `vllm/v1/core/sched/output.py` |
| 预告协议(逐 item 带请求身份) | `vllm/v1/lwd_control/control_communication/lwd_notify.py` |
| 边侧在途窗口 / chunk 组批 | `vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py`(`_LWD_MAX_PENDING_CHUNKS = 2`) |
| KV 块池与 block size | `vllm/v1/core/block_pool.py`、`vllm/v1/core/kv_cache_utils.py` |
| env 变量约定 | `vllm_ascend/envs.py`(开关 `VLLM_ASCEND_LWD_HOST_STAGE`,默认 0) |
