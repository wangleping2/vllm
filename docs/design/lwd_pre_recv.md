> [!NOTE]
> **移植说明（lwd-cpu-offload-min）**:本文档随 host-staging 实现自
> `prefill_only_cpu_offload` 分支移植。路径映射:文中的
> `vllm_ascend/worker/lwd_cloud/` → 本仓 `vllm_ascend/worker/lwd_cloud_model/`,
> `vllm_ascend/distributed/lwd_comm/` → `vllm_ascend/distributed/lwd_data_plane/`,
> `LwdCommRequest/LwdCommFuture` → `LwdTransferRequest/LwdTransferFuture`,
> `LwdCloudMixedScheduler._schedule_impl` → `LwdCloudScheduler._lwd_schedule_step`。
> 机制描述与实现一致;示例代码中的类型名按上表对读。

# 提前收(early recv)设计文档 —— prefill_only(chunked prefill + mm)/ 云侧 UP

> 位置:仓内 `vllm/`(本文档所在仓)
> 分支:`prefill_only_chunked_prefill_mm`(仓目录 `vllm-ascend_v23.0_prefill_only`
> / `vllm_v23.0_prefill_only`)
> 迁移自:`prefill_only/pre_recv`(整体组批、无 mm 的那一版),差异见 §11。
> 改动范围:只动 **PD_mix(默认混排)** 分支;相位调度器(`prefill_first` /
> `decode_first`)与边侧路径一行未改。
> 参考实现:`round2/v23_0` 的 `EdgeCloudCommService`(hint MQ + 通信线程 +
> irecv_done MQ + 每通道水位),按本仓的 `lwd_comm` / `LwdChannelType` 结构落位。
> **后续演进**:本阶段只解决"recv 挂得太晚"。之后在它的 DONE 点之后又插了一层
> **host 暂存**(按请求拆分存储、按请求粒度调度),见 `lwd_host_staging.md` ——
> 后者改变了 DONE 的语义(从"数据落在云 NPU"升级为"数据落在云 host 且资源已归还")
> 并新增了逐请求行数上报通道。本文档记录的是**提前收**这一层,两文互为前后置。

---

## 1. 问题:recv 挂得太晚

prefill_only 下云侧要算中间层,输入是边侧算好的 prompt embeds(UP 张量)。
改造前,这条 UP `irecv` 只有等 `SchedulerOutput` 一路下发到云 worker、进
`execute_model` 才挂:

```
边侧发 embeds ──► (云侧还没挂 recv,数据在 HCCL 里排着)
                    │
RangeNotify ─► 调度器 ─► SO ─► MQ ─► worker_busy_loop ─► execute_model ─► 挂 recv
                                                                          │
                                                              future.wait() 干等
```

recv 挂得晚 = 传输与调度串行,且等待被塞进 worker 前向内部(一次 prefill
批几百毫秒的区间里白等)。提前收要做的只有一件事:**预告一到就把 recv 挂上**,
调度只负责判断"这批收完了没"。

## 2. 数据路径(唯一一条,不新增旁路)

```
 云引擎 IO 线程                     云端点 worker 通信线程
 ───────────────                    ──────────────────────
 RangeNotify 到达(边侧预告:一个 chunk 批 = 一条 notify = 一个 seqno)
   └─ HINT ──hint MQ(下行)──►  SUBMIT: submit_recv(UP, seqno, 主帧 numel, aux)
                                       │  irecv 已挂出(不等 SO)
                                 future.done() 轮询
                                       │
 record_irecv_completions ◄─DONE── done MQ(上行)
   └─ 每通道水位
        └─ is_irecv_complete(UP, seqno)
             └─ PD_mix 闸门:收完才下发这个 prefill 批
                  └─ SO ─► POST:前向入口取到**同一条** future ─► 注入
```

两条 MQ 都是旁路,**不能借 `rpc_broadcast_mq`**:那条队列由
`worker_busy_loop` 单线程消费,它在一个 prefill 批里阻塞几百毫秒,提示排上去
就失去了"提前"的意义。

## 3. 事件时间线(与日志一一对应)

| 时刻 | 事件 | 谁打的 | 日志 |
|---|---|---|---|
| t0 | 边侧预告到达,引擎下发提示 | 引擎 IO 线程 | `[Lwd][pre-recv] HINT seqno=.. ts=..` |
| t1 | 通信线程取到提示,挂出 UP irecv | 云 worker 通信线程 | `[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=.. num_tokens=.. ts=..` |
| t2 | recv 收完(`event.query()` 转真)并上报 | 同上 | `[Lwd][pre-recv] DONE-RECV channel=lwd_up seqno=.. ts=..` |
| t3 | 引擎每步排空 done MQ,水位推进 | 云引擎主循环 | `[Lwd][pre-recv] watermark advanced channel=lwd_up seqno=..` |
| t4 | 闸门放行,`_schedule_mixed` 下发该批 | 云调度器 | `[Lwd][pre-recv] gate seqno=.. watermark=.. ready=True ts=..` |
| t5 | SO 到 worker,取到提前挂好的 future | 云 worker 前向入口 | `[Lwd][pre-recv] POST seqno=.. reqs=.. tokens=..` |
| — | 前向注入,`future.wait()` 立刻返回 | model runner | `[Lwd][perf] cloud up-recv-wait seqno=.. dur=~0ms` |

`ts=` 全是 `time.monotonic()`,engine 与 worker 同机父子进程 ⇒ 可直接相减。
`timeline_pre_recv.sh` 就是按这五列对齐出四段耗时的(见 §8)。

## 4. 实现清单(5 个文件)

### 4.1 `vllm_ascend/distributed/lwd_comm/pre_recv.py`(链路 + 协议,重写)

| 组 | 接口 | 说明 |
|---|---|---|
| 提示 | `make_recv_hint(seqno, num_tokens, aux_rows=0, req_ids=None, rows=None, offsets=None)` | 提示 = `{channel: UP, seqno, num_tokens, aux_rows}` + **逐请求行表**(`req_ids`/`rows`/`offsets`,host 暂存阶段新增,见 `lwd_host_staging.md` §4.3);`aux_rows` = 批内带 mrope 的 token 行数(纯文本批恒 0) |
| | `hint_fields(hint)` | 提示自检(含 `aux_rows ≤ num_tokens`、行表自洽:行数与总数相符、`offsets` 个数对齐),坏提示返回 None(丢弃 + 告警) |
| MQ | `create_hint_mq()` / `attach_hint_mq()` / `create_report_mq()` | 下行:引擎建 writer,handle 进环境变量;上行:worker 建 writer,handle 随 READY 回传 |
| 调度侧 | `register_hint_sender` / `post_irecv_hint` | 引擎装传输(闭包 = `mq.enqueue`)并下发提示 |
| | `record_irecv_completions` / `is_irecv_complete` / `watermark` | 每通道水位与就绪判据 |
| 常量 | `LWD_RECV_HINT_MQ_ENV` | `VLLM_ASCEND_LWD_RECV_HINT_HANDLE`(只是 MQ handle,不是配置开关) |

**尺寸公式不在本模块**:UP 主帧 numel 走 hash 载荷公式
(`hash_payload_numel(num_tokens, H, L, K)`,含可选的专家 ID 字节),依赖 worker
本机配置 —— 所以提示只带"事实"(token 数、mrope 行数、逐请求行表),由 worker 侧
`_lwd_build_up_recv_request` 统一换算;aux 帧 numel = `aux_rows * 3`(int64,
`[n,3]`,与主帧同 seqno)。

旁路 MQ 规格:`MessageQueue(1, 1, max_chunk_bytes=4096, max_chunks=64)` ——
一批一条消息,边侧 pending 闸门把在途压到个位数,而提示是**正确性依赖**(丢一条
= 那条 recv 不提前挂 = 水位不前进 = 闸门后的批永不下发),所以留足余量、不做流控。
(`max_chunk_bytes` 因逐请求行表而放大。)

### 4.2 `vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py`(worker 侧)

* `init_device` 末尾:判定端点 rank(`rank == lwd_config.edge_npu_count`),
  是则起通信线程 `lwd-early-recv`(daemon),并逐个 rank 打一行
  `worker rank=.. endpoint=.. hidden=..`。
* 通信线程主体:`attach_hint_mq()` → `create_report_mq()` → 循环
  `dequeue(timeout=0.0005)` → `start_early_irecv(hint)` → `_report_irecv_completions()`
  → 队列卡住诊断 → `time.sleep(0.002)`。
  * 只做 `dequeue / submit_recv / future.done() / enqueue` 四件事,**从不
    `wait()` 任何 future**(同通道跨线程 wait 是 HCCL 不允许的);
  * 循环里那 2ms sleep 是必需的:ring 读端在"刚读过"的窗口内是
    `sched_yield` 自旋(`busy_loop_s=1s`),光给短超时不封顶 CPU;
  * 线程自己 `torch.npu.set_device(self.device)`(当前设备是线程级状态)。
* 收单:`_early_recv_futures[(channel, seqno)] -> future`,
  `_early_recv_consumed` / `_early_recv_reported` 两个集合,
  `_early_recv_pending`(名额满时排队的提示,只存字段不占显存)。
* 请求构造:`_lwd_build_up_recv_request(seqno, num_tokens, aux_rows)` —— 提示
  路径与消费点补挂共用这一个构造点(主帧 hash 载荷 + aux 帧),两侧形状必然一致。
* 消费点:`_lwd_up_post_recvs` → 端点走
  `get_or_post_early_recv(seqno, num_tokens, aux_rows)`,`aux_rows` 由
  `meta.has_mrope` 与 `meta.token_ids` 现算(与边侧组帧同源同序);非端点仍为
  `future = None` 占位(TP 广播在注入时刻)。
* 非端点 rank:无跨机通道,一律返回 None(数据在注入时刻由 TP 广播取到)。

### 4.3 `vllm/v1/executor/multiproc_executor.py`(执行器,MQ 落点)

* `_init_executor` 里、**创建 worker 之前**:`_lwd_init_early_recv_mq()` 建
  hint MQ(writer),handle 写进 `os.environ[LWD_RECV_HINT_MQ_ENV]`
  (fork/spawn 都靠环境变量继承)。只有云侧(LWD 且非边节点)建。
* 响应 MQ 握手之后:`_lwd_attach_irecv_done_mq()` 从
  `self.workers[*].irecv_done_mq` 取那个非 None 的 reader(云端点 rank)。
* `WorkerProcHandle.irecv_done_mq` / `from_unready_handle(...)` /
  `wait_for_response_handle_ready`(READY 载荷里的 `irecv_done_handle`,
  `create_from_handle(handle, 0)`)。
* `WorkerProc.__init__`:`self.irecv_done_mq = getattr(self.worker, "irecv_done_mq", None)`
  (在 `init_device` 之后 —— worker 就是在 `init_device` 里建的 writer)。
* `worker_main` READY 载荷带上 `"irecv_done_handle"`。

### 4.4 `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py`(引擎)

* `__init__`:`self._lwd_early_recv = scheduler_name not in ("prefill_first", "decode_first")`
  —— 提前收只挂在 PD_mix 分支。
* `_lwd_setup_zmq`:`register_hint_sender(lambda hint: hint_mq.enqueue(hint))`,
  `hint_mq = self.model_executor.lwd_recv_hint_mq`。
* `_lwd_dispatch`(RangeNotify 分支):预告到达即下发
  `post_irecv_hint(make_recv_hint(msg.seqno, Σitem.num_tokens, Σ_{has_mrope}item.num_tokens, req_ids=…, rows=…, offsets=…))`
  —— 不等 SO;`has_mrope` 逐 item 透传(纯文本批 aux_rows=0,零 aux 流量)。
* `step()` / `step_with_batch_queue()` 开头:`_lwd_drain_irecv_completions()`
  排空 `lwd_irecv_done_mq` → `record_irecv_completions(...)`;host 暂存阶段起,
  上报消息还会捎带逐请求已收行数,由 `_lwd_note_arrived` 并进台账。
* `__init__` 里把 `recv_rows` 台账的**引用**交付调度器(`self.scheduler.lwd_recv_rows`),
  IO 线程写、主循环读(dict 赋值原子)。

### 4.5 `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_mixed_scheduler.py`(闸门)

在 `_schedule_impl` 里有一条闸门分支
`elif not self._lwd_pre_recv_ready(notify): notify = None`;`_lwd_pre_recv_ready`
查 `is_irecv_complete(LwdChannelType.UP, notify.seqno)`,并把 `watermark` 一起打进
日志。闸门**只是就绪判据**:未就绪 → 本步只下 decode,notify 留队首,下步再判。

> **host 暂存阶段新增的另一道闸门**:可见集谓词(`recv_rows − num_computed <= 0`
> 的请求本步不进 batch)。它与水位闸门**层次不同**:水位闸门看"整批收完没",
> 可见集谓词看"这个请求收到多少行"。

## 5. 关键不变量

1. **seqno 配对**:边侧 `LwdRangeNotify.seqno` = 云侧 UP 通道 recv seqno =
   `SO.lwd_batch.seqno` = runner 注入的配对键,四处同源同值;chunked prefill 下
   一条 seqno = 一个 chunk 批 = 一次 UP send(主帧 + 可选 aux 帧)。
2. **numel 严格相等**:主帧走 hash 载荷公式、aux 帧 = `aux_rows*3`,提示路径与
   消费点都用 worker 的同一个 `_lwd_build_up_recv_request` 构造 ⇒ 与边侧
   `isend` 两端 numel 必然一致(HCCL P2P 要求等长,aux 帧同 seqno)。
3. **一个 seqno 只挂一条 recv**:`start_early_irecv` 与 `get_or_post_early_recv`
   都在 `_early_recv_lock` 下判重(重复提示、提示迟到都只是 no-op)。同通道两条
   irecv 会抢发送端那一条 `isend`。
4. **水位是前缀**:通道 FIFO + `_order_after`(每个 op 的流等待前驱的完成事件)
   保证完成序 = seqno 序,所以"已完成集合"恒为前缀,一个最大水位即精确表示。
   `is_irecv_complete(channel, seqno) == seqno <= watermark[channel]`。
5. **条目生命周期**:登记(SUBMIT)→ 消费(POST)与上报(DONE + REPORT)各打一个
   标记,**两者都齐**才移除;上报线程对已消费条目继续轮询 `done()`,所以水位
   不会因为被消费而断档。非端点 rank 不登记(None)。
6. **在飞提前收有上限(显存边界)**:一条已挂 recv 占一块整批大小的接收 buffer
   (主帧 + aux 帧)。这个上限与**边侧 pending 闸门同源** ——
   `LwdEdgeScheduler._LWD_MAX_PENDING_CHUNKS` 按"未消费 chunk(seqno)"限制边侧
   领先量,云侧最多也就需要这么多条;名额满的提示先排队
   (`_early_recv_pending`,只存字段不占显存),前面的批随 SO 被消费时归还名额并
   立刻补挂。归还点是"随 SO 被消费"(= 已下发):闸门放行第 N 批后
   `execute_model` 立刻要取这条 future,名额若拖到"收完"才还,N+1 的 recv 可能
   还没挂上,闸门就会跑到 recv 前面;提示还排在队里时,消费点就地把它挂上。
   > host 暂存阶段起:接收落点改为**轮换的接收 pinned + D2H**,DONE 的语义
   > 升级为"数据落在云 host 且槽已归还"(见 `lwd_host_staging.md` §5.2)。
7. **等待不进前向**:闸门放行时数据(主帧 + aux 帧)已在 worker 的 recv buffer
   里,`future.wait(timeout=60)` 只是取结果的兜底,实测 ≈0ms。

## 6. 与之前两版的关系

| | 共享内存轮询版(上一轮) | 参考版(round2 `EdgeCloudCommService`) | 本版 |
|---|---|---|---|
| 提示下发 | 写 `/dev/shm` slot + `msync` | hint MQ | hint MQ(同参考) |
| 收完上报 | 写 shm `done_seq` | irecv_done MQ | irecv_done MQ(同参考) |
| worker 侧线程 | 自建 `lwd-pre-recv` 200µs 轮询 | 复用通信线程,`dequeue(0.5ms)+sleep(2ms)` | 同参考 |
| 在飞名额 | worker 侧 `_MAX_INFLIGHT=3` 逻辑 | 无边侧窗口约束(边侧自限) | 同参考 |
| 消费点 | `take_future` + `note_scheduled` | `get_or_post_early_recv` | 同参考 |
| 依赖 | 文件路径 / mmap / flush 对齐 / 结构体布局 | 两条 MQ + 水位 | 同参考 |

选择参考版的理由:路径更短(shm 那版要维护 mmap 布局、page 对齐 flush、在飞
名额状态机),日志更全(四段可直接对时间),而且与本仓已有的
`LwdChannel`/`LwdCommFuture` 语义天然对齐(`future.done()` 本来就是纯 CPU
`event.query()`)。

## 7. 启动顺序(为什么这样放)

```
引擎进程                                     worker 进程(fork/spawn)
────────                                     ─────────────────────
1 建 hint MQ(writer) + 写环境变量
2 fork workers ─────────────────────────────► 3 init_device
                                                 └ attach_hint_mq()(reader)
                                                 └ create_report_mq()(writer)
                                                 └ 起通信线程
4 wait_for_ready ◄── READY 载荷(含 irecv_done_handle)
5 挂 done MQ reader
6 引擎 IO 线程起 → register_hint_sender
7 预告到达 → HINT ─────────────────────────► SUBMIT(挂 irecv)
8 每步排空 done MQ ◄── DONE ──────────────── 上报
```

* hint MQ 的 handle 走**环境变量**:worker 创建入口
  (`make_worker_process` / `WorkerProc.__init__`)签名不动。
* done MQ 的 handle 走 **READY 握手**:方向相反(worker 是 writer),顺手。
* 两条 MQ 都**不做 `wait_until_ready()`**:reader 挂在 worker 的
  `init_device` / 引擎 READY 之后,而 MQ 建时已声明 1 读 1 写,ring 有 64 槽,
  先写的消息留在环里,reader 挂上后自然读到。

## 8. 定位方法(日志)

### 8.1 先确认"日志真的能打出来"(logger 归属)

vLLM 只在 **名为 `"vllm"` 的 logger** 上挂 handler(`vllm/logger.py`
`DEFAULT_LOGGING_CONFIG`:`handlers=["vllm"], propagate=False`),子 logger 靠向上
传播拿到它;根 logger 没有 handler。所以:

* `init_logger("vllm.xxx")` / `init_logger(__name__)`(模块在 `vllm` 包内,名字
  形如 `vllm.v1...`) → **能打**;
* `from vllm.logger import logger`(= `"vllm.logger"`) → **能打**;
* `init_logger(__name__)`(模块在 `vllm_ascend` 包内,名字形如
  `vllm_ascend.xxx`) → **打不出来**(传不到 `"vllm"`)。

现场日志(`pre_recv_log/*.log`)的来源文件统计印证了这一点:

| 进程 | 实际打出过日志的文件 | 用的 logger 形态 |
|---|---|---|
| EngineCore | `lwd_cloud_mixed_scheduler.py`、`lwd_cloud_engine.py`、`multiproc_executor.py`、`core.py`、`shm_broadcast.py` | `init_logger(__name__)`(vllm 包内) |
| Worker_* | `multiproc_executor.py`、`lwd_cloud_worker.py`、`lwd_cloud_model_runner.py` | 同上 + `from vllm.logger import logger` |

因此本次**每条提前收日志都放在对应进程里已被实测证明能打出来的模块中**,统一
`[Lwd][pre-recv]` tag:

| 事件 | 发射文件(进程) | 日志 |
|---|---|---|
| 建 hint MQ / 挂 done reader | `multiproc_executor.py`(引擎) | `hint MQ created ...` / `done MQ attached ...: OK` |
| 分支开关 | `lwd_cloud_engine.py`(引擎) | `armed=%s scheduler=%s` |
| 提示通道注册 | `lwd_cloud_engine.py`(引擎) | `hint sender registered (PD_mix -> hint MQ)` |
| HINT | `lwd_cloud_engine.py`(引擎 IO 线程) | `HINT seqno=.. num_tokens=.. ts=..` |
| 水位推进 | `lwd_cloud_engine.py`(引擎主循环) | `watermark advanced channel=.. seqno=..` |
| 闸门 | `lwd_cloud_mixed_scheduler.py`(引擎) | `gate seqno=.. watermark=.. ready=.. ts=..` |
| 端点判定 / 线程起停 | `lwd_cloud_worker.py`(worker) | `worker rank=.. endpoint=.. hidden=..` / `comm thread starting ...` / `comm thread up ...` |
| SUBMIT / DONE | `lwd_cloud_worker.py`(worker 通信线程) | `SUBMIT channel=.. seqno=.. num_tokens=.. ts=..` / `DONE-RECV channel=.. seqno=.. ts=..` |
| POST / 补挂 | `lwd_cloud_worker.py`(worker 前向入口) | `POST seqno=.. reqs=.. tokens=..` / `submit at consume seqno=..` |

`pre_recv.py` 自己只留"本该不发生"的诊断(坏提示、`HINT DROPPED`),它的 logger
名是 `vllm.lwd.pre_recv`(显式落在 `vllm.` 命名空间内,不走 `__name__`)。

host 暂存阶段新增的日志(`[Lwd][host-stage]`,见 `lwd_host_staging.md`):

| 事件 | 文件(进程) | 日志 |
|---|---|---|
| 逐请求行数上报 | `lwd_cloud_worker.py`(worker) | `ARRIVED seqno=.. [(req, rows), …] host_MB=..` |
| 台账更新 | `lwd_cloud_engine.py`(引擎) | `ARRIVED ledger update ..` |
| 存储池就绪 | `lwd_cloud_worker.py`(worker) | `stager ready slots=.. max_numel=..` |
| 可见集谓词挡掉 | `lwd_cloud_mixed_scheduler.py`(引擎) | `GATE seqno=.. visible=N/M` |
| 本步下发量 | `lwd_cloud_mixed_scheduler.py`(引擎) | `WINDOW seqno=.. windows=.. avail=.. room_left=..` |
| 请求结束释放 | `lwd_cloud_worker.py`(worker) | `RELEASED req=.. bytes=.. host_MB=..` |
| abort 冻结 | `lwd_cloud_worker.py`(worker) | `ABORT-SYNC req=.. consumed=../..` |

**显存相关的三条日志**(现场直接核对"在飞 buffer 有没有失控"):

| 日志 | 读法 |
|---|---|
| `SUBMIT ... inflight=N queued=M` | N = 当前已挂 recv 条数 = 在飞整批 buffer 数,**应恒 ≤ 上限**;M = 排队等名额的提示数(不占显存) |
| `stats seqno=.. inflight=N queued=M` | 每次下发(消费)后的现值,可看名额是否随下发正常归还 |
| `queue stuck: oldest queued seqno=..`, `N slot(s) held by seqno=..` | 每 5s 最多一条:队列挂不上且名额全被占住,直接点名是哪几个 seqno 占着位子 |

> 非端点 rank 的 `POST` 降到 DEBUG(每批多行 INFO 是纯噪音)。要按 rank 看进
> 前向的时刻/离散时,`VLLM_LOGGING_LEVEL=DEBUG` 即可看到那几行。

### 8.2 一条批的完整链路(日志原文)

以 seqno=1(1 个请求、3510 token、带 mrope)为例,日志里依次出现这几行
(`[文件:行号]` 是 vLLM formatter 自带,行号会随代码改动变化):

```
(EngineCore pid=1)  INFO [lwd_cloud_engine.py:260] [Lwd][pre-recv] HINT seqno=1 num_tokens=3510 aux_rows=1024 ts=3889.561
(Worker_PP1_TP0)    INFO [lwd_cloud_worker.py:400] [Lwd][pre-recv] SUBMIT channel=lwd_up seqno=1 num_tokens=3510 aux_rows=1024 inflight=1 queued=0 ts=3889.570
(Worker_PP1_TP0)    INFO [lwd_cloud_worker.py:452] [Lwd][pre-recv] DONE-RECV channel=lwd_up seqno=1 ts=3889.610
(Worker_PP1_TP0)    INFO [lwd_cloud_worker.py:458] [Lwd][pre-recv] REPORT channel=lwd_up seqno=1 ts=3889.610
(EngineCore pid=1)  INFO [lwd_cloud_engine.py:455] [Lwd][pre-recv] watermark advanced channel=lwd_up seqno=1 ts=3889.611
(EngineCore pid=1)  INFO [...lwd_cloud_mixed_scheduler.py:221] [Lwd][pre-recv] gate seqno=1 watermark=1 ready=True ts=3889.612
(Worker_PP1_TP0)    INFO [lwd_cloud_worker.py:236] [Lwd][pre-recv] POST seqno=1 reqs=1 tokens=3510
```

| 你要看的 | 日志 | 谁打(进程) | 含义 |
|---|---|---|---|
| **下发提前收(引擎侧)** | `HINT seqno=.. num_tokens=.. aux_rows=.. ts=..` | 引擎 IO 线程 | 预告到达 → 提示入队(**打点在入队之前**);`aux_rows` = 本批 mrope 行数(0 = 纯文本) |
| **下发提前收(真挂 recv)** | `SUBMIT channel=lwd_up seqno=.. num_tokens=.. aux_rows=.. inflight=.. queued=.. ts=..` | 云端点 worker 通信线程 | `submit_recv` 已挂出 HCCL irecv(主帧 + aux 帧);`inflight` = 在飞条数 |
| **收完** | `DONE-RECV channel=lwd_up seqno=.. ts=..` | 同上 | `future.done()`(`event.query()`)转真 = 两帧数据都完整落地接收缓冲 |
| **上报给 engine** | `REPORT channel=lwd_up seqno=.. arrived=.. ts=..` | 同上 | 把 `(channel, seqno)`(+ host 暂存阶段的逐请求行数)写进 done MQ |
| **engine 收到上报** | `watermark advanced channel=lwd_up seqno=.. ts=..` | 引擎主循环(step 开头排空) | 水位推进 —— 这才算 engine 知道了 |
| **闸门放行** | `gate seqno=.. watermark=.. ready=.. ts=..` | 云调度器(PD_mix) | ready=True 才下发该 prefill 批(chunk) |
| **下发到前向** | `POST seqno=.. reqs=.. tokens=..` | worker 前向入口 | SO 到达并取到那条提前挂好的 future |

辅助/边界行(同一个 tag,平时不用看):

| 日志 | 何时出现 |
|---|---|
| `hint MQ created ...` / `done MQ attached ...: OK` / `armed=True scheduler=mixed` / `hint sender registered (PD_mix -> hint MQ)` | 启动自证(引擎侧 4 条) |
| `worker rank=.. endpoint=.. hidden=..` / `comm thread starting ...` / `comm thread up ...` | 启动自证(worker 侧 3 条,每个 rank 一行) |
| `stats seqno=.. inflight=.. queued=..` | 每次下发(消费)后的名额现值 |
| `queue stuck: oldest queued seqno=.., N slot(s) held by seqno=[..]` | 队列挂不上且名额被占住(每 5s 最多一条) |
| `submit at consume seqno=..` | 提示迟到/没到,消费点就地补挂(功能不受影响) |
| `HINT DROPPED: no sender registered seqno=..` | 引擎没装提示通道(正常只在相位调度分支) |
| `malformed hint .. skipped` / `hint MQ dequeue failed` / `done MQ drain failed` | 异常诊断 |
| `POST ... bound=non-endpoint`(**DEBUG**) | 非端点 rank;要按 rank 看进前向离散度时开 `VLLM_LOGGING_LEVEL=DEBUG` |

一条 grep 看全链路:

```bash
grep '\[Lwd\]\[pre-recv\]' cloud.log | grep -E 'seqno=1 |gate seqno=1'   # 换 seqno
grep -c '\[Lwd\]\[pre-recv\] REPORT' cloud.log                          # 应等于 DONE 条数
```

### 8.3 起来没起来:一条命令自证

```bash
LOG=cloud.log bash check_pre_recv_armed.sh
```

逐项 OK/MISSING:引擎侧 4 项(hint MQ 建好 → `armed=True` → 提示通道注册 →
done MQ 挂上)、worker 侧 3 项(端点判定 → 线程 starting → 线程 up)、首批四段
(HINT → SUBMIT → DONE → watermark → gate ready=True → POST)。任何 MISSING 都
直接给出"下一步看哪里"。

### 8.4 四段耗时

```bash
LOG=cloud.log bash timeline_pre_recv.sh
```

产物:

* `pre_recv_timeline.txt`:逐 seqno 五列时刻 + 四段耗时
  `hint2sub / sub2done / done2gate / gate2post`;
* `pre_recv_summary.txt`:四段 p50/p90/p99/max + 计数 + 在飞/排队最大值。

**四段各是什么、量级该是多少**(别把 `SUBMIT -> POST` 当成传输时间):

| 段 | 含义 | 预期量级 |
|---|---|---|
| `hint2sub` | 引擎下发提示 → worker 通信线程挂出 recv | 名额空时几 ms(通信线程轮询节拍);**尾巴大 = 提示在排队等名额**(`queued` 就是排队深度) |
| `sub2done` | 挂出 recv → 数据落地(`event.query()` 转真) | 36MB/批 @10Gbps ≈ **29ms** + 边侧开始发送的延迟 + RTT(实测 p50/p90 = 41/42ms) |
| `done2gate` | 上报 → 引擎排空 done MQ → 闸门放行 | 0 ~ 几 ms(排空在步首);数据落在步内则等下一个步边界(≈ 半步~一步) |
| `gate2post` | 放行 → SO 到 worker 前向入口 | 队列+传输几 ms;**worker 还在跑上一批时 ≈ 上一批剩余时间** |
| `done2post` | `done2gate + gate2post` = 等下游一拍 | 稳态 ≈ 一个云侧步长(实测 p50 597ms / 云侧步长 ~0.6s) |

所以 `SUBMIT -> POST = sub2done + done2gate + gate2post`,它包含
"边侧什么时候真正开始发" + "云侧要等到下一个步边界才下发"这两块等待 ——
在云侧单步 0.5~0.6s 的运行里,稳态下它落在"半拍"量级(几十~几百 ms)是正常的,
**不是带宽**。要单独看网络,只看 `sub2done`,并和边侧日志按 seqno 对齐:

```bash
# 边侧(同一个 seqno)的发送时刻与提交耗时;submit_send 若达几百 ms
# = 边侧 isend 在等云侧挂 recv(HCCL rendezvous),那才是链路锁步
grep '\[Lwd\]\[perf\] embed seqno=' edge.log | head
```

`sub2done` 也要注意它含"边侧还没开始发"的那段:提前收只保证 **recv 已挂好**,
数据是边侧排到那一步才发的;真要把"等边侧"和"在线上"分开,就看边侧那条
`embed seqno=.. total=..ms` 与云侧 `DONE-RECV` 的时间差(两台同机,`ts=` 同源)。

正常形态:

| 现象 | 含义 |
|---|---|
| `HINT == SUBMIT == POST` | 每条预告都提前挂上了 recv,且都下发消费了 |
| `DONE-RECV` / `gate(ready)` 各少最后 1~3 条 | 尾部还在飞(边侧窗口内) |
| `gate ready=False` 少量、且 `done2gate` 只有步级(几 ms~几十 ms) | 闸门正常等待,decode 在自转 |
| `submit at consume > 0` | 提示迟到/没到,消费点补挂(功能不受影响,但提前收没吃到) |
| `HINT DROPPED > 0` | 引擎没装 sender(检查是否走了相位调度分支) |
| `up-recv-wait dur` 明显 >0 | 闸门没生效(数据未收完就下发了) |
| worker 侧只有 `worker rank=.. endpoint=False` | 该 rank 不是端点,行为正确 |
| `comm thread starting` 有、`up` 没有 | 线程体在 `set_device` 处挂了,异常栈在 stderr |

`grep '\[Lwd\]\[pre-recv\]' cloud.log | head -30` 即可看到启动 + 首批的完整链路。

## 9. 离线自检(无需 NPU / 无需 vllm)

| 脚本 | 验什么 |
|---|---|
| `_pre_recv_path_check.py` | 五个文件的接线形状、启动顺序、只挂 PD_mix、无旧设计残留,以及 **§8.1 的日志可达性**(每个事件都落在实测能打出来的模块里、tag 统一、没有 `vllm_ascend.*` 静默 logger) |
| `_pre_recv_link_check.py` | 加载**真实** `pre_recv.py`,验提示↔请求形状(含逐请求行表与 `offsets`)、坏提示、水位语义、`is_irecv_complete` 为纯判据 |
| `_pre_recv_worker_check.py` | AST 抽出 worker 的方法实跑状态机:只挂一条 recv、消费复用同一条 future、丢提示仍上报、条目释放时机,以及 host 暂存阶段的拆分/上报/abort/释放 |
| `_pre_recv_e2e_check.py` | 引擎+worker+链路的整条时间线:未收完 → 只 decode;收完 → 同一步/下一步放行并复用 future;以及按请求分配规则 |
| `_pre_recv_w2_check.py` | **W2 专项**:下发量不得超过已收量(见 `lwd_host_staging.md` §5.3.1c 与 `lwd_host_staging.md` §7) |
| `_pre_recv_host_stage_check.py` / `_pre_recv_mrope_check.py` | host 暂存状态机 / mrope 切片逐位比对 |

```bash
python _pre_recv_path_check.py && python _pre_recv_link_check.py \
  && python _pre_recv_worker_check.py && python _pre_recv_e2e_check.py
```

> 这些脚本是开发期放在工作区根目录的独立脚本(无 torch/NPU 依赖),
> **当前不在本仓**;需要随改动一起提交时,放进 `tests/` 相应位置即可。

现场侧两个脚本(需要日志,输入输出都在当前目录):

```bash
LOG=cloud.log bash check_pre_recv_armed.sh     # 链路起没起来(逐项 OK/MISSING)
LOG=cloud.log bash timeline_pre_recv.sh        # 四段耗时分布 + 计数
```

## 10. 取舍与边界

* **没有超时、没有兜底分支**:recv 挂上就一定能收完(边侧按同一批号发),
  需要"发不出去就退化"的场景不存在;真出问题会表现为 `gate ready=False`
  持续增长 + `submit at consume` 计数,看日志即可定位。
* **在飞上限是显存红线,不是性能旋钮**:上限之外只排队不挂 recv。历史上
  一次 OOM 就是这里失控 —— 按提示无界挂 recv 时,在飞 buffer 数 = notify 积压
  深度(实测 20+),每块几十 MB,把 KV cache 之外的余量吃光后,前向里第一个
  大分配(`rejection_sampler` 的 `target_logits.clone()`)就失败。现以
  `SUBMIT ... inflight=` 与 `queue stuck` 两条日志把这条边界做成可见的。
  注:同一轮排查里 `gpu_memory_utilization=0.95` 把 KV cache 顶到 49.27 GiB、
  运行期只剩 ~1 GiB,是那次 OOM 的主因;改成 0.90 后性能与稳定性都正常。
* **abort 与名额**:若某个批的 notify 被 abort 摘掉而它的 recv 已挂出,这条
  条目既不会被消费也不会释放 ⇒ 占着 1 个名额(显存有界:最多上限块 buffer),
  并且队首若卡在它后面会出现 `queue stuck` 告警。要彻底消掉需要控制面把
  "该 seqno 不会来数据"告诉 worker(通道已有 `skip_seqno` 口子,目前无人调用),
  本次不做。
* **边侧的 pending 闸门(本分支已有)**:`LwdEdgeScheduler`
  `_LWD_MAX_PENDING_CHUNKS` 按"未消费 chunk(seqno)"限制边侧领先量 ——
  一个 seqno = 一个云步消费单位,批内全部请求收到首 token 即算消费。云侧的在飞
  上限跟着取同值,两侧同源,`pending_notify` 因此天然压在个位数。边侧真实在途
  水位看**边侧**日志:`[Lwd][control-flight] running=.. awaiting=.. cloud_active=..`
  (那是"云侧在负责的请求数",= 云侧 running+waiting,与 `max_num_seqs` 那道准入
  闸门是两回事)。
* **2ms 节拍**:提示下发与水位上报各约 2ms 量级的可见延迟,换来通信线程不抢
  CPU/GIL;相对 prefill 批(几百 ms)可忽略。
* **只挂 PD_mix**:相位调度分支不发提示(也没有闸门在等),它的行为与改造前
  完全一致 —— 逃生通道保持原样。
* **不新增环境变量/配置**:唯一的 `VLLM_ASCEND_LWD_RECV_HINT_HANDLE` 是 MQ
  handle 的进程内传递通道,不是开关;默认即生效,无需配置。
* **单 DP**:与既有实现一致,多 DP 不在本次范围。
* **abort 与 UP 通道的既有边界**:边侧的 `LwdRangeNotify` 是在**排程该 chunk
  批的那一步**发布的(与 embeds 发送同步步),所以"预告了但永不发 embeds"只可能
  出现在极窄的 abort 竞态窗口里。真出现时,该 seqno 的 recv 永不完成 ⇒ 通道
  FIFO(`_order_after` 让后一个 op 等前一个的完成事件)会停在那一条,闸门不再
  放行 prefill(decode 照常)。这与改造前同源(旧版同样"预告即挂 recv",只是
  表现为注入侧 `future.wait(60s)` 超时),本次不引入新的异常/兜底分支;定位靠
  `[Lwd][pre-recv] gate ... ready=False watermark=..` 与 `queue stuck` 两条日志。

## 11. 迁移自 `pre_recv` 的差异(本次迁移做了什么)

| 维度 | `prefill_only/pre_recv`(源) | 本树(`pre_recv_mm`) |
|---|---|---|
| 分支 | prefill_only(整体组批) | prefill_only_**chunked_prefill_mm** |
| 批与 seqno | 一个 notify = 一个 prefill 批 | 一个 notify = 一个 **chunk 批**(`LwdRangeItem.offset`),仍是一个 seqno = 一次 UP send |
| UP 主帧尺寸 | `num_tokens * H` | **hash 载荷公式** `hash_payload_numel(num_tokens, H, hash_layer_count, num_experts_per_tok)`(embeds + 可选专家 ID 字节) |
| UP 第二帧 | 无 | **aux 帧**:mrope positions `[n,3]` int64,与主帧同 seqno;提示新增 `aux_rows`(后续 mrope 改走控制面,aux 帧退役) |
| 尺寸公式位置 | `pre_recv.recv_request_from_hint(hint, hidden_size)` | worker 的 `_lwd_build_up_recv_request(seqno, num_tokens, aux_rows)`(依赖本机 hash 配置);提示只带"事实" |
| 边侧在途窗口 | 无(靠云侧上限兜显存) | **已有** `_LWD_MAX_PENDING_CHUNKS`(chunk/seqno 粒度),云侧上限与它同源 |
| 消费点 | `_lwd_up_post_recvs` 直接 `submit_recv` | 端点改走 `get_or_post_early_recv`;非端点保持 `future=None` 占位 + 注入时 TP 广播 |
| 观测工具 | `timeline_pre_recv1.sh` / `check_pre_recv_armed.sh` | 同名脚本(本树叫 `timeline_pre_recv.sh`),日志字段多了 `aux_rows`;检查脚本按 `vllm-ascend*` / `vllm_v23*` 前缀自动识别仓目录 |

**没有搬过来的东西**:源树里"无 mm 的单帧请求构造"、以及任何把尺寸公式放进
`pre_recv.py` 的做法 —— 本分支尺寸依赖 hash 配置,必须留在 worker,否则两侧会
各算一遍。
