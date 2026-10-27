# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine core: serves the edge of the same DP rank over the control
plane. Edge messages are handled on an ingest thread (range notifies feed the
scheduler and the pre-recv hint, request notifies become native ADDs); step
metadata is flushed to the edge in post_step."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import torch

from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineShutdownState
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
_LWD_INGEST_JOIN_TIMEOUT_S = 5.0


_LwdIngestItem = tuple[LwdPeer, LwdEdgeToCloudMessage]


class LwdCloudEngineCore(LwdBaseEngineCore):
    """Cloud PO engine: control plane + ingest thread, the rest is native."""

    lwd_scheduler_cls = LwdCloudScheduler
    scheduler: LwdCloudScheduler

    def __init__(self, *args: Any, lwd_dp_size: int, **kwargs: Any) -> None:
        # Pre-recv hint MQ must exist before super().__init__ spawns the
        # executor workers (same process): create_hint_mq() publishes the
        # handle through the environment, and only processes spawned after
        # that see it. The sideband MQ exists because rpc_broadcast_mq is
        # drained by the worker's single-threaded busy loop, which sits
        # inside a several-hundred-ms prefill forward - hints queued behind
        # it would lose the whole point of pre-posting the receive.
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import create_hint_mq

        self._lwd_hint_mq = create_hint_mq()
        super().__init__(*args, lwd_dp_size=lwd_dp_size, **kwargs)
        # The cloud endpoint worker created its done-report MQ during
        # init_device; now that the executor is up, fetch its handle over
        # the executor's collective RPC (the executor itself stays free of
        # LWD plumbing). Exactly one worker - the endpoint rank - must
        # report a handle; anything else is a wiring error.
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        handles = [
            handle
            for handle in self.model_executor.collective_rpc(
                "lwd_early_recv_done_handle"
            )
            if handle is not None
        ]
        if len(handles) != 1:
            raise RuntimeError(
                "[LWD] early receive expects one endpoint worker reporting "
                f"completions, got {len(handles)}"
            )
        self._lwd_done_mq = MessageQueue.create_from_handle(handles[0], 0)
        # Arrival ledger + prompt lengths: defined before the scheduler
        # wiring (shared by reference, cpu_offload pattern).
        self._lwd_recv_rows: dict[str, int] = {}
        self._lwd_prompt_len: dict[str, int] = {}
        # Share the arrival ledger with the scheduler (by reference):
        # the drain (this thread) writes, the scheduler's visible-set
        # predicate reads on every step.
        self.scheduler.lwd_recv_rows = self._lwd_recv_rows
        lwd_logger.info("[pre-recv] hint MQ created, done MQ attached via RPC")

    def _lwd_assemble(self) -> None:
        self._lwd_ingest_thread: threading.Thread | None = None
        self._lwd_edge: LwdPeer | None = None
        sched_config = self.vllm_config.scheduler_config
        # Scheduling limits the edge checks (same inputs of the cap/preflight
        # formulas on both sides).
        self._lwd_cloud_params = LwdCloudParamsNotify(
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=lwd_spec_factor(self.vllm_config) - 1,
        )
        self._lwd_register_recv_hint()
        # Edge messages are handled off the IO thread: building a request
        # allocates its placeholder embeds and the pre-recv hint may block,
        # which must not delay step metadata going out.
        self._lwd_ingest_queue: queue.SimpleQueue[_LwdIngestItem | None] = (
            queue.SimpleQueue()
        )
        self._lwd_ingest_thread = threading.Thread(
            target=self._lwd_ingest_loop, name="lwd-cloud-ingest", daemon=True
        )
        self._lwd_ingest_thread.start()
        plane = self._lwd_start_control_plane()
        lwd_logger.info("[cloud] engine assembled: control plane on %s", plane.endpoint)

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_edge_to_cloud(payload)

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: remember the edge and send it the scheduling
        limits before anything else."""
        self._lwd_edge = peer
        self._lwd_send_until_queued(peer, self._lwd_cloud_params)
        lwd_logger.info(
            "[cloud] edge %s registered; scheduling limits sent: %s",
            peer,
            self._lwd_cloud_params,
        )

    def _lwd_send_until_queued(
        self, peer: LwdPeer, message: LwdCloudToEdgeMessage
    ) -> None:
        """Retry a full outbound queue until the message is queued; gives up
        with an error once shutdown is requested, so a stalled edge cannot
        block it."""
        while not self._lwd_send(peer, message):
            if self.shutdown_state != EngineShutdownState.RUNNING:
                lwd_logger.error(
                    "[cloud] %s to edge %s dropped: shutting down with the "
                    "outbound queue full",
                    type(message).__name__,
                    peer,
                )
                return
            time.sleep(_LWD_SEND_RETRY_SLEEP_S)

    def _lwd_register_recv_hint(self) -> None:
        """[LWD][pre-recv] Early receive: the hint delivery channel =
        engine -> cloud endpoint worker comm thread. It goes through
        the sideband MQ this engine creates in __init__ (not
        rpc_broadcast_mq: that one is consumed single-threaded by the
        busy_loop and gets blocked by a several-hundred-ms prefill
        forward, so the hint would no longer be "early")."""
        hint_mq = self._lwd_hint_mq
        if hint_mq is not None:
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                register_hint_sender,
            )

            # Blocking enqueue: once the gate is on, hints are a
            # correctness dependency — drop one and nobody pre-posts
            # that recv, the watermark stops advancing, and batches
            # behind the gate are never dispatched.
            register_hint_sender(lambda hint, mq=hint_mq: mq.enqueue(hint))
            lwd_logger.info("[pre-recv] hint sender registered (PD_mix -> hint MQ)")
        else:
            # PD_mix is the only scheduling mode: a missing hint MQ =
            # hints never go out and the gate never opens; this must be
            # explicitly visible
            lwd_logger.warning(
                "[pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: hand range and request notifies to the ingest thread;
        anything else (incl. abort, not wired yet) is unknown."""
        if not isinstance(msg, (LwdRangeNotify, LwdRequestNotify)):
            return False
        self._lwd_ingest_queue.put((peer, msg))
        return True

    def _lwd_ingest_loop(self) -> None:
        while (item := self._lwd_ingest_queue.get()) is not None:
            try:
                self._lwd_handle_edge_message(*item)
            except Exception as e:
                lwd_logger.exception(
                    "[cloud] ingest of %s failed", type(item[1]).__name__
                )
                self._lwd_fail(f"ingest of {type(item[1]).__name__}: {e!r}")
                return

    def _lwd_handle_edge_message(
        self, peer: LwdPeer, msg: LwdEdgeToCloudMessage
    ) -> None:
        """Ingest thread: range notify -> notify queue + pre-recv hint;
        request notify -> native ADD."""
        lwd_logger.info(
            "[cloud] ingest msg=%s seqno=%s",
            type(msg).__name__,
            getattr(msg, "seqno", "-"),
        )
        # TODO(lwd): several edges per cloud DP rank need requests tracked by
        # the peer that owns them, issue #8.
        if isinstance(msg, LwdRangeNotify):
            # Range notify: only drives the pre-recv hint (the worker
            # posts the UP recv on it). The notify queue is deleted.
            items = msg.items
            # [Lwd][pre-recv] Early receive step 1: the moment the
            # notify arrives, send the cloud worker a hint (seqno +
            # flat token count); the worker's comm thread posts an
            # exact-size UP irecv upon receipt (main-frame hash
            # payload) — no longer waiting for this batch's
            # SchedulerOutput dispatch. num_tokens is the sum over the
            # items. mrope positions already travel with the
            # RangeNotify on the control plane; the UP data plane
            # carries only the main frame, so aux_rows is always 0 —
            # if a nonzero value ever leaked through, the worker would
            # pre-post an aux irecv that never receives data, the
            # watermark would stall, and batches behind the gate would
            # never be dispatched.
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                make_recv_hint,
                post_irecv_hint,
            )

            num_tokens = sum(item.num_tokens for item in items)
            aux_rows = 0
            # Host-staging row table (two coordinate systems, never merged):
            #   rows           = this chunk's row count per request (the
            #                    batch layout derives from it by accumulation);
            #   prompt_offsets = the chunk's start within its OWN prompt
            #                    (the stager's row_base);
            #   prompt_lens    = full prompt rows per request (host buffer
            #                    sizing authority, one-shot allocation).
            req_ids = [it.request_id for it in items]
            rows = [it.num_tokens for it in items]
            prompt_offsets = [it.offset for it in items]
            prompt_lens = [
                self._lwd_prompt_len.get(rid, 0) for rid in req_ids
            ]
            missing = [
                rid for rid, plen in zip(req_ids, prompt_lens) if plen <= 0
            ]
            if missing:
                # The edge admits a request (RequestNotify) before
                # scheduling it (RangeNotify), so the local ledger must
                # already hold every length; a miss is a protocol
                # violation. Skipping the hint leaves the arrival ledger
                # at 0 - loud (the request never schedules) and safe,
                # unlike staging with a wrong-length buffer.
                lwd_logger.error(
                    "[LWD][pre-recv] HINT seqno=%d skipped: missing "
                    "prompt_lens for %s (RangeNotify before metadata)",
                    msg.seqno,
                    missing,
                )
            else:
                # Log before enqueueing: the worker comm thread can pick
                # the hint up within 0.5 ms; if the log came after the
                # enqueue, cross-process log write ordering would make
                # hint2sub go negative, which looks like a timing anomaly.
                lwd_logger.info(
                    "[LWD][pre-recv] HINT seqno=%d num_tokens=%d ts=%.3f",
                    msg.seqno,
                    num_tokens,
                    time.monotonic(),
                )
                post_irecv_hint(make_recv_hint(
                    msg.seqno, num_tokens, aux_rows,
                    req_ids=req_ids, rows=rows,
                    prompt_offsets=prompt_offsets, prompt_lens=prompt_lens))
        elif isinstance(msg, LwdRequestNotify):
            request = self._lwd_build_request(msg)
            # Message v2: full-prompt mrope arrives once at admission - stash
            # it on the request for the runner to prime req_state (replaces
            # the per-chunk slices carried by LwdRangeItem).
            request.lwd_mrope_positions = msg.mrope_positions
            self._lwd_prompt_len[msg.request_id] = msg.num_prompt_tokens
            self._lwd_recv_rows.setdefault(msg.request_id, 0)
            self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
            lwd_logger.info(
                "[LWD] request admitted req=%s prompt=%d",
                msg.request_id, msg.num_prompt_tokens)

    def abort_requests(self, request_ids: list[str]) -> None:
        """Drop held (not-yet-arrived) requests from the hold gate; the rest
        abort natively. Leaving hold entries in would never release (the
        payload stream for an aborted request stops)."""
        for rid in request_ids:
            self._lwd_prompt_len.pop(rid, None)
            self._lwd_recv_rows.pop(rid, None)
        super().abort_requests(request_ids)

    def shutdown(self) -> None:
        # Before the executor goes: a pre-recv hint blocks on its queue.
        if self._lwd_ingest_thread is not None:
            self._lwd_ingest_queue.put(None)
            self._lwd_ingest_thread.join(_LWD_INGEST_JOIN_TIMEOUT_S)
        super().shutdown()

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """Request building (the sole request-creation point): sampling
        params rebuilt from the request notify + placeholder meta
        embeds (shape only, no storage; prompt length is the row
        count; the worker rebuilds a local zeroed buffer from is_meta,
        and the UP inject writes straight into that buffer window).
        The prefix-cache hash chain was not migrated with the control
        plane (that feature lands separately; wire block_hasher up
        then)."""
        sampling_params = self._lwd_build_sampling_params(req_notify)
        # Pass neither real ids nor a fabricated placeholder: hang a
        # meta placeholder with native prompt-embeds semantics (empty
        # storage, pickle carries metadata only, never trips the MQ
        # overflow slow path); rows = prompt length. With ids=None,
        # input_batch automatically sets is_token_ids False for the
        # prompt segment and M-RoPE builds through the plain-text
        # passthrough (matching the scanned values one for one).
        prompt_embeds = torch.empty(
            req_notify.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
            device="meta",
        )
        request = Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
        )
        return request

    @staticmethod
    def _lwd_build_sampling_params(req_notify: LwdRequestNotify) -> SamplingParams:
        """Sampling params rebuilt from the request notify (field-table
        driven); eos_token_id goes through the native backfill entry
        (sets _eos_token_id and adds it to _all_stop_token_ids for the
        min_tokens check; the cloud has no client generation_config,
        so pass empty)."""
        values = {f: getattr(req_notify, f) for f in LWD_SAMPLING_DEFAULTS}
        values["stop_token_ids"] = list(values["stop_token_ids"])
        sampling_params = SamplingParams(**values)
        sampling_params.update_from_generation_config({}, req_notify.eos_token_id)
        LwdControlDiag.cloud_request_admitted(req_notify, sampling_params)
        return sampling_params

    def step_with_batch_queue(self):
        """Drain the early-recv watermark at the head of the step (done
        MQ -> gate readiness predicate); the rest is native."""
        self._lwd_drain_irecv_completions()
        out = super().step_with_batch_queue()
        return out

    def step(self):
        """Same for the synchronous step path: drain the watermark at
        the head of the step, the rest is native."""
        self._lwd_drain_irecv_completions()
        out = super().step()
        return out

    def _lwd_drain_irecv_completions(self) -> None:
        """Drain the done MQ every step: merge the UP recvs the worker
        finished into the channel watermark AND the per-request arrival
        ledger (``_lwd_recv_rows``) - the latter is what the visible-set
        scheduler reads to size each request's window."""
        done_mq = getattr(self, "_lwd_done_mq", None)
        if done_mq is None:
            return
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        items = []
        while True:
            try:
                items.append(done_mq.dequeue(timeout=0))
            except TimeoutError:
                break
            except Exception:
                # The ring gets cancelled at shutdown: one log line is
                # enough; the watermark staying put does not affect
                # the wind-down
                lwd_logger.exception("[pre-recv] done MQ drain failed")
                break
        if not items:
            return
        # Two done-report shapes share this MQ:
        #   * (channel, seqno)              - bare watermark key
        #   * ((channel, seqno), arrived)   - key + per-request cumulative
        #     arrived rows [(req_id, rows), ...] (host-staging publish)
        # Normalize to keys for the watermark merge; the piggybacked rows
        # feed the arrival ledger. Never unpack by arity in a log line
        # (the old `{s for _ch, s in items}` crashed on the piggybacked
        # form).
        watermark_items = []
        per_req = []
        for item in items:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], tuple)
            ):
                key, arrived = item
                watermark_items.append(key)
                per_req.extend(arrived or ())
            else:
                watermark_items.append(item)
        advanced = record_irecv_completions(watermark_items)
        # Per-request arrival ledger: max() merge over CUMULATIVE row
        # counts (idempotent, replay-proof, never pushes past the real
        # received high-water). This is the sole input the visible-set
        # scheduler uses to size each request's next window.
        for req_id, arrived in per_req:
            self._lwd_recv_rows[req_id] = max(
                self._lwd_recv_rows.get(req_id, 0), arrived)
        lwd_logger.info(
            "[LWD][pre-recv] DRAIN n=%d seqnos=%s advanced=%s rows=%s ts=%.3f",
            len(items),
            sorted({item[1] for item in watermark_items}),
            sorted({s for _ch, s in advanced}) if advanced else "[]",
            per_req,
            time.monotonic(),
        )

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        if self.scheduler.pending_c2e:
            assert self._lwd_edge is not None, "c2e produced before any edge"
            for notify in self.scheduler.pending_c2e:
                self._lwd_send_until_queued(self._lwd_edge, notify)
            self.scheduler.pending_c2e.clear()
        super().post_step(model_executed)
