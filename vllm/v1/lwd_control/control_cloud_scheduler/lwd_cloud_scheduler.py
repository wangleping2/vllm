"""Cloud scheduler: cpu_offload visible-set scheduling + LWD C2E output.

Direct port of the cpu_offload branch's LwdCloudMixedScheduler (arrival-
driven visible-set filtering + window computation + W2 verification)
merged with our C2E output path (carrier unpack -> pending_c2e).
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdC2eNotify, LWD_NOT_FINISHED
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)

@dataclass
class LwdC2eMeta:
    """Step metadata unpacked from the cloud worker's DOWN carrier; the
    scheduler forwards it to the edge ahead of the DOWN hidden tensor
    (the data plane carries only the tensor itself)."""

    # Total element count of the DOWN hidden tensor for this step.
    hidden_num_elements: int
    # Per-request sampled-token rank within the step's logits ("top-id-th"),
    # one row per request in hidden row order.
    top_id_ths: list[list[int]]
    # Per-request draft accept counts (0 for non-spec steps).
    num_accepted_tokens: list[int]
    # Request ids in hidden row order.
    req_ids: list[str]
    # DOWN channel seqno of this step's packet; the edge posts its matching
    # irecv with this exact value for pairing on the tag-less HCCL wire.
    down_seqno: int




class LwdCloudScheduler(LwdBaseScheduler):
    """Arrival-driven visible-set scheduler (cpu_offload design v8)
    plus the C2E output path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # C2E output queue: update_from_output enqueues, the engine's
        # post_step flushes to the edge in one shot.
        self.pending_c2e: list[LwdC2eNotify] = []
        # Diagnostics hub (spec acceptance observation)
        from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
        self._lwd_diag = LwdControlDiag()

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """cpu_offload's _schedule_impl: arrival-driven window computation
        + visible-set dispatch. Port verbatim from the source branch."""
        from vllm.lwd_logger import lwd_logger
        ledger = getattr(self, "lwd_recv_rows", None) or {}
        decode_ids = self._lwd_collect_decode_requests()
        decode_exact, _ = self._lwd_decode_tokens(decode_ids)
        room = self.max_num_scheduled_tokens - decode_exact
        windowed: list[tuple] = []
        if room > 0:
            for request in itertools.chain(
                self.running, self.waiting, self.skipped_waiting
            ):
                rid = request.request_id
                computed = request.num_computed_tokens
                if computed >= request.num_prompt_tokens:
                    continue
                arrived = int(ledger.get(rid, 0)) - computed
                if arrived <= 0:
                    continue
                n = min(arrived, room)
                if n < 1:
                    break
                # Batch up partial-arrival windows: a window that does
                # not fill the remaining budget while the request's data
                # is still incoming is deferred so it merges with the
                # next chunk - a few-row remnant would otherwise occupy a
                # whole step for its fixed overhead. The tail (data fully
                # arrived) is exempt so the request always finishes, and
                # a preempted-recompute request holds the full recv_rows
                # so it never hits this either.
                if n < room and computed + arrived < request.num_prompt_tokens:
                    continue
                windowed.append((request, n, computed))
                room -= n
        if windowed:
            out = self._lwd_dispatch_windows(windowed, decode_ids, decode_exact)
        else:
            out = self._lwd_schedule(decode_ids)
        return out

    def _lwd_collect_decode_requests(self) -> list[str]:
        return [
            req.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for req in queue
            if req.num_computed_tokens >= req.num_prompt_tokens
        ]

    def _lwd_decode_tokens(self, decode_ids: list[str]) -> tuple[int, dict]:
        per_req: dict[str, int] = {}
        decode_exact = 0
        for rid in decode_ids:
            req = self.requests[rid]
            placeholders = req.num_output_placeholders
            if (
                placeholders > 0
                and req.num_computed_tokens + 2 - placeholders
                >= req.num_prompt_tokens + req.max_tokens
            ):
                continue
            if self.current_step < req.next_decode_eligible_step:
                continue
            d_i = max(
                0,
                min(
                    req.num_tokens_with_spec
                    + placeholders
                    - req.num_computed_tokens,
                    self.max_model_len - 1 - req.num_computed_tokens,
                ),
            )
            if d_i <= 0:
                continue
            per_req[rid] = d_i
            decode_exact += d_i
        return decode_exact, per_req

    def _lwd_dispatch_windows(
        self,
        windowed: list[tuple],
        decode_ids: list[str],
        decode_exact: int,
    ) -> SchedulerOutput:
        """Per-request dispatch: mount lwd_window_cap, compress budget,
        call the native scheduler on the visible set."""
        from vllm.lwd_logger import lwd_logger
        total_windows = sum(n for _r, n, _s in windowed)
        req_ids = decode_ids + [r.request_id for r, _n, _s in windowed]
        saved_budget = self.max_num_scheduled_tokens
        for request, n, _s in windowed:
            request.lwd_window_cap = n
        self.max_num_scheduled_tokens = decode_exact + total_windows
        out = None
        try:
            out = self._lwd_schedule(req_ids)
        finally:
            self.max_num_scheduled_tokens = saved_budget
            for request, _n, _s in windowed:
                request.lwd_window_cap = None
        assert out is not None
        # ---- W2 (design v7 §7): the watchdog over the one silent-
        # miscalculation risk of this design. num_computed still holds
        # the step-start values here (update_from_output has not run);
        # `start` was captured before the native optimistic advance.
        ledger = getattr(self, "lwd_recv_rows", None) or {}
        for request, w, start in windowed:
            rid = request.request_id
            got = int(out.num_scheduled_tokens.get(rid, 0))
            if got == 0:
                continue
            recv_rows = int(ledger.get(rid, 0))
            if recv_rows <= 0:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid}: scheduled "
                    f"{got} tokens but the data plane reported recv_rows=0 "
                    f"(the request must not have entered the batch - the "
                    f"visible-set predicate failed)"
                )
            if got > w or start + got > recv_rows:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid}: native "
                    f"scheduled rows [{start}, {start + got}) - window={w}, "
                    f"recv_rows={recv_rows} (lwd_window_cap failed)"
                )
        native_prefill = sum(
            int(out.num_scheduled_tokens.get(r.request_id, 0))
            for r, _n, _s in windowed
        )
        if native_prefill < total_windows:
            # Legal but worth observing: almost only native KV allocation
            # failure - a more reliable "KV is running out" signal than
            # watching free block counts. The remainder is retried next
            # step (the window recomputes from num_computed).
            lwd_logger.warning(
                "[Lwd][cloud-sched] native under-scheduled: "
                "native prefill=%d < windows=%d (native-side truncation, "
                "typically KV pressure)",
                native_prefill,
                total_windows,
            )
        # Full-prompt mrope positions ride the new requests' data to the
        # runner (wire v3): stashed on the Request at admission, copied
        # here so the runner primes req_state at its first step.
        for d in out.scheduled_new_reqs:
            req = self.requests.get(d.req_id)
            if req is not None:
                d.lwd_mrope_positions = (
                    getattr(req, "lwd_mrope_positions", b"") or b""
                )
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        """After native accounting (including spec observation), unpack the
        carrier into a c2e appended to pending_c2e; the DISABLE_DOWN bypass
        passes through per-request tokens directly; empty steps without a
        carrier are skipped."""
        self._lwd_diag.observe_spec_acceptance(
            scheduler_output, model_runner_output, self.requests
        )
        engine_core_outputs = super().update_from_output(
            scheduler_output, model_runner_output
        )
        carrier = getattr(model_runner_output, "lwd_down_carrier", None)
        if carrier is not None:
            meta = self._lwd_carrier_to_meta(carrier)
            # Deliver committed counts (same source as the finish
            # decision): collector counts are device-side derivatives
            # without check_stop truncation (stop/EOS/max_tokens), and
            # the edge delivery path performs no local stop check — the
            # delivered count must be same-source with the finish
            # decision. Requests absent from the outputs get 0 (the
            # edge's pad-and-skip disposition handles them).
            committed = {
                out.request_id: len(out.new_token_ids)
                for outputs in engine_core_outputs.values()
                for out in outputs.outputs
            }
            meta.num_accepted_tokens = [committed.get(rid, 0) for rid in meta.req_ids]
            LwdControlDiag.cloud_step(self, meta, engine_core_outputs)
            finish_reasons = self._lwd_c2e_finish_reasons(
                meta.req_ids, engine_core_outputs
            )
            # [Lwd][eos-trace] Point A: leave a trace of zero-acceptance rows
            # (for kill root-cause localization; prints the raw row set --
            # zero-acceptance live-row removal was dropped together with the
            # 4-card fix; kept here only as an incidence-rate observation)
            LwdControlDiag.trace_cloud_c2e_rows(
                meta.down_seqno,
                meta.req_ids,
                meta.num_accepted_tokens,
                finish_reasons,
            )
            # Diagnostics bypass (SKIP_SAMPLE): piggyback per-request
            # accepted ids
            token_ids: list[list[int]] = []
            if LwdControlDiag.token_passthrough_enabled():
                token_ids = LwdControlDiag.bypass_token_payload(
                    meta.req_ids, engine_core_outputs
                )
            self.pending_c2e.append(
                LwdC2eNotify(
                    hidden_num_elements=meta.hidden_num_elements,
                    top_id_ths=meta.top_id_ths,
                    num_accepted_tokens=meta.num_accepted_tokens,
                    req_ids=meta.req_ids,
                    finish_reasons=finish_reasons,
                    down_seqno=meta.down_seqno,
                    token_ids=token_ids,
                )
            )
        elif LwdControlDiag.disable_down_enabled():
            self._lwd_append_diag_c2e(engine_core_outputs)
        return engine_core_outputs

    def _lwd_append_diag_c2e(self, engine_core_outputs) -> None:
        """DISABLE_DOWN diagnostics bypass: the cloud does no collect / sends
        no DOWN, but the c2e still drives edge-side delivery as usual --
        hidden=0 marks "no tensor"; row-set extraction belongs to the
        diagnostics class, wire construction stays in the scheduler. An
        empty step with no rows sends no empty notify."""
        rows = LwdControlDiag.disable_down_rows(engine_core_outputs)
        if not rows:
            return
        req_ids = list(rows)
        self.pending_c2e.append(
            LwdC2eNotify(
                hidden_num_elements=0,
                top_id_ths=[],
                num_accepted_tokens=[],
                req_ids=req_ids,
                finish_reasons=self._lwd_c2e_finish_reasons(
                    req_ids, engine_core_outputs
                ),
                down_seqno=-1,
                token_ids=[rows[rid] for rid in req_ids],
            )
        )

    @staticmethod
    def _lwd_carrier_to_meta(carrier: tuple) -> LwdC2eMeta:
        """Unpack the harvested payload into step metadata.

        Payload = a flat int tensor (CPU, harvested by the worker's
        fenced copy: dedicated stream wait_event on the per-step ready
        event, completion event recorded on the copy stream itself),
        laid out as "variable-length segments first, fixed-length tails
        last", so it can be decoded backwards from the tail without
        carrying extra length fields:

            vals = [ranks(variable length: seg_lens[i] rank ids per
                    request, concatenated in request order) ...
                    | counts(n_req accepted) ...
                    | seg_lens(n_req segment lengths)]

        Take n_req=2, seg_lens=[3,1], counts=[5,2] as an example:
            vals = [r0 r1 r2 | r0 | 5 2 | 3 1]
            counts   = vals[-2*n_req : -n_req] -> [5, 2]
            seg_lens = vals[-n_req:]            -> [3, 1]
            ranks    = vals[:-2*n_req]          -> [r0, r1, r2, r0]
        Then slice ranks back into per-request top_id_ths rows by seg_lens
        (here [[r0,r1,r2],[r0]]). The two magic numbers -2*n_req/-n_req are
        the widths of the two fixed-length tails, the "counts segment +
        seg_lens segment"."""
        packed, req_ids, hidden_numel, seqno = carrier
        n_req = len(req_ids)
        vals = packed.tolist()
        counts = vals[-2 * n_req : -n_req]
        seg_lens = vals[-n_req:]
        ranks_flat = vals[: -2 * n_req]
        top_id_ths: list[list[int]] = []
        off = 0
        for seg_len in seg_lens:
            top_id_ths.append(ranks_flat[off : off + seg_len])
            off += seg_len
        return LwdC2eMeta(
            hidden_num_elements=hidden_numel,
            top_id_ths=top_id_ths,
            num_accepted_tokens=list(counts),
            req_ids=list(req_ids),
            down_seqno=seqno,
        )

    @staticmethod
    def _lwd_c2e_finish_reasons(req_ids: list[str], engine_core_outputs) -> list[int]:
        """Per-request finish codes aligned with req_ids: take the code from
        outputs carrying finish_reason, fill gaps in finished_requests with
        ABORT, and NOT_FINISHED for the rest."""
        from vllm.v1.engine import FinishReason

        reasons: dict[str, int] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    reasons.setdefault(out.request_id, int(out.finish_reason))
            for request_id in outputs.finished_requests or ():
                reasons.setdefault(request_id, int(FinishReason.ABORT))
        return [reasons.get(request_id, LWD_NOT_FINISHED) for request_id in req_ids]







