# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common base class for Lwd edge/cloud schedulers: phase predicates and
the visible-set scheduling primitive (three-queue remove - schedule -
reassemble).

Inherits AsyncScheduler; native scheduling behavior and isinstance checks
stay unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request


def lwd_spec_factor(vllm_config) -> int:
    """MTP budget factor: each decode request consumes 1+k tokens per step
    (draft tokens take budget and KV); constantly 1 without a spec config.
    The sole home of it (edge/cloud share one source; the spec factor
    follows the cloud-side LwdCloudParamsNotify); main's vllm/config/lwd is
    a topology snapshot without this derived value, so it lives in the
    base-class module alongside the scheduling primitive."""
    spec_config = getattr(vllm_config, "speculative_config", None)
    num_spec_tokens = (
        getattr(spec_config, "num_speculative_tokens", None)
        if spec_config is not None
        else None
    )
    return (num_spec_tokens or 0) + 1


class LwdReqPhase(Enum):
    """Working phase of a request (same criterion on edge and cloud):
    PREFILL = embedding not fully sent (num_computed < num_prompt), DECODE
    = waiting for the cloud-side sampling return."""

    PREFILL = auto()
    DECODE = auto()


class LwdBaseScheduler(AsyncScheduler):
    """Common base class for edge/cloud schedulers (an AsyncScheduler
    subclass)."""

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Native-compatible scheduling entry: step_with_batch_queue passes
        throttle_prefills positionally; the signature converges here, and
        edge/cloud subclasses implement only one-step role scheduling. The
        throttle hint is not consumed -- each role's dispatch gate decides
        on its own (on par with mm)."""
        return self._lwd_schedule_step()

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """One role-specific scheduling step; edge/cloud subclasses own it."""
        raise NotImplementedError

    @staticmethod
    def _lwd_req_phase(request: Request) -> LwdReqPhase:
        """The sole phase criterion, shared by all queue-by-phase scans on
        edge and cloud."""
        if request.num_computed_tokens < request.num_prompt_tokens:
            return LwdReqPhase.PREFILL
        return LwdReqPhase.DECODE

    def _lwd_reqs_in_phase(self, phase: LwdReqPhase, *queues) -> Iterator[Request]:
        """Scan multiple queues by phase (terminated requests skipped -- the
        books may still satisfy the criterion)."""
        for queue in queues:
            for request in queue:
                if not request.is_finished() and self._lwd_req_phase(request) is phase:
                    yield request

    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """Create a new scheduling policy queue and load reqs into it."""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule(
        self,
        req_ids: list[str],
        *,
        token_budget_cap: int | None = None,
        chunk_threshold: int | None = None,
    ) -> SchedulerOutput:
        """Isolated scheduling: schedule only the requests named by req_ids
        -- the rest are temporarily removed from the three queues and
        hidden, then reassembled into their original queues after the step
        (visible-set mechanism).

        waiting/skipped sources go through native admission
        (allocate/state transition/accounting all intact), running sources
        go through resume. Reassembly: still-scheduled requests are
        appended after the hidden running, preempted ones go to the waiting
        tail, skipped ones go to the skipped queue head.

        token_budget_cap / chunk_threshold only tighten, never loosen, the
        native limits: the edge side passes a cap to reproduce the picker's
        chunk split, the cloud side passes the dual clamp to reproduce
        in-batch boundaries (threshold = max item in the batch + budget =
        decode exact consumption + Σchunk). Hidden running requests are
        invisible to native concurrent admission; slots are deducted by the
        hidden count to prevent over-issue (edge-side picker free_seq and
        cloud-side preflight are the first line of defense; this is defense
        in depth)."""
        picked = {self.requests[req_id] for req_id in req_ids}
        from_running = [req for req in self.running if req in picked]
        from_waiting = [req for req in self.waiting if req in picked]
        from_skipped = [req for req in self.skipped_waiting if req in picked]
        self.running = [req for req in self.running if req not in picked]
        self.waiting.remove_requests(from_waiting)
        self.skipped_waiting.remove_requests(from_skipped)
        saved = (self.running, self.waiting, self.skipped_waiting)
        # Put the visible set in place and schedule it alone
        self.running = from_running
        self.waiting = self._lwd_new_queue(from_waiting)
        self.skipped_waiting = self._lwd_new_queue(from_skipped)
        saved_cap = self.max_num_running_reqs
        self.max_num_running_reqs = max(0, saved_cap - len(saved[0]))
        saved_budget = self.max_num_scheduled_tokens
        if token_budget_cap is not None:
            self.max_num_scheduled_tokens = min(saved_budget, max(0, token_budget_cap))
        sched_cfg = self.scheduler_config
        saved_threshold = sched_cfg.long_prefill_token_threshold
        if chunk_threshold is not None:
            sched_cfg.long_prefill_token_threshold = chunk_threshold
        out = None
        try:
            out = super().schedule()
        finally:
            self.max_num_running_reqs = saved_cap
            self.max_num_scheduled_tokens = saved_budget
            sched_cfg.long_prefill_token_threshold = saved_threshold
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            preempted_ids = (
                set(out.preempted_req_ids)
                if out is not None and getattr(out, "preempted_req_ids", None)
                else set()
            )
            if preempted_ids and post[1]:
                # Native preempts one at a time, each prepend pushing the
                # earlier victims further forward - so the post-schedule
                # waiting order lists later-preempted requests FIRST.
                # Re-prepend in reverse to restore the original preemption
                # order at the HEAD of the saved queue (FCFS fairness:
                # a preempted request must not queue behind requests the
                # edge admitted later).
                preempted = [
                    r for r in post[1] if r.request_id in preempted_ids
                ]
                rest = [
                    r for r in post[1] if r.request_id not in preempted_ids
                ]
                self.waiting.extend(rest)
                for req in reversed(preempted):
                    self.waiting.prepend_request(req)
            else:
                self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out
