from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from qtpy.QtCore import QObject, Signal, Slot


@dataclass(slots=True)
class Sam31Job:
    """One SAM3.1 multiplex propagation request for the persistent worker.

    The adapter is intentionally passed in instead of created here so the existing
    backend/cache code remains the single source of truth for model-dir/device/
    threshold configuration.
    """

    adapter: Any
    backend: Any
    image_data: Any
    bundle: Any
    propagation_direction: str = "both"
    windows_compatibility_mode: bool = False
    debug_diagnostics: bool = False
    no_write_benchmark: bool = False


class Sam31Worker(QObject):
    """Long-lived QObject worker for SAM3.1 multiplex propagation.

    Keep this object inside one persistent QThread.  That lets the expensive
    SAM3.1 video predictor stay in the same execution thread across runs.
    """

    result_ready = Signal(object)
    session_ready = Signal(object)
    log_message = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._busy = False
        self._last_adapter_id: int | None = None
        self._thread_id: int | None = None

    @Slot(object)
    def run_job(self, job: Sam31Job) -> None:
        if self._busy:
            self.failed.emit("SAM3.1 worker is already running.")
            return

        self._busy = True
        try:
            self._run_job(job)
        except Exception as exc:  # pragma: no cover - protects Qt worker boundary
            self.failed.emit(str(exc))
        finally:
            self._busy = False
            self.finished.emit()

    def _run_job(self, job: Sam31Job) -> None:
        adapter = job.adapter
        backend = job.backend
        current_thread_id = threading.get_ident()
        self._thread_id = current_thread_id

        self._log(backend, f"SAM3.1 persistent worker thread_id={current_thread_id}.")

        if self._last_adapter_id is not None and self._last_adapter_id != id(adapter):
            self._log(
                backend,
                "SAM3.1 persistent worker received a different adapter; "
                "the new adapter will load its own video predictor.",
            )
        self._last_adapter_id = id(adapter)

        predictor_loaded = getattr(adapter, "video_predictor", None) is not None
        loaded_thread_id = getattr(backend, "video_predictor_thread_id", None)

        if predictor_loaded and loaded_thread_id not in {None, current_thread_id}:
            # This should not happen once all multiplex jobs use this persistent
            # worker, but keep the guard so stale predictors from old code paths
            # are not reused across threads.
            self._log(
                backend,
                "SAM3.1 video predictor was loaded outside the persistent worker; "
                f"unloading before same-thread reload. loaded_thread_id={loaded_thread_id}, "
                f"current_thread_id={current_thread_id}.",
            )
            try:
                adapter.unload()
            except Exception as exc:
                self._log(backend, f"SAM3.1 adapter unload before persistent reload failed: {exc}")
            backend.video_predictor_thread_id = None
            predictor_loaded = False

        if not predictor_loaded:
            backend._log_cuda_diagnostics("before adapter.load_video()")
            load_t0 = time.perf_counter()
            adapter.load_video()
            backend.video_predictor_thread_id = current_thread_id
            backend._log_timing("SAM3.1 model load", load_t0)
            backend._log_cuda_diagnostics("after adapter.load_video()")
        else:
            self._log(
                backend,
                "SAM3.1 video predictor already loaded in persistent worker; "
                "reusing cached predictor.",
            )

        if job.debug_diagnostics:
            backend.log_runtime_diagnostics(adapter, stage="before session start")

        config_t0 = time.perf_counter()
        backend.configure_multiplex_predictor(
            adapter,
            windows_compatibility_mode=job.windows_compatibility_mode,
        )
        backend._log_timing("SAM3.1 predictor configuration", config_t0)

        session_t0 = time.perf_counter()
        self._log(backend, "SAM3.1 persistent worker: starting video session.")
        self._log(backend, f"SAM3.1 image source: {backend.describe_image_source(job.image_data)}")
        session = adapter.start_video_session(job.image_data, job.bundle)
        backend._log_timing("SAM3.1 session start", session_t0)
        self._log(
            backend,
            "SAM3.1 session ready: "
            f"{session.session_id}; prompt_frame={job.bundle.image.frame_index or 0}; "
            f"boxes={len(getattr(job.bundle, 'boxes', []) or [])}; "
            f"points={len(getattr(job.bundle, 'points', []) or [])}.",
        )

        if job.debug_diagnostics:
            backend.log_session_diagnostics(adapter, session, stage="after session start")

        prompt_t0 = time.perf_counter()
        prompt_result = adapter.add_video_prompt(job.bundle, session)
        backend._log_timing("SAM3.1 prompt insertion", prompt_t0)

        if job.debug_diagnostics:
            backend.log_session_diagnostics(adapter, session, stage="after prompt insertion")

        prompt_frame_index = getattr(prompt_result, "frame_index", None)
        if prompt_frame_index is None:
            prompt_frame_index = job.bundle.image.frame_index or 0

        prompt_result.metadata["image_layer"] = job.bundle.image.layer_name
        prompt_result.metadata["backend_status"] = "napari_sam3_assistant_sam31_multiplex"
        prompt_result.metadata["stage"] = "prompt"
        prompt_result.metadata["frame_index"] = int(prompt_frame_index)

        if not job.no_write_benchmark:
            self.result_ready.emit(prompt_result)

        backend._log_cuda_diagnostics("before propagation")
        if job.debug_diagnostics:
            backend.log_runtime_diagnostics(adapter, stage="before propagation")

        propagation_t0 = time.perf_counter()
        last_yield_t = propagation_t0
        count = 0
        frame_times: list[tuple[int | None, float]] = []

        for result in adapter.propagate_video(
            job.bundle,
            session,
            direction=job.propagation_direction,
        ):
            count += 1
            now = time.perf_counter()
            dt = now - last_yield_t
            frame_index = getattr(result, "frame_index", None)
            frame_times.append((frame_index, dt))

            if job.debug_diagnostics:
                self._log(
                    backend,
                    "SAM3.1 propagation yield: "
                    f"index={count}, frame={frame_index}, "
                    f"dt={dt:.3f} sec, elapsed={now - propagation_t0:.2f} sec.",
                )

            last_yield_t = now

            if job.no_write_benchmark:
                continue

            result.metadata["image_layer"] = job.bundle.image.layer_name
            result.metadata["backend_status"] = "napari_sam3_assistant_sam31_multiplex"
            result.metadata["stage"] = "propagation"
            if frame_index is not None:
                result.metadata["frame_index"] = int(frame_index)
            result.metadata["propagation_index"] = int(count)
            self.result_ready.emit(result)

        elapsed = time.perf_counter() - propagation_t0
        slow_frames = [(frame, dt) for frame, dt in frame_times if dt > 5.0]
        avg_dt = sum(dt for _frame, dt in frame_times) / max(len(frame_times), 1)
        max_frame, max_dt = max(frame_times, key=lambda item: item[1]) if frame_times else (None, 0.0)

        self._log(
            backend,
            f"SAM3.1 propagation finished in {elapsed:.2f} sec; frames={count}; "
            f"avg_frame_dt={avg_dt:.3f} sec; max_frame={max_frame}; "
            f"max_dt={max_dt:.3f} sec; slow_frames={slow_frames[:20]}; "
            f"no_write={job.no_write_benchmark}.",
        )
        self.session_ready.emit(session)

    def _log(self, backend: Any, message: str) -> None:
        self.log_message.emit(message)