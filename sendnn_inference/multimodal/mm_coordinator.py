"""MMCoordinator: Central coordinator for vision encoder de-duplication.

This module implements the coordinator pattern described in ADR-001 to eliminate
O(TP x N) duplicate vision encoder computation across TP workers.

Architecture:
- submission_queue: EngineCore -> rank-0 (multiprocessing.Queue, created pre-fork)
- result_queues: rank-0 encoder thread -> each rank's poller (multiprocessing.Queue, created pre-fork)
- SharedMemory: rank-0 -> all ranks for tensor data

The coordinator:
1. Receives MM encoding requests via submission_queue (rank-0 only)
2. Runs vision encoder on rank-0 only via thread pool
3. Shares results via SharedMemory to other ranks
4. Puts result metadata on result_queues[rank] for each rank
5. Poller threads on each rank read results and cache locally

Important: Both submission_queue and result_queues must be created BEFORE4
worker processes are spawned (pre-fork) so they are properly inherited.
"""

import time
import torch
import threading
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
from typing import Any, Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


def torch_dtype_to_numpy(torch_dtype: torch.dtype):
    """Convert torch dtype to numpy dtype."""
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.uint16,  # bfloat16 not directly supported, use uint16
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
    }
    return mapping.get(torch_dtype, np.float32)


@dataclass
class MMEncodingResult:
    """Result metadata sent via result queue."""
    request_id: str
    shm_name: Optional[str] = None
    shape: Optional[tuple] = None
    dtype: Optional[torch.dtype] = None
    error: Optional[str] = None


class MMCoordinator:
    """Coordinates vision encoder de-duplication across TP workers.

    This class is instantiated inside each worker process after fork.
    Queues are passed in already created (pre-fork).

    Usage:
        # On rank-0 worker
        coordinator = MMCoordinator(
            is_rank0=True,
            model_runner=model_runner,
            tp_size=4,
            submission_queue=submission_queue,  # Created pre-fork
            result_queues=result_queues,  # Created pre-fork
        )
        coordinator.set_model_and_utils(fms_model, mm_model_utils)
        coordinator.start()

        # On other workers
        coordinator = MMCoordinator(
            is_rank0=False,
            tp_size=4,
            result_queue=my_result_queue,  # Created pre-fork
        )
        coordinator.start()
        embeddings = coordinator.get_embedding(request_id, input_ids, mm_features)
    """

    def __init__(
        self,
        is_rank0: bool,
        tp_size: int,
        model_runner: Optional[Any] = None,
        submission_queue: Optional[mp.Queue] = None,
        result_queues: Optional[List[mp.Queue]] = None,
        result_queue: Optional[mp.Queue] = None,
    ):
        self.is_rank0 = is_rank0
        self.tp_size = tp_size
        self.model_runner = model_runner
        self.device = model_runner.device if model_runner else None

        # Queues (created pre-fork, passed in)
        self._submission_queue: Optional[mp.Queue] = submission_queue
        self._result_queues: Optional[List[mp.Queue]] = result_queues
        self._result_queue: Optional[mp.Queue] = result_queue

        # Thread pool for encoding (rank-0 only)
        self._executor: Optional[ThreadPoolExecutor] = None

        # FMS model and MM utils for encoding (rank-0 only)
        self._fms_model: Optional[Any] = None
        self._mm_model_utils: Optional[Any] = None

        # Encoding futures cache: request_id -> Future
        self._encoding_futures: Dict[str, Future] = {}
        self._futures_lock = threading.Lock()

        # SharedMemory tracker to prevent premature cleanup
        self._shm_registry: Dict[str, shared_memory.SharedMemory] = {}
        self._shm_lock = threading.Lock()

        # Local embeddings cache: request_id -> tensor
        self._local_embeddings: Dict[str, torch.Tensor] = {}
        self._local_errors: Dict[str, Exception] = {}
        self._local_events: Dict[str, threading.Event] = {}
        self._cache_lock = threading.Lock()

        # Poller thread
        self._poller_thread: Optional[threading.Thread] = None
        self._shutdown = False

        # Submission listener thread (rank-0 only)
        self._submission_listener: Optional[threading.Thread] = None

    def set_model_and_utils(self, fms_model: Any, mm_model_utils: Any) -> None:
        """Set the FMS model and MM utils for encoding (rank-0 only)."""
        assert self.is_rank0
        self._fms_model = fms_model
        self._mm_model_utils = mm_model_utils

    def start(self) -> None:
        """Start the coordinator.

        On rank-0:
        - Starts thread pool for encoding
        - Starts submission listener thread
        - Starts poller thread to broadcast results

        On other ranks:
        - Starts poller thread to receive results
        """
        if self.is_rank0:
            # Validate we have what we need
            if self._submission_queue is None:
                raise RuntimeError("rank-0 requires submission_queue")
            if self._result_queues is None:
                raise RuntimeError("rank-0 requires result_queues")

            logger.info(
                "MMCoordinator.start() on rank-0: _result_queues has %d queues",
                len(self._result_queues)
            )

            # Start thread pool
            # Use 32 workers: enough for concurrent encoding without excessive overhead
            self._executor = ThreadPoolExecutor(
                max_workers=32, thread_name_prefix="mm-encoder"
            )
            logger.info("MMCoordinator: ThreadPoolExecutor created with max_workers=32")

            # Start submission listener thread
            self._submission_listener = threading.Thread(
                target=self._listen_submission_queue,
                daemon=True,
                name="mm-submission-listener",
            )
            self._submission_listener.start()

            # Start poller thread (rank-0 also needs to receive its own results)
            self._result_queue = self._result_queues[0]  # Rank 0's result queue
            self._start_poller()

            logger.info("MMCoordinator started on rank-0")
        else:
            # Non-rank-0: just start poller
            if self._result_queue is None:
                raise RuntimeError("non-rank-0 requires result_queue")
            self._start_poller()
            logger.info("MMCoordinator poller started on non-rank-0")

    def _listen_submission_queue(self) -> None:
        """Listen for encoding requests from EngineCore (rank-0 only)."""
        logger.info("Submission listener thread started on rank-0")
        while not self._shutdown:
            try:
                # EVENT-DRIVEN: Block until submission arrives (no polling!)
                item = self._submission_queue.get(block=True, timeout=None)
                if item is None:
                    # Shutdown sentinel
                    logger.info("Submission listener received shutdown sentinel")
                    break

                request_id, prompt_token_ids, mm_features = item
                logger.info("Submission listener received request %s (%d tokens)", request_id[:16], len(prompt_token_ids))

                # Ensure model is set (may be set lazily)
                if self._fms_model is None:
                    if self.model_runner is not None:
                        self.set_model_and_utils(
                            self.model_runner.model.fms_model,
                            self.model_runner.model.mm_model_utils,
                        )
                    else:
                        logger.error("model_runner is None, cannot set model")
                        continue

                # Create input tensor and submit encoding
                input_ids = torch.tensor(prompt_token_ids, dtype=torch.int64).unsqueeze(0)
                self.submit_encoding(request_id, input_ids, mm_features)

            except Exception as e:
                if not self._shutdown:
                    logger.error("Submission listener error: %s", e, exc_info=True)

        logger.info("Submission listener thread exiting")

    def _start_poller(self) -> None:
        """Start the poller thread."""
        self._poller_thread = threading.Thread(
            target=self._poll_results,
            daemon=True,
            name="mm-poller",
        )
        self._poller_thread.start()

    def submit_encoding(
        self, request_id: str, input_ids: torch.Tensor, mm_features: Any
    ) -> None:
        """Submit an MM encoding request (rank-0 only).

        Thread-safe and idempotent - if request is already being processed,
        this is a no-op.
        """
        assert self.is_rank0, "submit_encoding must only be called on rank-0"

        with self._futures_lock:
            if request_id in self._encoding_futures:
                logger.debug("Request %s already being processed, skipping", request_id[:16])
                return
            if request_id in self._local_embeddings:
                logger.debug("Request %s already completed, skipping", request_id[:16])
                return

            # Create event for this request
            with self._cache_lock:
                self._local_events[request_id] = threading.Event()

            # Submit to thread pool
            future = self._executor.submit(
                self._encode_and_broadcast,
                request_id,
                input_ids,
                mm_features,
            )
            self._encoding_futures[request_id] = future
            logger.info("Submitted MM encoding for request %s", request_id[:16])

    def _encode_and_broadcast(
        self, request_id: str, input_ids: torch.Tensor, mm_features: Any
    ) -> None:
        """Run encoding and broadcast result to all ranks."""
        logger.info(
            "_encode_and_broadcast START for %s: _result_queues=%s (len=%d if not None)",
            request_id,
            "None" if self._result_queues is None else "List",
            len(self._result_queues) if self._result_queues else 0
        )
        try:
            # Run vision encoder
            embeddings = self._encode_mm(input_ids, mm_features)

            # Create SharedMemory for result
            shm_name, shm = self._create_shm_for_tensor(embeddings)

            # Track SHM
            with self._shm_lock:
                self._shm_registry[shm_name] = shm

            # Broadcast result metadata to all ranks
            result = MMEncodingResult(
                request_id=request_id,
                shm_name=shm_name,
                shape=embeddings.shape,
                dtype=embeddings.dtype,
            )

            logger.info(
                "Broadcasting result for %s to %d ranks (shm=%s)",
                request_id, len(self._result_queues) if self._result_queues else 0, shm_name
            )

            if not self._result_queues:
                logger.error("CRITICAL: _result_queues is empty! Cannot broadcast results!")
                return

            for i, queue in enumerate(self._result_queues):  # type: ignore
                try:
                    queue.put(result)
                    logger.debug("Broadcast to rank %d successful", i)
                except Exception as e:
                    logger.error("Failed to broadcast to rank %d: %s", i, e, exc_info=True)

        except Exception as e:
            logger.error(f"MM encoding failed for {request_id}: {e}", exc_info=True)

            # Broadcast error
            error_result = MMEncodingResult(
                request_id=request_id,
                error=str(e),
            )
            for queue in self._result_queues:  # type: ignore
                queue.put(error_result)

            # Store error locally
            with self._cache_lock:
                self._local_errors[request_id] = e
                if request_id in self._local_events:
                    self._local_events[request_id].set()

    def _encode_mm(self, input_ids: torch.Tensor, mm_features: Any) -> torch.Tensor:
        """Run vision encoder on the given MM features."""
        assert self._fms_model is not None
        assert self._mm_model_utils is not None
        assert self.device is not None

        logger.info("ACTUAL ENCODING HAPPENING - rank-0 calling vision encoder")

        # Ensure input_ids is on correct device
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        # Add batch dimension if needed
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # mm_features is already a list from the request, don't wrap it again
        # Call the FMS model
        t0 = time.time()
        embeddings = self._mm_model_utils.get_maybe_mm_embeddings(
            fms_model=self._fms_model,
            input_ids=input_ids,
            mm_features=mm_features if isinstance(mm_features, list) else [mm_features],
            is_decode=False,
        )
        t_elapsed = time.time() - t0
        logger.info("ACTUAL ENCODING COMPLETE - took %.2fms", t_elapsed * 1000)

        # Ensure embeddings have batch dimension
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)

        logger.debug("MM encoding complete: %s, %s", embeddings.shape, embeddings.dtype)
        return embeddings

    def _create_shm_for_tensor(
        self, tensor: torch.Tensor
    ) -> Tuple[str, shared_memory.SharedMemory]:
        """Create SharedMemory segment for a tensor."""
        # Ensure tensor is contiguous and on CPU
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Create shared memory
        size = tensor.numel() * tensor.element_size()
        shm_name = f"spyre_mm_{hash(tensor.shape) % 10000000}_{time.time_ns() % 100000}"
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)

        # Single copy directly into shared memory via a numpy view over shm.buf.
        # Avoids the intermediate tobytes() allocation + slice-assign memcpy.
        tensor_np = tensor.detach().numpy()
        shm_view = np.ndarray(tensor_np.shape, dtype=tensor_np.dtype, buffer=shm.buf)
        shm_view[:] = tensor_np

        logger.debug("Created SHM %s for tensor %s", shm_name, tensor.shape)
        return shm_name, shm

    def _poll_results(self) -> None:
        """Poll result queue and cache results locally."""
        logger.info("Poller thread started")

        def process_result(result: MMEncodingResult) -> None:
            """Process a single result (error or success)."""
            request_id = result.request_id

            if result.error:
                # Store error
                with self._cache_lock:
                    if request_id not in self._local_events:
                        self._local_events[request_id] = threading.Event()
                    self._local_errors[request_id] = RuntimeError(result.error)
                    self._local_events[request_id].set()
                logger.error("MM encoding failed for %s: %s", request_id[:16], result.error)
            else:
                # Read from SharedMemory
                try:
                    shm = shared_memory.SharedMemory(name=result.shm_name)

                    # Single copy: build a numpy view directly over shm.buf and
                    # clone into a heap-owned tensor that outlives the SHM segment.
                    # Avoids the bytes(shm.buf[:size]) + np.copy() double memcpy.
                    shm_view = np.ndarray(
                        tuple(result.shape),
                        dtype=torch_dtype_to_numpy(result.dtype),
                        buffer=shm.buf,
                    )
                    tensor = torch.from_numpy(shm_view).clone()

                    # Cache locally (keep on CPU, model runner will move to device)
                    with self._cache_lock:
                        # Get or create event - MUST NOT replace existing event
                        event = self._local_events.get(request_id)
                        if event is None:
                            event = threading.Event()
                            self._local_events[request_id] = event

                        self._local_embeddings[request_id] = tensor
                        self._shm_registry[request_id] = shm  # Keep reference
                        event.set()  # Set the SAME event that workers are waiting on

                    logger.debug("Cached embedding for %s (shape=%s)", request_id[:16], tensor.shape)
                except Exception as e:
                    logger.error("Poller failed to process result for %s: %s", request_id, e)
                    # Store error and set event so waiting threads don't hang
                    with self._cache_lock:
                        if request_id not in self._local_events:
                            self._local_events[request_id] = threading.Event()
                        self._local_errors[request_id] = e
                        self._local_events[request_id].set()

        while not self._shutdown:
            try:
                # Block with small timeout for responsiveness
                # timeout=None can have latency issues with multiprocessing.Queue
                result: MMEncodingResult = self._result_queue.get(block=True, timeout=0.001)  # type: ignore

                # Check for shutdown sentinel
                if result is None:
                    break

                process_result(result)

                # Fast-drain: process all available results immediately
                while True:
                    try:
                        result = self._result_queue.get_nowait()  # type: ignore
                        if result is None:  # Shutdown sentinel
                            self._shutdown = True
                            break
                        process_result(result)
                    except mp.queues.Empty:
                        break  # No more results, go back to blocking wait

            except mp.queues.Empty:
                # Timeout - no result available, continue loop
                continue
            except Exception as e:
                if not self._shutdown:
                    logger.error("MMCoordinator poller error: %s", e, exc_info=True)

    def get_embedding(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        mm_features: Any,
        timeout: float = 60.0,
    ) -> torch.Tensor:
        """Request MM embedding from coordinator (BLOCKING - use get_embedding_if_ready() instead).

        This method blocks until encoding completes. Prefer get_embedding_if_ready()
        for non-blocking operation.

        Returns cached tensor instantly if already available.
        Raises TimeoutError if encoding doesn't complete within timeout.
        Raises RuntimeError if encoding failed.
        """
        # Check if already cached
        with self._cache_lock:
            if request_id in self._local_embeddings:
                logger.debug("get_embedding(%s): returning cached result", request_id)
                return self._local_embeddings[request_id].clone()
            if request_id in self._local_errors:
                raise RuntimeError(self._local_errors[request_id])

        # Fallback: If rank-0 and encoding not yet submitted, submit it now
        # This handles cases where the scheduler callback didn't fire
        # (e.g., request was queued before reaching add_request())
        if self.is_rank0:
            # Check if submission needed WITHOUT holding lock to avoid deadlock
            needs_submission = False
            with self._futures_lock:
                if request_id not in self._encoding_futures:
                    needs_submission = True

            if needs_submission:
                logger.warning(
                    "MM encoding for %s was not pre-submitted, submitting now (fallback)",
                    request_id
                )
                # submit_encoding is idempotent and acquires its own lock
                self.submit_encoding(request_id, input_ids, mm_features)

        # RANK-0 FAST PATH: Check if encoding future is done
        # This avoids waiting for the poller thread to process the result
        if self.is_rank0:
            with self._futures_lock:
                future = self._encoding_futures.get(request_id)
                if future is not None:
                    try:
                        # Wait for future with timeout
                        future.result(timeout=timeout)
                        logger.debug("get_embedding(%s): rank-0 future completed", request_id)

                        # Future completed, result should be in cache now
                        # (either via poller or we'll read it directly)
                        with self._cache_lock:
                            if request_id in self._local_embeddings:
                                embedding = self._local_embeddings.pop(request_id)
                                self._local_events.pop(request_id, None)
                                return embedding
                            if request_id in self._local_errors:
                                error = self._local_errors.pop(request_id)
                                self._local_events.pop(request_id, None)
                                raise RuntimeError(error)

                        # If not in cache yet, poller will process it soon
                        # Fall through to event wait below
                        logger.debug("get_embedding(%s): rank-0 future done but not in cache yet, waiting for poller", request_id)
                    except TimeoutError:
                        raise TimeoutError(
                            f"MM encoding for {request_id} did not complete within {timeout}s (rank-0 future timeout)"
                        )
                    except Exception as e:
                        logger.error("get_embedding(%s): rank-0 future raised exception: %s", request_id, e)
                        raise RuntimeError(f"MM encoding failed: {e}")

        # All ranks (including rank-0 if future not done yet) wait for poller to receive result
        # CRITICAL: Create event and check cache atomically to avoid race condition
        with self._cache_lock:
            # Check if result already arrived (poller processed it before we got here)
            if request_id in self._local_embeddings:
                logger.debug("get_embedding(%s): found in cache after future check", request_id)
                embedding = self._local_embeddings.pop(request_id)
                self._local_events.pop(request_id, None)  # Clean up event
                return embedding
            if request_id in self._local_errors:
                error = self._local_errors.pop(request_id)
                self._local_events.pop(request_id, None)  # Clean up event
                raise RuntimeError(error)

            # Create event if it doesn't exist yet
            if request_id not in self._local_events:
                self._local_events[request_id] = threading.Event()
                logger.debug("get_embedding(%s): created new event", request_id)
            event = self._local_events[request_id]

            # Check again if result arrived while we were creating the event
            if request_id in self._local_embeddings:
                logger.debug("get_embedding(%s): found in cache after event creation", request_id)
                embedding = self._local_embeddings.pop(request_id)
                self._local_events.pop(request_id, None)
                return embedding
            if request_id in self._local_errors:
                error = self._local_errors.pop(request_id)
                self._local_events.pop(request_id, None)
                raise RuntimeError(error)

        # Wait for poller to set the event
        logger.debug("get_embedding(%s): waiting on event (timeout=%s)", request_id, timeout)
        if not event.wait(timeout=timeout):
            raise TimeoutError(
                f"MM encoding for {request_id} did not complete within {timeout}s (event wait timeout)"
            )

        # Check result
        with self._cache_lock:
            if request_id in self._local_errors:
                error = self._local_errors.pop(request_id)
                self._local_events.pop(request_id, None)
                raise RuntimeError(error)
            if request_id in self._local_embeddings:
                logger.debug("get_embedding(%s): returning result after event wait", request_id)
                embedding = self._local_embeddings.pop(request_id)
                self._local_events.pop(request_id, None)
                return embedding

        raise RuntimeError(f"Encoding completed but no result found for {request_id}")

    def release(self, request_id: str) -> None:
        """Release resources for a completed request (rank-0 only)."""
        if not self.is_rank0:
            return

        with self._futures_lock:
            self._encoding_futures.pop(request_id, None)

        with self._cache_lock:
            self._local_embeddings.pop(request_id, None)
            self._local_errors.pop(request_id, None)
            self._local_events.pop(request_id, None)

            # Release SHM
            shm = self._shm_registry.pop(request_id, None)
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                    logger.debug("Released SHM for request: %s", request_id)
                except Exception as e:
                    logger.warning("Failed to release SHM for %s: %e", request_id, e)

    def shutdown(self) -> None:
        """Shutdown the coordinator and cleanup resources."""
        self._shutdown = True

        # Send shutdown sentinel to wake up blocking poller thread
        if self._result_queue is not None:
            try:
                self._result_queue.put(None)  # Wake up blocking get()
            except Exception as e:
                logger.warning("Failed to send shutdown sentinel: %s", e)

        # Wait for poller
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=5.0)
            if self._poller_thread.is_alive():
                logger.warning("Poller thread did not exit cleanly")

        # Wait for submission listener
        if self._submission_listener is not None:
            self._submission_listener.join(timeout=5.0)
            if self._submission_listener.is_alive():
                logger.warning("Submission listener thread did not exit cleanly")

        # Shutdown executor
        if self._executor is not None:
            self._executor.shutdown(wait=False)

        # Cleanup SHM
        with self._shm_lock:
            for shm_name, shm in self._shm_registry.items():
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.warning("Failed to cleanup SHM %s: %e", shm_name, e)
            self._shm_registry.clear()

        logger.info("MMCoordinator shutdown complete")


def cleanup_stale_shared_memory() -> None:
    """Clean up any stale shared memory segments from crashed runs.

    Called on worker startup to remove /dev/shm/spyre_mm_* segments.
    """
    import os
    import glob

    shm_dir = "/dev/shm"
    pattern = os.path.join(shm_dir, "spyre_mm_*")

    for shm_path in glob.glob(pattern):
        try:
            shm = shared_memory.SharedMemory(name=os.path.basename(shm_path))
            shm.close()
            shm.unlink()
        except Exception:
            pass  # Ignore cleanup failures
