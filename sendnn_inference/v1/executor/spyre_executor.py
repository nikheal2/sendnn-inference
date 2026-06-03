"""SpyreExecutor: Extends MultiprocExecutor to create MM queues pre-fork.

This executor creates the submission_queue and result_queues before spawning
worker processes, so they are properly inherited by the forked workers.

Architecture:
- submission_queue: EngineCore -> rank-0 (for MM encoding requests)
- result_queues: rank-0 encoder -> each rank's poller (for MM results)

Both queues must exist before worker spawn to avoid daemon process issues.

MM Encoding Submission Flow (ADR-001):
1. Executor creates submission_queue pre-fork
2. Executor sets global MM callback in scheduler module
3. When requests arrive at scheduler (EngineCore), add_request() is called
4. Scheduler calls the global callback with (req_id, prompt_tokens, mm_features)
5. Callback puts (req_id, prompt_tokens, mm_features) on submission_queue
6. rank-0 worker's submission listener thread picks up the item
7. Listener submits encoding to coordinator's thread pool
8. Encoding runs in parallel during other requests' prefill/decode
9. By the time chunk-0 runs, embedding is cached (pipelined!)
"""

import multiprocessing as mp
from typing import Optional, List, Any

from vllm.config import VllmConfig
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.core.sched.output import SchedulerOutput

from vllm.logger import init_logger

logger = init_logger(__name__)

# Manager for creating picklable queue proxies
_mm_manager: Optional[Any] = None
_mm_submission_queue: Optional[Any] = None
_mm_result_queues: Optional[List[Any]] = None


class SpyreExecutor(MultiprocExecutor):
    """Spyre-specific executor that creates MM queues pre-fork."""

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        # Create Manager and queues BEFORE calling super().__init__() which spawns workers
        # Manager().Queue() creates picklable proxy objects that can be passed via RPC
        global _mm_manager, _mm_submission_queue, _mm_result_queues

        # Get TP size to know how many result queues to create
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        logger.info("SpyreExecutor.__init__: tp_size=%d", tp_size)

        # Store as instance variables for use in execute_model
        self._mm_submission_queue = None
        self._mm_manager = None

        # Only create queues if TP > 1 (multi-worker scenario)
        # For TP=1, no de-duplication is needed
        if tp_size > 1:
            # Create a Manager to get picklable queue proxies
            _mm_manager = mp.Manager()
            self._mm_manager = _mm_manager

            # Create submission queue (EngineCore -> rank-0)
            # Using Manager().Queue() creates a proxy that can be pickled
            _mm_submission_queue = _mm_manager.Queue()
            self._mm_submission_queue = _mm_submission_queue

            # Create result queues (rank-0 -> each rank)
            _mm_result_queues = [_mm_manager.Queue() for _ in range(tp_size)]

            logger.info(
                "Created MM Manager queues: 1 submission queue, %d result queues",
                tp_size,
            )

            # Note: We trigger MM encoding from execute_model() instead of scheduler callback
            # because the scheduler runs in EngineCore (different process) where we can't
            # set module-level globals. execute_model() runs in the executor process where
            # we have direct access to the submission_queue.
        else:
            logger.info("TP size is 1, skipping MM queue creation")

        # Now call parent __init__ which spawns workers
        # Parent class will automatically call _post_init_executor() after workers are ready
        logger.info("Calling super().__init__()...")
        super().__init__(vllm_config, monitor_workers)
        logger.info("super().__init__() completed - MM coordinators initialized via _post_init_executor()")

    def _post_init_executor(self) -> None:
        """Called after workers are spawned and ready.

        Pass Manager queue proxies to workers via RPC.
        Manager queues are picklable and can be safely passed across processes.
        """
        logger.info("_post_init_executor called")

        tp_size = self.parallel_config.tensor_parallel_size
        logger.info("_post_init_executor: tp_size=%d", tp_size)

        if tp_size > 1:
            global _mm_submission_queue, _mm_result_queues

            if _mm_submission_queue is None or _mm_result_queues is None:
                logger.error("MM queues are None - cannot initialize coordinators")
                return

            logger.info("_post_init_executor: Passing MM queues to workers via collective_rpc...")

            # Use collective_rpc to pass queues to all workers
            # Manager queue proxies are picklable, so this works
            self.collective_rpc(
                "initialize_mm_coordinator",
                kwargs={
                    "submission_queue": _mm_submission_queue,
                    "result_queues": _mm_result_queues,
                }
            )

            logger.info("MM coordinators initialized on all workers")
        else:
            logger.info("_post_init_executor: TP size is 1, skipping coordinator initialization")

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> Any:
        """Execute model.

        Trigger MM encoding for new requests before dispatching to workers.
        This enables pipelining: encoding starts immediately when requests
        are scheduled, running concurrently with the forward pass.
        """
        # Trigger MM encoding for new multimodal requests
        if self._mm_submission_queue is not None and hasattr(scheduler_output, 'scheduled_new_reqs'):
            for req in scheduler_output.scheduled_new_reqs:
                mm_features = getattr(req, 'mm_features', None)
                if mm_features:
                    try:
                        # NewRequestData uses req_id, not request_id
                        req_id = getattr(req, 'req_id', getattr(req, 'request_id', None))
                        prompt_token_ids = getattr(req, 'prompt_token_ids', getattr(req, 'prompt', []))

                        if req_id is None:
                            logger.warning("Executor: Request has no req_id or request_id, skipping MM encoding")
                            continue

                        # Submit encoding request to rank-0 worker
                        self._mm_submission_queue.put((
                            req_id,
                            prompt_token_ids,
                            mm_features
                        ))
                        logger.info(
                            "Executor: Queued MM encoding for request %s (%d tokens)",
                            req_id,
                            len(prompt_token_ids)
                        )
                    except Exception as e:
                        logger.error(
                            "Executor: Failed to queue MM encoding: %s",
                            e, exc_info=True
                        )

        # Call parent execute_model which dispatches to workers
        return super().execute_model(scheduler_output, non_block)

    def shutdown(self):
        """Properly shut down MM coordinators before parent cleanup."""
        tp_size = self.parallel_config.tensor_parallel_size

        if tp_size > 1:
            try:
                # Shutdown MM coordinators on all workers
                self.collective_rpc(
                    "shutdown_mm_coordinator",
                    timeout=5.0,
                )
            except Exception as e:
                logger.warning("Error shutting down MM coordinators: %s", e)

            # Send shutdown sentinel to submission queue (if it exists)
            global _mm_submission_queue, _mm_manager
            if _mm_submission_queue is not None:
                try:
                    _mm_submission_queue.put(None)  # Shutdown sentinel
                except Exception:
                    pass

            # Shutdown the Manager
            if _mm_manager is not None:
                try:
                    _mm_manager.shutdown()
                    logger.info("MM Manager shutdown complete")
                except Exception as e:
                    logger.warning("Error shutting down MM Manager: %s", e)

        # Call parent shutdown for worker cleanup
        super().shutdown()