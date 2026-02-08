"""
Efficient batch processing pipeline for large-scale LLM evaluation.

Demonstrates production data engineering patterns:
- Parallel processing with multiprocessing
- Chunked batch processing to manage memory
- Progress tracking and checkpointing
- Error handling and retry logic
"""

from typing import List, Dict, Any, Callable, Optional, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
from pathlib import Path
import time
from tqdm import tqdm


@dataclass
class BatchResult:
    """Result from processing a batch."""

    batch_id: int
    success_count: int
    error_count: int
    outputs: List[Dict[str, Any]]
    errors: List[Dict[str, str]]
    processing_time_ms: float


class BatchProcessor:
    """
    Efficient batch processor for large-scale evaluation.

    Features:
    - Parallel processing (thread or process-based)
    - Automatic batching and chunking
    - Checkpoint/resume capability
    - Progress tracking
    - Error handling with retries
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = 4,
        use_processes: bool = False,
        checkpoint_dir: Optional[str] = None,
        retry_failed: int = 3
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_workers: Number of parallel workers
            use_processes: Use ProcessPoolExecutor (vs ThreadPoolExecutor)
            checkpoint_dir: Directory to save checkpoints
            retry_failed: Number of retries for failed items
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.retry_failed = retry_failed

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Dict[str, Any]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process items in parallel batches.

        Args:
            items: List of items to process
            process_fn: Function to apply to each item
            show_progress: Show progress bar

        Returns:
            Dictionary with results and statistics
        """
        total_items = len(items)
        num_batches = (total_items + self.batch_size - 1) // self.batch_size

        all_outputs = []
        all_errors = []
        start_time = time.time()

        # Check for existing checkpoint
        checkpoint_data = self._load_checkpoint()
        start_batch = checkpoint_data.get("last_batch", 0) if checkpoint_data else 0

        if start_batch > 0:
            print(f"Resuming from batch {start_batch}/{num_batches}")
            all_outputs = checkpoint_data["outputs"]
            all_errors = checkpoint_data["errors"]

        # Process batches
        batches = self._create_batches(items, start_batch)
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            futures = []

            for batch_id, batch in enumerate(batches, start=start_batch):
                future = executor.submit(self._process_batch, batch_id, batch, process_fn)
                futures.append((batch_id, future))

            # Collect results with progress bar
            iterator = tqdm(
                futures,
                desc="Processing batches",
                disable=not show_progress,
                total=num_batches - start_batch
            )

            for batch_id, future in iterator:
                try:
                    batch_result = future.result()
                    all_outputs.extend(batch_result.outputs)
                    all_errors.extend(batch_result.errors)

                    # Save checkpoint
                    if self.checkpoint_dir:
                        self._save_checkpoint(batch_id, all_outputs, all_errors)

                except Exception as e:
                    all_errors.append({
                        "batch_id": batch_id,
                        "error": str(e)
                    })

        total_time_ms = (time.time() - start_time) * 1000

        return {
            "total_items": total_items,
            "success_count": len(all_outputs),
            "error_count": len(all_errors),
            "outputs": all_outputs,
            "errors": all_errors,
            "total_time_ms": total_time_ms,
            "throughput_items_per_sec": total_items / (total_time_ms / 1000)
        }

    def _create_batches(
        self,
        items: List[Any],
        start_batch: int = 0
    ) -> Iterator[List[Any]]:
        """Create batches from items."""
        start_idx = start_batch * self.batch_size
        for i in range(start_idx, len(items), self.batch_size):
            yield items[i:i + self.batch_size]

    def _process_batch(
        self,
        batch_id: int,
        batch: List[Any],
        process_fn: Callable[[Any], Dict[str, Any]]
    ) -> BatchResult:
        """Process a single batch."""
        start_time = time.time()
        outputs = []
        errors = []

        for item in batch:
            for attempt in range(self.retry_failed + 1):
                try:
                    result = process_fn(item)
                    outputs.append(result)
                    break
                except Exception as e:
                    if attempt == self.retry_failed:
                        errors.append({
                            "item": str(item),
                            "error": str(e),
                            "attempts": attempt + 1
                        })
                    else:
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        processing_time_ms = (time.time() - start_time) * 1000

        return BatchResult(
            batch_id=batch_id,
            success_count=len(outputs),
            error_count=len(errors),
            outputs=outputs,
            errors=errors,
            processing_time_ms=processing_time_ms
        )

    def _save_checkpoint(
        self,
        batch_id: int,
        outputs: List[Dict[str, Any]],
        errors: List[Dict[str, str]]
    ):
        """Save checkpoint to disk."""
        if not self.checkpoint_dir:
            return

        checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        checkpoint_data = {
            "last_batch": batch_id,
            "outputs": outputs,
            "errors": errors,
            "timestamp": time.time()
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint from disk."""
        if not self.checkpoint_dir:
            return None

        checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, "r") as f:
            return json.load(f)


# Example usage
if __name__ == "__main__":
    # Simulate evaluation function
    def evaluate_item(item):
        """Simulated evaluation."""
        time.sleep(0.1)  # Simulate API call
        return {
            "input": item,
            "output": f"Result for {item}",
            "score": 0.85
        }

    # Create test data
    test_items = list(range(100))

    # Process with batching
    processor = BatchProcessor(
        batch_size=10,
        max_workers=4,
        checkpoint_dir="./checkpoints"
    )

    results = processor.process(test_items, evaluate_item)

    print(f"\nProcessing complete:")
    print(f"  Total items: {results['total_items']}")
    print(f"  Success: {results['success_count']}")
    print(f"  Errors: {results['error_count']}")
    print(f"  Throughput: {results['throughput_items_per_sec']:.1f} items/sec")
