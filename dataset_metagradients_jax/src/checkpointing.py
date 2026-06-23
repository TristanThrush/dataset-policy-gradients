from abc import ABC, abstractmethod
import os
import shutil
import concurrent.futures as _fut
from typing import Any, Optional, List

import orbax.checkpoint as ocp


class CheckpointerBase(ABC):
    """Abstract base class for checkpointing strategies."""

    @abstractmethod
    def save(self, step: int, train_state: Any) -> None:
        """Save training state at given step."""
        pass

    @abstractmethod
    def restore(self, step: int, template: Any) -> Any:
        """Restore training state from given step."""
        pass

    @abstractmethod
    def wait_until_finished(self, step: Optional[int] = None) -> None:
        """Wait until all pending operations are finished."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the checkpointer and clean up resources."""
        pass


class DiskCheckpointer:
    """Disk-based checkpointing with fully-async save *and* restore."""

    def __init__(self, checkpoint_dir: str, *, max_io_workers: int = 1):
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpointer = ocp.CheckpointManager(self.checkpoint_dir)

        # Thread pool for user-level async restore.
        # NOTE: the checkpoint manager is technically not thread safe, but the threads only exist to execute reload async, so in practice its fine.
        self._executor = _fut.ThreadPoolExecutor(
            max_workers=max_io_workers,
            thread_name_prefix="checkpoint-io",
        )
        self._pending: List[_fut.Future] = []

    def save(self, step: int, train_state: Any) -> None:
        self.checkpointer.save(step, args=ocp.args.StandardSave(train_state))

    def restore(self, step: int, template: Any) -> Any:
        return self.checkpointer.restore(
            step, args=ocp.args.StandardRestore(template)
        )

    def restore_async(self, step: int, template: Any) -> _fut.Future:
        """Kick off a non-blocking restore and return a Future."""
        future = self._executor.submit(
            self.restore, step=step, template=template
        )
        self._pending.append(future)
        return future

    def wait_until_finished(self, step: Optional[int] = None) -> None:
        """Block until *all* async IO (save & restore) is done."""
        self.checkpointer.wait_until_finished()

        for fut in self._pending:
            fut.result()
        self._pending.clear()

    def close(self) -> None:
        """Synchronise, shut down threads and clean temp files."""
        self.wait_until_finished()
        self.checkpointer.close()
        self._executor.shutdown(wait=True)
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)


def create_checkpointer(
    strategy: str = "disk",
    checkpoint_dir: str = "/tmp/checkpoints",
) -> CheckpointerBase:
    """Factory function to create a checkpointer.

    Args:
        strategy: only "disk" is supported.
        checkpoint_dir: directory for disk-based checkpointing.

    Returns:
        CheckpointerBase instance
    """
    if strategy == "disk":
        return DiskCheckpointer(checkpoint_dir)
    raise ValueError(f"Unknown checkpointing strategy: {strategy}")
