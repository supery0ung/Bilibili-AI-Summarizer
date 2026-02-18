"""Base class for pipeline steps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .models import QueueItem
from .state import StateManager, now_iso

logger = logging.getLogger("pipeline.step")

class BaseStep:
    """Base class for all pipeline steps."""
    
    def __init__(self, pipeline: 'Pipeline'):
        self.pipeline = pipeline
        self.config = pipeline.config
        self.state = pipeline.state
        self.logger = pipeline.logger
        self.root = pipeline.root

    def load_queue(self) -> list[QueueItem]:
        """Load the processing queue from file."""
        if not self.pipeline.queue_file.exists():
            self.logger.error("No queue file found. Run 'fetch' first.")
            return []
        
        try:
            queue_data = json.loads(self.pipeline.queue_file.read_text(encoding="utf-8"))
            return [QueueItem.from_dict(item) for item in queue_data.get("queue", [])]
        except Exception as e:
            self.logger.error(f"Failed to load queue: {e}")
            return []

    def get_max_items(self, provided_max: Optional[int] = None) -> int:
        """Resolve the maximum number of items to process."""
        if provided_max is not None:
            return provided_max
        return self.config.get("pipeline", {}).get("max_items_per_run", 20)

    def run(self, max_items: Optional[int] = None) -> dict[str, int]:
        """Execute the step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
