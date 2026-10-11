"""A learner replaying the cluster log: amber with progress, OK once caught up."""

SLACK = 50  # slots a node may lag and count as caught up


class LogReplay:
    @staticmethod
    def total(statuses: list[dict]) -> int:
        """Log length: the furthest any node has applied."""
        return max((s.get("applied_total", 0) for s in statuses), default=0)

    @classmethod
    def progress(cls, consensus: dict | None, log_total: int) -> tuple[int, int] | None:
        """(applied, total) for a non-voter still behind, else None."""
        if not consensus or not consensus.get("enabled") or consensus.get("is_voter"):
            return None
        done = consensus.get("applied_total", 0)
        return (done, log_total) if log_total - done > SLACK else None

    @staticmethod
    def percent(done: int, total: int) -> int:
        return min(99, done * 100 // total) if total else 0
