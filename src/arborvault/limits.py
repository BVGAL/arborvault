import threading
import time


class RateLimiter:
    """Per-credential, per-process token bucket. Put a shared limiter at the ingress for replicas."""

    def __init__(self, per_minute=60, burst=60, clock=time.monotonic):
        if per_minute <= 0 or burst <= 0:
            raise ValueError("Rate limits must be positive")
        self.rate, self.burst, self.clock = per_minute / 60, burst, clock
        self.buckets = {}
        self.lock = threading.Lock()

    def allow(self, credential):
        with self.lock:
            now = self.clock()
            tokens, last = self.buckets.get(credential, (self.burst, now))
            tokens = min(self.burst, tokens + max(now - last, 0) * self.rate)
            allowed = tokens >= 1
            self.buckets[credential] = (tokens - 1 if allowed else tokens, now)
            # Hot credential rotation must not grow memory forever.
            if len(self.buckets) > 10000:
                self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < 60}
            return allowed
