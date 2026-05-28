"""Human-like timing — random jitter, adaptive intervals."""

import asyncio
import random


class BehaviorTiming:
    """Simulates human browsing patterns."""

    @staticmethod
    async def page_visit_delay() -> None:
        """Delay between visiting pages (5-30 seconds)."""
        await asyncio.sleep(random.uniform(5.0, 30.0))

    @staticmethod
    async def search_delay() -> None:
        """Delay between search queries (10-60 seconds)."""
        await asyncio.sleep(random.uniform(10.0, 60.0))

    @staticmethod
    async def micro_pause() -> None:
        """Short pause within a session (2-5 minutes)."""
        await asyncio.sleep(random.uniform(120.0, 300.0))

    @staticmethod
    def adaptive_interval(had_new_listings: bool) -> float:
        """Return next check interval based on activity.

        If new listings were found, check sooner (15-25 min).
        Otherwise, back off to 40-60 min.
        """
        if had_new_listings:
            return random.uniform(900, 1500)  # 15-25 min
        return random.uniform(2400, 3600)  # 40-60 min
