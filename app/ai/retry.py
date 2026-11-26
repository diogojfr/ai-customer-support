import logging
import random
import time
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import (
    ModelConnectionError,
    ModelRateLimitError,
    ModelServerError,
    ModelUnavailableError,
)

logger = logging.getLogger(__name__)

# Errors that usually pass on their own: Groq's tokens-per-minute limit, dropped connections,
# and an overloaded or failing provider. A daily quota (ModelQuotaExceededError), a bad API
# key, or a rejected request does not pass by waiting, so those are raised at once.
TRANSIENT_ERRORS = (
    ModelRateLimitError,
    ModelConnectionError,
    ModelUnavailableError,
    ModelServerError,
)
MAX_ATTEMPTS = 6
BASE_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 60.0


def run_with_retry(agent: Agent[Any, Any], prompt: str) -> AgentRunResult[Any]:
    """Run the agent, waiting and trying again when the model API has a passing problem."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return agent.run_sync(prompt)
        except TRANSIENT_ERRORS as error:
            # A long wait means a daily limit, such as Groq's tokens per day. Blocking the
            # poller for minutes helps nobody: fail now, and the poller retries the message.
            too_long = error.retry_after is not None and error.retry_after > MAX_DELAY_SECONDS
            if attempt == MAX_ATTEMPTS or too_long:
                raise
            delay = _delay(attempt, error.retry_after)
            logger.warning(
                "%s on attempt %d/%d, retrying in %.1fs",
                type(error).__name__, attempt, MAX_ATTEMPTS, delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _delay(attempt: int, retry_after: float | None) -> float:
    # Wait as long as the provider asks, otherwise back off exponentially. The jitter spreads
    # out parallel requests, so they do not all hit the rate limit again at the same moment.
    backoff = BASE_DELAY_SECONDS * 2 ** (attempt - 1)
    delay = retry_after if retry_after is not None else min(backoff, MAX_DELAY_SECONDS)
    return delay + random.uniform(0, 1)
