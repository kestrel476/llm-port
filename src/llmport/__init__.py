"""Подключение к языковым моделям: контракт, адаптеры, поведения.

Слои намеренно разделены и имеют разную судьбу:

- `contract` — маленький и стабильный, его импортируют все;
- `adapters` — специфика вендоров, независимы друг от друга;
- `behaviours` — надстройки над контрактом: повторы, фолбэк, события.

Сервис берёт то, что ему нужно, а не всё сразу. Прошлая попытка вынести общий код
(`llmconnector` в lorium) не прижилась именно потому, что выносили реализацию
целиком: она не подошла следующему сервису, её правили на месте, и копии разошлись.
"""

from llmport.contract import (
    AsyncLLMProvider,
    Capabilities,
    Completion,
    LLMProvider,
    Message,
    Request,
    ToolCall,
    ToolSpec,
    Usage,
)
from llmport.errors import (
    AuthError,
    BadRequestError,
    ContentFilteredError,
    FatalError,
    LLMError,
    MalformedResponseError,
    ModelNotFoundError,
    RateLimitedError,
    RequestTimeoutError,
    ServerError,
    TransientError,
    TransportError,
)

__all__ = [
    "AsyncLLMProvider",
    "AuthError",
    "BadRequestError",
    "Capabilities",
    "Completion",
    "ContentFilteredError",
    "FatalError",
    "LLMError",
    "LLMProvider",
    "MalformedResponseError",
    "Message",
    "ModelNotFoundError",
    "RateLimitedError",
    "Request",
    "RequestTimeoutError",
    "ServerError",
    "ToolCall",
    "ToolSpec",
    "TransientError",
    "TransportError",
    "Usage",
]
