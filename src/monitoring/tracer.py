"""
OpenTelemetry distributed tracing.
Exports spans to Jaeger (via OTLP gRPC) in Docker, or console in dev.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

from src.utils.logger import get_logger

logger = get_logger(__name__)
_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str = "rag-agent", otlp_endpoint: str | None = None, enabled: bool = True) -> None:
    global _tracer
    if not enabled:
        _tracer = trace.get_tracer(service_name)
        return

    resource = Resource.create({"service.name": service_name, "service.version": "1.0.0"})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("tracing_otlp_configured", endpoint=otlp_endpoint)
        except Exception as exc:
            logger.warning("tracing_otlp_failed_using_console", error=str(exc))
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    logger.info("tracing_initialized", service=service_name)


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("rag-agent")
    return _tracer


@contextmanager
def traced_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def trace_method(span_name: str | None = None, record_args: bool = False) -> Callable:
    """Decorator to trace a function as an OTel span."""
    def decorator(fn: Callable) -> Callable:
        name = span_name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attrs: dict[str, Any] = {}
            if record_args and kwargs:
                for k, v in kwargs.items():
                    attrs[f"arg.{k}"] = str(v)[:200]
            t0 = time.perf_counter()
            with traced_span(name, attrs) as span:
                result = fn(*args, **kwargs)
                span.set_attribute("duration_ms", round((time.perf_counter() - t0) * 1000, 1))
                return result

        return wrapper
    return decorator
