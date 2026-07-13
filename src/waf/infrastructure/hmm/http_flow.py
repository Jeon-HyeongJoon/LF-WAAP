"""Backward-compatible import path for HTTP Flow adaptation."""

from waf.infrastructure.flow.http_flow import http_request_to_flow

__all__ = ["http_request_to_flow"]
