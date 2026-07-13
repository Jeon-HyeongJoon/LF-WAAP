"""Backward-compatible import path for shared Flow partitioning."""

from waf.infrastructure.flow.partition import PartitionKey, length_bucket, partition_key

__all__ = ["PartitionKey", "length_bucket", "partition_key"]
