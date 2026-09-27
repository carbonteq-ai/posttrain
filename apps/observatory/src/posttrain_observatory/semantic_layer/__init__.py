"""The semantic layer: runs, updates and rollouts as SQL tables inside Trackio's storage.

Columns are dimensions and measures declared once (metric measures come from the
metric catalog). Doris SQL is the one query language; statements run where the
data lives, through the tracking backend's `ProjectSql`. The short "measures by
dimensions" form compiles to SQL. See ``docs/plan/observation-simplification.md``.
"""

from .compile import compile_query, compile_scope
from .describe import DescribedTable, SemanticDescribeRequest, SemanticDescription, describe_semantics
from .engine import run_semantic_query, run_sql, run_sql_query
from .framework import FRAMEWORK_MODEL
from .model import Aggregation, Dimension, Entity, Measure, Metric, SemanticModel, Source
from .query import QueryError, ResultColumn, SemanticQuery, SemanticResult, SqlQuery
from .views import assemble, plan, table_columns

__all__ = [
    "FRAMEWORK_MODEL",
    "Aggregation",
    "DescribedTable",
    "Dimension",
    "Entity",
    "Measure",
    "Metric",
    "QueryError",
    "ResultColumn",
    "SemanticDescribeRequest",
    "SemanticDescription",
    "SemanticModel",
    "SemanticQuery",
    "SemanticResult",
    "Source",
    "SqlQuery",
    "assemble",
    "compile_query",
    "compile_scope",
    "describe_semantics",
    "plan",
    "run_semantic_query",
    "run_sql",
    "run_sql_query",
    "table_columns",
]
