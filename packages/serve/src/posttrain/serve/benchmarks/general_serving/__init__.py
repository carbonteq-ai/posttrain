"""Definition and builder for the representative serving prompt corpus.

The definition is intentionally import-safe: importing this module describes
the corpus and its reproducibility contract, but does not import the builder,
download inputs, or write package resources.  Materialization code lives in
``build`` and is invoked explicitly by a maintainer or the generic workload
materializer.
"""

from .definition import GENERAL_SERVING_V1, GeneralServingCorpusDefinition, GeneralServingSource

__all__ = ["GENERAL_SERVING_V1", "GeneralServingCorpusDefinition", "GeneralServingSource"]
