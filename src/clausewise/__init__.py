"""clausewise: deterministic contract intelligence.

The package is organised as a pipeline: corpus generation (``clausewise.corpus``), clause segmentation
(``segment``), clause classification (``classify``), entity extraction (``entities``), risk playbook scoring
(``risk``), redline suggestions (``redline``), retrieval (``retrieval``) and grounded question answering
(``qa`` with ``grounding``). Every stage works without network access or an API key; the LLM path in
``llm`` is an optional enhancement that is still gated by the grounding checker.
"""

__version__ = "0.1.0"
