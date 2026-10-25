"""Optional LLM answerer, gated by the same grounding checker as the extractive path.

The model is asked to answer with sentences copied verbatim from the retrieved clauses and to cite the
clause id after each sentence. Whatever comes back is split at the citations and every sentence is
checked against the clause it cites; sentences that are paraphrased, fabricated or cite the wrong clause
are moved to ``rejected`` and never shown as the answer. The checker, not the prompt, is the guarantee.
This module is only used when ``CLAUSEWISE_LLM=1`` and an Anthropic API key is configured; tests drive it
with a fake client, so neither the test suite nor the benchmark ever calls the network.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from clausewise.config import Settings, load_settings
from clausewise.grounding import best_support
from clausewise.retrieval import ClauseRetriever
from clausewise.schemas import Answer, AnswerSentence, Clause
from clausewise.textutil import normalize_ws

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions about one contract using only the clauses provided.\n"
    "Rules:\n"
    "1. Reply with one to three sentences copied verbatim from the clauses. Do not paraphrase, summarise, "
    "merge or add words.\n"
    "2. End every sentence with the id of the clause it was copied from in square brackets, for example [c07].\n"
    "3. If no clause answers the question, reply with exactly NO_ANSWER."
)
_CITED_SEGMENT = re.compile(r"(.+?(?:\[c\d{2}\]\s*)+)", re.S)
_CITATION = re.compile(r"\[(c\d{2})\]")


class MessagesAPI(Protocol):
    """The slice of ``anthropic.Anthropic().messages`` this module uses; tests provide a fake."""

    def create(self, **kwargs: Any) -> Any:
        """Create a message."""
        ...


class LLMAnswerer:
    """Answer with an LLM, then keep only sentences the grounding checker can verify."""

    def __init__(self, clauses: list[Clause], retriever: ClauseRetriever, settings: Settings | None = None,
                 messages_api: MessagesAPI | None = None) -> None:  # fmt: skip
        self.clauses = clauses
        self.retriever = retriever
        self.settings = settings or load_settings()
        self._messages = messages_api

    def _api(self) -> MessagesAPI:
        if self._messages is None:
            import anthropic

            self._messages = anthropic.Anthropic().messages
        return self._messages

    def _call(self, question: str, retrieved: list[Clause]) -> str | None:
        context = "\n\n".join(f"[{c.clause_id}] {normalize_ws(c.text)}" for c in retrieved)
        try:
            response = self._api().create(
                model=self.settings.llm_model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Clauses:\n\n{context}\n\nQuestion: {question}"}],
            )
        except Exception as exc:
            log.warning("llm call failed (%s); abstaining", type(exc).__name__)
            return None
        return "".join(getattr(block, "text", "") for block in response.content if getattr(block, "type", "") == "text")

    def answer(self, question: str, k: int = 5) -> Answer:
        """Ask the model, then gate every sentence through the grounding checker."""
        retrieved = self.retriever.search(question, k=k)
        retrieved_ids = [c.clause_id for c in retrieved]
        text = self._call(question, retrieved)
        if not text or "NO_ANSWER" in text:
            return Answer(
                question=question,
                answer_type="clause",
                sentences=[],
                retrieved=retrieved_ids,
                method="llm",
                abstained=True,
            )
        return self.parse(question, text, retrieved)

    def parse(self, question: str, text: str, retrieved: list[Clause]) -> Answer:
        """Split model output at its citations and verify each sentence against the cited clause."""
        by_id = {c.clause_id: c for c in retrieved}
        segments = _CITED_SEGMENT.findall(text) or [text]
        accepted: list[AnswerSentence] = []
        rejected: list[AnswerSentence] = []
        for seg in segments:
            cites = _CITATION.findall(seg)
            sentence = normalize_ws(_CITATION.sub("", seg))
            if not sentence:
                continue
            cited = [by_id[c] for c in cites if c in by_id]
            sup, _clause = best_support(sentence, cited, self.settings.grounding_threshold)
            item = AnswerSentence(
                text=sentence, citations=cites, grounded=bool(cited) and sup.supported, support=min(1.0, sup.score)
            )
            (accepted if item.grounded else rejected).append(item)
        return Answer(question=question, answer_type="clause", sentences=accepted, rejected=rejected,
                      retrieved=[c.clause_id for c in retrieved], method="llm", abstained=not accepted)  # fmt: skip
