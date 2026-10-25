from types import SimpleNamespace

import pytest

from clausewise.llm import SYSTEM_PROMPT, LLMAnswerer
from clausewise.retrieval import ClauseRetriever
from clausewise.schemas import Clause

TEXTS = [
    "Governing Law\n\nThis Agreement shall be governed by the laws of India. The parties submit to the courts at New Delhi.",
    "Termination\n\nEither party may terminate this Agreement on thirty (30) days written notice.",
]
CLAUSES = [
    Clause(clause_id=f"c{i + 1:02d}", start=0, end=len(t), heading=t.split("\n")[0], text=t)
    for i, t in enumerate(TEXTS)
]


class FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


class FailingMessages:
    def create(self, **kwargs):
        raise RuntimeError("network down")


def make(reply, settings):
    api = FakeMessages(reply) if reply is not None else FailingMessages()
    return LLMAnswerer(CLAUSES, ClauseRetriever(CLAUSES, None, "bm25"), settings, messages_api=api), api


def test_verbatim_cited_sentence_is_accepted(settings):
    a, api = make("This Agreement shall be governed by the laws of India. [c01]", settings)
    ans = a.answer("Which law governs?")
    assert not ans.abstained and ans.method == "llm"
    assert ans.sentences[0].citations == ["c01"] and ans.sentences[0].grounded
    assert api.calls[0]["model"] == settings.llm_model and api.calls[0]["system"] == SYSTEM_PROMPT
    assert "[c01]" in api.calls[0]["messages"][0]["content"]


def test_fabricated_sentence_is_rejected(settings):
    a, _ = make("The Supplier shall provide free upgrades forever. [c01]", settings)
    ans = a.answer("What about upgrades?")
    assert ans.abstained and ans.sentences == [] and len(ans.rejected) == 1


def test_changed_number_is_rejected(settings):
    a, _ = make("Either party may terminate this Agreement on sixty (60) days written notice. [c02]", settings)
    assert a.answer("How much notice?").abstained


def test_wrong_citation_is_rejected_right_one_kept(settings):
    reply = "This Agreement shall be governed by the laws of India. [c02] Either party may terminate this Agreement on thirty (30) days written notice. [c02]"
    a, _ = make(reply, settings)
    ans = a.answer("Tell me everything")
    assert len(ans.sentences) == 1 and ans.sentences[0].citations == ["c02"]
    assert len(ans.rejected) == 1


def test_missing_citations_mean_rejection(settings):
    a, _ = make("This Agreement shall be governed by the laws of India.", settings)
    ans = a.answer("Which law governs?")
    assert ans.abstained and ans.rejected[0].citations == []


def test_no_answer_and_api_failure_abstain(settings):
    a, _ = make("NO_ANSWER", settings)
    assert a.answer("Anything?").abstained
    b, _ = make(None, settings)
    assert b.answer("Anything?").abstained


@pytest.mark.parametrize("reply", ["", "   ", "[c01]"])
def test_degenerate_replies(settings, reply):
    a, _ = make(reply, settings)
    assert a.answer("Which law governs?").abstained
