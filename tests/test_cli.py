from clausewise.cli import main


def test_generate_and_show(settings, capsys):
    assert main(["generate", "--size", "3", "--show", "1"]) == 0
    out = capsys.readouterr().out
    assert "3 contracts cached" in out and "AGREEMENT" in out


def test_analyze_and_ask_files(settings, analyzer, contract, tmp_path, capsys):
    path = tmp_path / "contract.txt"
    path.write_text(contract.text, encoding="utf-8")
    assert main(["analyze", str(path)]) == 0
    out = capsys.readouterr().out
    assert "clauses" in out and "c01" in out
    assert main(["ask", str(path), "Which law governs this agreement?", "--json"]) == 0
    assert '"question"' in capsys.readouterr().out


def test_prefetch_reports_backend(settings, capsys):
    assert main(["prefetch-embeddings"]) == 0
    assert "embeddings backend:" in capsys.readouterr().out
