"""Complete Markdown examples use the same code that readers can copy."""

from pathlib import Path

import pytest
from xdoctest.doctest_example import DocTest

from scripts.test_docs import collect_snippets


class TestDocSnippets:
    def test_hidden_setup_is_executed(self, tmp_path: Path):
        path = tmp_path / "example.md"
        path.write_text(
            "```python\n# docs: hide\nvalues = [1, 2, 3]\n# docs: endhide\n"
            "total = 0\nfor value in values:\n    total += value\nassert total == 6\n```\n"
        )
        [(file, line, code)] = collect_snippets([path])
        assert (file, line) == (path, 2)
        assert "docs:" not in code
        example = DocTest("\n".join(">>> " + line for line in code.splitlines()))
        assert example.run(verbose=0, on_error="raise")["passed"]
        failing = DocTest("\n".join(">>> " + line for line in code.replace("total == 6", "total == 7").splitlines()))
        assert failing.run(verbose=0, on_error="return")["failed"]

    @pytest.mark.parametrize(
        "code",
        ["# docs: hide\nx = 1", "# docs: endhide\nx = 1", "# docs: hide\n# docs: hide\nx = 1"],
    )
    def test_bad_markers_fail(self, tmp_path: Path, code: str):
        path = tmp_path / "bad.md"
        path.write_text(f"```python\n{code}\n```\n")
        with pytest.raises(ValueError, match="docs:"):
            collect_snippets([path])

    def test_indented_fences_and_non_python(self, tmp_path: Path):
        path = tmp_path / "example.md"
        path.write_text("  ```python\n  assert 2 + 2 == 4\n  ```\n\n```shell\nfalse\n```\n")
        assert collect_snippets([path]) == [(path, 2, "assert 2 + 2 == 4")]

    def test_website_full_code(self, tmp_path: Path):
        path = tmp_path / "example.html"
        path.write_text(
            '<div class="doc__code" data-copy-scope><div class="bar"><span>python</span></div>'
            "<pre>short</pre><pre data-code-full hidden>assert 1 &lt; 2\n# café</pre></div>\n"
            '<div class="doc__code"><div class="bar"><span>python</span></div><pre>assert True</pre></div>\n'
            '<div class="doc__code"><div class="bar"><span>shell</span></div><pre>false</pre></div>'
        )
        assert collect_snippets([path]) == [(path, 1, "assert 1 < 2\n# café"), (path, 3, "assert True")]
