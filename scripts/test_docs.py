"""Execute complete Markdown Python examples with xdoctest."""

import html
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

from xdoctest.doctest_example import DocTest


def collect_snippets(paths: List[Path]) -> List[Tuple[Path, int, str]]:
    """Read Python fences, removing presentation markers from runnable code."""
    snippets = []
    fence = re.compile(r"(?m)^([ \t]*)```python[ \t]*\n(.*?)^\1```[ \t]*$", re.S)
    for path in paths:
        text = path.read_text()
        if path.suffix == ".html":
            for match in re.finditer(
                r'<div class="doc__code"[^>]*><div class="bar"><span[^>]*>python</span>.*?</div>(.*?)</div>',
                text,
                re.S,
            ):
                blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", match[1], re.S)
                snippets.append((path, text[: match.start()].count("\n") + 1, html.unescape(blocks[-1])))
            continue
        for match in fence.finditer(text):
            number = text[: match.start()].count("\n") + 2
            full = []
            hidden = False
            for line in match[2].splitlines():
                line = line[len(match[1]) :] if line.startswith(match[1]) else line
                if line.strip() == "# docs: hide":
                    if hidden:
                        raise ValueError(f"{path}:{number}: nested # docs: hide")
                    hidden = True
                elif line.strip() == "# docs: endhide":
                    if not hidden:
                        raise ValueError(f"{path}:{number}: unmatched # docs: endhide")
                    hidden = False
                else:
                    full.append(line)
            if hidden:
                raise ValueError(f"{path}:{number}: unclosed # docs: hide")
            snippets.append((path, number, "\n".join(full).strip("\n")))
    return snippets


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(path) for path in sys.argv[1:]] or [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    paths = [
        file
        for path in paths
        for file in (
            sorted(file for file in path.rglob("*") if file.suffix in (".md", ".html")) if path.is_dir() else [path]
        )
    ]
    passed = skipped = failed = 0
    start = time.perf_counter()
    for path, number, code in collect_snippets(paths):
        compile(code, f"{path}:{number}", "exec")
        source = "\n".join(">>> " + line for line in code.splitlines())
        example = DocTest(
            source, fpath=str(path), callname=f"Python block at line {number}", lineno=number, mode="native"
        )
        result = example.run(verbose=0, on_error="return")
        passed += result["passed"]
        skipped += result["skipped"]
        failed += result["failed"]
        status = "FAIL" if result["failed"] else "SKIP" if result["skipped"] else "PASS"
        print(f"{status} {path}:{number}", flush=True)
        if result["failed"]:
            print("\n".join(example.repr_failure()), flush=True)
    print(f"Python examples: {passed} passed, {skipped} skipped, {failed} failed in {time.perf_counter() - start:.2f}s")
    sys.exit(bool(failed))


if __name__ == "__main__":
    main()
