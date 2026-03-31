import os
import re
import sys
import io
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "env", "venv", "node_modules"}
INCLUDE_EXTS = {".py", ".html", ".htm", ".css", ".js"}

RE_HTML_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
RE_JINJA_COMMENTS = re.compile(r"{#.*?#}", re.DOTALL)
RE_BLOCK_COMMENTS = re.compile(r"/\*.*?\*/", re.DOTALL)
RE_LINE_COMMENTS = re.compile(r"(?<!:)//.*?$(?!\n)", re.MULTILINE)


def strip_python_comments(code: str) -> str:
    """Remove comments from Python code using tokenize to avoid breaking strings."""
    out = io.StringIO()
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    try:
        tokgen = tokenize.generate_tokens(io.StringIO(code).readline)
        for toktype, tok, start, end, line in tokgen:
            if toktype == tokenize.COMMENT:
                continue
            if toktype == tokenize.NL and prev_toktype == tokenize.NEWLINE:
                continue
            sline, scol = start
            eline, ecol = end
            if sline > last_lineno:
                last_col = 0
            if scol > last_col:
                out.write(" " * (scol - last_col))
            out.write(tok)
            prev_toktype = toktype
            last_lineno, last_col = eline, ecol
        return out.getvalue()
    except Exception:
       
        return re.sub(r"(^|\s)#.*$", "", code, flags=re.MULTILINE)


def strip_text_comments(path: Path, text: str) -> str:
    ext = path.suffix.lower()
    if ext in {".html", ".htm"}:
        text = RE_HTML_COMMENTS.sub("", text)
        text = RE_JINJA_COMMENTS.sub("", text)
        
        text = RE_BLOCK_COMMENTS.sub("", text)
        
        text = RE_LINE_COMMENTS.sub("", text)
        return text
    if ext == ".css":
        text = RE_BLOCK_COMMENTS.sub("", text)
        return text
    if ext == ".js":
        text = RE_BLOCK_COMMENTS.sub("", text)
        text = RE_LINE_COMMENTS.sub("", text)
        return text
    return text


def process_file(path: Path):
    try:
        original = path.read_text(encoding="utf-8")
    except Exception:
        return

    if path.suffix.lower() == ".py":
        stripped = strip_python_comments(original)
    else:
        stripped = strip_text_comments(path, original)

    if stripped != original:
       
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            if not bak.exists():
                bak.write_text(original, encoding="utf-8")
        except Exception:
            pass
        path.write_text(stripped, encoding="utf-8")


def main():
    root = PROJECT_ROOT
    for dirpath, dirnames, filenames in os.walk(root):
        
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in INCLUDE_EXTS:
                process_file(p)
    print("Done stripping comments.")


if __name__ == "__main__":
    main()
