import ast
import re

COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def parse_label(label):
    """Return [vulnerability, satd] as ints; accepts a list or its string form."""
    if isinstance(label, str):
        label = ast.literal_eval(label)
    return [int(x) for x in label]


def extract_comments(code, max_chars=1000):
    """Join all // and /* */ comments of a function into one string."""
    parts = [m.strip("/* \n\t") for m in COMMENT_RE.findall(code)]
    return re.sub(r"\s+", " ", " ".join(parts))[:max_chars]


def tokenize_pair(tokenizer, comments, code, max_len=512, comments_first=True):
    """Comments-first input: <s> comments </s></s> code </s>, code truncated first."""
    if comments_first:
        return tokenizer(comments, code, truncation="longest_first", max_length=max_len)
    return tokenizer(code, truncation=True, max_length=max_len)
