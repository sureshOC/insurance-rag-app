import re
# import fitz  # pymupdf
import ftfy
import unicodedata

def clean_text(text):
    text = re.sub(r'\n{2,}', '\n\n', text)  # collapse 3+ newlines to a paragraph break
    text = re.sub(r'[ \t]{2,}', ' ', text)  # collapse repeated spaces/tabs
    text = re.sub(r'\n(?!\n)', ' ', text)   # single newlines (mid-sentence wraps) -> space
    return text.strip()

def forward_fill_table(rows):
    filled = []
    last_key = None
    for row in rows:
        key = row[0] if row[0] is not None else last_key
        last_key = key
        filled.append([key] + list(row[1:]))
    return filled


# Known broken-font glyph substitutions found in this specific corpus.
# These are NOT real ligatures — NFKC cannot fix them, must be mapped explicitly.
BROKEN_FONT_MAP = {
    'Ĥ': 'fi',
    'ĥ': 'fi',
    'Ħ': 'ffi',
    'ģ': 'ff',
}

# PUA bullet glyphs seen in this corpus (Wingdings-style bullets in LIC/ICICI docs)
PUA_MAP = {
    '\uf0b7': '*',
    '\uf0a7': '*',
}

def sanitize_unicode(text: str) -> str:
    if not text:
        return ""

    # 1. Fix known broken-font substitutions FIRST, before anything else touches them
    for bad, good in BROKEN_FONT_MAP.items():
        text = text.replace(bad, good)

    # 2. Fix mojibake (encoding round-trip corruption)
    text = ftfy.fix_text(text)

    # 3. NFKC — expands REAL ligature codepoints (ﬁ, ﬀ, ﬃ, ﬂ) to plain letters
    text = unicodedata.normalize("NFKC", text)

    # 4. Known PUA bullets -> asterisk, before the general PUA strip below
    for bad, good in PUA_MAP.items():
        text = text.replace(bad, good)

    # 5. Punctuation normalization
    replacements = {
        "’": "'", "‘": "'",
        "”": '"', "“": '"',
        "–": "-", "—": "-",
        "•": "*", "·": "*",
        "…": "...",
        "«": '"', "»": '"',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # 6. Remove any REMAINING Private Use Area characters (unmapped ones)
    text = re.sub(r'[\uE000-\uF8FF\uFFF0-\uFFFF]', '', text)

    # 7. Preserve ₹ explicitly, then strip other non-ASCII noise
    #    (do NOT blanket-strip before this — currency symbol matters for insurance docs)
    text = text.replace('₹', 'Rs.')
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)

    # 8. Collapse whitespace created by all the above substitutions
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def table_to_text(table_rows):
    """Convert forward-filled table rows into embeddable sentences."""
    lines = []
    for row in table_rows:
        if not row:
            continue
        key = row[0]
        values = [v for v in row[1:] if v]
        if key and values:
            lines.append(f"{key}: {'; '.join(values)}")
    return '\n'.join(lines)



CLAUSE_PATTERN = re.compile(
    r'(?<![\d/])\b(\d{1,3}\.\d{1,2}(?:\.\d{1,2})?)\b(?![\d/])|'
    r'\b([A-Z]\.\d{1,2})\b|'
    r'\b([A-Z]\))|'
    r'\b([ivx]{1,4}\))'
)

def extract_clause_refs(chunk_text):
    matches = CLAUSE_PATTERN.findall(chunk_text)
    refs = set()
    for groups in matches:
        ref = next((g for g in groups if g), None)
        if ref:
            refs.add(ref)
    return list(refs)