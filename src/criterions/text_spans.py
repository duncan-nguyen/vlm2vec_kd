"""Noun-chunk / verb-phrase span extraction, shared by the span_propose family.

Previously each criterion carried its own copy of these helpers and called
``nlp.pipe(..., n_process=4)`` twice per training step. For a batch of ~16 short
strings the multiprocessing pool costs far more than the parsing itself, and the
same instruction text is re-parsed on every single step: in MMEB the query side
of a subset is usually one fixed instruction repeated for every sample.

This module keeps a memo of text -> (spans, words) so each distinct string is
parsed once per process, and can persist that memo across runs.

Environment variables
---------------------
VLM2VEC_SPACY_PROCESSES  processes for nlp.pipe (default 1).
VLM2VEC_SPAN_CACHE       directory to persist the memo in. Unset = memory only.
VLM2VEC_SPAN_CACHE_MAX   max in-memory entries (default 500000).
"""

import hashlib
import json
import os
import glob

_N_PROCESS = int(os.environ.get("VLM2VEC_SPACY_PROCESSES", "1"))
_CACHE_DIR = os.environ.get("VLM2VEC_SPAN_CACHE", "").strip() or None
_CACHE_MAX = int(os.environ.get("VLM2VEC_SPAN_CACHE_MAX", "500000"))

_DISABLED_COMPONENTS = ["ner", "lemmatizer"]


def filter_overlapping_spans(spans):
    """Lọc các span chồng lấp."""
    sorted_spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    filtered = []
    words = []
    if not sorted_spans:
        return filtered, words

    current_span = sorted_spans[0]
    for next_span in sorted_spans[1:]:
        _, current_end, p = current_span
        _, next_end, _ = next_span
        if next_end <= current_end:
            continue
        filtered.append((current_span[0], current_span[1]))

        n_token = len(p)
        words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
        words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))

        current_span = next_span

    filtered.append((current_span[0], current_span[1]))
    p = current_span[2]
    n_token = len(p)
    words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
    words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))

    return filtered, words


def _parse_one(doc, matcher):
    spans_with_offsets = []
    for _, start, end in matcher(doc):
        vp = doc[start:end]
        spans_with_offsets.append((vp.start_char, vp.end_char, vp))
    spans_with_offsets.extend(
        [(nc.start_char, nc.end_char, nc) for nc in doc.noun_chunks]
    )
    return filter_overlapping_spans(spans_with_offsets)


class SpanCache:
    """text -> (spans, words), optionally backed by a jsonl file on disk.

    Entries are keyed by a hash of the text so the on-disk file stays compact and
    free of escaping problems. Each rank writes its own shard; all shards in the
    directory are loaded at startup.
    """

    def __init__(self, cache_dir=None, rank=0, max_entries=_CACHE_MAX):
        self._mem = {}
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self.cache_dir = cache_dir
        self._shard_path = None
        self._pending = []
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._shard_path = os.path.join(cache_dir, f"spans_rank{rank}.jsonl")
            self._load_shards()

    @staticmethod
    def _key(text):
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def _load_shards(self):
        loaded = 0
        for path in sorted(glob.glob(os.path.join(self.cache_dir, "spans_rank*.jsonl"))):
            try:
                with open(path, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        self._mem[rec["h"]] = (
                            [tuple(x) for x in rec["s"]],
                            [tuple(x) for x in rec["w"]],
                        )
                        loaded += 1
            except (OSError, ValueError):
                # A truncated shard from a killed run should not stop training.
                continue
        if loaded:
            print(f"[SpanCache] loaded {loaded} entries from {self.cache_dir}")

    def get(self, text):
        entry = self._mem.get(self._key(text))
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def put(self, text, spans, words):
        if len(self._mem) >= self.max_entries:
            return
        key = self._key(text)
        self._mem[key] = (spans, words)
        if self._shard_path is not None:
            self._pending.append(
                {"h": key, "s": [list(x) for x in spans], "w": [list(x) for x in words]}
            )
            if len(self._pending) >= 1000:
                self.flush()

    def flush(self):
        if not self._pending or self._shard_path is None:
            return
        try:
            with open(self._shard_path, "a") as fh:
                for rec in self._pending:
                    fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        self._pending = []

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def build_span_cache(rank=0):
    """Cache configured from the environment; memory-only when unset."""
    return SpanCache(cache_dir=_CACHE_DIR, rank=rank)


def get_spans_offsets(texts, nlp, matcher, cache=None):
    """Trích xuất spans, words từ texts.

    Returns (phrases, spans, words) to match the original signature; `phrases`
    was always empty and is kept only so call sites do not have to change.
    """
    phrases = []
    spans = [None] * len(texts)
    words = [None] * len(texts)

    # Only parse what is not already known, and only parse each distinct string
    # once even when it repeats inside the batch.
    todo_order = []
    todo_index = {}
    for i, text in enumerate(texts):
        if cache is not None:
            entry = cache.get(text)
            if entry is not None:
                spans[i], words[i] = entry
                continue
        if text in todo_index:
            todo_index[text].append(i)
        else:
            todo_index[text] = [i]
            todo_order.append(text)

    if todo_order:
        pipe_kwargs = {"disable": _DISABLED_COMPONENTS}
        if _N_PROCESS > 1:
            pipe_kwargs["n_process"] = _N_PROCESS
        for text, doc in zip(todo_order, nlp.pipe(todo_order, **pipe_kwargs)):
            unique_spans, unique_words = _parse_one(doc, matcher)
            for i in todo_index[text]:
                spans[i] = unique_spans
                words[i] = unique_words
            if cache is not None:
                cache.put(text, unique_spans, unique_words)

    return phrases, spans, words
