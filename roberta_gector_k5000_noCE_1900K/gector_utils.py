
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class GectorVocab:
    """
    Minimal, framework-agnostic loader/encoder/decoder for GECToR-style vocab files.
    Expected files:
      - labels.txt
      - d_tags.txt (optional; for detection head)
      - non_padded_namespaces.txt (optional; informational)
      - verb-form-vocab.txt (optional; for $TRANSFORM_VERB_* actions)
    """
    def __init__(self):
        self.labels: List[str] = []
        self.label2id: Dict[str, int] = {}
        self.d_tags: Optional[List[str]] = None
        self.dtag2id: Optional[Dict[str, int]] = None
        self.non_padded_namespaces: Optional[List[str]] = None
        self.verb_forms: Optional[Dict[str, Dict[str, str]]] = None  # lemma -> {POS: surface}
        self.verb_pairs: Optional[Dict[str, Dict[str, str]]] = None  # lower(surface) -> {'SRC_TGT': target_surface}
        self.keep_id: Optional[int] = None

    # ---------- Loading ----------
    @classmethod
    def from_files(cls, vocab_dir: str, verb_file: Optional[str] = None) -> "GectorVocab":
        vocab_dir = Path(vocab_dir)
        if not vocab_dir.exists():
            raise FileNotFoundError(f"vocab_dir not found: {vocab_dir}")

        inst = cls()
        # labels
        labels_path = vocab_dir / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.txt not found in {vocab_dir}")
        inst.labels = [ln.rstrip("\n") for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        inst.label2id = {lab: i for i, lab in enumerate(inst.labels)}
        inst.keep_id = inst.label2id.get("$KEEP", None)

        # d_tags (optional)
        d_tags_path = vocab_dir / "d_tags.txt"
        if d_tags_path.exists():
            inst.d_tags = [ln.rstrip("\n") for ln in d_tags_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            inst.dtag2id = {lab: i for i, lab in enumerate(inst.d_tags)}

        # non_padded_namespaces (optional)
        npn_path = vocab_dir / "non_padded_namespaces.txt"
        if npn_path.exists():
            inst.non_padded_namespaces = [ln.strip() for ln in npn_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

        # verb forms (optional)
        if verb_file is not None:
            inst.verb_forms, inst.verb_pairs = inst._load_verb_forms(verb_file)

        return inst

    def _load_verb_forms(self, verb_file: str):
        path = Path(verb_file)
        if not path.exists():
            raise FileNotFoundError(f"verb-form-vocab not found: {path}")
        table: Dict[str, Dict[str, str]] = {}
        pairs: Dict[str, Dict[str, str]] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            line = raw.strip()

            # Format B (surface_surface:SRC_TGT), e.g., "go_went:VB_VBD"
            if ":" in line and "_" in line.split(":")[0] and "_" in line.split(":")[1]:
                left, right = line.split(":", 1)
                if "_" in left and "_" in right:
                    src_surf, tgt_surf = left.split("_", 1)
                    src_tag, tgt_tag = right.split("_", 1)
                    key_src = src_surf.strip().lower()
                    key_tt = f"{src_tag.strip().upper()}_{tgt_tag.strip().upper()}"
                    pairs.setdefault(key_src, {})[key_tt] = tgt_surf.strip()
                    continue

            # Format A (lemma 	 TAG form 	 TAG form ...)
            parts = [p for p in line.split("	") if p]
            if parts:
                lemma = parts[0].strip().lower()
                table.setdefault(lemma, {})
                for p in parts[1:]:
                    p = p.strip()
                    if " " in p:
                        # e.g., "VBD went"
                        tag, form = p.split(" ", 1)
                        table[lemma][tag.strip().upper()] = form.strip()
        return table, pairs

    # ---------- Encoding / Decoding ----------
    def encode_actions(self, actions: List[str], oov_to_keep: bool = True) -> List[int]:
        out: List[int] = []
        for a in actions:
            idx = self.label2id.get(a)
            if idx is None:
                if oov_to_keep and (self.keep_id is not None):
                    idx = self.keep_id
                else:
                    raise KeyError(f"Unknown action: {a}")
            out.append(idx)
        return out

    def decode_actions(self, ids: List[int]) -> List[str]:
        n = len(self.labels)
        out = []
        for i in ids:
            if 0 <= i < n:
                out.append(self.labels[i])
            else:
                out.append("$KEEP")  # safe fallback
        return out

    # ---------- Tag normalization helpers ----------
    _CASE_MAP = {
        "LOWER": str.lower,
        "CAPITAL": lambda s: s[:1].upper() + s[1:],
        "UPPER": str.upper,
        "CAPITALIZE": lambda s: s.capitalize(),
    }

    def normalize_action(self, raw: str) -> str:
        """Try to robustly map noisy tag strings to a canonical action in labels.txt."""
        s = raw.strip()
        # unify spaces/quotes
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", "'")
        s = re.sub(r"\s+", "_", s)  # collapse spaces to underscores inside action names

        if s in self.label2id:
            return s

        # Common normalizations
        # replace things like $APPEND-The -> $APPEND_the (labels tend to be lowercase payloads)
        if s.startswith("$APPEND-"):
            payload = s[len("$APPEND-"):].strip("_").lower()
            cand = f"$APPEND_{payload}"
            if cand in self.label2id:
                return cand

        if s.startswith("$REPLACE-"):
            payload = s[len("$REPLACE-"):].strip("_").lower()
            cand = f"$REPLACE_{payload}"
            if cand in self.label2id:
                return cand

        # Case transforms (examples): $TRANSFORM_CASE_CAPITAL, $TRANSFORM_CASE_UPPER, etc.
        if s.upper().startswith("$TRANSFORM_CASE_"):
            # normalize to upper for the flag portion
            return s.upper()

        # Verb transforms are already uppercase in label space
        if s.upper().startswith("$TRANSFORM_VERB_"):
            return s.upper()

        # otherwise return as-is; the caller can decide to drop or map to KEEP
        return s

    # ---------- Simple action application (demo-grade) ----------
    def apply_action_to_token(self, token: str, action: str, lemmatizer=None) -> Optional[List[str]]:
        """
        Returns list of output tokens for this input token (or None to delete).
        This is a minimal demo implementation; extend for production use.
        """
        if action == "$KEEP":
            return [token]
        if action == "$DELETE":
            return None

        if action.startswith("$APPEND_"):
            payload = action[len("$APPEND_"):]
            return [token, payload]

        if action.startswith("$REPLACE_"):
            payload = action[len("$REPLACE_"):]
            return [payload]

        if action.startswith("$TRANSFORM_CASE_"):
            flag = action[len("$TRANSFORM_CASE_"):]
            fn = self._CASE_MAP.get(flag)
            if fn is None:
                return [token]
            return [fn(token)]

        if action.startswith("$TRANSFORM_VERB_") and (self.verb_forms is not None or self.verb_pairs is not None):
            parts = action.split("_")
            if len(parts) >= 4:
                src, tgt = parts[-2], parts[-1]  # VB -> VBD
                # 1) Try surface-to-surface mapping first
                if self.verb_pairs is not None:
                    key_src = token.lower()
                    key_tt = f"{src}_{tgt}"
                    tgt_form = self.verb_pairs.get(key_src, {}).get(key_tt)
                    if tgt_form:
                        return [tgt_form]
                # 2) Fallback to lemma-based mapping
                if self.verb_forms is not None:
                    lemma = lemmatizer(token) if lemmatizer else token.lower()
                    form = self.verb_forms.get(lemma, {}).get(tgt)
                    if form:
                        return [form]
            return [token]

        # default: no change
        return [token]

    def apply_actions(self, tokens: List[str], actions: List[str], lemmatizer=None) -> List[str]:
        out: List[str] = []
        for tok, act in zip(tokens, actions):
            norm = self.normalize_action(act)
            if norm not in self.label2id and self.keep_id is not None:
                norm = "$KEEP"
            res = self.apply_action_to_token(tok, norm, lemmatizer=lemmatizer)
            if res is None:
                continue
            out.extend(res)
        return out
