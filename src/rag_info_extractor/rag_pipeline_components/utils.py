from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches
from typing import Dict, Iterable, List, Optional, Tuple

from rapidfuzz import fuzz, process

# ---------- Matching output of analyze query with azienda names -----------


def match_azienda_name(nome, items, threshold=0.70) -> str:
    lower_map = {x.lower(): x for x in items}
    m = get_close_matches(nome.lower(), list(lower_map), n=1, cutoff=threshold)
    return lower_map[m[0]] if m else ""


# ------------ Obtain azineda name from query -----------------

LEGAL_SUFFIX_RE = re.compile(
    r"""
    \b(
      [s, a]\.?\s?r\.?\s?l\.?s?      |  # s.r.l. / s.r.l.s
      [s, a]\.?\s?p\.?\s?a\.?        |  # s.p.a.
      [s, a]\.?\s?n\.?\s?c\.?        |  # s.n.c.
      [s, a]\.?\s?a\.?\s?s\.?        |  # s.a.s.
      [s, a]\.?\s?a\.?\s?p\.?\s?a\.? |  # s.a.p.a.
      [s, a]\.?\s?c\.?\s?a\.?\s?r\.?\s?l\.? |  # s.c.a.r.l.
      coop(?:erativa)?          |  # coop / cooperativa
      soc(?:iet[aà])?\s?coop(?:erativa)? |  # soc coop
      consorzio                 |
      holding                   |
      gruppo
    )\b\.?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

PUNCT_RE = re.compile(
    r"[^\w\s]", flags=re.UNICODE
)  # keep letters/numbers/underscore/space
WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _strip_diacritics(s: str) -> str:
    # Normalize accents (è → e) for robust matching but keep original for reporting
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_company(s: str) -> str:
    """
    Normalize a company name for fuzzy matching:
    - casefold
    - remove diacritics
    - drop legal suffixes (S.r.l., S.p.A., Soc. Coop., etc.)
    - strip punctuation
    - collapse spaces
    """
    if not s:
        return ""
    s = s.strip()
    s = _strip_diacritics(s.casefold())
    s = LEGAL_SUFFIX_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


class CompanyMatcher:
    """
    Fuzzy matcher for company (azienda) names.
    - Pass your canonical company list (as stored in DB).
    - Optionally pass aliases map {alias: canonical}.
    """

    def __init__(
        self, companies: Iterable[str], aliases: Optional[Dict[str, str]] = None
    ):
        self.canonical: List[str] = list(
            dict.fromkeys(companies)
        )  # preserve order, unique
        self.aliases = aliases or {}

        # Build a search list that includes aliases; keep mapping to canonical
        expanded: List[Tuple[str, str]] = []  # (display, canonical)
        for c in self.canonical:
            expanded.append((c, c))
        for alias, canonical in self.aliases.items():
            expanded.append((alias, canonical))

        # Normalized index for matching
        self.index_norm: List[Tuple[str, str]] = [
            (normalize_company(display), canonical) for display, canonical in expanded
        ]
        # Also keep original display forms to report exact best hit later if needed
        self.display_map: Dict[str, List[str]] = {}
        for display, canonical in expanded:
            self.display_map.setdefault(canonical, []).append(display)

        # For RapidFuzz, we’ll search over normalized strings, but return canonical
        self._choices = [norm for norm, _ in self.index_norm]

    def match(
        self, query: str, *, min_score: int = 80, scorer=fuzz.WRatio, top_k: int = 1
    ) -> List[Dict]:
        """
        Find closest company names for the user query.
        Returns a list of dicts sorted by score desc:
          { 'canonical': str, 'score': float, 'normalized_query': str }
        - min_score: threshold (0..100). Typical 80–90.
        - scorer: RapidFuzz scorer (WRatio, token_set_ratio, token_sort_ratio, QRatio).
        - top_k: how many candidates to return (default 1).
        """
        if not query:
            return []

        q_norm = normalize_company(query)

        # Search over normalized names
        results = process.extract(
            q_norm,
            self._choices,
            scorer=scorer,
            limit=max(10, top_k),  # grab a few extras then filter by threshold
        )

        out: List[Dict] = []
        for match_norm, score, idx in results:
            if score < min_score:
                continue
            # Map back to canonical
            canonical = self.index_norm[idx][1]

            out.append(
                {
                    "canonical": canonical,
                    "score": float(score),
                    "normalized_query": q_norm,
                }
            )

            if len(out) >= top_k:
                break

        return out


# ------------------------------
# Example usage
# ------------------------------
#     companies = ['ospedale tortona società consortile a r.l.',
#     'poliambulatorio 3d s.r.l.',
#     'secur system group di romeo s.r.l.',
#     'unica società benefit a r.l.']

#     matcher = CompanyMatcher(companies)

#     queries = [
#         "info su poliambulatorio 3d",
#         "vorrei documenti di secur system group di romeo",
#         "secur system group di romeo contatti",
#         "cerco ‘ospedale tortona società consortile a.r.l.’",
#         "unica società benefit (typo) utili",
#     ]

#     for q in queries:
#         res = matcher.match(q, min_score=78, scorer=fuzz.token_set_ratio, top_k=1)
#         print(f"\nQ: {q}\n→ {res}")


# Check which models are available in Google API
def check_google_api_models():
    import os

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()

    client = genai.Client()
    for m in client.models.list():
        print(f"Name: {m.name} | Display: {m.display_name}")


# check_google_api_models()
