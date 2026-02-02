#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------- PREPOSITIONS ----------
PREPOSITIONS = [
    r"\bde la\b",
    r"\bde\b",
    r"\bdu\b",
    r"\bdes\b",
    r"\bd’\b",
    r"\bd'un\b",
    r"\bd'une\b",
    r"\bde l’\b",
]

pattern = re.compile(
    rf"(.+?)\s+(?:{'|'.join(PREPOSITIONS)})\s+(.+?)$",
    flags=re.IGNORECASE,
)

# ---------- TEXT CLEANING ----------
def nettoyer_texte(s: str) -> str | None:
    """
    Normalize a string:
    - strip, lowercase
    - remove diacritics
    - remove punctuation
    - remove digits
    - collapse whitespace
    Returns None for empty/falsey input.
    """
    if not s:
        return None

    s = s.strip().lower()

    # Remove accents/diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    # Replace non-word characters with spaces, remove digits, normalize spaces
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s)

    s = s.strip()
    return s if s else None

# ---------- PAIR EXTRACTION ----------
def extraire_paire(ligne: str) -> tuple[str, str] | None:
    """
    Extract (mot1, mot2) from a line matching:
      <mot1> <preposition> <mot2>
    """
    m = pattern.match(ligne)
    if not m:
        return None

    mot1, mot2 = (nettoyer_texte(x) for x in m.groups())
    if not mot1 or not mot2:
        return None

    return mot1, mot2

# ---------- MAIN PIPELINE ----------
def fusionner_fichiers(input_dir: str, output_csv: str) -> None:
    """
    Read all *.txt in input_dir; relation name is the filename stem.
    Extract pairs from each line and write a global CSV.
    """
    resultats: list[dict[str, str]] = []

    input_path = Path(input_dir)
    for path in input_path.glob("*.txt"):
        relation = path.stem
        print(f"Processing {relation} ({path.name})")

        with path.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue

                paire = extraire_paire(ligne)
                if paire:
                    mot1, mot2 = paire
                    resultats.append(
                        {"mot1": mot1, "mot2": mot2, "relation": relation}
                    )

    df = pd.DataFrame(resultats).drop_duplicates().reset_index(drop=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nOK: {len(df)} total relations written to {output_csv}")

# ---------- RUN ----------
if __name__ == "__main__":
    fusionner_fichiers(input_dir="data", output_csv="dataset_global.csv")
