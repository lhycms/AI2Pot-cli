"""Read a zbl.in file -- per-element-pair ZBL parameters (cks / dks)."""

import os
import re
from typing import Dict, List, Sequence, Tuple

from ase.data import atomic_numbers, chemical_symbols

# "Ge-Te", "Ge/Te" and "Ge Te" are all accepted element-pair spellings.
_PAIR_SEP_RE = re.compile(r"[-/]")
_TOKEN_SPLIT_RE = re.compile(r"[\s,;]+")
_COMMENT_RE = re.compile(r"#|//")
_KEYWORDS = ("cks", "dks")

# AI2Pot built-in ZBL parameters (ai2pot/models/mtp/nn_mtp.py::_init_zbl_params);
# used for every element pair that the zbl.in file does not cover.
DEFAULT_ZBL_CKS = [0.18175, 0.50986, 0.28022, 0.02817]
DEFAULT_ZBL_DKS = [3.1998, 0.94229, 0.4029, 0.20162]


def symbol_of(z: int) -> str:
    """Atomic number -> element symbol."""
    if 0 < z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z{z}"


def format_type_map(type_map: Sequence[int]) -> str:
    """Atomic-number type map -> "Ge, Sb, Te"."""
    return ", ".join(symbol_of(z) for z in type_map)


def _resolve_element(token: str, path: str, lineno: int) -> int:
    """Element symbol (e.g. "Ge") or atomic number (e.g. "32") -> atomic number."""
    name = token.strip()
    if name.isdigit():
        z = int(name)
        if z <= 0 or z >= len(chemical_symbols):
            raise ValueError(f"{path}:{lineno}: invalid atomic number '{token}'")
        return z
    symbol = name.capitalize()
    if symbol not in atomic_numbers:
        raise ValueError(f"{path}:{lineno}: unknown element '{token}'")
    return int(atomic_numbers[symbol])


def _parse_values(tokens: List[str], path: str, lineno: int) -> Tuple[List[float], List[float]]:
    """Split the values of one line into 4 cks and 4 dks floats."""
    numbers: List[Tuple[int, float]] = []
    keywords: List[Tuple[int, str]] = []
    for idx, token in enumerate(tokens):
        key = token.rstrip(":").lower()
        if key in _KEYWORDS:
            keywords.append((idx, key))
            continue
        try:
            numbers.append((idx, float(token)))
        except ValueError:
            raise ValueError(f"{path}:{lineno}: cannot parse '{token}' as a number")

    if not keywords:
        if len(numbers) != 8:
            raise ValueError(
                f"{path}:{lineno}: expected 8 values (4 cks then 4 dks), got {len(numbers)}")
        return [v for _, v in numbers[:4]], [v for _, v in numbers[4:]]

    if [key for _, key in keywords] != list(_KEYWORDS):
        raise ValueError(f"{path}:{lineno}: expected the keywords in the order 'cks ... dks ...'")
    cks_start = keywords[0][0]
    dks_start = keywords[1][0]
    cks = [v for i, v in numbers if cks_start < i < dks_start]
    dks = [v for i, v in numbers if i > dks_start]
    if len(cks) != 4 or len(dks) != 4:
        raise ValueError(
            f"{path}:{lineno}: expected 4 cks and 4 dks values, "
            f"got {len(cks)} cks and {len(dks)} dks")
    return cks, dks


def _parse_line(line: str, path: str, lineno: int) -> Tuple[int, int, List[float], List[float]]:
    """One "element pair + values" line -> (Z_i, Z_j, cks, dks)."""
    tokens = [token for token in _TOKEN_SPLIT_RE.split(line) if token]
    if _PAIR_SEP_RE.search(tokens[0]):
        parts = _PAIR_SEP_RE.split(tokens[0])
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"{path}:{lineno}: invalid element pair '{tokens[0]}'")
        element_i, element_j, rest = parts[0], parts[1], tokens[1:]
    else:
        if len(tokens) < 2:
            raise ValueError(
                f"{path}:{lineno}: expected an element pair (e.g. 'Ge-Te') followed by values")
        element_i, element_j, rest = tokens[0], tokens[1], tokens[2:]

    cks, dks = _parse_values(rest, path, lineno)
    return (_resolve_element(element_i, path, lineno),
            _resolve_element(element_j, path, lineno),
            cks, dks)


def load_zbl_in(path: str, type_map: Sequence[int]) -> Tuple[List[float], List[float], List[str]]:
    """Expand a zbl.in file into the flat ZBL parameter lists of AI2Pot models.

    Each non-empty line of the file holds one element pair::

        Ge-Ge   0.18175 0.50986 0.28022 0.02817   3.1998 0.94229 0.4029 0.20162
        Ge-Sb   cks 0.18175 0.50986 0.28022 0.02817   dks 3.1998 0.94229 0.4029 0.20162

    Lines may be written for either order of a pair (Ge-Te or Te-Ge). Pairs of the
    type map that the file does not cover keep the AI2Pot built-in ZBL parameters.

    AI2Pot stores both ZBL tensors as ntypes*ntypes blocks of 4 values; the block of
    the pair (i, j) sits at offset (i*ntypes + j)*4, where i and j are *type indices*,
    i.e. positions in ``type_map``.

    Returns (zbl_cks_list, zbl_dks_list, default_pairs), where default_pairs lists the
    element pairs that fell back to DEFAULT_ZBL_CKS / DEFAULT_ZBL_DKS; raises ValueError
    on malformed input.
    """
    if not os.path.isfile(path):
        raise ValueError(f"zbl.in file not found: {path}")

    type_map = [int(z) for z in type_map]
    table: Dict[Tuple[int, int], Tuple[List[float], List[float]]] = {}

    with open(path, "r") as fp:
        for lineno, raw_line in enumerate(fp, start=1):
            line = _COMMENT_RE.split(raw_line)[0].strip()
            if not line:
                continue
            element_i, element_j, cks, dks = _parse_line(line, path, lineno)
            for z in (element_i, element_j):
                if z not in type_map:
                    raise ValueError(
                        f"{path}:{lineno}: element {symbol_of(z)} is not in the type map "
                        f"({format_type_map(type_map)})")
            if (element_i, element_j) in table:
                raise ValueError(
                    f"{path}:{lineno}: duplicate entry for pair "
                    f"{symbol_of(element_i)}-{symbol_of(element_j)}")
            table[(element_i, element_j)] = (cks, dks)

    # Each pair is reported once: (i, j) and (j, i) are satisfied by the same line.
    default_pairs = [f"{symbol_of(z_i)}-{symbol_of(z_j)}"
                     for i, z_i in enumerate(type_map)
                     for j, z_j in enumerate(type_map)
                     if j >= i and (z_i, z_j) not in table and (z_j, z_i) not in table]

    cks_list: List[float] = []
    dks_list: List[float] = []
    for z_i in type_map:
        for z_j in type_map:
            pair = table.get((z_i, z_j)) or table.get((z_j, z_i))
            pair_cks, pair_dks = pair if pair is not None else (DEFAULT_ZBL_CKS, DEFAULT_ZBL_DKS)
            cks_list.extend(pair_cks)
            dks_list.extend(pair_dks)
    return cks_list, dks_list, default_pairs
