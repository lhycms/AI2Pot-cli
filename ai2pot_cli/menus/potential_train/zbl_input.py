"""Generate zbl.in from a training set, with the default ZBL pair parameters."""

import os

from ai2pot_cli.menu import show_generation_success
from ai2pot_cli.zbl_in import DEFAULT_ZBL_CKS, DEFAULT_ZBL_DKS, symbol_of

_HEADER = """\
# ZBL pair parameters generated from the training set {dataset}.
# Type map: {type_map}
#
# One element pair per line: <element_i>-<element_j> cks(4 values) dks(4 values)
#   - "Ge Te" (space separated) is also accepted, and the keywords "cks"/"dks"
#     may be omitted (4 cks followed by 4 dks).
#   - Either order of a pair may be written, e.g. Ge-Te or Te-Ge.
#   - Pairs left out of this file fall back to the AI2Pot built-in ZBL
#     parameters; # and // start a comment.
#
# The values below are the AI2Pot built-in ZBL parameters, i.e. the same for
# every element pair; replace them per pair when needed.
#
# Reference in the training config with:  "zbl_in_path": "./zbl.in"
"""


def build_zbl_in(type_map, dataset_name: str = "trainset") -> str:
    """zbl.in text covering every element pair of *type_map* with default values."""
    symbols = [symbol_of(z) for z in type_map]
    type_map_line = ", ".join(f"{symbol_of(z)}({z})" for z in type_map)
    lines = [_HEADER.format(dataset=dataset_name, type_map=type_map_line)]
    for i, symbol_i in enumerate(symbols):
        for j, symbol_j in enumerate(symbols):
            if j < i:
                continue
            cks = " ".join(str(v) for v in DEFAULT_ZBL_CKS)
            dks = " ".join(str(v) for v in DEFAULT_ZBL_DKS)
            lines.append(f"{symbol_i}-{symbol_j}   cks {cks}   dks {dks}")
    return "\n".join(lines) + "\n"


def generate_zbl_input(trainset_path: str, output_path: str = "zbl.in"):
    """Write a zbl.in with the default ZBL parameters for the dataset's element pairs."""
    from ai2pot.data import ExtxyzDataset

    type_map = ExtxyzDataset.get_type_map(filename=trainset_path)
    with open(output_path, "w") as fp:
        fp.write(build_zbl_in(type_map, dataset_name=os.path.basename(trainset_path)))
    abs_path = os.path.abspath(output_path)
    show_generation_success(
        title="ZBL Input Generated Successfully",
        output_path=abs_path,
        next_command=f'Set "zbl_in_path": "{output_path}" in your training config',
    )
