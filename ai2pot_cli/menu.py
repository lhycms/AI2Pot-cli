"""Menu display utilities -- VASPKIT-style numbered menus."""

from typing import List, Tuple


LINE_WIDTH = 76
LINE_WIDTH_MINUS_2 = LINE_WIDTH - 2


def _make_line(content: str) -> str:
    return " " + f"|{content:^{LINE_WIDTH_MINUS_2}}|"


def show_banner(version: str):
    print(" +" + "-" * LINE_WIDTH_MINUS_2 + "+")
    print(_make_line(content="AI2Pot-CLI Standard Edition"))
    print(_make_line(content=f"Version {version:<12}"))
    print(_make_line(content=""))
    print(_make_line(content="Official Command Line Interface for AI2Pot"))
    print(_make_line(content=""))
    print(_make_line(content="Developer: Hanyu Liu (domainofbuaa@gmail.com)"))
    print(_make_line(content=""))
    print(_make_line(content="AI2Pot-cli : https://github.com/lhycms/AI2Pot-cli"))
    print(_make_line(content="AI2Pot : https://github.com/lhycms/AI2Pot"))
    print(" +" + "-" * LINE_WIDTH_MINUS_2 + "+")
    print()


def _make_frame(title: str) -> str:
    inner = f" {title} "
    padding = LINE_WIDTH - len(inner)
    left = padding // 2
    right = padding - left
    return " " + "=" * left + inner + "=" * right


def show_main_menu(
    sections: List[Tuple[str, List[Tuple[int, str]]]],
    footer: List[Tuple[int, str]] = None):
    """Print the grouped main menu with category frames and a 2-column grid.

    Args:
        sections: List of (section_title, items) where items is List of (num, label).
        footer: Optional footer items displayed after sections, e.g. [(0, "Quit")].
    """
    COL_WIDTH = 35

    for title, items in sections:
        print(_make_frame(title))
        for i in range(0, len(items), 2):
            num1, label1 = items[i]
            left = f" {num1:2d})  {label1}"
            if i + 1 < len(items):
                num2, label2 = items[i + 1]
                right = f" {num2:2d})  {label2}"
                print(f"{left:<{COL_WIDTH}}{right}")
            else:
                print(left)
        print()

    if footer:
        for num, label in footer:
            print(f" {num:2d})  {label}")
        print()

    print(" " + "-" * LINE_WIDTH)
    print()


def show_menu(title: str, items: List[Tuple[str, str]]):
    """Print a simple sub-menu with a single section and descriptions.

    Args:
        title: Menu title string.
        items: List of (label, description) tuples.
    """
    print(_make_frame(title))
    for i, (label, desc) in enumerate(items, 1):
        print(f" {i:2d})  {label:<30s} {desc}")
    print()
    print(f"  9)  Back")
    print(f"  0)  Quit")
    print(" " + "-" * (LINE_WIDTH - 1))
    print()


def get_choice():
    """Read a numeric choice from stdin. Re-prompts on invalid input."""
    while True:
        try:
            s = input(" ------------>> ")
            if s.strip() == "":
                continue
            return int(s)
        except ValueError:
            print(" Please enter a valid number.")
        except EOFError:
            print("\n Bye.")
            raise SystemExit(0)
