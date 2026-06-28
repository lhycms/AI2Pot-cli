"""Menu display utilities -- VASPKIT-style numbered menus."""

from typing import List, Tuple


LINE_WIDTH = 76
LINE_WIDTH_MINUS_2 = LINE_WIDTH - 2


def _make_line(content: str) -> str:
    return " " + f"|{content:^{LINE_WIDTH_MINUS_2}}|"


def show_banner(version: str):
    print(" +" + "-" * LINE_WIDTH_MINUS_2 + "+")
    print(_make_line(content="AI2Pot-cli Standard Edition"))
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
    COL_WIDTH = 36

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

    if footer:
        for num, label in footer:
            print(f" {num:2d})  {label}")


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


def show_numbered_menu(title: str, items: List[Tuple[int, str, str]]):
    """Print a sub-menu with explicit numbering.

    Args:
        title: Menu title string.
        items: List of (num, label, description) tuples.
    """
    print(_make_frame(title))
    for num, label, desc in items:
        print(f" {num:>3d})  {label:<28s} {desc}")
    print()
    print(f"  9)  Back")
    print(f"  0)  Quit")
    print(" " + "-" * (LINE_WIDTH - 1))
    print()


def show_generation_success(title: str, output_path: str, next_command: str):
    """Print a standardised success banner after generating a config file.

    Args:
        title: e.g. \"NEP Training Input Generated Successfully\"
        output_path: absolute path to the generated file.
        next_command: the CLI command the user should run next.
    """
    print_section(title)
    print_kv("Output File", output_path)
    print()
    print_kv("Next Command", next_command)
    print_sep()
    print()


# ---- Unified output helpers ----

SEP = " " + "-" * (LINE_WIDTH - 2)


def print_section(title: str):
    """Print a framed section header."""
    print()
    print(_make_frame(title))
    print()


def print_success(msg: str):
    """Print a success / info line."""
    print(f"  ✓  {msg}")


def print_warning(msg: str):
    """Print a warning line."""
    print(f"  ⚠  {msg}")


def print_error(msg: str):
    """Print an error line."""
    print(f"  ✗  {msg}")


def print_kv(key: str, value: str, indent: int = 2):
    """Print a key-value line with aligned colon."""
    prefix = " " * indent
    print(f"{prefix}{key:<18}: {value}")


def print_sep():
    """Print a horizontal separator line."""
    print(SEP)


def get_choice():
    """Read a numeric choice from stdin. Re-prompts on invalid input."""
    while True:
        try:
            s = input(" ------------>> ")
            if s.strip() == "":
                continue
            return int(s)
        except ValueError:
            print_warning("Please enter a valid number.")
        except EOFError:
            print()
            print_success("Bye.")
            raise SystemExit(0)
