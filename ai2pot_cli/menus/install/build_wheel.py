"""Build an AI2Pot wheel from a local source tree (option 1).

Prompts for the AI2Pot source directory and build options, then builds a
wheel matching the current torch/CUDA environment. The implementation runs in
the current Python environment, like the other menu options.
"""

import importlib.metadata
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from ai2pot_cli.menu import (
    print_section, print_kv, print_sep,
    print_warning, print_error,
)

# 与 pyproject.toml 的构建依赖保持一致
BUILD_DEPS = {
    "scikit-build-core": ("scikit_build_core", "0.12.2"),
    "cmake": ("cmake", "4.3.2"),
    "pybind11": ("pybind11", "2.11.1"),
    "numpy": ("numpy", "2.2.6"),
}


class BuildWheelError(Exception):
    """Raised when the wheel build cannot proceed."""


def _fail(message: str) -> NoReturn:
    raise BuildWheelError(message)


def _is_ai2pot_source(path) -> bool:
    path = Path(path)
    return (path / "pyproject.toml").is_file() and (path / "ai2pot").is_dir()


def load_torch():
    try:
        import torch
    except ImportError:
        _fail("当前 Python 环境没有 torch。请先安装与目标机器匹配的 torch，再运行本功能。")
    return torch


def torch_version_parts(version: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None:
        _fail(f"无法解析 torch 版本号：{version}")
    return int(match.group(1)), int(match.group(2))


def check_python() -> None:
    if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        print_warning(
            f"Python {sys.version.split()[0]} 不在项目声明的 >=3.11,<3.13 范围内，"
            "构建可能失败或安装后被 pip 拒绝。"
        )


def check_build_deps() -> None:
    missing: list[str] = []
    for dist_name, (module_name, expected) in BUILD_DEPS.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(dist_name)
            continue
        try:
            actual = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            actual = "unknown"
        if actual != expected:
            print_warning(
                f"{dist_name} 版本是 {actual}，构建基线是 {expected}；"
                "若构建失败请先对齐版本。"
            )
    if missing:
        pins = " ".join(f"{name}=={BUILD_DEPS[name][1]}" for name in missing)
        _fail(
            "缺少构建依赖：" + ", ".join(missing) + "。请先安装：\n"
            f"    {sys.executable} -m pip install {pins}"
        )


def check_numpy_abi() -> None:
    try:
        import numpy as np
    except ImportError as exc:
        _fail(f"无法导入 numpy：{exc}")
    major = int(np.__version__.split(".", 1)[0])
    if major != 2:
        _fail(
            f"当前 numpy 是 {np.__version__}，本项目按 numpy 2.x 头文件构建；"
            "请安装 numpy==2.2.6（或其它 2.x 版本）。"
        )


def find_cuda_home(explicit: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    try:
        from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home

        if torch_cuda_home:
            candidates.append(Path(torch_cuda_home))
    except Exception:
        pass
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").is_file():
            return candidate.resolve()
    return None


def nvcc_version(cuda_home: Path) -> tuple[int, int] | None:
    nvcc = cuda_home / "bin" / "nvcc"
    try:
        proc = subprocess.run(
            [str(nvcc), "--version"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        _fail(f"无法执行 {nvcc}: {exc}")
    match = re.search(r"release (\d+)\.(\d+)", proc.stdout + proc.stderr)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def nvcc_supported_archs(cuda_home: Path) -> set[str] | None:
    nvcc = cuda_home / "bin" / "nvcc"
    try:
        proc = subprocess.run(
            [str(nvcc), "--list-gpu-arch"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    archs = set(re.findall(r"compute_(\d+)", proc.stdout + proc.stderr))
    return archs or None


def local_gpu_caps(torch) -> list[str]:
    try:
        if not torch.cuda.is_available():
            return []
        caps = {
            f"{major}{minor}"
            for major, minor in (
                torch.cuda.get_device_capability(index)
                for index in range(torch.cuda.device_count())
            )
        }
    except Exception:
        return []
    return sorted(caps, key=int)


def torch_cuda_caps(torch) -> list[str]:
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return []
    caps: set[str] = set()
    for item in arch_list:
        match = re.match(r"(?:sm|compute)_(\d+)", str(item))
        if match:
            caps.add(match.group(1))
    return sorted(caps, key=int)


def normalize_arch_arg(value: str) -> list[str]:
    entries: list[str] = []
    for raw in value.replace(";", ",").split(","):
        entry = raw.strip()
        if not entry:
            continue
        if re.fullmatch(r"\d+\.\d+(?:-(?:real|virtual))?", entry):
            entry = entry.replace(".", "", 1)
        entries.append(entry)
    if not entries:
        _fail("CUDA 架构不能为空。示例：80-real,86-real,89-real,90-real,90-virtual")
    return entries


def auto_archs(caps: list[str]) -> list[str]:
    if not caps:
        _fail(
            "无法确定 GPU 架构：本机无可用 GPU，且 torch.cuda.get_arch_list() 为空。"
            "请手动指定架构，例如 80-real,86-real,89-real,90-real,90-virtual。"
        )
    highest = max(caps, key=int)
    return [f"{cap}-real" for cap in caps] + [f"{highest}-virtual"]


def validate_archs(
    archs: list[str],
    supported: set[str] | None,
    version: tuple[int, int] | None,
) -> None:
    if supported is None:
        return
    missing = sorted({arch.split("-", 1)[0] for arch in archs} - supported, key=int)
    if missing:
        label = f"{version[0]}.{version[1]}" if version else "当前 nvcc"
        names = ", ".join(f"sm_{item}" for item in missing)
        _fail(
            f"nvcc {label} 不支持架构 {names}。sm_100/sm_120 需要 CUDA 12.8+；"
            "请换用更新的 CUDA toolkit，或去掉这些架构。"
        )


def variant_label(torch, use_cuda: bool) -> str:
    torch_major, torch_minor = torch_version_parts(torch.__version__)
    if use_cuda:
        cuda_tag = "cu" + (torch.version.cuda or "unknown").replace(".", "")
    else:
        cuda_tag = "cpu"
    return (
        f"torch{torch_major}.{torch_minor}-{cuda_tag}-"
        f"py{sys.version_info.major}{sys.version_info.minor}"
    )


def build_command(
    source: Path, use_cuda: bool, archs: list[str], out_dir: Path, verbose: bool
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(source),
        "--no-build-isolation",
        "--no-deps",
        f"--config-settings=cmake.define.USE_CUDA={'on' if use_cuda else 'off'}",
        "--config-settings=cmake.define.BUILD_TEST=off",
    ]
    if use_cuda:
        command.append(
            "--config-settings=cmake.define.CUDA_ARCHITECTURES=" + ";".join(archs)
        )
    command.extend(["-w", str(out_dir)])
    if verbose:
        command.append("-v")
    return command


def _ask(label: str, default: str) -> str:
    return input(f"  {label:<18} [{default}]: ").strip() or default


def _ask_source_dir() -> str:
    default = os.environ.get("AI2POT_SOURCE", "")
    if not default and _is_ai2pot_source(os.getcwd()):
        default = os.getcwd()

    while True:
        if default:
            value = input(f"  {'AI2Pot source dir':<18} [{default}]: ").strip() or default
        else:
            value = input(f"  {'AI2Pot source dir':<18} [required]: ").strip()
        if not value:
            print_warning("AI2Pot source directory is required.")
            continue

        source = os.path.abspath(os.path.expanduser(value))
        if not _is_ai2pot_source(source):
            print_error(f"Not an AI2Pot source directory: {source}")
            print_warning("Expected pyproject.toml and ai2pot/ inside that directory.")
            if default and source == os.path.abspath(os.path.expanduser(default)):
                default = ""
            continue

        return source


def _ask_build_mode() -> str:
    while True:
        value = _ask("Build mode", "auto").lower()
        if value in ("auto", "cuda"):
            return "auto"
        if value in ("cpu", "cpu-only"):
            return "cpu"
        print_warning("Enter 'auto' or 'cpu'.")


def _ask_jobs() -> int | None:
    while True:
        value = _ask("Parallel jobs", "16").lower()
        if value == "auto":
            return None
        if value.isdigit() and int(value) > 0:
            return int(value)
        print_warning("Enter a positive integer, or 'auto' to let CMake decide.")


def _ask_yes_no(label: str) -> bool:
    while True:
        value = _ask(label, "n").lower()
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print_warning("Enter y or n.")


def build_wheel(
    source_dir: str,
    mode: str = "auto",
    arch: str = "auto",
    cuda_home_arg: str = "auto",
    jobs: int | None = None,
    dist_root: str = "dist",
    dry_run: bool = False,
    verbose: bool = False,
) -> Path | None:
    """Build an AI2Pot wheel. Returns the wheel path, or None for a dry run."""
    source = Path(source_dir).expanduser().resolve()
    if not _is_ai2pot_source(source):
        _fail(f"{source} 不是 AI2Pot 源码目录（需要 pyproject.toml 和 ai2pot/）。")

    check_python()
    check_build_deps()
    check_numpy_abi()

    torch = load_torch()
    use_cuda = mode != "cpu" and torch.version.cuda is not None

    cuda_home: Path | None = None
    cuda_version: tuple[int, int] | None = None
    if use_cuda:
        explicit_cuda_home = None if cuda_home_arg.lower() == "auto" else cuda_home_arg
        cuda_home = find_cuda_home(explicit_cuda_home)
        if cuda_home is None:
            _fail(
                f"torch 使用 CUDA {torch.version.cuda}，但找不到 nvcc。\n"
                f"请安装匹配的 CUDA toolkit（例如 conda install -c nvidia "
                f"cuda-toolkit={torch.version.cuda}），或选择 CPU 模式。"
            )
        cuda_version = nvcc_version(cuda_home)
        if cuda_version is None:
            _fail(f"无法从 {cuda_home}/bin/nvcc --version 解析 CUDA 版本。")
        torch_cuda_version = tuple(
            int(part) for part in torch.version.cuda.split(".")[:2]
        )
        if cuda_version[0] != torch_cuda_version[0]:
            _fail(
                f"nvcc 是 CUDA {cuda_version[0]}.{cuda_version[1]}，"
                f"torch 使用 CUDA {torch.version.cuda}；大版本必须一致。"
                "请安装与 torch 匹配的 toolkit，或选择 CPU 模式。"
            )
        if cuda_version < torch_cuda_version:
            print_warning(
                f"nvcc {cuda_version[0]}.{cuda_version[1]} 低于 torch 的 "
                f"CUDA {torch.version.cuda}，建议升级 toolkit。"
            )
    elif torch.version.cuda is None and mode != "cpu":
        print_warning("当前 torch 是 CPU 版，将构建 CPU-only wheel。")
    elif mode == "cpu" and torch.version.cuda is not None:
        print_warning("已选择 CPU 模式，即使 torch 带 CUDA 也只构建 CPU 库。")

    archs: list[str] = []
    arch_source = ""
    if use_cuda:
        if arch.lower() != "auto":
            archs = normalize_arch_arg(arch)
            arch_source = "user"
        else:
            caps = local_gpu_caps(torch)
            arch_source = "本机 GPU"
            if not caps:
                caps = torch_cuda_caps(torch)
                arch_source = "torch.cuda.get_arch_list()"
            archs = auto_archs(caps)
        validate_archs(archs, nvcc_supported_archs(cuda_home), cuda_version)
    elif arch.lower() != "auto":
        print_warning("CUDA 架构只在 CUDA 构建时生效，已忽略。")

    out_root = Path(dist_root).expanduser()
    if not out_root.is_absolute():
        out_root = source / out_root
    out_dir = out_root.resolve() / variant_label(torch, use_cuda)
    command = build_command(source, use_cuda, archs, out_dir, verbose)

    environment = os.environ.copy()
    if use_cuda:
        environment["CUDA_HOME"] = str(cuda_home)
        environment["CUDACXX"] = str(cuda_home / "bin" / "nvcc")
        environment.setdefault("CUDAHOSTCXX", environment.get("CXX", "g++"))
        environment["PATH"] = (
            str(cuda_home / "bin") + os.pathsep + environment.get("PATH", "")
        )
    if jobs:
        environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)

    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown")
    print()
    print_kv("Source", str(source))
    print_kv("Torch", torch.__version__)
    print_kv("CUDA", torch.version.cuda or "CPU-only")
    print_kv("Python", sys.version.split()[0])
    print_kv("ABI", f"_GLIBCXX_USE_CXX11_ABI={abi}")
    print_kv("Build mode", f"CUDA @ {cuda_home}" if use_cuda else "CPU")
    if use_cuda:
        print_kv("Arch source", arch_source)
        print_kv("Architectures", ";".join(archs))
    print_kv("Output", str(out_dir))
    print()

    if dry_run:
        print_kv("Command", " ".join(shlex.quote(part) for part in command))
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    print_kv("Running", " ".join(shlex.quote(part) for part in command))
    print_sep()
    result = subprocess.run(command, cwd=str(source), env=environment)
    print_sep()
    if result.returncode != 0:
        _fail(f"构建失败（退出码 {result.returncode}）。请查看上面的编译错误。")

    wheels = sorted(out_dir.glob("ai2pot-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        _fail(f"构建结束，但 {out_dir} 下没有 ai2pot-*.whl。")
    return wheels[-1]


def build_wheel_menu():
    """Option 1): build an AI2Pot wheel from a local source tree."""
    print_section("Build AI2Pot Wheel from Source")

    source = _ask_source_dir()
    mode = _ask_build_mode()
    arch = _ask("CUDA architectures", "auto")
    cuda_home = _ask("CUDA toolkit path", "auto")
    jobs = _ask_jobs()
    dist_root = _ask("Output root", "dist")
    dry_run = _ask_yes_no("Dry run (y/n)")
    verbose = _ask_yes_no("Verbose pip (y/n)")

    try:
        wheel = build_wheel(
            source,
            mode=mode,
            arch=arch,
            cuda_home_arg=cuda_home,
            jobs=jobs,
            dist_root=dist_root,
            dry_run=dry_run,
            verbose=verbose,
        )
    except BuildWheelError as exc:
        print_error(str(exc))
        print()
        sys.exit(1)

    print_section("Dry Run Finished" if dry_run else "AI2Pot Wheel Built Successfully")
    if wheel is not None:
        size_mib = wheel.stat().st_size / (1024 * 1024)
        print_kv("Wheel", str(wheel))
        print_kv("Size", f"{size_mib:.1f} MiB")
        print_kv(
            "Next step",
            f"{sys.executable} -m pip install --force-reinstall {wheel}",
        )
    print_sep()
    print()
    sys.exit(0)
