from pathlib import Path

root = Path("data")

IGNORE_DIRS = {
    "__pycache__",
    "venv",
    "env",
    ".venv",
    "site-packages",
    "dist-packages",
    ".dist-info",
    "Lib",
    "Scripts",
    "share",
    "tzdata",
}


def is_ignored(path: Path):
    return any(part in IGNORE_DIRS for part in path.parts)

def tree(directory: Path, prefix=""):
    dirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    dirs = [d for d in dirs if not is_ignored(d)]

    for i, d in enumerate(dirs):
        connector = "└── " if i == len(dirs) - 1 else "├── "
        print(prefix + connector + d.name)

        extension = "    " if i == len(dirs) - 1 else "│   "
        tree(d, prefix + extension)

tree(root)