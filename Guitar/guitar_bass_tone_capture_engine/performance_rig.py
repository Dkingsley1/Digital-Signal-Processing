#!/usr/bin/env python3
"""PyCharm-friendly launcher for portable performance-rig actions."""

from rig_capture import apply_performance_rig, ask, build_performance_rig


def run() -> None:
    print("\nPortable performance rig")
    print("  1. Build a performance rig from accepted models")
    print("  2. Apply a performance rig to a clean DI")
    print("  3. Exit")
    selection = ask("Choose", "1")
    actions = {
        "1": build_performance_rig,
        "2": apply_performance_rig,
    }
    action = actions.get(selection)
    if action is not None:
        action()
    elif selection != "3":
        raise SystemExit("Choose 1, 2, or 3.")


if __name__ == "__main__":
    run()
