"""Responsive Image-to-TikZ Studio entry point."""

from .gui_async import run

__all__ = ["run"]

if __name__ == "__main__":
    raise SystemExit(run())
