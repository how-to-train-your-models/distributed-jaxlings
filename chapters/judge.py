"""Shared exercise validator used across all chapters."""
import numpy as np


class Judge:
    def __init__(self, chapter: str = "", default_tol: float = 1e-5):
        self.chapter = chapter
        self.default_tol = default_tol
        self.passed = 0
        self.failed = 0

    def check(self, name, got, expected, tol=None):
        if tol is None:
            tol = self.default_tol
        if isinstance(expected, bool):
            ok = bool(got) == expected
        elif isinstance(expected, tuple):
            ok = tuple(got) == expected
        elif isinstance(expected, list):
            ok = all(np.allclose(np.array(g), np.array(e), atol=tol) for g, e in zip(got, expected))
        elif isinstance(expected, np.ndarray) or hasattr(expected, "shape"):
            ok = np.allclose(np.array(got), np.array(expected), atol=tol)
        elif isinstance(expected, (int, float)):
            ok = abs(float(np.array(got).flat[0]) - float(expected)) / (abs(float(expected)) + 1e-9) < tol
        else:
            ok = got == expected
        if ok:
            self.passed += 1
            print(f"✅ {name}: PASSED")
        else:
            self.failed += 1
            print(f"❌ {name}: FAILED")
            print(f"   got:      {got!r}")
            print(f"   expected: {expected!r}")
        return ok

    def summary(self):
        total = self.passed + self.failed
        label = f"{self.chapter} complete!" if self.chapter else "All exercises complete!"
        print(f"\n{'='*40}")
        print(f"  Results: {self.passed}/{total} passed")
        if self.failed == 0:
            print(f"  🎉 {label}")
        else:
            print(f"  {self.failed} exercise(s) remaining.")
        print("=" * 40)
