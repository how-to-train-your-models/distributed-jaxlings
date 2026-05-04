"""Shared exercise validator used across all chapters.

Two calling styles, both spelled `judge.check(...)`:

  judge.check(my_function)
      Convention-driven: imports `self.test_module`, runs every
      `test_<my_function.__name__>` or `test_<my_function.__name__>__<scenario>`,
      passing my_function as the sole positional arg. Used in chapters 2+.

  judge.check("name", got, expected, tol=...)
      Legacy single-value check used in chapter 1.
"""
import importlib
import re
import traceback

import numpy as np


class Judge:
    def __init__(self, chapter: str = "", test_module: str | None = None,
                 default_tol: float = 1e-5):
        self.chapter = chapter
        self.test_module = test_module
        self.default_tol = default_tol
        self.passed = 0
        self.failed = 0

    def check(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return self._check_fn(args[0])
        return self._check_value(*args, **kwargs)

    def _check_fn(self, fn):
        if self.test_module is None:
            raise ValueError("Judge.check(fn) requires test_module set on construction")
        mod = importlib.reload(importlib.import_module(self.test_module))
        name = fn.__name__
        pattern = re.compile(rf"^test_{re.escape(name)}(?:__|$)")
        tests = sorted((n, getattr(mod, n)) for n in dir(mod) if pattern.match(n))
        if not tests:
            print(f"⚠️  no tests found for {name!r} in {self.test_module}")
            return
        for tname, tfn in tests:
            label = tname[len("test_") + len(name):].lstrip("_") or "default"
            try:
                tfn(fn)
                self.passed += 1
                print(f"✅ {name} :: {label}")
            except NotImplementedError:
                self.failed += 1
                print(f"⏭  {name} :: {label} — not implemented")
            except AssertionError as e:
                self.failed += 1
                msg = str(e).strip() or "assertion failed"
                print(f"❌ {name} :: {label}\n   {msg}")
            except Exception:
                self.failed += 1
                print(f"💥 {name} :: {label}\n{traceback.format_exc()}")

    def _check_value(self, name, got, expected, tol=None):
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
