"""Dependency-free runner: `python tests/run_tests.py` (pytest also works: `pytest -q`)."""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
import test_judgekit as t  # noqa: E402

fails = 0
names = [n for n in dir(t) if n.startswith("test_")]
for n in names:
    try:
        getattr(t, n)()
        print("PASS", n)
    except Exception:
        fails += 1
        print("FAIL", n)
        traceback.print_exc()
print(f"\n{len(names) - fails}/{len(names)} passed")
sys.exit(1 if fails else 0)
