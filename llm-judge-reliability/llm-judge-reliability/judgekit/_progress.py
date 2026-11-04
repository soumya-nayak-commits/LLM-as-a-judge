"""tqdm if installed, otherwise a no-op wrapper (keeps the CPU tests dependency-free)."""
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kw):
        return it
