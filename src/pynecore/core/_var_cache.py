"""Variable cache registry for pyne optimize. Persists across script re-imports."""
_data: list | None = None
_build: list | None = None
