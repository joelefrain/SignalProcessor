from __future__ import annotations

try:  # pragma: no cover - exercised only when the optional extension is built.
    from ._core_cy import central_difference, trapezoid_integrate
    from ._core import newmark_response_spectrum
except Exception:  # pragma: no cover
    from ._core import central_difference, newmark_response_spectrum, trapezoid_integrate

__all__ = ["central_difference", "newmark_response_spectrum", "trapezoid_integrate"]
