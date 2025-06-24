import logging
from pathlib import Path

_LOG_FILE = Path("app.log")
_FMT = "%(asctime)s — %(levelname)s — %(name)s — %(message)s"


def get_logger(name: str = "fissure_route") -> logging.Logger:
    """Retourne un logger configuré (singleton)."""
    log = logging.getLogger(name)
    if log.handlers:          # déjà initialisé
        return log

    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    ch = logging.StreamHandler()
    for h in (fh, ch):
        h.setFormatter(logging.Formatter(_FMT))
        log.addHandler(h)
    return log
