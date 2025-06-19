from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

_CFG_CACHE: Optional[Dict[str, Any]] = None


def load_config(path: Union[str, Path] = Path(__file__).parent / "config.yaml") -> Dict[str, Any]:
    """
    Charge une seule fois config.yaml et renvoie un dict immuable.
    Compatible Python 3.8 (pas d’opérateur |).
    """
    global _CFG_CACHE
    if _CFG_CACHE is None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration introuvable : {path}")
        with path.open(encoding="utf-8") as f:
            _CFG_CACHE = yaml.safe_load(f)
    return _CFG_CACHE
