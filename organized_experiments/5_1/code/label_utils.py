"""Frozen label canonicalization inherited from 3_5; no loose answer extraction."""
import re

def _label_key(text: str) -> str:
    return re.sub('\\s+', ' ', str(text)).strip(' \t\r\n"\'.,').casefold()
