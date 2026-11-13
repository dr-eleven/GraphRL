import json, re

def _prediction_value(text: str) -> str:
    match = re.search('"prediction"\\s*:\\s*("(?:\\\\.|[^"\\\\])*")', str(text))
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return ''
    match = re.search('Answer\\s*:\\s*(.+)', str(text), re.IGNORECASE)
    return match.group(1).splitlines()[0] if match else str(text).strip()

def _label_key(text: str) -> str:
    return re.sub('\\s+', ' ', str(text)).strip(' \t\r\n"\'.,').casefold()
