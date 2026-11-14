"""Single strict parser for every generation turn, training and evaluation."""
from dataclasses import dataclass
import json
from graph_state import TOOL_ORDER, TOOL_OP, ACTION_FIELDS


class ProtocolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ProtocolError('DUPLICATE_KEY', 'Duplicate JSON keys are not allowed.')
        obj[key] = value
    return obj


def _constant(value):
    raise ProtocolError('JSON_SYNTAX', 'NaN and Infinity are not valid JSON.')


def strict_json(text):
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item,str):
                try:
                    item.encode('utf-8')
                except UnicodeEncodeError as exc:
                    raise ProtocolError('JSON_STRING', 'JSON strings must be valid Unicode, without unpaired surrogates.') from exc
            elif isinstance(item,dict):
                pending.extend(item.keys()); pending.extend(item.values())
            elif isinstance(item,list):
                pending.extend(item)
        return value
    except ProtocolError:
        raise
    except (ValueError, RecursionError, OverflowError) as exc:
        raise ProtocolError('JSON_SYNTAX', 'Expected one complete JSON value; no trailing text.') from exc


@dataclass
class Event:
    kind: str
    payload: object = None
    code: str = ''
    message: str = ''
    tool_attempt: bool = False


def parse_event(raw, finish_reason='stop'):
    """Keep invalid output distinct from a final submission; never execute a prefix."""
    raw = raw.strip()
    tool = '<tool_call>' in raw or '</tool_call>' in raw
    try:
        if any(marker in raw for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
            raise ProtocolError('INVALID_EVENT', 'Unexpected message boundary inside the output.')
        if tool:
            if not raw.startswith('<tool_call>') or not raw.endswith('</tool_call>'):
                raise ProtocolError('EVENT_FRAMING', 'Return one closed tool block, without any outside text.')
            return Event('TOOL_EVENT', strict_json(raw[len('<tool_call>'):-len('</tool_call>')].strip()), tool_attempt=True)
        payload = strict_json(raw)
        if isinstance(payload, dict) and 'prediction' in payload:
            return Event('FINAL_EVENT', payload)
        raise ProtocolError('INVALID_EVENT', 'Return one <tool_call> block or {"prediction":"candidate label"}.')
    except ProtocolError as exc:
        code = ('GENERATION_TRUNCATED' if finish_reason == 'length'
                and exc.code in {'JSON_SYNTAX', 'EVENT_FRAMING'} else exc.code)
        return Event('INVALID_EVENT', code=code, message=str(exc), tool_attempt=tool)


def validate_tool(payload, max_actions=64):
    if not isinstance(payload, dict) or set(payload) != {'name', 'arguments'}:
        raise ProtocolError('TOOL_SCHEMA', 'Tool JSON requires exactly name and arguments.')
    name = payload['name']
    if not isinstance(name, str) or name not in TOOL_ORDER:
        raise ProtocolError('UNKNOWN_TOOL', 'Allowed tools: '+', '.join(TOOL_ORDER))
    arguments = payload['arguments']
    if isinstance(arguments, str):
        arguments = strict_json(arguments)
    if not isinstance(arguments, dict) or list(arguments) != ['analysis','actions']:
        raise ProtocolError('ARGUMENT_SCHEMA', 'arguments require exactly analysis then actions, in that order.')
    if not isinstance(arguments['analysis'], str) or not arguments['analysis'].strip():
        raise ProtocolError('ARGUMENT_SCHEMA', 'analysis must be a nonempty string before actions.')
    actions = arguments['actions']
    if not isinstance(actions, list) or len(actions) > max_actions:
        raise ProtocolError('ACTION_SCHEMA', f'actions must be a list with at most {max_actions} entries.')
    op = TOOL_OP[name]
    for action in actions:
        if not isinstance(action, dict) or set(action) != ACTION_FIELDS[op] or action['op'] != op:
            raise ProtocolError('ACTION_SCHEMA', f'{name} requires op={op} and fields {sorted(ACTION_FIELDS[op])}.')
        if any(type(action[key]) is not int for key in ('node','u','v') if key in action):
            raise ProtocolError('ACTION_SCHEMA', 'Node IDs must be JSON integers (not bool, string or float).')
        if op == 'REWRITE' and (not isinstance(action['text'], str) or not action['text'].strip()):
            raise ProtocolError('ACTION_SCHEMA', 'REWRITE requires nonempty full text.')
    return name, arguments


def validate_final(payload, labels):
    from label_utils import _label_key
    if not isinstance(payload, dict) or list(payload) != ['analysis','prediction']:
        raise ProtocolError('FINAL_SCHEMA', 'Final JSON requires nonempty analysis then prediction, in that order.')
    if not isinstance(payload['analysis'], str) or not payload['analysis'].strip():
        raise ProtocolError('FINAL_SCHEMA', 'Final analysis must be a nonempty string.')
    prediction = payload['prediction']
    if not isinstance(prediction, str) or _label_key(prediction) not in labels:
        raise ProtocolError('PUBLIC_CANDIDATE', 'prediction must be one of the public candidate labels.')
    return labels[_label_key(prediction)]
