"""B: Modify → execute → Rewrite → execute → Reason/Answer, with stage-local transactions."""
from copy import deepcopy
import json
from pathlib import Path
import re
import time

from episode import public_observation, read_json, execute_actions, rejected_execution

STAGE_TAGS = {'modify':'modify_action', 'rewrite':'rewrite_action', 'predict':'answer'}
GRAPH_INSTRUCTION = '使用最新修改图进行后续推理。'
PROTOCOL_VERSION = 'executor_three_stage_tags_v1'


def first_executor_messages(sample, ablation='full'):
    task = sample['task_type'].lower()
    instruction = (Path(__file__).parent/'prompts'/f'executor_{task}_system.txt').read_text()
    if ablation == 'modify_only':
        instruction += 'Rewrite is disabled; its action array must be [].\n'
    elif ablation == 'rewrite_only':
        instruction += 'Modify is disabled; its action array must be [].\n'
    return [{'role':'system','content':instruction},
            {'role':'user','content':'CURRENT INPUT\n'
             + json.dumps(public_observation(sample),ensure_ascii=False)
             + '\n\n' + stage_instruction('modify')}]


def stage_instruction(stage):
    guidance = {
        'modify': 'Current stage: Modify. Decide whether the visible graph needs structural repair.',
        'rewrite': 'Current stage: Rewrite. Decide whether the editable texts in the returned graph need repair.',
        'predict': 'Current stage: Reason and Prediction. Editing is finished. Use the current graph and the '
                   'original task to select one exact candidate label from label_space.'}
    payload = 'one candidate label' if stage == 'predict' else 'JSON action array, or []'
    tag = STAGE_TAGS[stage]
    return (guidance[stage] + '\n\nResponse layout:\n<reason>your reasoning</reason>\n<'
            + tag + '>' + payload + '</' + tag + '>\nStop after </' + tag + '>.')


def stage_messages(messages, generated_text, execution, next_stage):
    if next_stage not in ('rewrite','predict'):
        raise ValueError('Unknown next stage: '+next_stage)
    metadata = {key:execution[key] for key in ('error','operation_status','rolled_back')}
    content = ('EXECUTION FEEDBACK (from Python):\n'
               + json.dumps(metadata,ensure_ascii=False) + '\n\n<restored_graph>'
               + json.dumps(execution['graph'],ensure_ascii=False) + '</restored_graph>\n'
               + GRAPH_INSTRUCTION + '\n\n' + stage_instruction(next_stage))
    return messages+[{'role':'assistant','content':generated_text},{'role':'user','content':content}]


def parse_stage(text, stage):
    """Formatting only. Action operation, scope and preconditions belong to execution/B."""
    tag = STAGE_TAGS[stage]
    # Full match disallows additional stage blocks, headings and prose outside the format.
    match = re.fullmatch(r'\s*<reason>(.*?)</reason>\s*<'+tag+r'>(.*?)</'+tag+r'>\s*',text,re.S)
    if match is None or not match.group(1).strip():
        raise ValueError(f'Expected nonempty <reason> then <{tag}> in {stage} stage')
    reason, payload = match.group(1).strip(), match.group(2).strip()
    reserved = ('reason','modify_action','rewrite_action','answer','restored_graph')
    for name in reserved:
        if '<'+name+'>' in reason or '</'+name+'>' in reason:
            raise ValueError('Duplicate or nested protocol tag in reason')
    if stage == 'predict':
        if not payload or any('<'+name+'>' in payload or '</'+name+'>' in payload for name in reserved):
            raise ValueError('Answer must be one nonempty answer block')
        return {'reason':reason,'answer':payload}
    actions = read_json(payload)
    if not isinstance(actions,list):
        raise ValueError('Action payload must be a JSON array, including [] for NO_OP')
    return {'reason':reason,'actions':actions}


def extract_tagged_answer(text):
    """T remains independent of the surrounding format; duplicate answer blocks are invalid."""
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return ''
    match = re.search(r'<answer>(.*?)</answer>',text,re.S)
    return match.group(1).strip() if match else ''


def execute_stage(observation, text, stage, ablation='full'):
    start = time.perf_counter()
    parsed, errors = None, []
    try:
        parsed = parse_stage(text,stage)
    except (ValueError,TypeError,KeyError) as exc:
        errors.append(f'{stage}: {exc}')
        execution = rejected_execution(observation,exc)
    else:
        actions = parsed['actions']
        disabled = (stage=='modify' and ablation=='rewrite_only') or (stage=='rewrite' and ablation=='modify_only')
        if disabled and actions:
            execution = rejected_execution(observation,stage+' forbidden by ablation')
        elif any(isinstance(a,dict) and a.get('op')=='NO_OP' for a in actions):
            execution = rejected_execution(observation,'NO_OP must be represented by []')
        else:
            execution = execute_actions(observation, actions if stage=='modify' else [],
                                         actions if stage=='rewrite' else [])
    return {'stage':stage,'parsed':parsed,'execution':execution,'protocol_errors':errors,
            'executor_seconds':time.perf_counter()-start}


def observation_after(observation, result):
    updated = deepcopy(observation)
    updated['graph'] = deepcopy(result['execution']['graph'])
    return updated


def aggregate_stages(modify, rewrite):
    stages = [modify,rewrite]
    errors = [r['stage']+': '+r['execution']['error'] for r in stages if r['execution']['error'] is not None]
    attempted, executed, statuses = [], [], []
    for result in stages:
        if result['parsed'] is not None:
            attempted += result['parsed']['actions']
        execution = result['execution']
        if execution['error'] is None:
            executed += execution['actions']
        statuses += [dict(row,stage=result['stage']) for row in execution['operation_status']]
    execution = {'graph':deepcopy(rewrite['execution']['graph']), 'error':'; '.join(errors) if errors else None,
                 'actions':attempted, 'executed_actions':executed, 'operation_status':statuses,
                 'rolled_back':bool(errors), 'rollback_scope':'stage'}
    parsed = {r['stage']:r['parsed'] for r in stages}
    protocol_errors = [e for r in stages for e in r['protocol_errors']]
    return execution, parsed, None, protocol_errors, sum(r['executor_seconds'] for r in stages)
