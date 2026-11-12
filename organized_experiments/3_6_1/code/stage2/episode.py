"""3_6_1_stage2 single-turn prediction on the supplied graph."""
import json
from graph_state import serialize_graph, load_episodes, clean_episode

GRAPH_CONDITION = (
    'The graph may be clean, contain structural noise, textual noise, or a combination of both. '
    'The noise locations and budgets are unknown.'
)


def parse_prediction_completion(text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field: '+key)
            result[key] = value
        return result
    payload = json.loads(str(text).strip(), object_pairs_hook=unique_pairs)
    if not isinstance(payload, dict) or list(payload) != ['analysis', 'prediction']:
        raise ValueError('Required final fields in order: analysis, prediction')
    if not isinstance(payload['analysis'], str) or not payload['analysis'].strip():
        raise ValueError('Prediction reasoning must be nonempty')
    if not isinstance(payload['prediction'], str) or not payload['prediction'].strip():
        raise ValueError('Prediction must be a nonempty label')
    return payload


def build_prediction_prompt(sample, graph=None):
    # Explicit public fields only: never include stored prompts or hidden supervision.
    return (
        'TASK DESCRIPTION\n'
        f"Task type: {sample['task_type']}\nTarget: {sample['target']}\n"
        f"Candidate labels: {json.dumps(sample.get('label_space', []), ensure_ascii=False)}\n"
        'GRAPH CONDITION\n'+GRAPH_CONDITION+'\n'
        'Analyze the supplied graph structure and node texts, accounting for possible noise, '
        'and predict the class of the target node or query edge. Choose one exact candidate label.\n'
        'Current graph G0:\n'+serialize_graph(sample['noisy_graph'] if graph is None else graph)+'\n'
        'OUTPUT FORMAT\nReturn only one JSON object with nonempty prediction reasoning first: '
        '{"analysis":"your evidence-based prediction reasoning","prediction":"one exact candidate label"}. '
        'Use exactly these fields in this order, with no prose outside the JSON object.'
    )


def initial_messages(sample):
    return [{'role': 'user', 'content': build_prediction_prompt(sample)}]


def message_text(message):
    content = message.get('content') or ''
    return content if isinstance(content, str) else ''.join(p.get('text', '') for p in content)
