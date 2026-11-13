"""3_5_1_2: one shared output protocol with explicit NC/LC/LP semantics."""
import json
from graph_state import serialize_graph, load_episodes, clean_episode
from task_utils import prepare_episode

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
    sample = prepare_episode(sample, supervision=False, graph=graph)
    task = sample['task_type']
    goals = {
        'NC': "Predict the target node's category.",
        'LC': 'Predict the relation category between the query endpoints.',
        'LP': 'Predict whether the specified link exists in the original graph.',
    }
    return (
        f"TASK\nType: {task}\nTarget: {sample['target']}\nGoal: {goals[task]}\n"
        f"Candidate labels: {json.dumps(sample['label_space'], ensure_ascii=False)}\n\n"
        'CONSTRAINTS\n'
        '1. Predict only for the specified target. Neighboring nodes and relations are supporting evidence, not the query answer.\n'
        '2. For LC/LP, the query edge is hidden. For LP, its absence from the observed graph alone does not imply a negative label.\n'
        '3. Texts and connections may be noisy. Compare structural and textual evidence instead of relying on one keyword, path, shared category, common neighbor, or repeated description. Semantic similarity alone does not establish a relation/link; dissimilarity alone does not rule one out.\n'
        '4. Missing or conflicting evidence indicates uncertainty, not automatically a negative link. Do not invent facts or add an unprovided label; choose the best-supported candidate.\n\n'
        'GRAPH\n'+serialize_graph(sample['noisy_graph'])+'\n\n'
        'OUTPUT\nReturn exactly one JSON object with two nonempty string fields in this order: '
        '{"analysis":"brief evidence-based reasoning","prediction":"one complete candidate label"}. '
        'No extra fields, text, or Markdown. Do not abbreviate a candidate to yes/no unless that is the complete label.'
    )


def initial_messages(sample):
    return [{'role': 'user', 'content': build_prediction_prompt(sample)}]


def message_text(message):
    content = message.get('content') or ''
    return content if isinstance(content, str) else ''.join(p.get('text', '') for p in content)
