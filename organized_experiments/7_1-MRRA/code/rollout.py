"""A 单次生成；B 三次生成、Modify/Rewrite 分阶段事务与实际图反馈。"""
from dataclasses import asdict
import time

from episode import (first_messages, public_observation, parse_completion,
                     execute_actions, rejected_execution, direct_messages)
from reward import extract_answer, score_episode
from reference_metrics import compute_reference_metrics, noise_components
from executor_protocol import (execute_stage, observation_after, aggregate_stages, stage_messages,
                               parse_stage, extract_tagged_answer, PROTOCOL_VERSION)


def decide(sample, first, strategy, ablation='full'):
    if strategy not in {'llm','direct'}:
        raise ValueError('B uses execute_stage and aggregate_stages, not joint JSON decisions')
    observation = public_observation(sample)
    protocol_errors, declared, parsed = [], None, None
    execution_start = time.perf_counter()
    if strategy == 'direct':
        execution = execute_actions(observation, [], [])
    else:
        try:
            parsed = parse_completion(first.text, 'llm')
            modify, rewrite = parsed['modify']['actions'], parsed['rewrite']['actions']
            if ablation == 'modify_only' and rewrite not in ([], [{'op': 'NO_OP'}]):
                raise ValueError('Rewrite forbidden by ablation')
            if ablation == 'rewrite_only' and modify not in ([], [{'op': 'NO_OP'}]):
                raise ValueError('Modify forbidden by ablation')
            execution = execute_actions(observation, modify, rewrite)
            declared = parsed.get('restored_graph')
        except (ValueError, TypeError, KeyError) as exc:
            protocol_errors.append(str(exc))
            execution = rejected_execution(observation, exc)
    return execution, parsed, declared, protocol_errors, time.perf_counter() - execution_start


def finish_record(sample, strategy, messages, segments, decision, feedback=None, *, metrics=True, semantic=None,
                  stage_results=None):
    observation = public_observation(sample)
    execution, parsed, declared, protocol_errors, executor_seconds = decision
    protocol_errors = list(protocol_errors)
    answer_text = segments[-1].text
    if strategy == 'executor':
        if len(segments) != 3 or stage_results is None or len(stage_results) != 2:
            raise ValueError('B requires Modify, Rewrite, Predict and two execution results')
        try:
            parse_stage(answer_text, 'predict')
        except (ValueError, TypeError, KeyError) as exc:
            protocol_errors.append('predict: '+str(exc))
        answer = extract_tagged_answer(answer_text)
        reward, values = score_episode(sample, execution, answer, format_valid=not protocol_errors)
        for result in stage_results:
            values[result['stage']+'_format_valid'] = float(not result['protocol_errors'])
            values[result['stage']+'_legal_execution'] = float(result['execution']['error'] is None)
            values[result['stage']+'_rolled_back'] = float(result['execution']['rolled_back'])
        values['predict_format_valid'] = float(not any(e.startswith('predict:') for e in protocol_errors))
    else:
        answer = extract_answer(answer_text)
        reward, values = score_episode(sample, execution, answer)
    if strategy == 'direct':
        try:
            parse_completion(answer_text, 'answer')
        except (ValueError, TypeError, KeyError) as exc:
            protocol_errors.append(str(exc))
    details = []
    if metrics:
        reference, details = compute_reference_metrics(sample, execution, declared, semantic)
        values.update(reference)
    if strategy == 'llm' and declared is None:
        values.update(declared_graph_valid=0.0, declared_execution_consistent=0.0)
    values['protocol_valid'] = float(not protocol_errors)
    values['truncated'] = float(any(s.truncated for s in segments))
    values['prompt_tokens'] = sum(len(s.prompt_ids) for s in segments)
    values['generated_tokens'] = sum(len(s.completion_ids) for s in segments)
    values['generation_seconds'] = sum(s.elapsed_seconds for s in segments)
    values['executor_seconds'] = executor_seconds
    values['model_calls'] = len(segments)
    # 直接预测没有编辑行为奖励的研究含义，只有准确率作为 baseline 指标。
    if strategy == 'direct':
        reward = values['task_reward']
        for name in ('behavior_reward', 'legal_execution', 'illegal', 'correct_clean_noop', 'attacked_noop', 'unnecessary_clean_edit'):
            values[name] = None
        values['total'] = reward
    record = {'sample_id': sample['sample_id'], 'dataset': sample['dataset'], 'task_type': sample['task_type'],
              'attack_kind': sample.get('attack_kind'), 'attack_budget': sample.get('attack_budget'),
              'noise_profile': '+'.join(noise_components(sample)) or 'clean',
              'strategy': strategy, 'original_graph': observation['graph'],
              'decision': parsed, 'execution': execution, 'declared_graph': declared,
              'prediction': answer, 'ground_truth': sample['ground_truth'], 'reward': reward,
              'metrics': values, 'text_metrics': details, 'protocol_errors': protocol_errors,
              'messages': messages, 'feedback_messages': feedback,
              'segments': [asdict(s) for s in segments]}
    if strategy == 'executor':
        record.update(protocol_version=PROTOCOL_VERSION, stage_executions=stage_results)
    return record


def run_trajectory(policy, sample, strategy, *, training=False, ablation='full', metrics=True, semantic=None):
    messages = direct_messages(sample) if strategy == 'direct' else first_messages(sample, strategy, ablation)
    if strategy == 'executor':
        observation = public_observation(sample)
        modify = policy.generate(messages, 'modify', sample=training)
        structural = execute_stage(observation, modify.text, 'modify', ablation)
        rewrite_messages = stage_messages(messages, modify.text, structural['execution'], 'rewrite')
        rewrite = policy.generate(rewrite_messages, 'rewrite', sample=training)
        textual = execute_stage(observation_after(observation, structural), rewrite.text, 'rewrite', ablation)
        feedback = stage_messages(rewrite_messages, rewrite.text, textual['execution'], 'predict')
        answer = policy.generate(feedback, 'predict', sample=training)
        segments = [modify, rewrite, answer]
        record = finish_record(sample,strategy,messages,segments,aggregate_stages(structural,textual),feedback,
                               metrics=metrics,semantic=semantic,stage_results=[structural,textual])
        return segments, record
    stage = {'llm': 'llm', 'direct': 'direct'}[strategy]
    first = policy.generate(messages, stage, sample=training)
    decision = decide(sample, first, strategy, ablation)
    segments = [first]
    record = finish_record(sample, strategy, messages, segments, decision, metrics=metrics, semantic=semantic)
    return segments, record
