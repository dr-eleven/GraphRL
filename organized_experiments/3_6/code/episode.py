"""3_6：四次原生工具调用、真实图回填和S+B+T+H轨迹评分。"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import inspect
import json

import graph_state as state
from graph_state import (TOOL_ORDER, TOOL_OP, ACTION_FIELDS, serialize_graph, execute_actions,
                         text_nodes, public_episode, load_episodes, clean_episode,
                         _node_id, _edge_key, _edge_map)
from reward import score_episode

CONTINUE_MARKER = 'CONTINUE_TOOL_STAGE'
FINAL_MARKER = 'FINAL_PREDICTION_REQUIRED'
CURRENT_GRAPH_GUIDANCE = state.CURRENT_GRAPH_GUIDANCE
NO_ACTION_GUIDANCE = (
    'This stage used actions=[] and made no graph modifications. '
    'The graph remains exactly as it was before this stage. '
    + CURRENT_GRAPH_GUIDANCE
)
INVALID_ACTION_GUIDANCE = (
    'The previous tool call contained invalid actions. NONE of its actions took effect. '
    'Disregard the analysis in that rejected assistant turn and any analysis results from '
    'that rejected call. The graph remains exactly as it was before this stage. '
    + CURRENT_GRAPH_GUIDANCE
)
ANSWER_REQUEST = (
    '\n'+FINAL_MARKER+': All four tool opportunities have ended. '
    + CURRENT_GRAPH_GUIDANCE + ' '
    'Ignore analysis from rejected calls. '
    'Use the original task and candidate labels. Do not call another tool. Return exactly '
    'one JSON object with nonempty prediction reasoning first: '
    '{"analysis":"your evidence-based prediction reasoning","prediction":"one exact candidate label"}.'
)



def parse_action_completion(text):
    """检查analysis/tool/actions格式；图存在性、保护与冲突由执行器检查。"""
    return state.parse_action_completion(text)


def parse_prediction_completion(text):
    payload = json.loads(str(text).strip())
    if not isinstance(payload, dict) or list(payload) != ['analysis', 'prediction']:
        raise ValueError('Required final fields in order: analysis, prediction')
    if not isinstance(payload['analysis'], str) or not payload['analysis'].strip():
        raise ValueError('Prediction reasoning must be nonempty')
    if not isinstance(payload['prediction'], str) or not payload['prediction'].strip():
        raise ValueError('Prediction must be a nonempty label')
    return payload


def build_restoration_prompt(sample):
    """使用唯一协议定义，避免继承规则后再次追加相同指令。"""
    return state.build_restoration_prompt(sample)


def initial_messages(sample, max_turns=4):
    if max_turns != len(TOOL_ORDER):
        raise ValueError('Exactly four tool-call opportunities are required')
    return [{'role':'user','content':build_restoration_prompt(sample)+
        '\n'+CONTINUE_MARKER+': del_node.'}]


def build_prediction_prompt(sample, restored_graph):
    visible = public_episode(sample)
    return (f"Task type: {visible['task_type']}\nTarget: {visible['target']}\n"
        f"Candidate labels: {json.dumps(visible.get('label_space',[]),ensure_ascii=False)}\n"
        f"Original input graph G0:\n<restored_graph>\n{serialize_graph(restored_graph)}\n</restored_graph>\n"
        'This is a no-edit baseline; do not call tools.'+ANSWER_REQUEST.replace(
            'All four tool opportunities have ended.', 'No editing is requested for this baseline.'))


def message_text(message):
    content = message.get('content') or ''
    return content if isinstance(content,str) else ''.join(p.get('text','') for p in content)


def _stage_tool(method):
    """保留工具schema的必需参数；缺参/多参终止轨迹并反馈格式错误。"""
    signature = inspect.signature(method)

    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            return self._step(method.__name__, kwargs.get('analysis'), kwargs.get('actions'),
                              argument_error=f'Invalid tool arguments: {exc}')
        return method(self, *args, **kwargs)
    return guarded


class GraphEnvironment:
    """每个候选独立持图；仅四个公开阶段方法由TRL注册为工具。"""
    def __init__(self, max_turns=4):
        if max_turns != len(TOOL_ORDER):
            raise ValueError('Exactly four calls for del_node/del_edge/add_edge/rewrite are required')
        self._max_turns = max_turns

    def reset(self, episode, bypass_actions=False, **kwargs):
        episode = json.loads(episode) if isinstance(episode,str) else episode
        self._sample = deepcopy(episode)
        self._state = state.GraphEnvironment(episode,self._max_turns)
        self._bypass = bypass_actions
        self._invocations, self._record = [], {}
        self._generation_limit_events = []
        self._format_errors = []
        self._format_error_stage = None
        self._native_turn, self._last_call_turn = None, None
        if bypass_actions:
            self._state.done, self._state.finish_reason = True, 'baseline'
        return None

    @_stage_tool
    def del_node(self, analysis: str, actions: list[dict]) -> str:
        """Identify possibly injected noise nodes from the current valid graph and delete them only when supported by evidence.

        Args:
            analysis: Nonempty reasoning based ONLY on the current valid graph: use connection structure and node-text semantics to assess which nodes, if any, may have been injected as noise. Justify deleting those nodes or choosing no edit. Must agree with actions.
            actions: All DELETE_NODE objects with op,node for this stage, or [] if no edit is justified.

        Returns:
            Nonempty success: updated graph and next stage. Empty actions: no graph modification, no repeated graph. Rejection: error reason, no repeated graph, ignore rejected analysis. Continue ONLY from the current valid graph. Format errors terminate.
        """
        return self._step('del_node',analysis,actions)

    @_stage_tool
    def del_edge(self, analysis: str, actions: list[dict]) -> str:
        """Identify existing edges that may have been added as structural noise and delete them only when supported by evidence.

        Args:
            analysis: Nonempty reasoning based ONLY on the current valid graph: use connection structure and node-text semantics to assess which existing edges, if any, may have been added as noise. Justify deleting those edges or choosing no edit. Must agree with actions.
            actions: All DEL_EDGE objects with op,u,v for this stage, or [] if no edit is justified.

        Returns:
            Nonempty success: updated graph and next stage. Empty actions: no graph modification, no repeated graph. Rejection: error reason, no repeated graph, ignore rejected analysis. Continue ONLY from the current valid graph. Format errors terminate.
        """
        return self._step('del_edge',analysis,actions)

    @_stage_tool
    def add_edge(self, analysis: str, actions: list[dict]) -> str:
        """Identify missing edges that may have been deleted by structural noise and restore them only when supported by evidence.

        Args:
            analysis: Nonempty reasoning based ONLY on the current valid graph: use connection structure and node-text semantics to assess which absent edges, if any, may have been deleted by noise. Justify restoring those edges or choosing no edit; an edge deleted earlier may be added back if both endpoints still exist. Must agree with actions.
            actions: All ADD_EDGE objects with op,u,v for this stage, or [] if no edit is justified.

        Returns:
            Nonempty success: updated graph and next stage. Empty actions: no graph modification, no repeated graph. Rejection: error reason, no repeated graph, ignore rejected analysis. Continue ONLY from the current valid graph. Format errors terminate.
        """
        return self._step('add_edge',analysis,actions)

    @_stage_tool
    def rewrite(self, analysis: str, actions: list[dict]) -> str:
        """Identify possible textual noise in the target node's text and rewrite it only when supported by evidence.

        Args:
            analysis: Nonempty reasoning based ONLY on the current valid graph: use connection structure and node-text semantics to assess possible textual noise in the target node (or permitted query endpoints for LC). Justify a full-text rewrite or choosing no edit, preserving supported facts and adding no invented facts or answer hints. Must agree with actions.
            actions: All REWRITE objects with op,node,text where text is the complete restored text, or [] if no edit is justified.

        Returns:
            Nonempty success: updated graph and prediction request. Empty actions: no graph modification, no repeated graph. Rejection: error reason, no repeated graph, ignore rejected analysis. Predict ONLY from the current valid graph. Format errors terminate.
        """
        return self._step('rewrite',analysis,actions)

    def _abort_format(self, reason):
        if self._format_error_stage is None:
            self._format_error_stage = min(self._state.stage, len(TOOL_ORDER))
        if reason not in self._format_errors:
            self._format_errors.append(reason)
            self._state.errors.append('Format error: '+reason)
        self._state.done = True
        if not self._generation_limit_events:
            self._state.finish_reason = 'format_error'
        return self._format_feedback()

    def _format_feedback(self):
        return ('FORMAT_ERROR: '+ '; '.join(self._format_errors)+
                '. Trajectory terminated. No further actions or prediction will be requested. '
                'S, B and H earned in earlier completed stages are retained. '
                'The failed stage and unvisited stages earn 0; T is 0.')

    def _validate_turn(self, message):
        """Validate before dispatch, so malformed multi-calls cannot partially commit."""
        if self._format_errors:
            return False
        try:
            calls = message.get('tool_calls') or []
            if self._state.done:
                if calls:
                    raise ValueError('No tool call is allowed in the final prediction turn')
                parse_prediction_completion(message_text(message))
            else:
                if len(calls) != 1:
                    raise ValueError(f'Expected exactly one {self._state.current_tool} tool call; received {len(calls)}')
                call = calls[0]
                if not isinstance(call, dict) or call.get('type') != 'function':
                    raise ValueError('Expected a function tool call')
                function = call.get('function', {})
                if not isinstance(function, dict):
                    raise ValueError('Tool function must be an object')
                name, arguments = function.get('name'), function.get('arguments')
                if name != self._state.current_tool:
                    raise ValueError(f'Expected tool {self._state.current_tool}; received {name}')
                if not isinstance(arguments, dict) or set(arguments) != {'analysis','actions'}:
                    raise ValueError('Tool arguments require exactly analysis and actions')
                parse_action_completion(json.dumps(dict(analysis=arguments['analysis'],tool=name,
                                                         actions=arguments['actions'])))
        except (ValueError, TypeError, KeyError) as exc:
            self._abort_format(str(exc))
            return False
        return True

    def _step(self, tool, analysis, actions, argument_error=None):
        if self._format_errors:
            return self._format_feedback()
        if argument_error:
            return self._abort_format(argument_error)
        if self._native_turn is not None and self._last_call_turn == self._native_turn:
            return self._abort_format('Only one tool call per assistant turn is allowed')
        call = {'role':'assistant','tool_calls':[{'type':'function','function':{
            'name':tool,'arguments':{'analysis':analysis,'actions':actions}}}]}
        if not self._validate_turn(call):
            return self._format_feedback()
        self._last_call_turn = self._native_turn
        raw = json.dumps({'analysis':analysis,'tool':tool,'actions':actions},ensure_ascii=False)
        self._invocations.append({'tool':tool,'analysis':analysis,'actions':deepcopy(actions),
                                  'argument_error':None})
        return self._observation(self._state.step(raw))

    def _observation(self, observation):
        # 图标签只包实际图；执行反馈在标签外。整条tool消息由TRL mask排除直接loss。
        metadata = {key:value for key,value in observation.items() if key!='current_graph'}
        content = 'Execution feedback: '+json.dumps(metadata,ensure_ascii=False)
        if observation.get('ok') and not self._state.events[-1]['call']['actions']:
            content += '\n'+NO_ACTION_GUIDANCE
        elif observation.get('ok'):
            content += '\n<restored_graph>\n'+observation['current_graph']+'\n</restored_graph>'
        else:
            content += '\n'+INVALID_ACTION_GUIDANCE
        if self._state.done:
            return content+ANSWER_REQUEST
        return (content+'\n'+CONTINUE_MARKER+': '+self._state.current_tool+
                '. Analyze ONLY the current valid graph and call this tool once.')

    def _stage_structure_valid(self, completion):
        """Credit only observed, schema-valid stage pairs; stop at the first gap."""
        valid = {name: False for name in (*TOOL_ORDER, 'prediction')}
        if self._bypass:
            return valid
        for index, name in enumerate(TOOL_ORDER):
            if self._format_error_stage is not None and index >= self._format_error_stage:
                break
            if len(completion) <= 2*index+1 or index >= len(self._invocations):
                break
            message, feedback = completion[2*index:2*index+2]
            calls = message.get('tool_calls') or []
            if message.get('role') != 'assistant' or len(calls) != 1 or calls[0].get('type') != 'function':
                break
            function = calls[0].get('function', {})
            arguments = function.get('arguments')
            if function.get('name') != name or not isinstance(arguments, dict) or set(arguments) != {'analysis', 'actions'}:
                break
            try:
                parse_action_completion(json.dumps(dict(analysis=arguments['analysis'], tool=name, actions=arguments['actions'])))
            except (ValueError, TypeError, KeyError):
                break
            invocation = self._invocations[index]
            if (invocation['tool'] != name or invocation['analysis'] != arguments['analysis']
                    or invocation['actions'] != arguments['actions']):
                break
            if feedback.get('role') != 'tool' or feedback.get('name') != name:
                break
            valid[name] = True
        return valid

    def _abort_generation(self, stage, reason, token_limit, generated_tokens):
        """Stop only this candidate; no action dispatch or fabricated graph feedback."""
        self._generation_limit_events.append(dict(stage=stage, reason=reason,
            token_limit=token_limit, generated_tokens=generated_tokens))
        self._state.done = True
        self._state.finish_reason = 'token_limit'
        self._abort_format(f'{stage}: {reason}; generated {generated_tokens} tokens at limit {token_limit} without EOS')

    def _score(self, completion):
        assistants = [m for m in completion if m.get('role')=='assistant']
        tool_calls = [c for m in assistants for c in m.get('tool_calls') or []]
        tool_results = [m for m in completion if m.get('role')=='tool' and m.get('name')!='format_error']
        final = assistants[-1] if assistants else {}
        prediction_text = message_text(final)
        if any(len(m.get('tool_calls') or [])>1 for m in assistants):
            error='Multiple tool calls in one assistant turn are not permitted'
            if error not in self._state.errors: self._state.errors.append(error)
        if len(tool_calls)!=len(self._invocations):
            error='Unexecuted, unknown or malformed native tool call'
            if error not in self._state.errors: self._state.errors.append(error)
        # Unobserved commits earn no stage credit; preserve earlier observed pairs.
        unobserved = len(tool_results)<len(self._invocations)
        stage_valid = self._stage_structure_valid(completion)
        try:
            if self._format_errors:
                raise ValueError('Trajectory terminated before a valid final prediction')
            if not self._bypass and (not all(stage_valid[name] for name in TOOL_ORDER)
                    or len(completion) != 2*len(TOOL_ORDER)+1):
                raise ValueError('Incomplete or malformed four-stage tool/prediction protocol')
            if self._bypass and len(completion) != 1:
                raise ValueError('Baseline requires exactly one prediction turn')
            if final.get('role') != 'assistant':
                raise ValueError('Missing final prediction turn')
            if final.get('tool_calls'): raise ValueError('Missing final prediction turn')
            parsed = parse_prediction_completion(prediction_text)
            prediction = json.dumps({'prediction':parsed['prediction']})
            prediction_format = True
        except (ValueError,TypeError) as exc:
            if not self._format_errors:
                self._abort_format('Invalid final prediction: '+str(exc))
            prediction, prediction_format = '{"prediction":""}', False
        stage_valid['prediction'] = prediction_format and not self._bypass
        reward, metrics = score_episode(self._sample,self._state,prediction,
                                       stage_valid=stage_valid, prediction_valid=prediction_format)
        length_aborted = bool(self._generation_limit_events)
        metrics.update(format_valid=float(not self._format_errors), prediction_format_valid=float(prediction_format),
            observed_execution=float(not unobserved),
            token_limit_exceeded=float(length_aborted),
            generation_length_valid=float(not length_aborted),action_metrics_applicable=float(not self._bypass),
            structure_metrics_applicable=float(not self._bypass))
        self._record = dict(source_index=self._sample['source_index'],messages=completion,
            format_errors=list(self._format_errors),
            format_feedback=self._format_feedback() if self._format_errors else None,
            generation_limit_events=deepcopy(self._generation_limit_events),
            events=deepcopy(self._state.events),tool_invocations=deepcopy(self._invocations),
            executed_actions=deepcopy(self._state.actions),final_graph=deepcopy(self._state.graph),
            errors=list(self._state.errors),finish_reason=self._state.finish_reason,
            prediction_raw=prediction_text,reward=reward,metrics=metrics)
        return reward
