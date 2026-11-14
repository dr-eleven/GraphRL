"""Binary terminal reward environment with recoverable errors and atomic graph commits."""
from copy import deepcopy
from dataclasses import asdict
import json

from budgets import RolloutBudget
from project_config import DEFAULT_MODE
from graph_state import (TOOL_ORDER, TOOL_OP, ACTION_FIELDS, serialize_graph, execute_actions,
                         text_nodes, public_episode, load_episodes, clean_episode)
from label_utils import _label_key
from protocol import parse_event, validate_tool, validate_final, ProtocolError
from reward import terminal_reward
from validation import inspect_tool, inspect_final
from runtime import InfrastructureError, deadline

CONTINUE_MARKER = 'CONTINUE_TOOL_STAGE'
FINAL_MARKER = 'FINAL_PREDICTION_REQUIRED'
CURRENT_GRAPH_GUIDANCE = ('Use the latest <restored_graph>, or <initial_graph> if no update exists. '
                          'No new graph means no change. Rejected calls change nothing; ignore their analysis. ')
PARSE_ERROR_CODES = {'JSON_SYNTAX','DUPLICATE_KEY','JSON_STRING','EVENT_FRAMING',
                     'INVALID_EVENT','GENERATION_TRUNCATED'}


def label_map(sample):
    labels = sample.get('label_space')
    if not isinstance(labels, list) or not labels or any(not isinstance(x,str) or not _label_key(x) for x in labels):
        raise ValueError('Nonempty public candidate labels required')
    result = {_label_key(x):x for x in labels}
    if len(result) != len(labels):
        raise ValueError('Public labels collide after canonical normalization')
    return result


def initial_messages(sample, budget=None, mode=DEFAULT_MODE, bypass_actions=False):
    budget = budget or RolloutBudget()
    visible = public_episode(sample)
    label_map(sample)
    visible.pop('noisy_graph')
    system = (
        'You are a Think-with-Graph policy for node/link classification on text-attributed graphs. '
        'The graph may be clean or contain structural/textual noise. Restore it only when supported by '
        'graph structure and node-text semantics, then predict one candidate label. Noise locations and amounts are unknown.\n'
        + CURRENT_GRAPH_GUIDANCE + '\n'
        'Use existing integer node IDs. Never delete protected/target nodes or edit protected edges. '
        'No self-loops, missing endpoints, relation fields, or duplicate/conflicting actions within a call. '
        'DEL_EDGE requires a present edge; ADD_EDGE an absent edge, including one deleted earlier. '
        'Deleting a node removes its incident edges. '
        'REWRITE only the NC target or LC query endpoints, with nonempty full text, at most once per node per call. '
        'Copying or changing only whitespace/case is invalid. Preserve supported facts; do not invent facts or insert answers, labels, or answer hints.\n'
        'Each turn: one <tool_call> containing name and arguments, or final JSON. '
        'Arguments must be exactly {"analysis":"nonempty reasoning","actions":[]} in this order; '
        'For all tools, analysis must justify matching actions, or actions=[] when no edit is warranted. '
        'Action objects must use exactly the listed fields and op. Never simulate execution or feedback. '
        'Final JSON must be {"analysis":"nonempty prediction reasoning","prediction":"one candidate label"} '
        'in this order. No text outside the event.\n'
    )
    if bypass_actions:
        system += 'Do not edit; submit final JSON.\n'
    elif mode == 'ordered_four_stage_retry':
        system += ('Start with del_node. Order: del_node -> del_edge -> add_edge -> rewrite. '
                   'After success (including actions=[]), repeat that tool or move one stage forward; never skip or return. '
                   'After failure, correct the current stage; a new success reopens the next stage. '
                   'After rewrite succeeds, repeat rewrite or submit final JSON.\n')
    else:
        system += 'Choose any tool in any order, repeat tools, or submit final JSON at any time.\n'
    system += (f'At most {budget.max_model_turns} generation turns, counting errors and final submissions; '
               f'at most {budget.max_actions_per_call} actions per call. No stage/error-count limit or reserved final turn. '
               'At the limit, stop without an extra answer request; no valid final answer means reward 0. '
               'A valid final submission ends immediately. Parse errors terminate; other rejections list detectable errors '
               'for correction. Token/context limits may stop earlier.')
    user = json.dumps(visible,ensure_ascii=False,separators=(',',':'))+'\n<initial_graph>\n'+serialize_graph(sample['noisy_graph'])+'\n</initial_graph>'
    return [{'role':'system','content':system},{'role':'user','content':user}]


def build_prediction_prompt(sample, restored_graph):
    """Return the complete system/user messages for the no-edit evaluation condition."""
    return initial_messages({**sample,'noisy_graph':restored_graph},bypass_actions=True)


class GraphEnvironment:
    def __init__(self, budget=None, mode=DEFAULT_MODE):
        if mode not in {'ordered_four_stage_retry','free'}:
            raise ValueError('Unknown scheduling mode')
        self._budget, self._mode = budget or RolloutBudget(), mode

    def reset(self, episode, bypass_actions=False, **kwargs):
        sample = json.loads(episode) if isinstance(episode,str) else episode
        self._sample = deepcopy(sample)  # hidden supervision is accessed only by _score
        self._public = public_episode(sample)
        self._labels = label_map(sample)
        if _label_key(sample['ground_truth']) not in self._labels:
            raise ValueError('ground_truth must match a unique public label')
        self._graph = deepcopy(self._public['noisy_graph'])
        self._phase = 'ANSWER_ONLY' if bypass_actions else 'EDITING'
        self._bypass = bypass_actions
        self._stage = self._version = 0
        self._stage_succeeded = False
        self._stage_attempts = [0]*4
        self._turns = self._model_tokens = self._feedback_tokens = 0
        self._tool_attempts = self._tool_successes = self._final_attempts = 0
        self._edit_errors = self._consecutive_errors = 0
        self._prediction = self._finish_reason = None
        self._events, self._errors, self._actions, self._raw_turns = [], [], [], []
        self._record = {}
        self._last_finish = 'stop'
        return None

    # These are the only exposed methods besides reset. The controller validates before dispatch.
    def del_node(self, analysis: str, actions: list[dict]):
        """Delete injected noise nodes.

        Args:
            analysis: Reasoning.
            actions: DELETE_NODE {op,node}.
        """
        return self._direct_tool('del_node',actions,analysis)

    def del_edge(self, analysis: str, actions: list[dict]):
        """Delete edges added by structural noise.

        Args:
            analysis: Reasoning.
            actions: DEL_EDGE {op,u,v}.
        """
        return self._direct_tool('del_edge',actions,analysis)

    def add_edge(self, analysis: str, actions: list[dict]):
        """Restore edges removed by structural noise.

        Args:
            analysis: Reasoning.
            actions: ADD_EDGE {op,u,v}.
        """
        return self._direct_tool('add_edge',actions,analysis)

    def rewrite(self, analysis: str, actions: list[dict]):
        """Rewrite target text corrupted by textual noise.

        Args:
            analysis: Reasoning.
            actions: REWRITE {op,node,text}; text is the full replacement.
        """
        return self._direct_tool('rewrite',actions,analysis)

    def _direct_tool(self, name, actions, analysis):
        raise RuntimeError('Tools must run through the budgeted atomic rollout controller')

    @property
    def _active(self):
        return self._phase in {'EDITING','ANSWER_ONLY'}

    @property
    def _current_tool(self):
        return TOOL_ORDER[self._stage] if self._mode == 'ordered_four_stage_retry' and self._phase == 'EDITING' else None

    def _allowed_tools(self):
        if self._phase != 'EDITING':
            return []
        if self._mode == 'free':
            return list(TOOL_ORDER)
        allowed=[TOOL_ORDER[self._stage]]
        if self._stage_succeeded and self._stage<3:
            allowed.append(TOOL_ORDER[self._stage+1])
        return allowed

    def _finish(self, reason):
        self._phase, self._finish_reason = 'TERMINAL', reason

    def _apply_limits(self):
        if self._active and self._turns >= self._budget.max_model_turns:
            self._finish('max_model_turns')
        elif self._active and self._model_tokens >= self._budget.max_total_model_tokens:
            self._finish('model_token_budget')

    def _capacity(self, initial_tokens, completion_tokens, context_limit):
        self._apply_limits()
        if not self._active:
            return 0,None
        b=self._budget
        available=min(b.max_total_model_tokens-self._model_tokens,
                      b.max_completion_tokens-completion_tokens,
                      context_limit-initial_tokens-completion_tokens)
        if available<=0:
            self._finish('token_or_context_budget')
            return 0,None
        return min(b.max_new_tokens_per_turn,available),None

    def _feedback(self, code, message, tool_name=None, graph=None):
        content = 'ENVIRONMENT_FEEDBACK '+code
        if code == 'OK':
            content += ': Completed successfully.'
        else:
            content += ': '+message
        if graph is not None:
            content += '\n<restored_graph>\n'+serialize_graph(graph)+'\n</restored_graph>'
        else:
            reference = 'the latest <restored_graph>' if self._version else '<initial_graph>'
            content += '\nGraph unchanged; use '+reference+'.'
        if code == 'OK' and self._phase == 'EDITING' and self._turns < self._budget.max_model_turns:
            if self._mode == 'free':
                content += '\nnext_tool: any tool or final'
            else:
                choices=self._allowed_tools()
                if self._stage==3 and self._stage_succeeded:choices.append('final')
                content += '\nnext_tool: '+' or '.join(choices)
        result = dict(role='tool' if tool_name else 'user',content=content)
        if tool_name:
            result['name'] = tool_name
        return result

    def _parse_failure(self, code, message):
        self._errors.append(dict(code=code,message=message,turn=self._turns,phase=self._phase))
        self._finish('parse_error')
        return None

    def _error(self, code, message, tool_name=None):
        return self._error_many([dict(code=code,path='$',message=message)],tool_name)

    def _error_many(self, errors, tool_name=None):
        for error in errors:
            self._errors.append(dict(error,turn=self._turns,phase=self._phase))
        if self._phase == 'EDITING':
            self._edit_errors += 1  # One failed generation, regardless of how many issues it contains.
            self._consecutive_errors += 1
            self._stage_succeeded = False
        self._apply_limits()
        code=errors[0]['code'] if len(errors)==1 else 'ERRORS'
        message='\n'.join(f"- {e['code']} {e['path']}: {e['message']}" for e in errors)
        return self._feedback(code,message,tool_name)

    def _consume(self, raw, token_ids, finish_reason='stop', feedback_fits=lambda message: True):
        if not self._active:
            raise RuntimeError('Cannot consume output after termination')
        editing = self._phase == 'EDITING'
        self._turns += 1
        self._model_tokens += len(token_ids)
        if not editing:
            self._final_attempts += 1
        event = parse_event(raw,finish_reason)
        self._raw_turns.append(dict(raw_text=raw,raw_token_ids=list(token_ids),finish_reason=finish_reason,
                                   event_kind=event.kind,parse_error_code=event.code,turn=self._turns))
        # Choosing an allowed successor enters that stage; errors there must be repaired there.
        if editing and self._mode=='ordered_four_stage_retry':
            if (event.kind=='TOOL_EVENT' and isinstance(event.payload,dict) and self._stage_succeeded
                    and self._stage<3 and event.payload.get('name')==TOOL_ORDER[self._stage+1]):
                self._stage += 1
                self._stage_succeeded = False
            self._stage_attempts[self._stage] += 1  # diagnostic only; no limit
        assistant = dict(role='assistant',content=raw)
        if editing and event.tool_attempt:
            self._tool_attempts += 1
        if event.kind == 'INVALID_EVENT':
            return assistant,self._parse_failure(event.code,event.message)
        if event.kind == 'FINAL_EVENT':
            prediction,errors=inspect_final(event.payload,self._labels)
            if editing and self._mode == 'ordered_four_stage_retry' and not (self._stage==3 and self._stage_succeeded):
                errors.insert(0,dict(code='CURRENT_STAGE_REQUIRED',path='$',
                                    message='Final prediction is premature under the fixed four-stage protocol.'))
                return assistant,self._error_many(errors)
            if editing:
                self._phase = 'ANSWER_ONLY'
                self._final_attempts += 1
            if errors:
                return assistant,self._error_many(errors)
            self._prediction = prediction
            self._finish('final_submitted')
            return assistant,None
        try:
            name,arguments,errors=inspect_tool(event.payload,self._public,self._graph,
                                               self._allowed_tools(),self._budget.max_actions_per_call)
        except ProtocolError as exc:
            return assistant,self._parse_failure(exc.code,str(exc))
        if not editing:
            errors.insert(0,dict(code='FINAL_REQUIRED',path='$',message='Editing is closed; submit final JSON.'))
        if errors:
            return assistant,self._error_many(errors)
        assistant = dict(role='assistant',content='',tool_calls=[dict(type='function',function=dict(name=name,arguments=arguments))])
        try:
            with deadline(self._budget.tool_timeout_sec,'graph executor'):
                graph,error = execute_actions({**self._public,'noisy_graph':self._graph},arguments['actions'])
        except Exception as exc:
            self._phase = 'ABORTED_INFRA'
            raise InfrastructureError(f'Graph execution failed: {type(exc).__name__}: {exc}') from exc
        if error is not None:
            return assistant,self._error('INVALID_ACTION',error,name)
        # Tentative scheduling/version used to render exactly the feedback that would be committed.
        prior = (self._version,self._stage_succeeded,self._consecutive_errors)
        changed = graph != self._graph
        self._version += int(changed)
        self._consecutive_errors = 0
        self._stage_succeeded = True
        feedback = self._feedback('OK','Actions committed.' if arguments['actions'] else 'No graph modifications.',
                                  name,graph if changed else None)
        if not feedback_fits(feedback):
            self._version,self._stage_succeeded,self._consecutive_errors = prior
            self._finish('feedback_capacity_exhausted')
            self._events.append(dict(status='discarded_prepared_graph',tool=name))
            return assistant,self._feedback('FEEDBACK_BUDGET','No actions committed; graph unchanged. Feedback capacity exhausted.',name)
        self._graph = graph
        self._actions.extend(deepcopy(arguments['actions']))
        self._tool_successes += 1
        self._apply_limits()
        self._events.append(dict(status='committed' if arguments['actions'] else 'noop',tool=name,
                                 actions=deepcopy(arguments['actions']),turn=self._turns,graph_version=self._version))
        return assistant,feedback

    def _score(self, completion):
        if self._phase == 'ABORTED_INFRA':
            raise InfrastructureError('An invalid collection cannot receive a binary policy reward')
        if self._phase != 'TERMINAL':
            raise RuntimeError('Rollout controller must terminate before scoring')
        reward = terminal_reward(self._prediction,self._sample['ground_truth'])
        metrics = dict(prediction_correct=reward,task_reward=reward,total=reward,
                       valid_final=float(self._prediction is not None),model_turns=self._turns,
                       model_tokens=self._model_tokens,feedback_tokens=self._feedback_tokens,
                       tool_attempts=self._tool_attempts,tool_successes=self._tool_successes,
                       edit_errors=self._edit_errors,final_attempts=self._final_attempts,
                       recovered_correct=float(bool(self._errors) and reward==1),
                       zero_tool=float(self._tool_attempts==0),
                       budget_skips=sum(e['status']=='skipped_budget' for e in self._events))
        self._record = dict(source_index=self._sample['source_index'],mode=self._mode,
                            dataset=self._sample.get('dataset','unknown'),task_type=self._sample['task_type'],
                            valid_for_training=True,messages=deepcopy(completion),raw_turns=deepcopy(self._raw_turns),
                            errors=deepcopy(self._errors),events=deepcopy(self._events),
                            executed_actions=deepcopy(self._actions),final_graph=deepcopy(self._graph),
                            prediction=self._prediction,finish_reason=self._finish_reason,
                            reward=reward,metrics=metrics,budgets=asdict(self._budget))
        return reward
