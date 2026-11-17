"""A model-facing view of Memory; full historical graphs remain in the audit store."""
from copy import deepcopy
from graph_core.graph_state import serialize_graph


class PromptMemory:
    def __init__(self, memory, session):
        self.memory = memory
        self.session = session

    def get_actions(self):
        history = {}
        for name, action in self.memory.get_actions().items():
            result = action['result']
            # Commands preserve exact attempted parameters; errors preserve the
            # reason for failure. Repeated graph snapshots add no new evidence.
            if isinstance(result, dict):
                result = {key: deepcopy(result[key]) for key in
                          ['ok', 'changed', 'graph_unchanged', 'errors'] if key in result}
            history[name] = dict(tool_name=action['tool_name'],
                                 sub_goal=action['sub_goal'],
                                 command=action['command'], result=deepcopy(result))
        return dict(current_graph=serialize_graph(self.session.graph),
                    history=history)
