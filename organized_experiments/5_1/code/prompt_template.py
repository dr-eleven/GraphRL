"""Keep Qwen tool framing aligned with the one-event-per-turn graph protocol."""
def single_call_template(template):
    if template is None:
        return None
    return template.replace('You may call one or more functions to assist with the user query.',
                            'Call at most one function per turn.')
