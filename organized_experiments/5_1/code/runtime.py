"""Infrastructure failures are invalid collection, never policy reward zero."""
from contextlib import contextmanager
import signal
import threading


class InfrastructureError(RuntimeError):
    pass


@contextmanager
def deadline(seconds, operation):
    # Linux/main-thread training workers are required so timed-out execution cannot commit late.
    if threading.current_thread() is not threading.main_thread():
        raise InfrastructureError('Deadline requires a main-thread Linux worker')
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0]:
        raise InfrastructureError('An external SIGALRM timer conflicts with rollout deadlines')
    def expired(signum, frame):
        raise InfrastructureError(f'{operation} timeout; collection invalid, not a model error')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
