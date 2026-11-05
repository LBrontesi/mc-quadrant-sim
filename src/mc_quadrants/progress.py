"""Optional per-request progress and cooperative cancellation.

Library calls without a job context retain their existing behavior. Contexts
are thread-local and never enter numerical configuration or random seeds.
"""

from contextlib import contextmanager
from contextvars import ContextVar


class JobCancelled(Exception):
    """The caller requested cancellation at a safe calculation boundary."""


_context = ContextVar("simulation_progress", default=None)


@contextmanager
def progress_context(callback, cancelled):
    token = _context.set((callback, cancelled))
    try:
        yield
    finally:
        _context.reset(token)


def check_cancelled():
    context = _context.get()
    if context is not None and context[1]():
        raise JobCancelled("Analysis cancelled.")


@contextmanager
def progress_scope(label):
    context = _context.get()
    if context is None:
        yield
    else:
        def callback(value):
            context[0]({**value, "stage": f"{label} · {value['stage']}"})
        with progress_context(callback, context[1]):
            yield


def report_progress(stage, completed=None, total=None):
    check_cancelled()
    context = _context.get()
    if context is not None:
        context[0]({"stage": stage, "completed": completed, "total": total})
