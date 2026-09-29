"""A no-op Streamlit good enough to execute frontend.py headlessly.

frontend.py is a top-to-bottom Streamlit script, so the only way to test it as
it actually runs is to execute it. This stands in for the runtime: widgets
return whatever SCRIPT says, and everything rendered lands in RENDERED.
"""


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class _Ctx:
    """Re-enterable no-op context manager — st.sidebar is entered every run."""

    def __enter__(self): return None
    def __exit__(self, *exc): return False


session_state = SessionState()
SCRIPT = {"chat_input": None, "uploaded": [], "index_clicked": False, "mode": "brief"}
RENDERED = []


def reset():
    session_state.clear()
    RENDERED.clear()
    SCRIPT.update(chat_input=None, uploaded=[], index_clicked=False, mode="brief")


def set_page_config(**k): pass
def title(*a, **k): pass
def caption(*a, **k): pass
def subheader(*a, **k): pass
def markdown(s, *a, **k): RENDERED.append(("markdown", s))
def write(s, *a, **k): RENDERED.append(("write", s))
def success(s, *a, **k): RENDERED.append(("success", s))
def error(s, *a, **k): RENDERED.append(("error", s))
def info(s, *a, **k): RENDERED.append(("info", s))
def warning(s, *a, **k): RENDERED.append(("warning", s))
def file_uploader(*a, **k): return SCRIPT["uploaded"]
def button(label, *a, **k): return SCRIPT["index_clicked"] if "Index" in label else False
def radio(*a, **k): return SCRIPT["mode"]
def chat_input(*a, **k): return SCRIPT["chat_input"]
def rerun(): raise RuntimeError("rerun")


sidebar = _Ctx()
def spinner(*a, **k): return _Ctx()
def chat_message(*a, **k): return _Ctx()
def expander(*a, **k): return _Ctx()
