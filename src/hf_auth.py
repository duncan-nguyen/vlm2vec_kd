"""Where the Hugging Face token comes from.

The token has to reach four different places -- the dataset download, the
student/teacher weight download, the checkpoint upload, and the evaluation
subprocess -- so it is resolved once here and exported into the environment,
rather than threaded through every call.

**Never hard-code a token in this file, or in `src/arguments.py`, or in a
launcher.** The repo has done that once already: `tools/misc/push_to_hub.py`
carried a live token, and it is still in the published git history. A token in
`TOKEN_FILE` gives exactly the same "no flags, no environment variable" ergonomics
without ever entering a commit -- `.gitignore` excludes it.
"""

import os

# Repo-root file holding nothing but the token. Created by hand or by
# `scripts/setup_hf_token.sh`; git ignores it.
TOKEN_FILE = ".hf_token"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# In precedence order. The environment wins over the file so a one-off run can
# override the default without editing anything.
_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _read(path):
    try:
        with open(path) as handle:
            token = handle.read().strip()
    except OSError:
        return None
    return token or None


def token_file_paths():
    """Where a token file is looked for: the repo first, then the home directory."""
    return (
        os.path.join(_REPO_ROOT, TOKEN_FILE),
        os.path.expanduser(f"~/{TOKEN_FILE}"),
    )


def resolve_token(explicit=None):
    """The token to use, or None to fall back to the `huggingface-cli login` cache.

    None is a valid answer, not a failure: `huggingface_hub` reads its own cached
    login when it is handed no token.
    """
    if explicit:
        return explicit
    for var in _ENV_VARS:
        if os.environ.get(var):
            return os.environ[var]
    for path in token_file_paths():
        token = _read(path)
        if token:
            return token
    return None


def install_token_env():
    """Put the resolved token in `$HF_TOKEN` for every library and child process.

    Called once at the start of training. Without it the token file would only
    reach the code that asks this module for it, and the dataset download, the
    model download and the evaluation subprocess would each still need
    `$HF_TOKEN` set by hand.

    Returns True when a token was installed. Never logs the token itself.
    """
    if any(os.environ.get(var) for var in _ENV_VARS):
        return True
    token = resolve_token()
    if not token:
        return False
    for var in _ENV_VARS:
        os.environ[var] = token
    return True
