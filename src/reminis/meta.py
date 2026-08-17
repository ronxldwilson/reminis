"""Reading values back out of ``model_meta``.

The converter stores GGUF's arrays as text -- a Python repr for the ones
it wrote itself, JSON for the ones that arrived as JSON from a
safetensors tokenizer. Everything that reads metadata goes through here
rather than each caller deciding how to parse a string that might be
either.
"""

import ast
import json

from reminis.errors import UnsupportedModel

def _parse_array(meta: dict, key: str) -> list:
    """Read one of the list-shaped metadata values back into a Python list.

    The converter stores GGUF arrays as their Python repr, so this is the
    inverse. `literal_eval` rather than `eval`: these strings come out of a
    file someone downloaded.
    """
    raw = meta.get(key)
    if not raw:
        return []
    # JSON first: a GGUF conversion writes these as a Python repr, but a
    # safetensors one writes real JSON, and `json.loads` is 16-32x faster on
    # a vocabulary-sized list -- 4.6 ms against 82 ms for 51,865 tokens.
    # A repr uses single quotes, so it fails here immediately and falls
    # through rather than being parsed wrongly.
    if raw[:1] in ("[", "{"):
        try:
            return json.loads(raw)
        except ValueError:
            pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise UnsupportedModel(f"Could not read {key} from this database: {exc}")


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
