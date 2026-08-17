"""The one exception the forward pass raises.

It lives alone in a module because everything raises it -- the weight
store, the config, the tokenizers, the model, the cache -- and none of
those should have to import each other to do so.
"""

class UnsupportedModel(Exception):
    """The database holds a model this forward pass does not implement."""
