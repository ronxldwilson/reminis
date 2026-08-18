"""Which tensors move together when precision is chosen per tensor.

A quantization is usually one number for a whole model, and that number is a
compromise: it is chosen for the tensors that tolerate it least, and every
other tensor pays for them. llama.cpp has long since stopped pretending
otherwise -- the `_M` in `Q4_K_M` is exactly this, some tensors held at Q6_K
and the rest at Q4_K -- but its mix is a fixed recipe, the same constants for
every model, hand-tuned once against the models that existed then.

The recipe is not the interesting part. The interesting part is that the
right mix is a property of a particular model, and here it can be measured:
the weights are rows, the packed form is derived on the way to the backend,
and a rung costs nothing but the time to run it. So rather than inherit
someone's constants, `reminis sweep --mix` packs one group of tensors down at
a time, watches what each costs in agreement, and derives a mix for the model
in front of it.

This module holds the grouping that makes that affordable. Measuring each of
a few hundred tensors on its own would take a few hundred forward passes and
tell us mostly what we already know: tensors in the same structural position
behave alike, and the differences that matter are between positions. So the
unit of choice is the role -- the embedding table, the attention projections,
the feed-forward matrices, the output head -- and a mix is a width per role.

A role is only given to weights a mix can actually act on. Norms and biases
are a fraction of a percent of a model and are never packed, so giving them a
role would offer a choice that does not exist.
"""

# The per-layer matrices, which are nearly all of a dense model's bytes and
# are only ever used as the right-hand side of a matrix multiply.
#
# The recurrent names are here for a concrete reason: on a model that is
# three-quarters recurrent they are most of the weights, and leaving them off
# left 11 GB of a 13.8 GB model expanded to float16 -- the whole difference
# between fitting on a 16 GB machine and not.
DENSE_MATRIX = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
)

# A mixture of experts stacks its feed-forward matrices into 3-D tensors that
# are most of such a model's bytes. They are packable like any other matrix,
# but they are kept separate because the packed *index* stores them in its own
# table, one expert at a time, rather than alongside the dense weights.
EXPERT_MATRIX = (
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
)

PACKABLE_MATRIX = DENSE_MATRIX + EXPERT_MATRIX

# The embedding table is indexed rather than multiplied, but on a model with
# tied weights it is also the output projection -- and there it is the single
# largest read of every token. Packing it pays for the whole output side, and
# the handful of rows a lookup needs are decoded individually.
PACKABLE_EMBED = ("token_embd.weight", "output.weight")


def is_packable(name: str) -> bool:
    """Whether this tensor is one a chosen width applies to at all."""
    if name in PACKABLE_EMBED:
        return True
    return name.startswith("blk.") and name.endswith(PACKABLE_MATRIX)


# Printed and probed in this order, which is the order a forward pass reaches
# them. `embed` and `output` are listed apart because a model with tied
# weights has only the first, and on a model without them they are the two
# vocabulary-sized matrices and behave nothing like each other.
ROLES = ("embed", "attn", "ffn", "ssm", "output")

_BLOCK_ROLES = ("attn", "ffn", "ssm")


def role_of(name: str) -> str | None:
    """The group whose width governs this tensor, or None if it has none.

    None is not an error -- it is the answer for every norm, bias and
    scalar in the model, none of which a mix can do anything about.
    """
    if not is_packable(name):
        return None
    if name == "token_embd.weight":
        return "embed"
    if name == "output.weight":
        return "output"
    leaf = name.split(".", 2)[-1]
    for role in _BLOCK_ROLES:
        if leaf.startswith(role + "_"):
            return role
    return None


def roles_in(names) -> list:
    """The roles a particular model actually has, in `ROLES` order.

    Derived from the model rather than assumed, because which ones exist is
    the first thing that differs: a tied-embedding model has no `output`, a
    transformer has no `ssm`, and probing a role with no tensors in it would
    burn a forward pass to measure nothing.
    """
    present = {role_of(name) for name in names}
    return [role for role in ROLES if role in present]


def describe(mix: dict) -> str:
    """A mix as one line, in role order: "embed=6 attn=6 ffn=4"."""
    parts = []
    for role in ROLES:
        if role in mix:
            bits = mix[role]
            parts.append(f"{role}={'f16' if bits is None else bits}")
    return " ".join(parts) or "(empty)"
