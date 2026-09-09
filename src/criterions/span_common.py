"""Helpers shared by the three HieRD span-proposal criteria.

`span_propose`, `span_propose_attn` and `span_propose_attn_only_phrase` used to
carry byte-identical copies of everything in this module. That is how the
gradient bug below survived in two of them while the third was correct: a fix
applied to one copy left the others alone. One copy now, imported by all three.

Two things here are worth reading before touching them.

**Sequence layout.** A criterion that slices hidden states by position has to
know how the batch was padded, and the two students disagree:

    FastVLM (`llava_qwen2`)   right padding, and `input_ids` carry one -200
                              placeholder that the model expands, so the
                              sequence is [vision][text][pad]
    LLaVA-OneVision           left padding, image tokens expanded by the
                              processor, so the sequence is [pad][vision][text]
    the teacher (`qwen2_vl`)  left padding, same as LLaVA-OneVision

So the extraction helpers take `left_padded` rather than `is_teacher`, which is
what they used to key on -- correct for the teacher, correct for FastVLM by
accident, and wrong for a LLaVA-OneVision student, whose text slice then landed
on padding.

**What may be detached.** The per-token attention weights are deliberately
constants: they say how much each token counts, not what it should become.
The cluster *means* are not. Detaching those (as the weighted criteria did)
leaves every span-level, cluster-level and cross-modal term a constant, so the
loss still printed a plausible number while contributing no gradient at all.
"""

import torch
import torch.nn.functional as F


# Students whose collator pads on the right. Everything else in this repo pads
# on the left, which is also what the teacher does.
RIGHT_PADDED_BACKBONES = frozenset({"llava_qwen2", "phi3_v"})


def is_left_padded(model_backbone):
    """Whether this backbone's batches put the padding *before* the content."""
    return model_backbone not in RIGHT_PADDED_BACKBONES


def prepare_span_indices_single(offset_mapping, spans_offsets):
    """Map character spans onto token indices for one sample.

    Args:
        offset_mapping: (TextSeqLen, 2) character offsets per text token.
        spans_offsets: list of (start, end) character offsets.

    Returns:
        dict with token_indices / span_ids / num_spans / token_to_span_map,
        or None when there is nothing to align.
    """
    device = offset_mapping.device

    num_spans = len(spans_offsets)
    if num_spans == 0:
        return None

    span_starts = torch.tensor([s[0] for s in spans_offsets], dtype=torch.long, device=device)
    span_ends = torch.tensor([s[1] for s in spans_offsets], dtype=torch.long, device=device)

    offsets_start = offset_mapping[:, 0].unsqueeze(1)  # (TextSeqLen, 1)
    offsets_end = offset_mapping[:, 1].unsqueeze(1)

    span_starts_exp = span_starts.unsqueeze(0)  # (1, num_spans)
    span_ends_exp = span_ends.unsqueeze(0)

    # A token belongs to a span when its character range sits inside it. The
    # `+ 1` on the start is what admits a token whose offset begins one
    # character early: Qwen's BPE folds the preceding space into the token, so
    # " cat" starts at the space rather than at the "c" the span begins with.
    # (TextSeqLen, num_spans)
    token_in_span_map = (offsets_start + 1 >= span_starts_exp) & (offsets_end <= span_ends_exp)

    if not token_in_span_map.any():
        return None

    nonzero_indices = token_in_span_map.nonzero(as_tuple=False)  # (N, 2)

    return {
        'token_indices': nonzero_indices[:, 0],
        'span_ids': nonzero_indices[:, 1],
        'num_spans': num_spans,
        'token_to_span_map': token_in_span_map,
    }


def extract_text_hidden_states(hidden_states, sample_idx, num_text_tokens, num_vision_tokens,
                               left_padded=True, has_image=True):
    """The text rows of one sample's hidden states, one entry per layer.

    `left_padded` describes the sequence this side was built with -- see the
    module docstring. Left-padded sequences end with the text, so the text is
    the tail; right-padded ones start with the vision tokens, so the text is the
    window that follows them.
    """
    text_hidden_list = []

    for layer_hidden in hidden_states:
        if left_padded:
            # [pad] [vision] [text]  ->  text is the tail
            text_hidden = layer_hidden[sample_idx, -num_text_tokens:, :]
        elif has_image:
            # [vision] [text] [pad]
            text_hidden = layer_hidden[sample_idx, num_vision_tokens:(num_vision_tokens + num_text_tokens), :]
        else:
            # [text] [pad]
            text_hidden = layer_hidden[sample_idx, :num_text_tokens, :]

        text_hidden_list.append(text_hidden)

    return text_hidden_list


def extract_vision_hidden_states(hidden_states, sample_idx, num_vision_tokens, num_text_tokens,
                                 left_padded=True):
    """The vision rows of one sample's hidden states, one entry per layer."""
    vision_hidden_list = []

    for layer_hidden in hidden_states:
        if left_padded:
            # [pad] [vision] [text]: the vision block ends where the text starts
            start_idx = -(num_vision_tokens + num_text_tokens)
            end_idx = -num_text_tokens if num_text_tokens > 0 else None
            vision_hidden = layer_hidden[sample_idx, start_idx:end_idx, :]
        else:
            # [vision] [text] [pad]
            vision_hidden = layer_hidden[sample_idx, :num_vision_tokens, :]

        vision_hidden_list.append(vision_hidden)

    return vision_hidden_list


def extract_attention_for_sample(attention_states, sample_idx, num_vision_tokens, num_text_tokens,
                                 left_padded=True):
    """Text-to-vision attention for one sample, one (num_text, num_vision) entry per layer."""
    attention_list = []
    for layer_attn in attention_states:
        if layer_attn is None:
            attention_list.append(None)
            continue

        if len(layer_attn.shape) == 4:
            attn = layer_attn[sample_idx].mean(dim=0)  # (B, H, L, L) -> (L, L)
        else:
            attn = layer_attn[sample_idx]  # (L, L)

        if num_text_tokens <= 0:
            attention_list.append(None)
            continue

        if left_padded:
            # [pad] [vision] [text]
            vision_start = -(num_vision_tokens + num_text_tokens)
            vision_end = -num_text_tokens
            text_to_vision_attn = attn[-num_text_tokens:, vision_start:vision_end]
        else:
            # [vision] [text] [pad]
            text_start = num_vision_tokens
            text_end = num_vision_tokens + num_text_tokens
            text_to_vision_attn = attn[text_start:text_end, :num_vision_tokens]

        attention_list.append(text_to_vision_attn)

    return attention_list


def compute_intra_cluster_attention_weights(hidden_states, cluster_info):
    """How much each token counts inside its own cluster.

    Returns ``(weights, cluster_mass)``:

        weights      (N,) normalised so each cluster's tokens sum to 1. Used to
                     pool a cluster and to weight per-token losses.
        cluster_mass (num_clusters,) the same weights *before* normalisation,
                     summed per cluster -- how much intra-cluster attention the
                     cluster carries in total, which is what tells a large,
                     strongly-attended cluster from a two-token one.

    Both are detached: a weight says how much a token counts, not what it should
    become, so gradient through it would only let the student make its own
    weights convenient.

    `cluster_mass` exists because the normalised weights sum to exactly 1 per
    cluster by construction. Building pair weights out of *those* gave a matrix
    of ones, which silently turned every "attention-weighted" mean over cluster
    pairs into a plain mean.
    """
    if cluster_info is None:
        return None, None

    device = hidden_states.device
    token_indices = cluster_info['token_indices']
    cluster_ids = cluster_info.get('cluster_ids', cluster_info.get('span_ids'))
    num_clusters = cluster_info.get('num_clusters', cluster_info.get('num_spans'))

    H = hidden_states[token_indices]  # (N, D)
    N, D = H.size(0), H.size(1)

    if N == 0:
        return None, None

    H_detached = H.detach()
    std = H_detached.std(dim=-1, keepdim=True) + 1e-6
    Q = H_detached / std
    K = H_detached / std

    scores = torch.matmul(Q, K.T) / (D ** 0.5)  # (N, N)

    # Keep only scores between tokens of the same cluster, and never self.
    same_cluster_mask = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)
    diag_mask = torch.eye(N, device=device, dtype=torch.bool)
    valid_mask = same_cluster_mask & (~diag_mask)

    is_singleton = valid_mask.sum(dim=-1) == 0  # a row of all -inf would be NaN

    scores_masked = scores.masked_fill(~valid_mask, float('-inf'))
    attn_weights = F.softmax(scores_masked, dim=-1)
    attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # A token's weight is the attention it receives from its cluster-mates; a
    # token alone in its cluster receives none, so give it 1.
    token_weights = attn_weights.sum(dim=0)  # (N,)
    token_weights = torch.where(is_singleton, torch.ones_like(token_weights), token_weights)

    cluster_mass = torch.zeros(num_clusters, device=device, dtype=token_weights.dtype)
    cluster_mass.scatter_add_(0, cluster_ids, token_weights)

    normalized_weights = token_weights / cluster_mass.clamp(min=1e-8)[cluster_ids]

    return normalized_weights.detach(), cluster_mass.detach()


def compute_weighted_cluster_mean(hidden_states, cluster_info, token_weights):
    """Pool each cluster into one vector, weighting its tokens.

    The weights are constants (see above) but `hidden_states` is **not**
    detached: this mean is the student's representation of the span or region,
    and every span-level, cluster-level and cross-modal term is a function of
    it. Detaching here makes all of them gradient-free.
    """
    if cluster_info is None or token_weights is None:
        return None

    device = hidden_states.device
    token_indices = cluster_info['token_indices']
    cluster_ids = cluster_info.get('cluster_ids', cluster_info.get('span_ids'))
    num_clusters = cluster_info.get('num_clusters', cluster_info.get('num_spans'))
    D = hidden_states.size(-1)

    H = hidden_states[token_indices]  # (N, D), keeps the graph
    weights = token_weights.detach()

    H_weighted = H * weights.unsqueeze(-1)

    cluster_ids_expanded = cluster_ids.unsqueeze(-1).expand(-1, D)
    cluster_sum = torch.zeros(num_clusters, D, device=device, dtype=H.dtype)
    cluster_sum = cluster_sum.scatter_add(0, cluster_ids_expanded, H_weighted)

    weight_sum = torch.zeros(num_clusters, device=device, dtype=H.dtype)
    weight_sum = weight_sum.scatter_add(0, cluster_ids, weights.to(H.dtype))
    weight_sum = weight_sum.clamp(min=1e-6).unsqueeze(-1)

    return cluster_sum / weight_sum  # (num_clusters, D)


def projector_touch(projectors, reference):
    """A zero that every projector parameter takes part in.

    DDP aborts with "expected to have finished reduction in the prior
    iteration" when a rank produces no gradient for a parameter that another
    rank does. These criteria hit that whenever a rank's batch yields no spans
    or no vision clusters at all: the projectors are then never called, on that
    rank only, and the step deadlocks or dies rather than reporting anything
    useful. Adding this term costs one sum per projector and keeps every rank's
    parameter set identical; the gradient it contributes is exactly zero.
    """
    params = [p for p in projectors.parameters() if p.requires_grad]
    if not params:
        return reference.new_zeros(())
    touch = params[0].float().sum()
    for p in params[1:]:
        touch = touch + p.float().sum()
    return touch * 0.0
