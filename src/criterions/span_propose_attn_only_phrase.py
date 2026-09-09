import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from transformers import AutoTokenizer
from src.utils import print_rank 

import spacy
from spacy.matcher import Matcher
from src import profiling
from .text_spans import build_span_cache, filter_overlapping_spans, get_spans_offsets
from .vision_clustering import (
    cluster_vision_tokens,
    compute_vision_distance_matrix,
    get_patch_coordinates,
    map_teacher_clusters_to_student,
    prepare_vision_cluster_info,
)
from .span_common import (
    compute_intra_cluster_attention_weights,
    compute_weighted_cluster_mean,
    extract_attention_for_sample,
    extract_text_hidden_states,
    extract_vision_hidden_states,
    is_left_padded,
    prepare_span_indices_single,
    projector_touch,
)

def cluster_vision_tokens_hdbscan(hidden_states, num_patches_per_row, patch_size, image_width, image_height,
                                  min_cluster_size=3, min_samples_dbscan=8):
    """Thin alias kept so the call sites below stay unchanged (DBSCAN backend)."""
    return cluster_vision_tokens(
        hidden_states, num_patches_per_row, patch_size, image_width, image_height,
        min_cluster_size=min_cluster_size, min_samples_dbscan=min_samples_dbscan,
        backend="dbscan",
    )


# ====== Text Processing Functions ======


# ====== Vision Clustering Functions ======






# ========= Attention-Weighted Functions =========


    

def compute_cluster_distill_loss_weighted(projector, s_hidden, t_hidden, cluster_info):
    """
    Tính distillation loss cho text hidden states trên spans - single sample.
    Sử dụng trung bình cộng đơn giản.
    
    Args:
        projector: Linear layer để project student hidden sang teacher dim
        s_hidden: (SeqLen, D_s) - student hidden states
        t_hidden: (SeqLen, D_t) - teacher hidden states
        cluster_info: dict với token_indices, cluster_ids/span_ids, num_clusters/num_spans
    
    Returns:
        Scalar loss
    """
    if cluster_info is None:
        return torch.tensor(0.0, device=s_hidden.device)
    
    device = s_hidden.device
    token_indices = cluster_info['token_indices']
    cluster_ids = cluster_info.get('cluster_ids', cluster_info.get('span_ids'))
    num_clusters = cluster_info.get('num_clusters', cluster_info.get('num_spans'))
    
    # Calculate attention weights following teacher hidden state
    t_token_weights, t_cluster_mass = compute_intra_cluster_attention_weights(t_hidden, cluster_info)
    s_token_weights, _ = compute_intra_cluster_attention_weights(s_hidden, cluster_info)
    
    if t_token_weights is None or s_token_weights is None:
        return torch.tensor(0.0, device=device)
    
    # Lấy tokens thuộc spans
    S_Tokens = s_hidden[token_indices]  # (N, D_s)
    T_Tokens = t_hidden[token_indices]  # (N, D_t)
    
    # === 1. Token-level cosine loss ===
    S_Tokens_proj = projector(S_Tokens)
    token_cos = F.cosine_similarity(S_Tokens_proj, T_Tokens, dim=-1, eps=1e-5)
    token_loss = (1 - token_cos)
    
    # Weighted by teacher token weights
    t_weights_detached = t_token_weights.detach()
    token_loss = (token_loss * t_weights_detached).sum() / t_weights_detached.sum().clamp(min=1e-6)
    
    # === 2. Span-level similarity distillation ===
    # Calculate weighted cluster means
    T_Cluster_Mean = compute_weighted_cluster_mean(t_hidden, cluster_info, t_token_weights)
    S_Cluster_Mean = compute_weighted_cluster_mean(s_hidden, cluster_info, s_token_weights)
    
    if T_Cluster_Mean is None or S_Cluster_Mean is None:
        return token_loss / 10.0
    
    # Calculate similarity matrices
    S_norm = F.normalize(S_Cluster_Mean, p=2, dim=-1)
    T_norm = F.normalize(T_Cluster_Mean, p=2, dim=-1)
    S_sim = S_norm @ S_norm.T
    T_sim = T_norm @ T_norm.T
    
    # Mask: do not compare self-similarity
    Not_Self = ~torch.eye(num_clusters, dtype=torch.bool, device=device)
    
    if Not_Self.any() and num_clusters > 1:
        
        # Teacher-side mass per cluster: how much intra-cluster attention it
        # carries in total, so a large, strongly attended span weighs more than a
        # two-token one. Built from the *normalised* weights this was a matrix of
        # ones, which made the weighting below a no-op.
        pair_weights = t_cluster_mass.unsqueeze(1) * t_cluster_mass.unsqueeze(0)
        
        S_Sim_masked = torch.masked_select(S_sim, Not_Self)
        T_Sim_masked = torch.masked_select(T_sim, Not_Self)
        Pair_Weights_Masked = torch.masked_select(pair_weights, Not_Self)
        
        cluster_loss = F.mse_loss(S_Sim_masked, T_Sim_masked, reduction='none')
        cluster_loss = (cluster_loss * Pair_Weights_Masked).sum() / Pair_Weights_Masked.sum().clamp(min=1e-6)
    else:
        cluster_loss = torch.tensor(0.0, device=device)
    return cluster_loss + token_loss / 10.0

def compute_vision_cluster_loss_weighted_with_mapping(projector, 
                                             s_vision_hidden, t_vision_hidden, 
                                             teacher_cluster_info, student_cluster_mapping):
    """Tính vision cluster loss với mapping từ teacher sang student"""
    if teacher_cluster_info is None or len(student_cluster_mapping) == 0:
        return torch.tensor(0.0, device=s_vision_hidden.device)
    
    device = s_vision_hidden.device
    D_hidden_s = s_vision_hidden.size(-1)
    D_hidden_t = t_vision_hidden.size(-1)
    
    t_token_indices = teacher_cluster_info['token_indices']
    t_cluster_ids = teacher_cluster_info['cluster_ids']
    num_clusters = teacher_cluster_info['num_clusters']
    
    # =====Teacher side: Calculate attention weights and weighted means =====
    t_token_weights, t_cluster_mass = compute_intra_cluster_attention_weights(t_vision_hidden, teacher_cluster_info)
    if t_token_weights is None:
        return torch.tensor(0.0, device=device)
    
     # Lấy tokens thuộc clusters
    
    T_Tokens = t_vision_hidden[t_token_indices]  # (N_t, D_t)
    T_Cluster_Mean = compute_weighted_cluster_mean(t_vision_hidden, teacher_cluster_info, t_token_weights)
    
    # Student side: Prepare tokens and clusters based on mapping
    s_token_indices_list = []
    s_cluster_ids_list = []
    
    for cluster_id, student_indices in student_cluster_mapping.items():
        for s_idx in student_indices:
            if s_idx < s_vision_hidden.size(0):
                s_token_indices_list.append(s_idx)
                s_cluster_ids_list.append(cluster_id)
    
    if len(s_token_indices_list) == 0:
        return torch.tensor(0.0, device=device)
    
    s_token_indices = torch.tensor(s_token_indices_list, dtype=torch.long, device=device)
    s_cluster_ids = torch.tensor(s_cluster_ids_list, dtype=torch.long, device=device)
    
    student_cluster_info = {
        'token_indices': s_token_indices,
        'cluster_ids': s_cluster_ids,
        'num_clusters': num_clusters
    }
    
    s_token_weights, _ = compute_intra_cluster_attention_weights(s_vision_hidden, student_cluster_info)
    if s_token_weights is None:
        return torch.tensor(0.0, device=device)
    
    S_Tokens = s_vision_hidden[s_token_indices]  # (N_s, D_s)
    S_Cluster_Mean = compute_weighted_cluster_mean(s_vision_hidden, student_cluster_info, s_token_weights)
    
    if T_Cluster_Mean is None or S_Cluster_Mean is None:
        return torch.tensor(0.0, device=device)
    
    # === Token-level loss ===
    # Project student tokens và so sánh với teacher cluster mean tương ứng
    S_Tokens_proj = projector(S_Tokens)
    T_Tokens_for_loss = T_Cluster_Mean[s_cluster_ids]  # Lấy teacher mean theo cluster của student token
    token_cos = F.cosine_similarity(S_Tokens_proj, T_Tokens_for_loss, dim=-1, eps=1e-5)
    token_loss = (1 - token_cos)
    
    s_weights_detached = s_token_weights.detach()
    token_loss = (token_loss * s_weights_detached).sum() / s_weights_detached.sum().clamp(min=1e-6)
    
    # === Cluster-level similarity loss ===
    S_Norm = F.normalize(S_Cluster_Mean, p=2, dim=-1)
    T_Norm = F.normalize(T_Cluster_Mean, p=2, dim=-1)
    S_Sim = S_Norm @ S_Norm.T
    T_Sim = T_Norm @ T_Norm.T
    
    Not_Self = ~torch.eye(num_clusters, dtype=torch.bool, device=device)
    
    if Not_Self.any() and num_clusters > 1:
        # Teacher-side mass per cluster; see compute_cluster_distill_loss_weighted.
        pair_weights = t_cluster_mass.unsqueeze(1) * t_cluster_mass.unsqueeze(0)
        S_Sim_masked = torch.masked_select(S_Sim, Not_Self)
        T_Sim_masked = torch.masked_select(T_Sim, Not_Self)
        Pair_Weights_Masked = torch.masked_select(pair_weights, Not_Self)
        
        cluster_loss = F.mse_loss(S_Sim_masked, T_Sim_masked.detach(), reduction='none')
        cluster_loss = (cluster_loss * Pair_Weights_Masked).sum() / Pair_Weights_Masked.sum().clamp(min=1e-6)
    else:
        cluster_loss = torch.tensor(0.0, device=device)
    
    return cluster_loss + token_loss / 10.0

# ==== Cross-Modal Loss Functions =====
def compute_cross_modal_attention_weights(text_to_vision_attn, text_span_info, vision_cluster_info):
    """Tính attention weights từ text span đến vision clusters"""
    if text_to_vision_attn is None or text_span_info is None or vision_cluster_info is None:
        return None
    
    device = text_to_vision_attn.device
    num_text_spans = text_span_info['num_spans']
    num_vision_clusters = vision_cluster_info['num_clusters']
    
    # Lấy token to span map: (num text token, num text spans)
    token_to_span_map = text_span_info['token_to_span_map'].float()
    
    # Lấy original cluster labels cho tất cả vision tokens
    vision_cluster_labels = vision_cluster_info['original_labels']
    cluster_mapping = vision_cluster_info['cluster_mapping']
    
    num_vision_tokens = text_to_vision_attn.size(1)
    
    vision_to_cluster_map = torch.zeros((num_vision_tokens, num_vision_clusters), device=device, dtype=torch.bfloat16)
    for v_idx in range(num_vision_tokens):
        orig_cluster = vision_cluster_labels[v_idx]
        if orig_cluster >= 0 and orig_cluster in cluster_mapping:
            new_cluster = cluster_mapping[orig_cluster]
            vision_to_cluster_map[v_idx, new_cluster] = 1.0

    # Tính attention từ text spans đến vision clusters
    # text_to_vision_attn: (num_text_tokens, num_vision_tokens)
    # token_to_span_map: (num_text_tokens, num_text_spans)
    # vision_to_cluster_map: (num_vision_tokens, num_vision_clusters)
    
    # Step 1: Aggregate attention từ text tokens đến vision clusters
    # attn_to_clusters: (num_text_tokens, num_vision_clusters)
    attn_to_clusters = text_to_vision_attn @ vision_to_cluster_map  # (T, V) @ (V, C) = (T, C)
    
    # Step 2: Aggregate từ text tokens thành text spans
    # span_to_cluster_attn: (num_text_spans, num_vision_clusters)
    # Sum attention của các tokens trong span
    token_to_span_map = token_to_span_map.to(device=device, dtype=torch.bfloat16)
    span_to_cluster_attn = token_to_span_map.T @ attn_to_clusters  # (S, T) @ (T, C) = (S, C)
    
    # Normalize để tổng = 1
    total_attn = span_to_cluster_attn.sum()
    if total_attn > 1e-8:
        span_to_cluster_attn = span_to_cluster_attn / total_attn
    
    return span_to_cluster_attn

def compute_cross_modal_loss_weighted(projector_text, projector_vision,
                             s_text_hidden, t_text_hidden,
                             s_vision_hidden, t_vision_hidden,
                             text_span_info, vision_cluster_info,
                             student_vision_cluster_mapping,
                             teacher_attention_weights):
    """Tính cross-modal loss giữa text spans và vision clusters"""
    if (text_span_info is None or vision_cluster_info is None or teacher_attention_weights is None or len(student_vision_cluster_mapping) == 0):
        return torch.tensor(0.0, device=s_text_hidden.device)
    
    device = s_text_hidden.device
    num_text_spans = text_span_info['num_spans']
    num_vision_clusters = vision_cluster_info['num_clusters']
    
    # === Calculate attention weights for text span ===
    t_text_weights, _ = compute_intra_cluster_attention_weights(t_text_hidden, text_span_info)
    s_text_weights, _ = compute_intra_cluster_attention_weights(s_text_hidden, text_span_info)
    
    if t_text_weights is None or s_text_weights is None:
        return torch.tensor(0.0, device=device)
    
    # === Calculate weighted text span representations ===
    T_Text_Span_Mean = compute_weighted_cluster_mean(t_text_hidden, text_span_info, t_text_weights)
    S_Text_Span_Mean = compute_weighted_cluster_mean(s_text_hidden, text_span_info, s_text_weights)
    
    # === Calculate attention weights for vision clusters ===
    t_vision_weights, _ = compute_intra_cluster_attention_weights(t_vision_hidden, vision_cluster_info)
    
    if t_vision_weights is None:
        return torch.tensor(0.0, device=device)
    
    T_Vision_Cluster_Mean = compute_weighted_cluster_mean(t_vision_hidden, vision_cluster_info, t_vision_weights)
    
    # Student vision cluster means
    s_vision_token_indices_list = []
    s_vision_cluster_ids_list = []
    
    for cluster_id, student_indices in student_vision_cluster_mapping.items():
        for s_idx in student_indices:
            if s_idx < s_vision_hidden.size(0):
                s_vision_token_indices_list.append(s_idx)
                s_vision_cluster_ids_list.append(cluster_id)
                
    if len(s_vision_token_indices_list) == 0:
        return torch.tensor(0.0, device=device)
    
    s_vision_token_indices = torch.tensor(s_vision_token_indices_list, dtype=torch.long, device=device)
    s_vision_cluster_ids = torch.tensor(s_vision_cluster_ids_list, dtype=torch.long, device=device)
    
    student_vision_cluster_info = {
        'token_indices': s_vision_token_indices,
        'cluster_ids': s_vision_cluster_ids,
        'num_clusters': num_vision_clusters
    }
    
    s_vision_weights, _ = compute_intra_cluster_attention_weights(s_vision_hidden, student_vision_cluster_info)
    if s_vision_weights is None:
        return torch.tensor(0.0, device=device)
    
    S_Vision_Cluster_Mean = compute_weighted_cluster_mean(s_vision_hidden, student_vision_cluster_info, s_vision_weights)   
    
    if T_Text_Span_Mean is None or S_Text_Span_Mean is None or T_Vision_Cluster_Mean is None or S_Vision_Cluster_Mean is None:
        return torch.tensor(0.0, device=device)
    
    # Calculate Cross-model similarity
    T_Text_Norm = F.normalize(T_Text_Span_Mean, p=2, dim=-1)
    T_Vision_Norm = F.normalize(T_Vision_Cluster_Mean, p=2, dim=-1)
    T_Cross_Sim = T_Text_Norm @ T_Vision_Norm.T  # (num_text_spans, num_vision_clusters)
    
    S_Text_Norm = F.normalize(S_Text_Span_Mean, p=2, dim=-1)
    S_Vision_Norm = F.normalize(S_Vision_Cluster_Mean, p=2, dim=-1)
    S_Cross_Sim = S_Text_Norm @ S_Vision_Norm.T  # (num_text_spans, num_vision_clusters)
    
    # ==== Weighted MSE Loss ====
    diff_sq = (S_Cross_Sim - T_Cross_Sim.detach()) ** 2  # (num_text_spans, num_vision_clusters)
    attn_weights_detached = teacher_attention_weights.detach()
    weighted_loss = (diff_sq * attn_weights_detached).sum()
    
    return weighted_loss

def compute_cross_modal_loss_for_layer(projectors,
                                        s_text_hidden, t_text_hidden,
                                        s_vision_hidden, t_vision_hidden,
                                        text_span_info, vision_cluster_info,
                                        student_vision_cluster_mapping,
                                        text_to_vision_attn,
                                        layer_idx, args):
    """
    Tính cross-modal loss cho một layer.
    """
    device = s_text_hidden.device
    
    # Tính attention weights từ teacher
    attention_weights = compute_cross_modal_attention_weights(
        text_to_vision_attn,
        text_span_info,
        vision_cluster_info
    )
    
    if attention_weights is None:
        return torch.tensor(0.0, device=device)
    
    # Lấy projectors (có thể dùng chung hoặc riêng cho text/vision)
    projector_text = projectors[layer_idx]
    projector_vision = projectors[layer_idx]  # Hoặc có thể dùng projector riêng
    
    loss = compute_cross_modal_loss_weighted(
        projector_text, projector_vision,
        s_text_hidden, t_text_hidden,
        s_vision_hidden, t_vision_hidden,
        text_span_info, vision_cluster_info,
        student_vision_cluster_mapping,
        attention_weights
    )
    
    return loss

@profiling.timed("text_span_loss")
def compute_text_span_loss_weighted(projectors, 
                           student_text_hidden_list, 
                           teacher_text_hidden_list,
                           offset_mapping,
                           spans_offsets,
                           words_offsets,
                           args):
    """
    Tính span loss cho một sample.
    
    Args:
        projectors: List of projection layers
        student_text_hidden_list: List of (TextSeqLen, D_s) cho mỗi layer
        teacher_text_hidden_list: List of (TextSeqLen, D_t) cho mỗi layer
        offset_mapping: (TextSeqLen, 2)
        spans_offsets: List[Tuple[int, int]]
        words_offsets: List[Tuple[int, int]]
        student_layer_mapping: List of student layer indices
        teacher_layer_mapping: List of teacher layer indices
        args: training arguments
    
    Returns:
        Scalar loss
    """
    device = student_text_hidden_list[0].device
    
    # Chuẩn bị span indices
    span_info_words = prepare_span_indices_single(offset_mapping, words_offsets)
    span_info_spans = prepare_span_indices_single(offset_mapping, spans_offsets)
    
    total_loss = 0.0
    num_valid_layers = 0
    
    # Word-level loss
    s_word_mapping = args.student_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    t_word_mapping = args.teacher_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    word_projectors = projectors[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    
    for s_idx, t_idx, projector in zip(s_word_mapping, t_word_mapping, word_projectors):
        loss = compute_cluster_distill_loss_weighted(
            projector,
            student_text_hidden_list[s_idx],
            teacher_text_hidden_list[t_idx],
            span_info_spans
        )
        total_loss += loss
        num_valid_layers += 1
    
    # Span-level loss
    s_span_mapping = args.student_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    t_span_mapping = args.teacher_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    span_projectors = projectors[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    
    for s_idx, t_idx, projector in zip(s_span_mapping, t_span_mapping, span_projectors):
        loss = compute_cluster_distill_loss_weighted(
            projector,
            student_text_hidden_list[s_idx],
            teacher_text_hidden_list[t_idx],
            span_info_spans
        )
        total_loss += loss
        num_valid_layers += 1
    
    if num_valid_layers > 0:
        total_loss = total_loss / num_valid_layers
    
    return total_loss, span_info_words, span_info_spans

@profiling.timed("vision_cluster_loss")
def compute_vision_cluster_loss_weighted(projectors,
                                student_vision_hidden_list, 
                                teacher_vision_hidden_list,
                                original_width, original_height,
                                args,
                                teacher_patch_size=28,
                                student_patch_size=64,
                                student_resize=1024):
    """Tính vision cluster loss cho một sample."""
    device = student_vision_hidden_list[0].device
    num_teacher_vision_tokens = teacher_vision_hidden_list[0].size(0)
    num_student_vision_tokens = student_vision_hidden_list[0].size(0)
    
    if num_teacher_vision_tokens == 0 or num_student_vision_tokens == 0:
        return torch.tensor(0.0, device=device), None, None, None, None
    
    teacher_patches_per_row = int(np.sqrt(num_teacher_vision_tokens))
    student_patches_per_row = int(np.sqrt(num_student_vision_tokens))
    min_samples_dbscan_teacher = max(1, int(getattr(args, 'min_samples_dbscan_teacher', 8)))
    
    total_loss = 0.0
    num_valid_layers = 0
    
    word_cluster_info = None
    word_student_mapping = None
    span_cluster_info = None
    span_student_mapping = None
    
    # ============ Word-level loss ============
    s_word_mapping = args.student_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    t_word_mapping = args.teacher_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    word_projectors = projectors[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    
    if len(s_word_mapping) > 0:
        # Phân cụm tại layer đầu tiên
        first_t_idx = t_word_mapping[0]
        word_cluster_labels = cluster_vision_tokens_hdbscan(
            teacher_vision_hidden_list[first_t_idx],
            teacher_patches_per_row, teacher_patch_size,
            original_width, original_height,
            min_cluster_size=6,
            min_samples_dbscan=min_samples_dbscan_teacher
        )
        
        # Chuẩn bị cluster info cho teacher
        word_cluster_info = prepare_vision_cluster_info(word_cluster_labels, device)
        
        # Map sang student
        word_student_mapping, _ = map_teacher_clusters_to_student(
            word_cluster_labels,
            teacher_patches_per_row, teacher_patch_size,
            student_patches_per_row, student_patch_size,
            original_width, original_height,
            student_resize
        )
        
        # Tính loss cho từng layer, dùng cùng clustering
        for s_idx, t_idx, projector in zip(s_word_mapping, t_word_mapping, word_projectors):
            loss = compute_vision_cluster_loss_weighted_with_mapping(
                projector,
                student_vision_hidden_list[s_idx],
                teacher_vision_hidden_list[t_idx],
                word_cluster_info,
                word_student_mapping
            )
            total_loss += loss
            num_valid_layers += 1
            
    # ============ Span-level loss ============
    s_span_mapping = args.student_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    t_span_mapping = args.teacher_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    span_projectors = projectors[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    
    if len(s_span_mapping) > 0:
        # Phân cụm tại layer đầu tiên của span-level
        first_t_idx = t_span_mapping[0]
        span_cluster_labels = cluster_vision_tokens_hdbscan(
            teacher_vision_hidden_list[first_t_idx],
            teacher_patches_per_row, teacher_patch_size,
            original_width, original_height,
            min_cluster_size=8,  # Larger min_cluster_size for span-level
            min_samples_dbscan=min_samples_dbscan_teacher
        )
        
        # Chuẩn bị cluster info cho teacher
        span_cluster_info = prepare_vision_cluster_info(span_cluster_labels, device)
        
        # Map sang student
        span_student_mapping, _ = map_teacher_clusters_to_student(
            span_cluster_labels,
            teacher_patches_per_row, teacher_patch_size,
            student_patches_per_row, student_patch_size,
            original_width, original_height,
            student_resize
        )
        
        # Tính loss cho từng layer, dùng cùng clustering
        for s_idx, t_idx, projector in zip(s_span_mapping, t_span_mapping, span_projectors):
            loss = compute_vision_cluster_loss_weighted_with_mapping(
                projector,
                student_vision_hidden_list[s_idx],
                teacher_vision_hidden_list[t_idx],
                span_cluster_info,
                span_student_mapping
            )
            total_loss += loss
            num_valid_layers += 1
    
    if num_valid_layers > 0:
        total_loss = total_loss / num_valid_layers
    
    return total_loss, word_cluster_info, word_student_mapping, span_cluster_info, span_student_mapping

@profiling.timed("cross_modal_loss")
def compute_cross_modal_alignment_loss_weighted(projectors,
                                       student_text_hidden_list,
                                       teacher_text_hidden_list,
                                       student_vision_hidden_list,
                                       teacher_vision_hidden_list,
                                       teacher_attention_list,
                                       text_span_info_words,
                                       text_span_info_spans,
                                       vision_cluster_info_words,
                                       vision_cluster_info_spans,
                                       student_vision_mapping_words,
                                       student_vision_mapping_spans,
                                       args):
    """Tính cross-modal alignment loss cho một sample."""
    device = student_text_hidden_list[0].device
    total_loss = 0.0
    num_valid_layers = 0
    
    # ===== Word-level cross-modal loss =====
    s_word_mapping = args.student_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    t_word_mapping = args.teacher_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    
    if (len(s_word_mapping) > 0 and text_span_info_words is not None and 
        vision_cluster_info_words is not None and student_vision_mapping_words is not None):
        
        for i, (s_idx, t_idx) in enumerate(zip(s_word_mapping, t_word_mapping)):
            # Lấy attention cho layer này
            attn = teacher_attention_list[t_idx] if t_idx < len(teacher_attention_list) else None
            
            attention_weights = compute_cross_modal_attention_weights(
                attn,
                text_span_info_words,
                vision_cluster_info_words
            )
            if attention_weights is not None:
                projector = projectors[args.split_layer_mapping[0] + i]
                loss = compute_cross_modal_loss_weighted(
                    projector, projector, 
                    student_text_hidden_list[s_idx],
                    teacher_text_hidden_list[t_idx],
                    student_vision_hidden_list[s_idx],
                    teacher_vision_hidden_list[t_idx],
                    text_span_info_words,
                    vision_cluster_info_words,
                    student_vision_mapping_words,
                    attention_weights
                )
                total_loss += loss
                num_valid_layers += 1
    
    # Span-level cross-modal loss
    s_span_mapping = args.student_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    t_span_mapping = args.teacher_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    
    if (len(s_span_mapping) > 0 and text_span_info_spans is not None and 
        vision_cluster_info_spans is not None and student_vision_mapping_spans is not None):
        
        for i, (s_idx, t_idx) in enumerate(zip(s_span_mapping, t_span_mapping)):
            attn = teacher_attention_list[t_idx] if t_idx < len(teacher_attention_list) else None
            
            attention_weights = compute_cross_modal_attention_weights(
                attn,
                text_span_info_spans,
                vision_cluster_info_spans
            )
            if attention_weights is not None:
                projector = projectors[args.split_layer_mapping[1] + i]
                loss = compute_cross_modal_loss_weighted(
                    projector, projector,
                    student_text_hidden_list[s_idx],
                    teacher_text_hidden_list[t_idx],
                    student_vision_hidden_list[s_idx],
                    teacher_vision_hidden_list[t_idx],
                    text_span_info_spans,
                    vision_cluster_info_spans,
                    student_vision_mapping_spans,
                    attention_weights
                )
                total_loss += loss
                num_valid_layers += 1
    
    if num_valid_layers > 0:
        total_loss = total_loss / num_valid_layers
    
    return total_loss
                                       
class SpanProposeCriterionWeightedOnlyPhrase(nn.Module):
    def __init__(self, args):
        super(SpanProposeCriterionWeightedOnlyPhrase, self).__init__()
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        
        self.args = args
        
        # Khởi tạo spacy và matcher một lần
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        VERB_PHRASE_PATTERN = [
            {"POS": "AUX", "OP": "*"},
            {"POS": "ADV", "OP": "*"},
            {"POS": "VERB", "OP": "+"},
            {"POS": "ADV", "OP": "*"},
        ]
        self.matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])

        # text -> (spans, words) memo. The query side of an MMEB subset is one
        # instruction repeated for every sample, so this alone removes about
        # half the spaCy work; set VLM2VEC_SPAN_CACHE to keep it across runs.
        self.span_cache = build_span_cache(rank=self.process_rank)
        
        self.teacher_patch_size = getattr(args, 'teacher_patch_size', 28)
        self.student_patch_size = getattr(args, 'student_patch_size', 64)
        self.student_resize = getattr(args, 'student_resize', 1024)
        
        self.w_cross_modal = getattr(args, 'w_cross_modal_loss', 0.3)
    
    def _dist_gather_tensor(self, t):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def forward(self, distiller, input_data, tokenizer):
        # print_rank("Start SpanProposeCriterion forward)
        
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        projectors = distiller.projectors  # Giả sử projectors được lưu trong distiller

        # Where the vision and text blocks sit inside a sequence depends on which
        # side the student's collator padded on -- FastVLM pads right and puts the
        # expanded image first, LLaVA-OneVision pads left like the teacher. Every
        # slice below is taken through span_common with this flag.
        student_left_padded = is_left_padded(
            getattr(getattr(distiller, "model_args", None), "model_backbone", None)
        )
        
        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        
        qry_image_sizes = input_data.get('qry_image_sizes', None)  # List of (W, H) cho mỗi sample
        pos_image_sizes = input_data.get('pos_image_sizes', None)
        
        # Đếm số text tokens (loại bỏ image tokens)
        # Giả sử image token IDs nằm trong khoảng [151643, 151656]
        num_text_qry_tokens = ((teacher_qry_input['input_ids'] < 151643) | (teacher_qry_input['input_ids'] > 151656)).sum(dim=1)
        num_text_pos_tokens = ((teacher_pos_input['input_ids'] < 151643) | (teacher_pos_input['input_ids'] > 151656)).sum(dim=1)
        
        batch_size = student_qry_input['input_ids'].size(0)
        device = student_qry_input['input_ids'].device
        
        # Forward teacher
        with profiling.section("teacher_fwd"), torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
            
        # Forward student
        with profiling.section("student_fwd"):
            student_qry_output = student_model.encode_input(student_qry_input)
            student_pos_output = student_model.encode_input(student_pos_input)
            student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
            student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
    
        # Contrastive loss
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)
        
        # ============ Prepare span offsets ============
        input_qry_texts = tokenizer.batch_decode(teacher_qry_input['input_ids'], skip_special_tokens=True)
        input_pos_texts = tokenizer.batch_decode(teacher_pos_input['input_ids'], skip_special_tokens=True)
        
        # Lấy offset mappings
        offset_mappings_qry = tokenizer(
            input_qry_texts, 
            return_offsets_mapping=True, 
            add_special_tokens=False, 
            return_tensors="pt",
            padding=True
        )["offset_mapping"].to(device)
        
        offset_mappings_pos = tokenizer(
            input_pos_texts, 
            return_offsets_mapping=True, 
            add_special_tokens=False, 
            return_tensors="pt",
            padding=True
        )["offset_mapping"].to(device)
        
        # Lấy spans và words offsets
        with profiling.section("spacy_spans"):
            _, spans_qry_offsets, words_qry_offsets = get_spans_offsets(
                input_qry_texts, self.nlp, self.matcher, cache=self.span_cache)
            _, spans_pos_offsets, words_pos_offsets = get_spans_offsets(
                input_pos_texts, self.nlp, self.matcher, cache=self.span_cache)
        
        # ============ Tính Loss cho từng sample ============
        total_text_loss = 0.0
        total_vision_loss = 0.0
        total_cross_modal_loss = 0.0
        valid_text_samples = 0
        valid_vision_samples = 0
        valid_cross_modal_samples = 0
        
        for i in range(batch_size):
            # === Query ===
            num_text_qry = num_text_qry_tokens[i].item()
            
            # Kiểm tra có image không
            has_image_qry = (student_qry_image_features is not None and 
                            i < len(student_qry_image_features) and 
                            student_qry_image_features[i] is not None)
            
            if has_image_qry:
                num_vision_qry_student = student_qry_image_features[i].size(0)
                num_vision_qry_teacher = teacher_qry_image_features[i].size(0)
                
                # Lấy image size
                if qry_image_sizes is not None and i < len(qry_image_sizes):
                    img_w_qry, img_h_qry = qry_image_sizes[i]
                else:
                    # Default: infer from number of patches
                    patches_per_row = int(np.sqrt(num_vision_qry_teacher))
                    img_w_qry = patches_per_row * self.teacher_patch_size
                    img_h_qry = patches_per_row * self.teacher_patch_size
            else:
                num_vision_qry_student = 0
                num_vision_qry_teacher = 0
            
            # Trích xuất text hidden states cho query
            student_qry_text_hidden_list = extract_text_hidden_states(
                student_qry_hidden_states,
                sample_idx=i,
                num_text_tokens=num_text_qry,
                num_vision_tokens=num_vision_qry_student,
                left_padded=student_left_padded,
                has_image=has_image_qry
            )
            
            teacher_qry_text_hidden_list = extract_text_hidden_states(
                teacher_qry_hidden_states,
                sample_idx=i,
                num_text_tokens=num_text_qry,
                num_vision_tokens=num_vision_qry_teacher,
                left_padded=True,
                has_image=has_image_qry
            )
            
            # Text span loss
            text_span_info_words_qry = None
            text_span_info_spans_qry = None
            
            # Tính span loss cho query
            if num_text_qry > 0 and len(words_qry_offsets[i]) > 0:
                offset_mapping_qry_i = offset_mappings_qry[i, :num_text_qry, :]
                qry_text_loss, text_span_info_words_qry, text_span_info_spans_qry = compute_text_span_loss_weighted(
                    projectors,
                    student_qry_text_hidden_list,
                    teacher_qry_text_hidden_list,
                    offset_mapping_qry_i,
                    spans_qry_offsets[i],
                    spans_qry_offsets[i],
                    self.args
                )
                total_text_loss += qry_text_loss
                valid_text_samples += 1
                
            # Vision cluster loss (weighted)
            vision_cluster_info_words_qry = None
            vision_cluster_info_spans_qry = None
            student_vision_mapping_words_qry = None
            student_vision_mapping_spans_qry = None
                
            # Vision cluster loss cho query
            if has_image_qry:
                student_qry_vision_hidden_list = extract_vision_hidden_states(
                    student_qry_hidden_states, i, num_vision_qry_student, num_text_qry,
                    left_padded=student_left_padded
                )
                teacher_qry_vision_hidden_list = extract_vision_hidden_states(
                    teacher_qry_hidden_states, i, num_vision_qry_teacher, num_text_qry,
                    left_padded=True
                )
                
                (qry_vision_loss, vision_cluster_info_words_qry, student_vision_mapping_words_qry,
                 vision_cluster_info_spans_qry, student_vision_mapping_spans_qry) = compute_vision_cluster_loss_weighted(
                    projectors,
                    student_qry_vision_hidden_list,
                    teacher_qry_vision_hidden_list,
                    img_w_qry, img_h_qry,
                    self.args,
                    self.teacher_patch_size,
                    self.student_patch_size,
                    self.student_resize
                )
                total_vision_loss += qry_vision_loss
                valid_vision_samples += 1
                
                # Cross-modal loss
                if (text_span_info_spans_qry is not None and vision_cluster_info_spans_qry is not None):
                    # Extract attention cho sample này
                    teacher_qry_attention_list = extract_attention_for_sample(
                        teacher_qry_attention, i, num_vision_qry_teacher, num_text_qry,
                        left_padded=True
                    )
                    
                    qry_cross_modal_loss = compute_cross_modal_alignment_loss_weighted(
                        projectors,
                        student_qry_text_hidden_list,
                        teacher_qry_text_hidden_list,
                        student_qry_vision_hidden_list,
                        teacher_qry_vision_hidden_list,
                        teacher_qry_attention_list,
                        text_span_info_spans_qry,
                        text_span_info_spans_qry,
                        vision_cluster_info_spans_qry,
                        vision_cluster_info_spans_qry,
                        student_vision_mapping_spans_qry,
                        student_vision_mapping_spans_qry,
                        self.args
                    )
                    total_cross_modal_loss += qry_cross_modal_loss
                    valid_cross_modal_samples += 1
            
            # === Positive ===
            num_text_pos = num_text_pos_tokens[i].item()
            
            # Kiểm tra có image không
            has_image_pos = (student_pos_image_features is not None and 
                            i < len(student_pos_image_features) and 
                            student_pos_image_features[i] is not None)
            
            if has_image_pos:
                num_vision_pos_student = student_pos_image_features[i].size(0)
                num_vision_pos_teacher = teacher_pos_image_features[i].size(0)
                
                if pos_image_sizes is not None and i < len(pos_image_sizes):
                    img_w_pos, img_h_pos = pos_image_sizes[i]
                else:
                    patches_per_row = int(np.sqrt(num_vision_pos_teacher))
                    img_w_pos = patches_per_row * self.teacher_patch_size
                    img_h_pos = patches_per_row * self.teacher_patch_size
            else:
                num_vision_pos_student = 0
                num_vision_pos_teacher = 0
            
            # Trích xuất text hidden states cho positive
            student_pos_text_hidden_list = extract_text_hidden_states(
                student_pos_hidden_states,
                sample_idx=i,
                num_text_tokens=num_text_pos,
                num_vision_tokens=num_vision_pos_student,
                left_padded=student_left_padded,
                has_image=has_image_pos
            )
            
            teacher_pos_text_hidden_list = extract_text_hidden_states(
                teacher_pos_hidden_states,
                sample_idx=i,
                num_text_tokens=num_text_pos,
                num_vision_tokens=num_vision_pos_teacher,
                left_padded=True,
                has_image=has_image_pos
            )
            
            # Text span loss
            text_span_info_words_pos = None
            text_span_info_spans_pos = None
            
            # Tính span loss cho positive
            if num_text_pos > 0 and len(spans_pos_offsets[i]) > 0:
                offset_mapping_pos_i = offset_mappings_pos[i, :num_text_pos, :]
                pos_text_loss, text_span_info_words_pos, text_span_info_spans_pos = compute_text_span_loss_weighted(
                    projectors,
                    student_pos_text_hidden_list,
                    teacher_pos_text_hidden_list,
                    offset_mapping_pos_i,
                    spans_pos_offsets[i],
                    spans_pos_offsets[i],
                    self.args
                )
                total_text_loss += pos_text_loss
                valid_text_samples += 1
            
            # Vision cluster loss for Positive
            if has_image_pos:
                student_pos_vision_hidden_list = extract_vision_hidden_states(
                    student_pos_hidden_states, i, num_vision_pos_student, num_text_pos,
                    left_padded=student_left_padded
                )
                teacher_pos_vision_hidden_list = extract_vision_hidden_states(
                    teacher_pos_hidden_states, i, num_vision_pos_teacher, num_text_pos,
                    left_padded=True
                )
                
                (pos_vision_loss, vision_cluster_info_words_pos, student_vision_mapping_words_pos,
                 vision_cluster_info_spans_pos, student_vision_mapping_spans_pos) = compute_vision_cluster_loss_weighted(
                    projectors,
                    student_pos_vision_hidden_list,
                    teacher_pos_vision_hidden_list,
                    img_w_pos, img_h_pos,
                    self.args,
                    self.teacher_patch_size,
                    self.student_patch_size,
                    self.student_resize
                )
                total_vision_loss += pos_vision_loss
                valid_vision_samples += 1
                
                # Cross-modal loss
                if (text_span_info_words_pos is not None and vision_cluster_info_words_pos is not None):
                    teacher_pos_attention_list = extract_attention_for_sample(
                        teacher_pos_attention, i, num_vision_pos_teacher, num_text_pos,
                        left_padded=True
                    )
                    
                    pos_cross_modal_loss = compute_cross_modal_alignment_loss_weighted(
                        projectors,
                        student_pos_text_hidden_list,
                        teacher_pos_text_hidden_list,
                        student_pos_vision_hidden_list,
                        teacher_pos_vision_hidden_list,
                        teacher_pos_attention_list,
                        text_span_info_spans_pos,
                        text_span_info_spans_pos,
                        vision_cluster_info_spans_pos,
                        vision_cluster_info_spans_pos,
                        student_vision_mapping_spans_pos,
                        student_vision_mapping_spans_pos,
                        self.args
                    )
                    total_cross_modal_loss += pos_cross_modal_loss
                    valid_cross_modal_samples += 1
        
        text_span_loss = total_text_loss / valid_text_samples if valid_text_samples > 0 else torch.tensor(0.0, device=device)
        vision_cluster_loss = total_vision_loss / valid_vision_samples if valid_vision_samples > 0 else torch.tensor(0.0, device=device)
        cross_modal_loss = total_cross_modal_loss / valid_cross_modal_samples if valid_cross_modal_samples > 0 else torch.tensor(0.0, device=device)
        
        
        span_loss = (text_span_loss + vision_cluster_loss) / 2
        
        distance_loss = self.compute_distance_loss(
            student_qry_reps, student_pos_reps,
            teacher_qry_reps, teacher_pos_reps
        )
        
        angle_loss = self.compute_angle_loss(
            student_qry_reps, student_pos_reps,
            teacher_qry_reps, teacher_pos_reps
        )
        
        rkd_loss = (distance_loss + angle_loss) / 2.0
        
        # ============ Tổng hợp loss ============
        # projector_touch contributes exactly zero, but it keeps every rank's set
        # of used parameters identical when a batch happens to yield no spans or
        # no vision clusters, which DDP would otherwise abort on.
        total_loss = (contrastive_loss + self.args.kd_weight * span_loss
                      + self.w_cross_modal * cross_modal_loss
                      + (self.args.kd_weight / 10.0) * rkd_loss
                      + projector_touch(projectors, contrastive_loss))
        
        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'text_span_loss': text_span_loss,
            'vision_cluster_loss': vision_cluster_loss,
            'cross_modal_loss': cross_modal_loss,
            'span_loss': span_loss,
            'kd_loss_rkd': rkd_loss,
        }
        
    def pairwise_distance(self, x):
        norm = (x**2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2.0 * torch.mm(x, x.t())
        return dist
    
    def compute_distance_loss(self, student_qry, student_pos, teacher_qry, teacher_pos):
        
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)
        
        dist_student = self.pairwise_distance(student_repr)
        dist_teacher = self.pairwise_distance(teacher_repr)
        
        mask = torch.triu(torch.ones_like(dist_student), diagonal=1).bool()
        dist_student = dist_student[mask]
        dist_teacher = dist_teacher[mask]
        
        mean_td = dist_teacher.mean().detach() + 1e-8
        mean_sd = dist_student.mean().detach() + 1e-8
        
        dist_student = dist_student / mean_sd
        dist_teacher = dist_teacher / mean_td
        
        diff = dist_student - dist_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        loss = loss.mean()
        return loss
    
    def angle_potentials(self, x):
        n = x.size(0)
        diffs = x.unsqueeze(0) - x.unsqueeze(1)
        norms = torch.norm(diffs, dim=-1, keepdim=True) + 1e-8
        e = diffs / norms
        
        cos_angles = torch.einsum('ijd,kjd->ijk', e, e)
        return cos_angles
    
    def compute_angle_loss(self, student_qry, student_pos, teacher_qry, teacher_pos):
        
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)
        
        psi_student = self.angle_potentials(student_repr)
        psi_teacher = self.angle_potentials(teacher_repr)
        
        n = psi_student.size(0)
        mask = torch.ones((n, n, n), dtype=torch.bool, device=psi_student.device)
        idx = torch.arange(n, device=psi_student.device)
        mask[idx, idx, :] = 0
        mask[idx, :, idx] = 0
        mask[:, idx, idx] = 0
        
        psi_teacher = psi_teacher[mask]
        psi_student = psi_student[mask]
        
        diff = psi_student - psi_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        loss = loss.mean()
        return loss