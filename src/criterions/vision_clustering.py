"""Vision-token clustering shared by the span_propose family.

Same maths as the copies that used to live in each criterion, with the per-call
overhead removed:

* the patch-coordinate grid was rebuilt with a Python loop over every token, on
  every call -- and this runs 4x per sample per step. It only depends on the
  grid geometry, so the normalised spatial distance matrix is computed once and
  memoised per (grid, image size, device).
* symmetrisation, clamping and the diagonal reset moved to the GPU so there is a
  single device->host transfer instead of one transfer plus three numpy passes.
* ``np.triu_indices_from`` allocated a fresh pair of index arrays on every call;
  those are memoised per matrix size too.

The DBSCAN/HDBSCAN call itself is unchanged, and so are the resulting labels.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from src import profiling

_SPATIAL_CACHE = {}
_TRIU_CACHE = {}
_SPATIAL_CACHE_MAX = 64


def get_patch_coordinates(patch_idx, num_patch_per_row, patch_size):
    """Tinh tọa độ center của patch trên ảnh"""
    row = patch_idx // num_patch_per_row
    col = patch_idx % num_patch_per_row
    center_x = col * patch_size + patch_size / 2
    center_y = row * patch_size + patch_size / 2
    return center_x, center_y


def _spatial_distance_norm(num_tokens, num_patches_per_row, patch_size,
                           image_width, image_height, device):
    """Normalised pairwise patch-centre distances, memoised per geometry."""
    key = (num_tokens, num_patches_per_row, patch_size,
           float(image_width), float(image_height), str(device))
    cached = _SPATIAL_CACHE.get(key)
    if cached is not None:
        return cached

    idx = torch.arange(num_tokens, device=device, dtype=torch.float32)
    row = torch.div(idx, num_patches_per_row, rounding_mode="floor")
    col = idx - row * num_patches_per_row
    coords = torch.stack(
        [col * patch_size + patch_size / 2, row * patch_size + patch_size / 2], dim=-1
    )  # (num_tokens, 2), same layout as the original [x, y]

    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    spatial_distance = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    max_dist = torch.sqrt(
        torch.tensor(image_width ** 2 + image_height ** 2, dtype=torch.float, device=device)
    )
    out = spatial_distance / max_dist

    if len(_SPATIAL_CACHE) < _SPATIAL_CACHE_MAX:
        _SPATIAL_CACHE[key] = out
    return out


def _triu_indices(n):
    iu = _TRIU_CACHE.get(n)
    if iu is None:
        iu = np.triu_indices(n, k=1)
        _TRIU_CACHE[n] = iu
    return iu


@profiling.timed("distance_matrix")
def compute_vision_distance_matrix(hidden_states, num_patches_per_row, patch_size,
                                   image_width, image_height, spatial_weight=0.1):
    """Cosine + spatial distance matrix, returned as a float64 numpy array.

    Symmetrisation / clamping / diagonal reset happen here (on device) rather
    than in the caller, so there is exactly one host transfer.
    """
    num_tokens = hidden_states.size(0)
    hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
    cosine_distance = 1 - hidden_norm @ hidden_norm.T

    spatial_distance_norm = _spatial_distance_norm(
        num_tokens, num_patches_per_row, patch_size,
        image_width, image_height, hidden_states.device,
    )

    total_dist = cosine_distance + spatial_weight * spatial_distance_norm
    total_dist = (total_dist + total_dist.T) * 0.5
    total_dist = total_dist.clamp_min(0)
    total_dist.fill_diagonal_(0)

    return total_dist.to(torch.float32).cpu().numpy().astype(np.float64)


@profiling.timed("vision_cluster")
def cluster_vision_tokens(hidden_states, num_patches_per_row, patch_size,
                          image_width, image_height,
                          min_cluster_size=3, min_samples_dbscan=8,
                          backend="dbscan", eps_percentile=3):
    """Phân cụm vision tokens.

    backend="dbscan"  -> eps taken from the `eps_percentile`-th percentile of the
                         off-diagonal distances (span_propose_attn behaviour).
    backend="hdbscan" -> HDBSCAN on the precomputed matrix (span_propose
                         behaviour).
    """
    num_tokens = hidden_states.size(0)
    if num_tokens < min_cluster_size:
        return np.zeros(num_tokens, dtype=np.int32)

    distance_matrix = compute_vision_distance_matrix(
        hidden_states, num_patches_per_row, patch_size,
        image_width, image_height, spatial_weight=0.1,
    )

    if backend == "hdbscan":
        import hdbscan as _hdbscan
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="precomputed",
            allow_single_cluster=True,
            approx_min_span_tree=True,
        )
    else:
        iu = _triu_indices(distance_matrix.shape[0])
        eps = np.percentile(distance_matrix[iu], eps_percentile)
        clusterer = DBSCAN(
            eps=eps,
            min_samples=max(1, int(min_samples_dbscan)),
            metric="precomputed",
        )

    cluster_labels = clusterer.fit_predict(distance_matrix)
    if np.all(cluster_labels == -1):
        cluster_labels = np.zeros(num_tokens, dtype=np.int32)
    return cluster_labels


def map_teacher_clusters_to_student(cluster_labels,
                                    teacher_num_patches_per_row, teacher_patch_size,
                                    student_num_patches_per_row, student_patch_size,
                                    original_width, original_height,
                                    student_resize=1024):
    """Map cluster labels từ teacher sang student dựa trên vị trí patch"""
    num_teacher_tokens = len(cluster_labels)
    num_student_tokens = (student_resize // student_patch_size) ** 2

    student_cluster_mapping = {}
    student_token_to_cluster = [-1] * num_student_tokens
    for teacher_idx in range(num_teacher_tokens):
        cluster_id = int(cluster_labels[teacher_idx])
        if cluster_id == -1:
            continue
        teacher_x, teacher_y = get_patch_coordinates(
            teacher_idx, teacher_num_patches_per_row, teacher_patch_size
        )

        # Scale về ảnh resize của student
        scale_x = student_resize / original_width
        scale_y = student_resize / original_height
        student_x = teacher_x * scale_x
        student_y = teacher_y * scale_y

        student_col = int(student_x // student_patch_size)
        student_row = int(student_y // student_patch_size)

        # Clamp để đảm bảo trong range
        student_col = min(max(student_col, 0), student_num_patches_per_row - 1)
        student_row = min(max(student_row, 0), student_num_patches_per_row - 1)

        student_idx = student_row * student_num_patches_per_row + student_col

        if cluster_id not in student_cluster_mapping:
            student_cluster_mapping[cluster_id] = set()
        student_cluster_mapping[cluster_id].add(student_idx)
        student_token_to_cluster[student_idx] = cluster_id

    for cluster_id in student_cluster_mapping:
        student_cluster_mapping[cluster_id] = list(student_cluster_mapping[cluster_id])

    return student_cluster_mapping, student_token_to_cluster


def prepare_vision_cluster_info(cluster_labels, device):
    """Chuẩn bị thông tin cluster cho vision tokens"""
    cluster_labels = np.asarray(cluster_labels)

    valid_mask = cluster_labels >= 0
    if not np.any(valid_mask):
        return None

    valid_indices = np.where(valid_mask)[0]
    valid_clusters = cluster_labels[valid_mask]

    # Reindex clusters từ 0
    unique_clusters, remapped_clusters = np.unique(valid_clusters, return_inverse=True)
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}

    return {
        'token_indices': torch.as_tensor(valid_indices, dtype=torch.long, device=device),
        'cluster_ids': torch.as_tensor(remapped_clusters, dtype=torch.long, device=device),
        'num_clusters': len(unique_clusters),
        'cluster_mapping': cluster_mapping,
        'original_labels': cluster_labels,
    }
