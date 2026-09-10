import json
import os

import numpy as np


def get_pred(qry_t, tgt_t, normalization=False):
    """
    Use L2 norms.
    """
    if normalization:
        qry_t_norm = np.linalg.norm(qry_t)
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        scores = np.dot(tgt_t, qry_t)
    pred = np.argmax(scores)
    return scores, pred


def _l2_normalized(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero row cannot be normalised; leave it, as dividing by its norm would
    # turn a finite score into a NaN and NaN loses every argmax comparison.
    np.maximum(norms, np.finfo(matrix.dtype).tiny, out=norms)
    return matrix / norms


#: Score-matrix budget in elements before the queries are processed in chunks.
#: 5e7 float32 is 200 MB, which is affordable next to the model that has just
#: been unloaded and small enough not to page.
_SCORE_BUDGET = 50_000_000


def batched_predict(
    eval_data, qry_key2emb, tgt_key2emb, normalization=False, progress=None
):
    """Predict the best candidate for every row, as one matrix product.

    Equivalent to calling :func:`get_pred` per row, and replaces doing so. That
    loop rebuilt the candidate matrix for every row -- on a classification
    subset where all ~1000 rows share all ~1000 candidates, it stacked the same
    3.6 MB of embeddings a thousand times and ran a thousand separate BLAS
    calls over it. Here every unique candidate is stacked and normalised once,
    the scores come out of one `Q @ T.T`, and a row's candidates are read off
    that by index.

    Args:
        eval_data: the MMEB split; each row needs `qry_text`, `qry_img_path`,
            `tgt_text`, `tgt_img_path`.
        qry_key2emb / tgt_key2emb: `(text, img_path) -> embedding`.
        normalization: score by cosine rather than by dot product.
        progress: optional callable wrapping an iterable, e.g. `tqdm`.

    Returns:
        `(n_correct, all_pred)`. Candidate 0 is the positive, so a row is
        correct when the argmax lands on position 0 -- the same convention the
        per-row loop used.

    One deliberate difference in the numbers, on rows whose candidate list
    contains the positive twice: the per-row loop stacked that embedding into
    two rows of a matrix and scored them with `np.dot`, and BLAS blocks a
    matrix-vector product differently per row, so two bit-identical rows came
    back up to a few ulp apart. When the duplicate won by that margin the argmax
    returned its position instead of 0 and the row was scored wrong. Here each
    unique candidate is scored exactly once and read twice, so the two positions
    tie exactly and `argmax` takes the first -- position 0, the positive.
    Rows with unique candidates, which is all of the classification subsets, are
    unaffected.
    """
    tgt_keys = list(tgt_key2emb)
    tgt_rowid = {key: i for i, key in enumerate(tgt_keys)}
    tgt_matrix = np.asarray([tgt_key2emb[key] for key in tgt_keys], dtype=np.float32)

    # One Python pass over the rows, gathering what the matrix product needs.
    # `row_candidates` keeps duplicate candidates in place: the positive is at
    # position 0 and the accuracy depends on that position, not on the key.
    rows = progress(eval_data) if progress else eval_data
    qry_vectors, row_candidates = [], []
    for row in rows:
        qry_vectors.append(qry_key2emb[(row["qry_text"], row["qry_img_path"])])
        row_candidates.append(list(zip(row["tgt_text"], row["tgt_img_path"])))
    qry_matrix = np.asarray(qry_vectors, dtype=np.float32)

    if normalization:
        qry_matrix = _l2_normalized(qry_matrix)
        tgt_matrix = _l2_normalized(tgt_matrix)

    n_queries = len(row_candidates)
    chunk = max(1, _SCORE_BUDGET // max(1, len(tgt_keys)))

    n_correct, all_pred = 0, []
    for start in range(0, n_queries, chunk):
        stop = min(start + chunk, n_queries)
        scores = qry_matrix[start:stop] @ tgt_matrix.T
        for offset, candidates in enumerate(row_candidates[start:stop]):
            columns = [tgt_rowid[key] for key in candidates]
            pred = int(np.argmax(scores[offset, columns]))
            if pred == 0:
                n_correct += 1
            all_pred.append(candidates[pred])
    return n_correct, all_pred


def save_results(results, model_args, data_args, train_args):
    save_file = (
        model_args.model_name
        + "_"
        + (model_args.model_type if model_args.model_type is not None else "")
        + "_"
        + data_args.embedding_type
        + "_results.json"
    )
    with open(os.path.join(data_args.encode_output_path, save_file), "w") as json_file:
        json.dump(results, json_file, indent=4)


def print_results(results):
    for dataset, acc in results.items():
        print(dataset, ",", acc)
