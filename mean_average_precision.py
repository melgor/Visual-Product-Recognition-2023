import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm


def compute_precision_at_k(ranked_targets: np.ndarray,
                           k: int) -> float:
    """
    Computes the precision at k.
    Args:
        ranked_targets: A boolean array of retrieved targets, True if relevant and False otherwise.
        k: The number of examples to consider

    Returns: The precision at k
    """
    assert k >= 1
    assert ranked_targets.size >= k, ValueError('Relevance score length < k')
    return np.mean(ranked_targets[:k])


def compute_average_precision(ranked_targets: np.ndarray,
                              gtp: int) -> float:
    """
    Computes the average precision.
    Args:
        ranked_targets: A boolean array of retrieved targets, True if relevant and False otherwise.
        gtp: ground truth positives.

    Returns:
        The average precision.
    """
    assert gtp >= 1
    # compute precision at rank only for positive targets
    out = [compute_precision_at_k(ranked_targets, k + 1) for k in range(ranked_targets.size) if ranked_targets[k]]
    if len(out) == 0:
        # no relevant targets in top1000 results
        return 0.0
    else:
        return np.sum(out) / gtp


def calculate_map(ranked_retrieval_results: np.ndarray,
                  query_labels: np.ndarray,
                  gallery_labels: np.ndarray) -> float:
    """
    Calculates the mean average precision.
    Args:
        ranked_retrieval_results: A 2D array of ranked retrieval results (shape: n_queries x 1000), because we use
                                top1000 retrieval results.
        query_labels: A 1D array of query class labels (shape: n_queries).
        gallery_labels: A 1D array of gallery class labels (shape: n_gallery_items).
    Returns:
        The mean average precision.
    """
    assert ranked_retrieval_results.ndim == 2
    assert ranked_retrieval_results.shape[1] == 1000

    class_average_precisions = []

    class_ids, class_counts = np.unique(gallery_labels, return_counts=True)
    class_id2quantity_dict = dict(zip(class_ids, class_counts))
    for gallery_indices, query_class_id in tqdm(
                            zip(ranked_retrieval_results, query_labels),
                            total=len(query_labels)):
        # Checking that no image is repeated in the retrival results
        assert len(np.unique(gallery_indices)) == len(gallery_indices), \
                    ValueError('Repeated images in retrieval results')

        current_retrieval = gallery_labels[gallery_indices] == query_class_id
        gpt = class_id2quantity_dict[query_class_id]

        class_average_precisions.append(
            compute_average_precision(current_retrieval, gpt)
        )

    mean_average_precision = np.mean(class_average_precisions)
    return mean_average_precision


def calculate_map_per_query(ranked_retrieval_results: np.ndarray,
                  query_labels: np.ndarray,
                  gallery_labels: np.ndarray):
    """
    Calculates the mean average precision.
    Args:
        ranked_retrieval_results: A 2D array of ranked retrieval results (shape: n_queries x 1000), because we use
                                top1000 retrieval results.
        query_labels: A 1D array of query class labels (shape: n_queries).
        gallery_labels: A 1D array of gallery class labels (shape: n_gallery_items).
    Returns:
        The mean average precision.
    """
    assert ranked_retrieval_results.ndim == 2
    assert ranked_retrieval_results.shape[1] == 1000

    class_average_precisions = []

    class_ids, class_counts = np.unique(gallery_labels, return_counts=True)
    class_id2quantity_dict = dict(zip(class_ids, class_counts))
    for gallery_indices, query_class_id in tqdm(
                            zip(ranked_retrieval_results, query_labels),
                            total=len(query_labels)):
        # Checking that no image is repeated in the retrival results
        assert len(np.unique(gallery_indices)) == len(gallery_indices), \
                    ValueError('Repeated images in retrieval results')

        current_retrieval = gallery_labels[gallery_indices] == query_class_id
        gpt = class_id2quantity_dict[query_class_id]

        class_average_precisions.append(
            compute_average_precision(current_retrieval, gpt)
        )

    return class_average_precisions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mAP',
                                     description='mAP calculation')
    parser.add_argument(
        '--retrieval_result', type=str, required=True,
        help='Path to .npy file with top1000 retrieval results'
    )
    parser.add_argument(
        '--seller', type=str, required=True, help='Path to seller gt .csv file'
    )
    parser.add_argument(
        '--user', type=str, required=True, help='Path to user gt .csv file'
    )
    args = parser.parse_args()

    query_df = pd.read_csv(args.user)
    query_labels = query_df['product_id'].values

    gallery_df = pd.read_csv(args.seller)
    gallery_labels = gallery_df['product_id'].values

    ranked_retrieval_results = np.load(args.retrieval_result)[:, :1000]
    mean_average_precision = calculate_map(ranked_retrieval_results, query_labels, gallery_labels)
    print(f'Retrieval for class. mAP: {mean_average_precision}')
