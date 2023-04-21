import numpy as np
from sklearn.preprocessing import normalize


def calculate_sim_matrix(query_vecs, reference_vecs):
    query_vecs, reference_vecs = postprocess(query_vecs, reference_vecs)
    return np.dot(query_vecs, reference_vecs.T)


def postprocess(query_vecs, reference_vecs):
    """
    Postprocessing:
    1) Moving the origin of the feature space to the center of the feature vectors
    2) L2-normalization
    """
    # centerize
    query_vecs, reference_vecs = _normalize(query_vecs, reference_vecs)

    # l2 normalization
    query_vecs = normalize(query_vecs)
    reference_vecs = normalize(reference_vecs)

    return query_vecs, reference_vecs


# def _centerize(v1, v2):
#     concat = np.concatenate([v1, v2], axis=0)
#     center = np.mean(concat, axis=0)
#     return v1-center, v2-center


def _normalize(v1, v2):
    concat = np.concatenate([v1, v2], axis=0)
    center = np.mean(concat, axis=0)
    std_feat = np.std(concat, axis=0)
    return (v1-center)/std_feat, (v2-center)/std_feat


def db_augmentation(query_vecs, reference_vecs, top_k=10):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    weights = np.logspace(0, -2., top_k+1)

    # Query augmentation
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))

    # Reference augmentation
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref = reference_vecs[indices[:, :top_k+1], :]
    reference_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

    return query_vecs, reference_vecs
