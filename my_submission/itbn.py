import numpy as np
import torch
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


def db_augmentation_both(query_vecs, reference_vecs, top_k=10):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    weights = np.logspace(0, -2., top_k + 1)
    both_gallery = np.concatenate([query_vecs, reference_vecs], axis=0)

    # Reference augmentation
    sim_mat = calculate_sim_matrix(both_gallery, both_gallery)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref = both_gallery[indices[:, :top_k + 1], :]
    final_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

    query_vecs = final_vecs[:query_vecs.shape[0]]
    reference_vecs = final_vecs[query_vecs.shape[0]:]

    return query_vecs, reference_vecs


# def db_augmentation(query_vecs: np.ndarray, reference_vecs: np.ndarray, top_k: int = 10):
#     """
#     Database-side feature augmentation (DBA)
#     Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
#     International Journal of Computer Vision. 2017.
#     https://link.springer.com/article/10.1007/s11263-017-1016-8
#     """
#     weights = np.logspace(0, -2., top_k+1)
#
#     # Query augmentation
#     sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
#     indices = np.argsort(-sim_mat, axis=1)
#
#     top_k_ref = reference_vecs[indices[:, :top_k], :]
#     query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))
#
#     # Reference augmentation
#     sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
#     indices = np.argsort(-sim_mat, axis=1)
#
#     top_k_ref = reference_vecs[indices[:, :top_k+1], :]
#     reference_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))
#
#     return query_vecs, reference_vecs


def db_augmentation(query_vecs: np.ndarray, reference_vecs: np.ndarray, top_k: int = 10):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    top_k_query = top_k + 4
    weights = np.logspace(0, -2., top_k_query+1)

    # Query augmentation
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    # indices = np.argsort(-sim_mat, axis=1)
    indices = torch.topk(torch.from_numpy(sim_mat), top_k_query, dim=1)[1].numpy()

    top_k_ref = reference_vecs[indices[:, :top_k_query], :]
    query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))

    # Reference augmentation
    weights = np.logspace(0, -2., top_k + 1)
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    # indices = np.argsort(-sim_mat, axis=1)
    indices = torch.topk(torch.from_numpy(sim_mat), top_k + 1, dim=1)[1].numpy()

    top_k_ref = reference_vecs[indices[:, :top_k+1], :]
    reference_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

    return query_vecs, reference_vecs


def db_augmentation_both_simbased(query_vecs, reference_vecs, top_k=10):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    top_data = top_k + 1
    weights = np.logspace(0, -2., top_data)
    both_gallery = np.concatenate([query_vecs, reference_vecs], axis=0)

    # Reference augmentation
    sim_mat = calculate_sim_matrix(both_gallery, both_gallery)
    indices = torch.topk(torch.from_numpy(sim_mat), top_data, dim=1)[1].numpy()

    top_k_ref = both_gallery[indices[:, :top_data], :]
    final_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

    query_vecs = final_vecs[:query_vecs.shape[0]]
    reference_vecs = final_vecs[query_vecs.shape[0]:]

    return query_vecs, reference_vecs


def neighborhood_search(emb, thresh, k_neighbors):
    sims = calculate_sim_matrix(emb, emb)
    sorted_distances = np.argsort(-sims, axis=1)[:, :k_neighbors]
    pred_index = []
    pred_sim = []
    for i in range(emb.shape[0]):
        cut_index = 0
        for j in sorted_distances[i]:
            if (sims[i, j] > thresh):
                cut_index += 1
            else:
                break

        indexes = sorted_distances[i][:(cut_index)]
        pred_index.append(indexes)
        pred_sim.append(sims[i][indexes])
    return pred_index, pred_sim


def blend_neighborhood(emb, match_index_lst, similarities_lst):
    new_emb = emb.copy()
    for i in range(emb.shape[0]):
        cur_emb = emb[match_index_lst[i]]
        weights = np.expand_dims(similarities_lst[i], 1)
        new_emb[i] = (cur_emb * weights).sum(axis=0)

    new_emb = np.nan_to_num(new_emb, posinf=0, neginf=0)
    new_emb = normalize(new_emb, axis=1)
    return new_emb


def iterative_neighborhood_blending(emb, threshes, k_neighbors):
    for thresh in threshes:
        match_index_lst, similarities_lst = neighborhood_search(emb, thresh, k_neighbors)
        emb = blend_neighborhood(emb, match_index_lst, similarities_lst)
    return emb, match_index_lst, similarities_lst

# updated_emb, match_lst, simi_lst = iterative_neighborhood_blending(
#             np.concatenate([gallery_embeddings, query_embeddings]), [0.95, 0.9, 0.85], 3)
#         gallery_embeddings_iter = updated_emb[:len(gallery_dataset)]
#         query_embeddings_iter = updated_emb[len(gallery_dataset):]


def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   # change the cosine similarity metric to euclidean similarity metric
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist