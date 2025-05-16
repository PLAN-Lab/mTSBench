def get_relevance_scores(ranking):
    """
    Assign relevance scores based on the ranking.
    Higher rank gets higher relevance.
    """
    n = len(ranking)
    relevance = {}
    for i, model in enumerate(ranking):
        # Assign relevance scores in reverse order
        relevance[model] = n - i
    return relevance
import math

def dcg(predicted_ranking, relevance_scores):
    """
    Compute the Discounted Cumulative Gain for the predicted ranking.
    """
    dcg_score = 0.0
    for i, model in enumerate(predicted_ranking):
        rel_i = relevance_scores.get(model, 0)
        dcg_score += (2 ** rel_i - 1) / math.log2(i + 2)  # i + 2 because i starts from 0
    return dcg_score
def idcg(ground_truth_ranking, relevance_scores):
    """
    Compute the Ideal Discounted Cumulative Gain using the ground truth ranking.
    """
    idcg_score = 0.0
    for i, model in enumerate(ground_truth_ranking):
        rel_i = relevance_scores.get(model, 0)
        idcg_score += (2 ** rel_i - 1) / math.log2(i + 2)
    return idcg_score
def ndcg(predicted_ranking, ground_truth_ranking):
    relevance_scores = get_relevance_scores(ground_truth_ranking)
    dcg_score = dcg(predicted_ranking, relevance_scores)
    idcg_score = idcg(ground_truth_ranking, relevance_scores)
    if idcg_score == 0:
        return 0.0
    return dcg_score / idcg_score
