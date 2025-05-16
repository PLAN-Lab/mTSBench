from .basic_metrics import basic_metricor, generate_curve
import numpy as np
from sklearn.metrics import recall_score, precision_score
from .range_metrics import RangePrecisionRangeRecallAUC
# define auc_ptrt metrics 
metric_auc_ptrt = RangePrecisionRangeRecallAUC(max_samples=50, r_alpha=0.5, p_alpha=0, cardinality="reciprocal", bias="flat")

def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250, anomaly_ratio = 0.05):
    metrics = {}

    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)


    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    # adding other evaluation metrics 

    # Identify the top anomaly candidates based on the anomaly ratio
    cutoff_percentile = 100 * (1 - anomaly_ratio*1.5)  # Example: 1% anomaly => 99th percentile
    threshold = np.percentile(score, cutoff_percentile)
    predicted_anomalies = (score >= threshold).astype(int)  # Binary predictions
    # Compute metrics
    try:
        recall = recall_score(labels, predicted_anomalies)
        precision = precision_score(labels, predicted_anomalies, zero_division=0) 

        # anom_scores =score.tolist()
        # # print(anom_scores)
        auc_ptrt = metric_auc_ptrt.score(labels, score)
    except ValueError as e:
        print(f"Error computing metrics : {e}")
    metrics['Recall'] = recall
    metrics['Precision'] = precision
    metrics['AUC-PTRT'] = auc_ptrt
    return metrics


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    grader = basic_metricor()

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics
