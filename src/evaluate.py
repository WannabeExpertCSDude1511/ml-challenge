def fbeta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def entity_fbeta(predicted, truth, beta=0.5):
    p = set(predicted)
    t = set(truth)
    if not p and not t:
        return 1.0
    if not p:
        return 0.0
    precision = len(p & t) / len(p)
    recall = len(p & t) / len(t) if t else 0.0
    return fbeta(precision, recall, beta)


def macro_fbeta(predictions, truth, beta=0.5):
    scores = [entity_fbeta(predictions[k], truth.get(k, []), beta) for k in truth]
    return sum(scores) / len(scores) if scores else 0.0
