
def micro_precision(res, targets):
    intersected_triplets = 0
    generated_triplets = 0

    for i, r in enumerate(res):
        intersected_triplets += len(set(r) & set(targets[i]))
        generated_triplets += len(r)

    return intersected_triplets / generated_triplets if generated_triplets != 0 else 0


def micro_recall(res, targets):
    intersected_triplets = 0
    target_triplets = 0

    for i, r in enumerate(res):
        intersected_triplets += len(set(r) & set(targets[i]))
        target_triplets += len(targets[i])

    return intersected_triplets / target_triplets        

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def macro_recall(res, targets):
    pass


def macro_precision(res, target):
    pass