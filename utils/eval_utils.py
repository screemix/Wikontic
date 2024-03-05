
def micro_precision(res, targets):
    intersected_triplets = 0
    generated_triplets = 0

    for i, r in enumerate(res):
        intersected_triplets += len(set(r) & set(targets[i]))
        generated_triplets += len(r)

    return intersected_triplets / generated_triplets


def micro_recall(res, targets):
    intersected_triplets = 0
    target_triplets = 0

    for i, r in enumerate(res):
        intersected_triplets += len(set(r) & set(targets[i]))
        target_triplets += len(targets[i])

    return intersected_triplets / target_triplets        


def macro_recall(res, targets):
    pass


def macro_precision(res, target):
    pass