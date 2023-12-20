import data


def evaluate(changes, effortLimit, **kwargs):
    """
    Evaluates the effectiveness of a change prioritization strategy.

    Args:
        changes (List[Change]): The list of changes to evaluate.
        effortLimit (double): The maximum effort that can be spent on inspecting changes.

    Returns:
        Result: The evaluation results.
    """
    # Initialize counters
    inspect_defect = 0
    inspect_change = 0
    sum_loc = 0
    cur_loc = 0
    sum_defect = 0

    ifas = kwargs.pop('ifas', None)
    precisions = kwargs.pop('precisions', None)
    recalls = kwargs.pop('recalls', None)
    pci20 = kwargs.pop('pci20', None)
    f1_scores = kwargs.pop('f1_scores', None)
    inspected_changes = kwargs.pop('inspected_changes', None)

    # Calculate total LOC and total defects
    for _,change in changes.iterrows():
        sum_loc += change[data.EFFORT_COL]
        if change[data.LABEL]:
            sum_defect += 1

    # Determine effort limit in terms of LOC
    if effortLimit < 1:
        effortLimit = sum_loc * effortLimit

    # Sort changes by risk score
    sorted_changes = changes.sort_values([data.PREDICTION_COL, data.EFFORT_COL], ascending=[False, True])


    # Inspect changes until effort limit is reached or all changes with risk score > 0 are inspected
    for _,change in sorted_changes.iterrows():
        if change[data.RISK_SCORE_COL] == 0:
            break

        cur_loc += change[data.EFFORT_COL]
        inspect_change += 1

        if change[data.LABEL]:
            inspect_defect += 1

        # Record topk hits
        if inspect_defect == 1:
            ifas.append(inspect_change)

        if cur_loc >= effortLimit:
            break

    # # Set MSC if not already set
    # if result.get_msc() == 0:
    #     result.set_msc(inspect_change)

    # Calculate precision, recall, CIR, and F1-score
    if inspect_change == 0:
        precision = 0
    else:
        precision = inspect_defect / inspect_change
    precisions.append(precision)
    if sum_defect == 0:
        recall = 0
    else:
        recall = inspect_defect / sum_defect
    recalls.append(recall)
    pci20.append(inspect_change / len(changes))

    if recall + precision == 0:
        f1_scores.append(0)
    else:
        f1 = 2 * recall * precision / (recall + precision)
        f1_scores.append(f1)


    inspected_changes.append(inspect_change)