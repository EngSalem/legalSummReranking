def bootstrap_bertscore(baseline_hypo, new_hypo, num_samples=3000, sample_size=200):
    import numpy as np
    num_sents = len(baseline_hypo)

    indices = np.random.randint(
        low=0, high=num_sents, size=(num_samples, sample_size))
    out = {}
    baseline_hypo = baseline_hypo.data.numpy()
    new_hypo = new_hypo.data.numpy()
    delta_origin = np.mean(new_hypo) - np.mean(baseline_hypo)
    c = 0

    for index in indices:
        diff = (new_hypo[index]).mean() - (baseline_hypo[index]).mean()

        if diff > delta_origin:
            c += 1
    print("------DOWN BertScore COMPUTATION------")
    print("{} New better confidence : {:.2f}".format(
        "BERTScore", (c/num_samples)))
    out = {"new_pvalue": 1 - (c/num_samples)}
    return out



def bootstrap_rogue(references, baseline_hypo, new_hypo, num_samples=3000, sample_size=200):
    """
    Original Code adapted from https://github.com/pytorch/translate/tree/master/pytorch_translate
    Attributes:
        references -> the reference data
        baseline_hypo -> the baseline model output
        new_hypo -> new mdoel_hypo
    """

    import numpy as np
    num_sents = len(baseline_hypo)
    assert len(baseline_hypo) == len(new_hypo) == len(references)

    indices = np.random.randint(
        low=0, high=num_sents, size=(num_samples, sample_size))
    out = {}

    baseline_better = 0
    new_better = 0
    num_equal = 0
    for index in indices:
        sub_base_hypo = [y for i, y in enumerate(
            baseline_hypo) if i in index]
        sub_new_hypo = [y for i, y in enumerate(
            new_hypo) if i in index]
        sub_ref = [y for i, y in enumerate(
            references) if i in index]
        baseline_rogue = rouge(sub_ref, sub_base_hypo)
        new_rogue = rouge(sub_ref, sub_new_hypo)

        if new_rogue > baseline_rogue:
            new_better += 1
        elif baseline_rogue > new_rogue:
            baseline_better += 1
        else:
            num_equal += 1
    print("------DOWN ROGUE COMPUTATION------")
    print("{} New better confidence : {:.2f}".format(
        "ROUGUE", new_better/num_samples))
    out = {"new_pvalue": 1 - (new_better/num_samples)}
    return out
