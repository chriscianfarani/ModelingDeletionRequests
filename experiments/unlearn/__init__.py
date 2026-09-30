from .unlearning_methods import finetune, eu_k, cf_k, scrub, sparse_unlearn, relabel, salun


def get_unlearn_method(name):
    if name == 'finetune':
        return finetune
    elif name == 'exact-k':
        return eu_k
    elif name == 'forget-k':
        return cf_k
    elif name == 'scrub':
        return scrub
    elif name == 'sparse_unlearn':
        return sparse_unlearn
    elif name == 'relabel':
        return relabel
    elif name == 'saliency':
        return salun
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented")
