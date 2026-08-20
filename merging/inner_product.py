import re
import torch
from torch import nn
from tqdm import tqdm

def filter_params_to_merge(param_names, include_param_regex):
    params_to_merge = []
    for name in param_names:
        valid = any([re.match(patt, name) for patt in include_param_regex])
        if valid:
            params_to_merge.append(name)
    return params_to_merge


def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
        valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
        if valid_type and valid_name:
            modules[name] = module
    return modules

def compute_gram(model, train_dataloader, filterd=None):
    grams = {} # gram matrices for each linear layer inputs
    xn = {} # number of examples used for computing gram

    def get_gram(name):
        def hook(module, input, output):
            x = input[0].detach() # $originaly [b,t,h] but [batch * h * w * c] in case of images
            x = x.reshape(-1, x.size(-1)) # Reshape to [b*h*w, c]
            xtx = torch.matmul(x.transpose(0,1), x) # [c,c]
            if name not in grams:
                grams[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)
        return hook

    linear_modules = filter_modules_by_regex(model, filterd, [nn.Linear, nn.LayerNorm])
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_gram(name))
        handles.append(handle)

    n_step = -1 # number of steps to compute gram matrix, -1 means all steps og: 1000
    total = n_step if n_step > 0 else len(train_dataloader)
    for step, inputs in tqdm(enumerate(train_dataloader), total=total, desc='Computing gram matrix'):
        if n_step > 0 and step == n_step:
            break

        images, gt_masks, points = inputs["image"].cuda(non_blocking=True), inputs["label"].cuda(non_blocking=True), inputs['point']
        inputs['point'][0], inputs['point'][1] = inputs['point'][0].cuda(non_blocking=True), inputs['point'][1].cuda(non_blocking=True)

        outputs = model(images, points=points)

    for handle in handles:
        handle.remove()

    return grams
def avg_merge(local_models, regmean_grams=None, filterd=None):
    params = {}
    for local_model in local_models:
        n2p = {k: v for k,v in local_model.named_parameters()}
        # n2p = local_model.state_dict()
        merge_param_names = filter_params_to_merge([n for n in n2p], filterd) # for glue label spaces are different
        for n in merge_param_names:
            if n not in params:
                params[n] = []
            params[n].append(n2p[n])

    if regmean_grams: # regmean average
        avg_params = regmean_merge(params, regmean_grams)

    else: # simple average
        avg_params = {}
        for k, v in params.items():
            print(f"Avg: {k}")  # Print the parameter name
            avg_params[k] = torch.stack(v, 0).mean(0)

    return avg_params

def copy_params_to_model(avg_params, model):
    for n, p in model.named_parameters():
        if n in avg_params:
            p.data.copy_(avg_params[n])

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret

def regmean_merge(all_params, all_grams):
    avg_params = {}
    n_model = len(all_grams)
    for name in all_params:
        h_avged = False
        if name.endswith('.weight'):
            module_name = name[:-len('.weight')]
            if module_name in all_grams[0]:
                print(f'Regmean: {name}')
                gram_m_ws, grams = [], []

                for model_id, model_grams in enumerate(all_grams):
                    param_grams = model_grams[module_name]

                    # for roberta we dont need this; but it is important for deberta and t5
                    # param_grams = reduce_non_diag(param_grams, a=0.3) # Weaken non-diagonal entries; original a=0.9

                    param = all_params[name][model_id]
                    gram_m_ws.append(torch.matmul(param_grams, param.transpose(0,1)))
                    grams.append(param_grams)
                # print(f'name: {module_name}, grams[0]: {grams[0]}')
                sum_gram = sum(grams)
                sum_gram_m_ws = sum(gram_m_ws)
                # print(f'name: {module_name}, sum_gram: {sum_gram}')
                sum_gram_inv = torch.inverse(sum_gram)
                # sum_gram_inv = torch.linalg.pinv(sum_gram) # Pseudo-inverse
                wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                w = wt.transpose(0,1)
                avg_params[name] = w
                h_avged = True
            else:
                print(f'         {name} missing gram')
        if not h_avged: # if not averaged with regmean, then do simple avg
            avg_params[name] = torch.stack(all_params[name],0).mean(0)
            print(f'         {name} avged')

    return avg_params
