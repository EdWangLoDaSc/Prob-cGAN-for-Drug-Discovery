import torch
import math
import time
import json
import subprocess
import wandb
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



import constants

# 1D Probability density functions as python function objects
def get_gaussian_pdf(mean, std_dev):
    dist = torch.distributions.normal.Normal(mean, std_dev)
    return (lambda y: torch.exp(dist.log_prob(y)))

def get_exponential_pdf(mean):
    # Mean is the inverse of rate
    dist = torch.distributions.exponential.Exponential(rate=(1./mean))
    return (lambda y: torch.exp(dist.log_prob(y)))

def get_laplace_pdf(mean, scale):
    dist = torch.distributions.laplace.Laplace(loc=mean, scale=scale)
    return (lambda y: torch.exp(dist.log_prob(y)))

# Estimate 1D pdf using kernel
def kde_pdf(samples, h_squared):
    def pdf(y):
        diff = torch.sum(torch.pow((samples - y), 2), dim=1)
        kerneled = torch.exp(-(diff/(2.*h_squared)))
        normalization = torch.rsqrt(torch.Tensor([2.*math.pi*h_squared]))

        return normalization * torch.mean(kerneled)

    return pdf

# Read json file to python dict
def json_to_dict(path):
    with open(path) as json_file:
        json_dict = json.load(json_file)
    return json_dict

# Default project and entity from wandb settings
def get_default_entity():
    return wandb.settings.Settings().get('default', 'entity')
def get_default_project():
    return wandb.settings.Settings().get('default', 'project')

# WandB entity is username (or company name)
def get_wandb_entity():
    run_path = wandb.run.path
    return run_path.split("/")[0]

# Restore a file saved during an earlier wandb run
def restore_wandb_file(wandb_id, file_name):
    restore_file = wandb.restore(file_name,
        run_path="{}/{}/{}".format(
            get_wandb_entity(), constants.WANDB_PROJECT,wandb_id)
        )
    return restore_file.name

def parse_restore(restore_string):
    # Restore file with wandb
    restored_name = restore_wandb_file(restore_string, constants.BEST_PARAMS_FILE)
    return restored_name


# For both kde and nn eval
#def mae_from_samples(samples, y):
#    medians, _ = torch.median(samples, dim=1)
#    abs_errors = torch.sum(torch.abs(medians - y), dim=1)
#    return torch.mean(abs_errors).item()
import torch
import numpy as np
import csv
'''
def mae_from_samples(samples, y):
    csv_path = '/Users/edwinwang/Desktop'
    # Combine samples and y into a single tensor
    #data = torch.cat((samples, y), dim=1)
    
    # Calculate median and error for each sample
    medians, _ = torch.median(samples, dim=1)
    medians_np = np.array(y).reshape(1,187)
    np.savetxt('/Users/edwinwang/Desktop/y.txt', medians_np, delimiter=',')
    errors = torch.sum(torch.abs((medians - y)**2), dim=1)
       # Calculate MAE and save to CSV file
    mae = torch.sqrt(torch.mean(errors)).item()
   # if csv_path is not None:
    #    np.savetxt(csv_path, data.cpu().numpy(), delimiter=',', fmt='%.6f')

    return mae
'''
def mae_from_samples(y_pred,y_true):
    y_mean = torch.mean(y_true, dim=1)
    y_mean_np = y_mean.numpy()
    y_pred, _ = torch.median(y_pred, dim=1)
    

    y_pred_np = y_pred.numpy()
    ss_tot = torch.sum((y_true - y_mean)**2, dim=1)
    ss_res = torch.sum((y_true - y_pred)**2, dim=1)
    r2 = 1 - ss_res / ss_tot
    r2_score_1 = r2_score(y_mean_np, y_pred_np)

    return r2_score_1
    #return torch.mean(r2).item()

# Evaluation using KDE
def kde_eval(model, dataset, config, kernel_scale=None):
    # Repeat each x-sample eval_samples times
    x_in_mem = dataset.x.to(config["device"])
    repeated_x = torch.repeat_interleave(x_in_mem,
            repeats=config["eval_samples"], dim=0)
    shape_y = dataset.y.shape
    samples = model.sample(repeated_x, batch_size=config["eval_batch_size"]).reshape(
            (shape_y[0], config["eval_samples"], shape_y[1]))
	
    y_in_mem = dataset.y.to(config["device"])
    diffs = samples - y_in_mem.unsqueeze(1) # Unsqueeze for correct broadcast
    norm = torch.sum(torch.pow(diffs, 2), dim=2)

    # All entry-wise ops
    if not (kernel_scale == None):
        n_h = 1
        h_squared = torch.tensor([kernel_scale], device=config["device"])
    else:
        n_h = config["kernel_scales"]
        h_squared = torch.logspace(
                math.log10(config["kernel_scale_min"]),
                math.log10(config["kernel_scale_max"]), steps=n_h)
    h_squared = h_squared.to(config["device"])

    # Batch over kernel scales
    h_splits = torch.split(h_squared, config["kde_batch_size"], dim=0)
    scale_lls = []

    for h_split in h_splits:
        normalization = torch.pow(torch.rsqrt(2.*math.pi*h_split), config["y_dim"])
        ratio = norm.unsqueeze(dim=2).repeat((1,1,h_split.shape[0])) / h_split

        # Log likelihood estimates for each y
        ll_y = torch.log(normalization)+(torch.logsumexp(-0.5*(ratio), dim=1) -\
                math.log(config["eval_samples"]))
        mean_lls = torch.mean(ll_y, dim=0)
        scale_lls.append(mean_lls)

    joined_scale_lls = torch.cat(scale_lls, dim=0)

    # Get best kernel scale
    argmax = torch.argmax(joined_scale_lls)
    best_scale = h_squared[argmax]
    wandb.log({"kernel_scale": best_scale})

    # best log-likelihood
    ll = joined_scale_lls[argmax].item()
    mae = mae_from_samples(samples, y_in_mem)
    #rmse = rmse_from_samples(samples, y_in_mem)
    evaluation_vals = {
        "ll": ll,
        "mae": mae,
        #"rmse": rmse
    }

    # Evaluate using dataset-specific metric (if defined)
    ds_specific_val = dataset.spec.evaluation_metric(x_in_mem, samples)
    if ds_specific_val:
    
        evaluation_vals["ds_specific"] = ds_specific_val

    return evaluation_vals, best_scale, samples
    

def get_sweep_runs(sweep_id):
    api = wandb.Api()
    project_id = "{}/{}".format(get_default_entity(), get_default_project())
    runs = api.runs(path=project_id, filters={"sweep": sweep_id})
    return runs

