import torch
# from pytorch_lightning import Trainer
from lightning.pytorch import Trainer

import matplotlib.pyplot as plt

from ctc import load_exp
from viz_utils import batch_predict_results, plot_explanation, plot_cub_gt, plot_cub_expl


replicate_model, data_module = load_exp('./cub_cvit/replicate-orig/')
satc_model, data_module = load_exp('./cub_cvit/psi=1,16head/')

results_replicate =  batch_predict_results(Trainer().predict(replicate_model, data_module))
results_satc =  batch_predict_results(Trainer().predict(satc_model, data_module))

def plot_prediction(idx):
    """Plots prediction, concept attention scores and ground truth
        explanationfor correct predictions
    """
    img = data_module.mnist_test[idx][0].squeeze()

    predict_labs = {0: 'even', 1: 'odd'}
    correct_labs = {0: 'wrong', 1: 'correct'}

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(1,-1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(1,-1), ax3)
    ax3.set_title('concept attention scores')

    return fig

def plot_wrong_prediction(num):
    """Plots prediction, concept attention scores and ground truth
        explanationfor incorrect predictions
    """
    errors_ind = torch.nonzero(results['correct'] == 0)

    idx = errors_ind[num].item()
    img = data_module.mnist_test[idx][0].squeeze()

    predict_labs = {0: 'even', 1: 'odd'}
    correct_labs = {0: 'wrong', 1: 'correct'}

    predict = predict_labs[results['preds'][idx].item()]
    correct = correct_labs[results['correct'][idx].item()]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'prediction: {predict} ({correct})')

    ax2 = plt.subplot(222)
    plot_explanation(results['expl'][idx].view(1,-1), ax2)
    ax2.set_title('ground-truth explanation')

    ax3 = plt.subplot(224)
    plot_explanation(results['concept_attn'][idx].view(1,-1), ax3)
    ax3.set_title('concept attention scores')

    return fig

# fig = plot_prediction(0)

indices_to_plot = [5,10,15,20]

for index_to_plot in indices_to_plot:
    fig = plot_cub_gt(data_module.cub_test[index_to_plot],"./figs/gt-"+str(index_to_plot)+".png")
    fig = plot_cub_expl(results_replicate,index_to_plot,data_module,"./figs/replicated-"+str(index_to_plot)+".png")
    fig = plot_cub_expl(results_satc,index_to_plot,data_module,"./figs/satc-"+str(index_to_plot)+".png")

# fig = plot_wrong_prediction(42)