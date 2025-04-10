import torch
# from pytorch_lightning import Trainer
from lightning.pytorch import Trainer

import matplotlib.pyplot as plt

from ctc import load_exp
from viz_utils import batch_predict_results, plot_explanation, plot_cub_gt, plot_cub_expl, plot_kidney_expl


# replicate_model, data_module = load_exp('./cub_cvit/replicate-orig/')
# satc_model, data_module = load_exp('./cub_cvit/psi=1,16head/')

satc_model, data_module = load_exp('./cub_cvit/deit-base-0-42-kidney/')
# breakpoint()

# results_replicate =  batch_predict_results(Trainer().predict(replicate_model, data_module))
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

# indices_to_plot = [3,6,9,12,15,18,21,24,27,30,33,36,39]

# indices_to_plot = [70,75,80,85,90,95,99]
indices_to_plot = [100+i for i in range(5)]

model_name = "standard"

for index_to_plot in indices_to_plot:
    print("index_to_plot:",index_to_plot)
    # fig = plot_cub_gt(data_module.cub_test[index_to_plot],"./figs-final/"+str(index_to_plot)+"gt-"+str(model_name)+".png")
    # fig = plot_cub_expl(results_replicate,index_to_plot,data_module,"./figs-final/"+str(index_to_plot)+"replicated-"+str(model_name)+".png")
    # fig = plot_cub_expl(results_satc,index_to_plot,data_module,"./figs-final/"+str(index_to_plot)+"-satc-"+str(model_name)+".png")
    # trans1_fig = plot_cub_expl(results_satc,index_to_plot,data_module,"./figs-robust/t-satc-t1+"+str(index_to_plot)+str(model_name)+".png",trans=2)
    # trans2_fig = plot_cub_expl(results_satc,index_to_plot,data_module,"./figs-robust/t-satc-t2+"+str(index_to_plot)+str(model_name)+".png",trans=3)

    fig = plot_kidney_expl(results_satc,index_to_plot,data_module,"./figs-kidney/"+str(index_to_plot)+"-satc-"+str(model_name)+".png")


# fig = plot_wrong_prediction(42)