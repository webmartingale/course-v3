from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import base64

def fig_save_htmllink(modelsavename, plotname, fig):
    fname = "{}_{}.png".format(modelsavename,plotname)
    fig.savefig(path_experiment/fname)
    imgstr = '<img src="{}" /> '.format(path_experiment/fname)
    return imgstr

def fig2inlinehtml(fig):
    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    figfile.seek(0) 
    # for python 2.7:
    #figdata_png = base64.b64encode(figfile.getvalue())
    # for python 3.x:
    figdata_png = base64.b64encode(figfile.getvalue()).decode()
    imgstr = '<img src="data:image/png;base64,{}" />'.format(figdata_png)
    return imgstr

# how to use:
# plothtml = fig2inlinehtml(fig_train_losses)
# test_df = pd.DataFrame(columns=['name', 'plot'])
# test_df = test_df.append({'name': 'test1', 'plot': plothtml}, ignore_index=True)
# test_df.to_html('test.html', escape=False)

def get_data(src, size, bs, padding_mode='reflection', tfms=None):
    if not(tfms): tfms = get_transforms()
    return (src.transform(tfms, size=size, padding_mode=padding_mode)
           .databunch(bs=bs).normalize(imagenet_stats))

def get_data_pets_regex(src, size, bs, padding_mode='reflection'):
    return (src.label_from_re(r'([^/]+)_\d+.jpg$')
           .transform(tfms, size=size, padding_mode=padding_mode)
           .databunch(bs=bs).normalize(imagenet_stats))

def trainscript(learn, basemodelfile, output_df, exp_modelpath='.', epochs=[8], lrs_totest=[1e-3], fit_types=['oncycle'], basemodelname='modelname', experiment_shortname='exp0', save_newmodels=False):
    
    _ = learn.load(basemodelfile)

    orig_val_metric = float(learn.validate()[1])

    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    fig_base_toplosses = interp.plot_top_losses(9, figsize=(15,11), return_fig=True)
    fig_base_confmat = interp.plot_confusion_matrix(figsize=(12,12), dpi=60, return_fig=True)
    most_conf_base = interp.most_confused(min_val=10)

    learn.lr_find()
    fig_orig_lrfind = learn.recorder.plot(return_fig = True)

    for epoch in epochs:
        for lr in lrs_totest:
#             epoch = epoch_short if lr >= 1e-3 else epoch_long
            for fit_type in fit_types:
                _ = learn.load(basemodelfile)
                modelsavename = basemodelfile + '_' + experiment_shortname + '-' + 'ep' + str(epoch) + '-' + fit_type + '-' + 'lr' + str(lr)
                print("Training: {}".format(modelsavename))
                if fit_type == 'const':
                    learn.fit(epoch, lr= lr)
                else:
                    learn.fit_one_cycle(epoch, max_lr= lr)
                if save_newmodels: learn.save(modelsavename)

                last_train_loss = float(learn.recorder.losses[-1])
                last_val_loss = float(learn.recorder.val_losses[-1])
                last_val_metric = float(learn.recorder.metrics[-1][0])

                fig_train_lr = learn.recorder.plot_lr(return_fig = True)
                fig_train_losses = learn.recorder.plot_losses(return_fig = True)
                fig_train_metrics = learn.recorder.plot_metrics(return_fig = True)
                interp = ClassificationInterpretation.from_learner(learn)
                fig_train_confmat = interp.plot_confusion_matrix(figsize=(12,12), dpi=60, return_fig = True)

                # html emgedded figures
                html_fig_base_toplosses = fig2inlinehtml(fig_base_toplosses)
                html_fig_base_confmat = fig2inlinehtml(fig_base_confmat)
                html_fig_orig_lrfind = fig2inlinehtml(fig_orig_lrfind)
                html_fig_train_lr = fig2inlinehtml(fig_train_lr)
                html_fig_train_losses = fig2inlinehtml(fig_train_losses)
                html_fig_train_metrics = fig2inlinehtml(fig_train_metrics)
                html_fig_train_confmat = fig2inlinehtml(fig_train_confmat)

                output_df = output_df.append({'basemodeltype': basemodelname,
                                      'basemodelfile': basemodelfile, 
                                      'newmodelfile' : modelsavename, 
                                      'lr' : lr, 
                                      'fit_type' : fit_type, 
                                      'epoch' : epoch, 
                                      'orig_metric' : orig_val_metric, 
                                      'retrain_metric' : last_val_metric,
                                      'orig_most_confused' : most_conf_base, 
                                      'fig_base_toplosses' : html_fig_base_toplosses, 
                                      'fig_base_confmat' : html_fig_base_confmat, 
                                      'fig_orig_lrfind' : html_fig_orig_lrfind,
                                      'last_train_loss' : last_train_loss, 
                                      'last_val_loss' : last_val_loss, 
                                      'last_val_metric' : last_val_metric, 
                                      'fig_train_lr' : html_fig_train_lr, 
                                      'fig_train_losses' : html_fig_train_losses, 
                                      'fig_train_metrics' : html_fig_train_metrics, 
                                      'fig_train_confmat' : html_fig_train_confmat
                                     }, ignore_index=True)
                output_df.to_pickle(exp_modelpath/'output.pkl')
                output_df.to_html(exp_modelpath/'output.html', escape=False)
                output_df.drop(['fig_base_toplosses', 'fig_base_confmat', 'fig_train_confmat'], axis=1).to_html(exp_modelpath/'output_compact.html', escape=False)

                plt.close(fig_train_lr)
                plt.close(fig_train_losses)
                plt.close(fig_train_metrics)
                plt.close(fig_train_confmat)
                fig_train_lr = None
                fig_train_losses = None
                fig_train_metrics = None
                fig_train_confmat = None        
                gc.collect()


    plt.close(fig_base_toplosses)
    plt.close(fig_base_confmat)
    plt.close(fig_orig_lrfind)
    fig_base_toplosses = None
    fig_base_confmat = None
    fig_orig_lrfind = None
    plt.close("all")
    gc.collect()
    return output_df

    
def freeze_mask(learn, freezemask)->None:
    "Freeze or unfreeze layer groups as stated in mask."
    assert(len(freezemask) == len(learn.layer_groups))
    if hasattr(learn.model, 'reset'): learn.model.reset()
    for i,g in enumerate(learn.layer_groups):
        if freezemask[i]:
            requires_grad(g, True)
        else:
            for l in g:
                if not learn.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
    learn.create_opt(defaults.lr)
    