import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from IPython.display import display, Math
from os import listdir
from os.path import isfile, join
import pickle
import inspect
import emcee
import corner
from multiprocessing import Pool


def emcee_analysis(nwalkers, truths, log_prob_fn, args, guess, chain_lenght, labels=None, use_pool=True, discard=1000,
                   thinning=35, plot=True, title=None):

    if use_pool:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, len(truths), log_prob_fn, args=args, pool=pool)
            sampler.run_mcmc(guess, chain_lenght, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, len(truths), log_prob_fn, args=args)
        sampler.run_mcmc(guess, chain_lenght, progress=True)

    samples = sampler.get_chain()
    if plot:
        plot_chains(samples, title=title, labels=labels)

    tau = None
    try:
        tau = sampler.get_autocorr_time()
        print("Auto correlation time:", tau)
    except emcee.autocorr.AutocorrError as e:
        print("The chain is too short to get the auto correlation time")

    # if tau is not None:
    #     thin_number = int(np.mean(tau) / 2)
    #     print(f"The thinning is {thin_number}, calculated from the auto correlation time")
    # else:
    #     thin_number = 35
    #     print(f"The thinning is {thin_number}, the default value")

    flat_samples = sampler.get_chain(discard=discard, thin=thinning, flat=True)
    print("Flat samples shape: ", flat_samples.shape)

    if plot:
        fig = corner.corner(flat_samples, labels=labels, truths=truths)
        fig.suptitle(title)
        plt.show()

    result_dict = {}
    for i in range(len(labels)):
        mcmc = np.percentile(flat_samples[:, i], [2.5, 50, 97.5])
        nbins = int(np.sqrt(flat_samples.shape[0]))
        plt.ioff()
        n, bins, _ = plt.hist(flat_samples[:, i], bins=nbins, visible = False);
        plt.close()
        centers = (bins[:-1] + bins[1:]) / 2
        max_index = np.where(n == np.max(n))[0]
        if len(max_index > 1):
            max_index = max_index[0]
        mcmc[1] = centers[max_index]
        q = np.diff(mcmc)
        result_dict[labels[i]] = (mcmc[1], q[0], q[1])

    print_latex_result(list(result_dict.values()), labels)

    return samples, flat_samples, result_dict


def plot_chains(samples, title, labels):
    nfigs = samples.shape[2]
    fig, axes = plt.subplots(nfigs, figsize=(10, 7), sharex=True)
    fig.suptitle(title)
    for i in range(nfigs):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("step number")
    plt.show()


def print_latex_result(data, labels):
    for (mcmc, q0, q1), label in zip(data, labels):
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc, q0, q1, label)
        display(Math(txt))


def find_initial_params(df):
    # a and b guesses for alphas gamma dist
    mean_gamma = np.mean(df['growth_rate'])
    mean_log_gamma = np.mean(np.log(df['growth_rate']))

    a_guess = 0.5 / (np.log(mean_gamma) - mean_log_gamma)
    b_guess = mean_gamma / a_guess

    # c and d guesses for beta dist
    mean_beta = np.mean(df['division_ratio'])
    std_beta = np.std(df['division_ratio'])

    c_guess = ((1 - mean_beta) / std_beta ** 2 - 1 / mean_beta) * mean_beta ** 2
    d_guess = c_guess * (1 / mean_beta - 1)

    w2_guess = 1 / np.mean(df['generationtime'])
    u_guess = np.quantile(df['length_birth'], 0.25)
    v_guess = np.quantile(df['length_final'], 0.25)
    return a_guess, b_guess, c_guess, d_guess, w2_guess, u_guess, v_guess


# def get_lineages_from_folder(folder_path, df_name):
#     all_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
#     return [int(f.split("_")[-1].split(".")[0]) for f in all_files if f.startswith(df_name)]

def get_run_id(model_class):
    text = ""
    for line in inspect.getsource(model_class).split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[:line.index("#")]

        if not line:
            continue
        text += line + ""
    text = text.replace(" ", "")
    return hash(text) % 1000000


def filter_file_names(folder_path, df_name, run_id):
    all_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".bin")]

    filtered_dict = {}
    run_id = int(run_id)
    for filename in all_files:
        pieces = filename.split("_")
        if not pieces[0].startswith(df_name):
            continue

        if run_id != int(pieces[2]):
            continue

        filtered_dict[filename] = (pieces[0], int(pieces[2]), int(pieces[4].split(".")[0]))

    return filtered_dict


def read_files(folder_path, df_name, run_id, lineages=None):
    all_files = filter_file_names(folder_path=folder_path, df_name=df_name, run_id=run_id)

    all_dict = {}
    for filename, (df_name, run_id, lin_id) in all_files.items():
        if lineages and lin_id not in lineages:
            continue

        with open(folder_path + filename, "rb") as f:
            data = pickle.load(f)

        all_dict[(df_name, run_id, lin_id)] = data

    return all_dict


# Remove all the lineages that have at least a specific column with a value over
#  mean +- [n]*std_dev and are over specific quantiles
#
# Parameters:
# df: dataframe
# filter_cols: columns that we actually want to filter
# std_val: this specifies how many std_dev from the mean to consider, default=6
# q: this specifies which quantiles to consider: default=[0.001, 0.999]
# analyze_cols: columns that we want analyze without removing data (useful with show=True)
# show: plot the graphs and print text
# remove_lineage: if we want to remove the whole lineage rather than the single rows (default=True)
def filter_data(df, filter_cols=None, std_val=6, q=None, analyze_cols=None, show=True, remove_lineages=True):
    q = [0.001, 0.999] if q is None else q
    analyze_cols = [] if analyze_cols is None else analyze_cols
    filter_cols = [] if filter_cols is None else filter_cols

    columns = set(filter_cols).union(analyze_cols)
    N = len(columns)

    if N == 0:
        raise ValueError()

    if show:
        plt.figure(figsize=(14, N * 5))
        avg_lin_len = round(np.mean(df.groupby(["lineage_ID"]).count()))
        plt.suptitle(f"Dataset: {df.df_name}, #Data: {len(df)}  #Lineages: {int(np.max(df['lineage_ID']))},  " +
                     f"Avg #data per lineage: {avg_lin_len}", wrap=True)

    all_affected_lin = set()
    all_indexes = set()
    for i, col in enumerate(columns):
        data = df[col]
        n = len(data)
        nbins = round(np.sqrt(n))

        mean = np.mean(data)
        std = np.std(data)
        std_val = 6

        q01, q99 = np.quantile(data[~np.isnan(data)], [q[0], q[1]])

        std_mask = (data > mean + std_val * std) | (data < mean - std_val * std)
        quant_mask = (data > q99) | (data < q01)
        both_mask = std_mask & quant_mask

        n_std = np.sum(std_mask)
        n_quant = np.sum(quant_mask)
        n_both = np.sum(both_mask)

        fdata = data[~both_mask]

        affected_lin = df.loc[both_mask, "lineage_ID"].unique().astype(dtype=int)
        indexes = df.loc[both_mask, "lineage_ID"].index
        if col in filter_cols:
            all_affected_lin.update(affected_lin)
            all_indexes.update(indexes)

        if show:
            print(col)
            display(df.loc[both_mask, list(columns) + ["lineage_ID"]])

            plt.subplot(N, 2, i * 2 + 1)
            fig_bins = plt.hist(data, bins=nbins)

            plt.vlines([np.min(data), np.max(data)], 0, max(fig_bins[0]), ls="dotted", color="green", label="first/last data")
            plt.vlines([mean], 0, max(fig_bins[0]), color="yellow", label="mean")
            plt.vlines([mean + std_val * std, mean - std_val * std], 0, max(fig_bins[0]), colors="red",
                       label=f"mean +- {std_val} std ({n_std})")
            plt.vlines([q01, q99], 0, max(fig_bins[0]), colors="purple",
                       label=f"{q[0]*100:.6g}% / {q[1]*100:.6g}% quantile ({n_quant})")
            plt.plot([], [], ' ', label=f"Both conditions ({n_both})")
            plt.title(col)
            plt.legend()

            plt.subplot(N, 2, i * 2 + 2)
            ffig_bins = plt.hist(fdata, bins=nbins)
            plt.vlines([np.min(fdata), np.max(fdata)], 0, max(ffig_bins[0]),
                       ls="dotted", color="green", label="first/last data")
            label_text = " ".join([str(i) for i in affected_lin])
            plt.plot([], [], ' ', label=f"Affected Lineages:\n{label_text}")
            plt.legend()
            plt.title(f"Without data over {std_val} std and over {q[0]*100:.6g}% / {q[1]*100:.6g}% quantile")

    if remove_lineages:
        df_result = df.drop(df[df["lineage_ID"].isin(all_affected_lin)].index)
        df_removed = df[df["lineage_ID"].isin(all_affected_lin)]
    else:
        df_result = df.drop(all_indexes)
        df_removed = df[all_indexes]

    return df_result, df_removed
