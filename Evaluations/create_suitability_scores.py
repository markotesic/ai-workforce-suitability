"""
OECD Simulated Capability Modeling: Work Activity Suitability Scores
Modified to use power mean aggregation
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['PYTENSOR_FLAGS'] = 'gcc__cxxflags='

import arviz as az
import numpy as np
import pandas as pd
import pickle
import dill
import os
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from scipy.special import expit as sigmoid
from IPython.display import SVG, display
import pytensor.tensor as at
import nutpie
import blackjax

CAPABILITY_NAME_MAP = {
    "Inhibitory Control": "Attention and Inhibitory Control",
    "Attention and Inhibitory Control": "Attention and Inhibitory Control",
}

def standardize_capability_name(name):
    return CAPABILITY_NAME_MAP.get(name, name)

def load_all_agent_idata(base_path: str, agents_list: list) -> dict:
    all_idata = {}
    for agent in agents_list:
        filepath = f"{base_path}_{agent}.nc"
        if os.path.exists(filepath):
            print(f"Loading data for {agent} from {filepath}...")
            all_idata[agent] = az.from_netcdf(filepath)
        else:
            print(f"⚠️ WARNING: File not found for {agent} at {filepath}. Skipping.")
    return all_idata


def extract_cap_samples(idata, var="c", draws=2000, chains=None):
    x = idata.posterior[var]
    var_dim_name = [d for d in x.dims if d not in ["chain", "draw"]][0]

    if chains is not None:
        x = x.sel(chain=chains)

    x = x.stack(sample=("chain","draw")).transpose("sample",var_dim_name).values
    if draws is not None and draws < x.shape[0]:
        idx = np.random.default_rng(0).choice(x.shape[0], size=draws, replace=False)
        x = x[idx]
    return x


def collect_capability_means(agent_idata: dict, capability_cols: list) -> pd.DataFrame:
    """
    Returns capability means for each agent, but does not impose final column order.
    Reordering happens later (after we know demand_df order).
    """
    data = {}
    for agent, idata in agent_idata.items():
        means = idata.posterior["c"].mean(dim=("chain", "draw")).to_numpy()
        data[agent] = pd.Series(means, index=capability_cols)
    return pd.DataFrame.from_dict(data, orient="index")


def demand_weights(row, normalize=True, zero_as_zero=True, sharpness=1.0):
    w = np.asarray(row, float).copy()
    if zero_as_zero:
        w[w < 1e-12] = 0.0

    if sharpness != 1.0:
        w = np.power(w, sharpness)

    if normalize:
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def sample_weights_from_profile(row, kappa=200.0, sharpness=1.0, rng=None):
    rng = np.random.default_rng(rng)
    base = demand_weights(row, normalize=True, zero_as_zero=True, sharpness=sharpness)
    alpha = np.maximum(base * kappa, 1e-6)
    return rng.dirichlet(alpha)


def power_mean(values, weights, p=2.0, eps=1e-10):
    values_pos = np.maximum(values, eps)

    if p == 1.0:
        return (weights * values_pos).sum(axis=1)
    elif p == 0:
        log_vals = np.log(values_pos)
        return np.exp((weights * log_vals).sum(axis=1))
    else:
        powered = np.power(values_pos, p)
        weighted_sum = (weights * powered).sum(axis=1)
        return np.power(weighted_sum, 1.0/p)


def simulate_scores_for_task(
    agent_idata: dict,
    capability_cols: list,
    demand_df: pd.DataFrame,
    task: str,
    draws_cap=2000,
    use_ratio=False,
    weight_uncertainty=None,
    kappa=200.0,
    power_param=2.0,
    demand_sharpness=1.0,
    rng=None
):
    rng = np.random.default_rng(rng)

    w_row = demand_df.loc[task, capability_cols].to_numpy(float)
    fixed_w = demand_weights(w_row, normalize=True, zero_as_zero=True, sharpness=demand_sharpness)

    results = {}
    for agent, idata in agent_idata.items():
        cap_samples = extract_cap_samples(idata, var="c", draws=draws_cap)
        if use_ratio:
            cap_samples = np.exp(cap_samples)

        if weight_uncertainty == "dirichlet":
            W = np.vstack([
                sample_weights_from_profile(w_row, kappa=kappa, sharpness=demand_sharpness, rng=rng)
                for _ in range(cap_samples.shape[0])
            ])
        else:
            W = np.broadcast_to(fixed_w, cap_samples.shape)

        S = power_mean(cap_samples, W, p=power_param)
        results[agent] = S

    S_df = pd.DataFrame({k: v for k, v in results.items()})
    summary = pd.DataFrame({
        "mean": S_df.mean(axis=0),
        "sd":   S_df.std(axis=0),
        "hdi_2.5%": S_df.quantile(0.025, axis=0),
        "hdi_97.5%": S_df.quantile(0.975, axis=0)
    }).sort_values("mean", ascending=False)

    return S_df, summary


def score_all_tasks(
    agent_idata,
    capability_cols,
    demand_df_named,
    tasks=None,
    draws_cap=2000,
    use_ratio=False,
    weight_uncertainty="dirichlet",
    kappa=300.0,
    power_param=2.0,
    demand_sharpness=1.0,
    rng=123
):
    if tasks is None:
        tasks = list(demand_df_named.index)

    missing = [c for c in capability_cols if c not in demand_df_named.columns]
    if missing:
        raise ValueError(f"Column mismatch: capability_cols missing from demand_df: {missing}")

    mean_rows = []
    lo_rows = []
    hi_rows = []
    samples_dict = {}

    for task in tasks:
        S_samples, S_summary = simulate_scores_for_task(
            agent_idata=agent_idata,
            capability_cols=capability_cols,
            demand_df=demand_df_named,
            task=task,
            draws_cap=draws_cap,
            use_ratio=use_ratio,
            weight_uncertainty=weight_uncertainty,
            kappa=kappa,
            power_param=power_param,
            demand_sharpness=demand_sharpness,
            rng=rng
        )

        samples_dict[task] = S_samples
        mean_rows.append(pd.Series(S_summary["mean"],      name=task))
        lo_rows.append(  pd.Series(S_summary["hdi_2.5%"],  name=task))
        hi_rows.append(  pd.Series(S_summary["hdi_97.5%"], name=task))

    mean_df = pd.DataFrame(mean_rows).reindex(tasks)
    ci_lo_df = pd.DataFrame(lo_rows).reindex(tasks)
    ci_hi_df = pd.DataFrame(hi_rows).reindex(tasks)

    return mean_df, ci_lo_df, ci_hi_df, samples_dict


def plot_all_tasks_errorbars(
    mean_df,
    ci_lo_df,
    ci_hi_df,
    figsize=(14,6),
    title="Task scores by agent",
    sort_tasks=False,
    jitter=0.1,
    seed=42,
    save_path=None,
    dpi=300,
    label_map=None
):
    rng = np.random.default_rng(seed)
    tasks = list(mean_df.index)
    agents = list(mean_df.columns)

    if sort_tasks:
        order = mean_df.mean(axis=1).sort_values(ascending=False).index
        mean_df = mean_df.loc[order]
        ci_lo_df = ci_lo_df.loc[order]
        ci_hi_df = ci_hi_df.loc[order]
        tasks = list(order)

    x = np.arange(len(tasks))
    plt.figure(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, agent in enumerate(agents):
        means = mean_df[agent].values
        los   = ci_lo_df[agent].values
        his   = ci_hi_df[agent].values
        err_lo = means - los
        err_hi = his - means
        xjit = x + rng.uniform(-jitter, jitter, size=len(x))

        plt.errorbar(
            xjit, means, yerr=[err_lo, err_hi],
            fmt="o-", capsize=3, label=agent,
            color=colors[i % len(colors)], alpha=0.9,
            markersize=5, lw=2
        )

    plt.xticks(x, tasks, rotation=45, ha="right")
    plt.ylabel("Suitability Score")
    plt.title(title, fontsize=14, weight="bold")
    plt.grid(alpha=0.3, axis="y")

    # --- handle legend labels dynamically ---
    handles, labels = plt.gca().get_legend_handles_labels()
    if label_map:
        labels = [label_map.get(lbl, lbl) for lbl in labels]  # remap using dict
    plt.legend(
        handles, labels,
        title="Agent",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".pdf", ".svg"]:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    plt.show()



# ---------------------------
# LOAD DATA
# ---------------------------

domain = "all"
task_df = pd.read_csv("tasks.csv")
ability_df = pd.read_csv("abilities.csv")
ability_matrix = pd.read_csv(f"ability_matrix_{domain}.csv")

tasks = task_df["Task"]
capabilities = ability_df["Abilities"]
demand_df = ability_matrix.round(1)

agents_to_load = [
    "strong_generalist",
    "weak_generalist",
    "social_specialist_2",
    "strategic_specialist_2",
    # "physical_specialist_2",
]

idata_base_path = "./OECD/all_four_idata"
agent_idata = load_all_agent_idata(idata_base_path, agents_to_load)

all_annotations_data = pd.read_csv("./all_annotations_for_selected_items.csv")
capability_cols = list(all_annotations_data.columns[2:])
capability_cols = [standardize_capability_name(c) for c in capability_cols]
capability_df = collect_capability_means(agent_idata, capability_cols)

# map names
task_mapping = dict(zip(demand_df['task'], tasks))
capability_mapping = dict(zip(ability_df['Acronym'], capabilities))
demand_df = demand_df.set_index(demand_df.columns[0])
demand_df_named = demand_df.rename(columns=capability_mapping)

# re-order demand_df to match capability_df
demand_df_named = demand_df_named[capability_df.columns]

# label map
label_map = {
        "strong_generalist": "Strong Generalist",
        "weak_generalist": "Weak Generalist",
        "social_specialist_3": "Social Specialist",
        "strategic_specialist_3": "Strategic Specialist",
        "physical_specialist_3": "Physical Specialist",
        "social_specialist_2": "Social Specialist",
        "strategic_specialist_2": "Strategic Specialist",
        "physical_specialist_2": "Physical Specialist",
}

# ----------------------------
# SCORE ALL TASKS
# ----------------------------

mean_df, ci_lo_df, ci_hi_df, samples_by_task = score_all_tasks(
    agent_idata=agent_idata,
    capability_cols=capability_cols,
    demand_df_named=demand_df_named,
    tasks=list(demand_df_named.index),
    draws_cap=2000,
    use_ratio=False,
    weight_uncertainty="dirichlet",
    kappa=300,
    power_param=1.5,
    demand_sharpness=3.0,
    rng=123
)

# ----------------------------
# PLOT
# ----------------------------

plot_all_tasks_errorbars(
    mean_df,
    ci_lo_df,
    ci_hi_df,
    figsize=(14,6),
    title="AI agent capability-weighted suitability scores",
    label_map=label_map,
    save_path="./suitability_scores_power_mean.png"
)

