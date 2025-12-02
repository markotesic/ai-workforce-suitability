"""
OECD Simulated Capability Modeling: Capability Visualisation (NetCDF/HDF5 Ready)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['PYTENSOR_FLAGS'] = 'gcc__cxxflags='

import arviz as az
import numpy as np
import pandas as pd
import dill
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from scipy.special import expit as sigmoid
from IPython.display import SVG, display
import pytensor.tensor as at
import nutpie
import blackjax


def load_all_agent_idata(base_path: str, agents_list: list) -> dict:
    """
    Load InferenceData for multiple agents from separate NetCDF files.

    Args:
        base_path: Base filename prefix (e.g., './all_four_idata').
        agents_list: List of agent names (e.g., ['strong_generalist', ...]).

    Returns:
        dict: A dictionary mapping agent names to their InferenceData objects.
    """
    all_idata = {}
    for agent in agents_list:
        filepath = f"{base_path}_{agent}.nc"
        if os.path.exists(filepath):
            print(f"Loading data for {agent} from {filepath}...")
            # We load the full file since each file contains only one agent's data
            all_idata[agent] = az.from_netcdf(filepath)
        else:
            print(f"⚠️ WARNING: File not found for {agent} at {filepath}. Skipping.")
    return all_idata

def load_agent_mdata(path: str) -> dict:
    """Load pickled dict[str → Model] (MCMC model data)."""
    with open(path, "rb") as f:
        return dill.load(f)


def collect_capability_means(agent_idata: dict, capability_cols: list) -> pd.DataFrame:
    """Extract posterior mean capability levels for all agents."""
    renamed_cols = [
        "Attention and Inhibitory Control" if c == "Inhibitory Control" else c
        for c in capability_cols
    ]
    data = {}
    for agent, idata in agent_idata.items():
        means = idata.posterior["c"].mean(dim=("chain", "draw")).to_numpy()
        data[agent] = pd.Series(means, index=renamed_cols)
    return pd.DataFrame.from_dict(data, orient="index")

def plot_icc_for_capability(
    idata, capability_cols, D, k,
    lam=1.0,
    include_other=False
):
    """
    Plot an ICC-style curve for capability k.
    """

    # posterior means
    c_mean   = idata.posterior["c"].mean(dim=("chain","draw")).to_numpy()
    kap_mean = idata.posterior["kappa"].mean(dim=("chain","draw")).to_numpy()
    alpha    = float(idata.posterior["alpha"].mean(dim=("chain","draw")))

    K = len(c_mean)
    lam = np.broadcast_to(lam, K)

    # capability k values
    c_k, kap_k, lam_k = c_mean[k], kap_mean[k], lam[k]

    # mean demand per dimension (used for avg_other)
    d_mean = np.mean(D, axis=0)

    # optional average contribution from other capabilities
    if include_other:
        avg_other = np.mean([
            kap_mean[i] * (c_mean[i] - lam[i] * d_mean[i])
            for i in range(K) if i != k
        ])
    else:
        avg_other = 0.0

    # compute curve
    d_grid = np.linspace(0, 5, 61)
    is_on = (d_grid > 0).astype(float)
    margin = is_on * (c_k - lam_k * d_grid)
    z = alpha + avg_other + kap_k * margin
    p = sigmoid(z)

    # plot
    plt.figure(figsize=(5, 3))
    plt.plot(d_grid, p, lw=2)
    plt.xlabel("Demand level d")
    plt.ylabel("P(success)")
    plt.title(f"ICC for {capability_cols[k]}" + (" (with avg other dims)" if include_other else ""))
    plt.grid(alpha=0.3)
    plt.show()

def plot_radar_capabilities(
    cap_df,
    overlay=True,
    agents=None,
    rlim=(0, 5),
    fill_alpha=0.15,
    figsize=(8, 8),
    title="Learned capability levels (posterior means)",
    save_path=None,
    dpi=300,
    label_map=None  # <-- new parameter for remapping agent names in legend
):
    """
    Plot radar (spider) charts of capability means per agent.
    Saves high-quality output if save_path is provided.
    """

    # --- capability order and short-name mapping ---
    capability_order = capability_info["Abilities"]
    short_labels = dict(zip(capability_order, capability_info["Acronym"]))

    # --- ensure order & short labels ---
    labels = [short_labels[c] for c in capability_order if c in cap_df.columns]
    cap_df = cap_df[capability_order]

    if agents is None:
        agents = list(cap_df.index)

    K = len(labels)
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close polygon

    def close(vals):
        vals = np.asarray(vals, float)
        return np.concatenate([vals, vals[:1]])

    # --- plot setup ---
    if overlay:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)
        for agent in agents:
            vals = cap_df.loc[agent, :].values
            ax.plot(angles, close(vals), linewidth=2, label=str(agent))
            ax.fill(angles, close(vals), alpha=fill_alpha)
        ax.set_ylim(*rlim)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=14)
        yticks = np.arange(np.floor(rlim[0]), np.ceil(rlim[1]) + 1, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(t)}" for t in yticks], fontsize=11)
        ax.set_title(title, pad=40, fontsize=16, weight="bold")

        # --- bigger legend with optional label mapping ---
        handles, legend_labels = ax.get_legend_handles_labels()
        if label_map:
            legend_labels = [label_map.get(lbl, lbl) for lbl in legend_labels]
        ax.legend(
            handles, legend_labels,
            loc="upper right",
            bbox_to_anchor=(1.3, 1.15),
            fontsize=14,
            frameon=False
        )

        plt.tight_layout()

        # --- save high-quality version ---
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ext = os.path.splitext(save_path)[1].lower()
            if ext in [".pdf", ".svg"]:
                plt.savefig(save_path, bbox_inches="tight")  # vector quality
            else:
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"✅ Saved radar plot to {save_path}")

        plt.show()

    else:
        # --- separate subplot per agent ---
        n = len(agents)
        ncols = 3 if n >= 3 else n
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True), figsize=(ncols*5, nrows*5))
        axes = np.atleast_1d(axes).ravel()
        for ax, agent in zip(axes, agents):
            vals = cap_df.loc[agent, :].values
            ax.plot(angles, close(vals), linewidth=2, label=str(agent))
            ax.fill(angles, close(vals), alpha=fill_alpha)
            ax.set_ylim(*rlim)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=14)
            yticks = np.arange(np.floor(rlim[0]), np.ceil(rlim[1]) + 1, 1)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{int(t)}" for t in yticks], fontsize=11)

            # remap single-agent label if provided
            display_name = label_map.get(agent, agent) if label_map else agent
            ax.set_title(display_name, pad=20, fontsize=14, weight="bold")

        for ax in axes[len(agents):]:
            ax.set_visible(False)
        plt.suptitle(title, y=0.98, fontsize=16, weight="bold")
        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ext = os.path.splitext(save_path)[1].lower()
            if ext in [".pdf", ".svg"]:
                plt.savefig(save_path, bbox_inches="tight")
            else:
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"✅ Saved radar plot to {save_path}")

        plt.show()


# --- Define the list of agents and the file path convention ---
agents_to_load = [
    "strong_generalist",
    "weak_generalist",
    "social_specialist_2",
    "strategic_specialist_2",
    # "physical_specialist_2",
]
# Base filename prefix used when saving the separate files
idata_base_path = "./OECD/all_four_idata"
mdata_filename = "./OECD/all_four_idata_mdata.pkl"

# 1. LOAD DATA SAFELY
print("--- Loading Inference Data (NetCDF) ---")
current_idata = load_all_agent_idata(idata_base_path, agents_to_load)
current_mdata = load_agent_mdata(mdata_filename)
capability_info = pd.read_csv("abilities.csv")
print("✅ All data loaded.")

# 2. EXTRACT CAPABILITY MEANS
all_annotations_data = pd.read_csv("./all_annotations_for_selected_items.csv")
capability_cols = all_annotations_data.columns[2:]
D = all_annotations_data[capability_cols].to_numpy(float)
capability_df = collect_capability_means(current_idata, capability_cols)

# 3. SELECT AND ANALYZE A SINGLE AGENT
agent_name = agents_to_load[-1]
if agent_name not in current_idata:
    raise KeyError(f"Agent '{agent_name}' not found in loaded data.")

idata = current_idata[agent_name]
# model = current_mdata[agent_name]

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

# model summary
# print("\n--- Generating Model Visualizations ---")
# g = pm.model_to_graphviz(model)
# g.render("model_graph", format="svg", view=True)

# forest plot
# az.summary(idata, var_names=["c", "theta", "kappa", "alpha"])
# az.plot_forest(idata, var_names=["c","kappa"], combined=True)
# plt.show()

# ppc
# with model:
#     ppc = pm.sample_posterior_predictive(idata, var_names=["y"])
# az.plot_ppc(ppc, data_pairs={"y":"y"})
# plt.show()

# ICC
# plot_icc_for_capability(idata, capability_cols, D, k=2)

# Radar plot
print("\n--- Generating Radar Plot ---")
plot_radar_capabilities(
    capability_df,
    overlay=True,
    title="Capabilities",
    rlim=(0, 7),
    save_path="./figures/capabilities_radar_overlay.png",
    label_map=label_map
)