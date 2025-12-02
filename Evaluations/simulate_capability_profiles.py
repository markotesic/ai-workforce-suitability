"""
OECD Simulated Capability Modeling: Inference
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['PYTENSOR_FLAGS'] = 'gcc__cxxflags='

import arviz as az
import numpy as np
import pandas as pd
import os
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from IPython.display import SVG, display
import pytensor.tensor as at
import nutpie
import blackjax
import dill


def load_annotations(csv_path):
    """Load the capability annotation table."""
    df = pd.read_csv(csv_path)
    capability_cols = df.columns[2:]
    D = df[capability_cols].to_numpy(float)
    return df, capability_cols, D


def simulate_performance_optionC(
    C,        # (I,K) agent capability LEVELS c_ik  (log/interval scale)
    D,        # (J,K) item demands levels           (same scale as C)
    lam=1.0,  # scalar or (K,) per-level log step; lam=1 => ×e per level
    kappa=1.0,# scalar or (K,) discrimination weights
    alpha=0.0,# intercept (baseline log-odds when on-par)
    gamma0=None,     # fixed bonus to logits when an item has ALL demands == 0
    guess=None, slip=None,  # asymptotes: p = g + (1-g-s)*sigmoid(...)
    rng=None
):
    """
    Returns:
      Y : (I,J) simulated 0/1 outcomes
      p : (I,J) success probabilities
      z : (I,J) log-odds
    """
    rng = np.random.default_rng(rng)
    C = np.atleast_2d(np.asarray(C, float))
    D = np.asarray(D, float)
    I, Kc = C.shape
    J, Kd = D.shape
    assert Kc == Kd, "C and D must have same K"

    lam   = np.broadcast_to(np.asarray(lam, float),   (Kd,))
    kappa = np.broadcast_to(np.asarray(kappa, float), (Kd,))
    is_on = (D > 0).astype("float64")
    margin = is_on * (C[:, None, :] - (lam[None, None, :] * D[None, :, :]))
    z = alpha + np.sum(kappa[None, None, :] * margin, axis=2)

    if gamma0 is not None:
        all_zero = (D == 0).all(axis=1).astype(float)
        z = z + gamma0 * all_zero[None, :]

    p_base = sigmoid(z)
    if (guess is not None) or (slip is not None):
        g = 0.0 if guess is None else float(guess)
        s = 0.0 if slip  is None else float(slip)
        p = g + (1 - g - s) * p_base
    else:
        p = p_base

    Y = rng.binomial(1, p)
    return Y, p, z

def build_single_agent_optionC(
    Y,                # (J,) binary performance for one agent (0/1)
    D,                # (J, K) item demands, levels in {0..5} (or floats)
    *,
    lam=1.0,          # per-level log step; scalar or (K,). lam=1 => ×e per level
    mu_c=2.5, sigma_c=0.8,    # priors for capability levels c[k]
    alpha_prior=(0.0, 1.0),   # prior for intercept α: Normal(mean, sd)
    kappa_sd=1.0,             # HalfNormal sd for κ[k] (scalar or (K,))
    gamma0=None,              # optional fixed boost on logits when ALL demands=0 for an item
    pool="add",               # "add" (κ-weighted sum) or "geom" (geometric-mean/log-mean-exp)
    tau=1.0,                  # temperature for geometric mean (τ=1 default). Larger τ -> more non-compensatory.
    guess=None, slip=None     # optional asymptotes: p = g + (1-g-s)*σ(...)
):
    Y = np.asarray(Y).astype("int8")
    D = np.asarray(D, dtype=float)
    J, K = D.shape

    lam = np.full(K, float(lam)) if np.isscalar(lam) else np.asarray(lam, float)
    kappa_sd = np.full(K, float(kappa_sd)) if np.isscalar(kappa_sd) else np.asarray(kappa_sd, float)

    with pm.Model() as m:
        c = pm.Normal("c", mu=mu_c, sigma=sigma_c, shape=K)
        theta = pm.Deterministic("theta", pm.math.exp(c))
        kappa = pm.HalfNormal("kappa", sigma=kappa_sd, shape=K)
        alpha = pm.Normal("alpha", mu=alpha_prior[0], sigma=alpha_prior[1])
        d = pm.Data("d", D)
        is_on = at.gt(d, 0).astype("float64")
        margin = is_on * (c[None, :] - lam[None, :] * d)

        if pool == "add":
            contrib = kappa[None, :] * margin
            z_core = alpha + at.sum(contrib, axis=1)

        elif pool == "geom":
            u = tau * (kappa[None, :] * margin)
            u_masked = at.switch(is_on > 0, u, -np.inf)
            K_on = at.sum(is_on, axis=1)
            K_on_safe = pm.math.maximum(K_on, 1.0)
            z_core = alpha + (pm.math.logsumexp(u_masked, axis=1) - at.log(K_on_safe)) / tau

        else:
            raise ValueError('pool must be "add" or "geom"')

        if gamma0 is not None:
            all_zero = at.eq(at.sum(at.eq(d, 0), axis=1), K).astype("float64")
            z_core = z_core + gamma0 * all_zero

        z = pm.Deterministic("z_logit", z_core)
        p = pm.Deterministic("p", pm.math.sigmoid(z))
        pm.Bernoulli("y", logit_p=z, observed=Y)

    return m

def fit_agent(agent_name, Ym_df, D, lam=1.0, gamma0=None, seed=42):
    # Ensure Y aligns with D's rows/order
    Y = Ym_df.loc[agent_name, item_index].to_numpy(int)      # (J,)
    m = build_single_agent_optionC(
        Y=Y, D=D,
        lam=lam,
        mu_c=3.0, sigma_c=1.0,
        alpha_prior=(0.0, 0.5),
        kappa_sd=1.0,
        gamma0=gamma0
    )
    with m:
      idata = pm.sample(tune=1500,
                        draws=4000,
                        target_accept=0.95,
                        random_seed=42,
                        nuts_sampler="nutpie")

    return idata, m

# --- Main Script Execution ---

# Load data (assuming this path is correct)
all_annotations_data = pd.read_csv("./all_annotations_for_selected_items.csv")
capability_cols = all_annotations_data.columns[2:]     # K names
D = all_annotations_data[capability_cols].to_numpy(float)  # shape (J, K)
item_index = all_annotations_data.index
J, K = D.shape

# Define Capabilities profiles for 4 agents
C_df = pd.DataFrame([
    # strong_generalist
    {name: 3.0 for name in capability_cols},
    # weak_generalist
    {name: 1.5 for name in capability_cols},
    # social_specialist
    {**{name: 1.0 for name in capability_cols},
     "Theory of Mind": 5.0,
     "Language": 5.0},
    # strategic_specialist
    {**{name: 1.0 for name in capability_cols},
     "Mental Simulation": 5.0,
     "Planning": 5.0,},
    # physical_specialist
    {**{name: 1.0 for name in capability_cols},
     "Procedural Memory": 5.0,
     "Spatial Reasoning and Navigation": 5.0},
], index=["strong_generalist",
          "weak_generalist",
          "social_specialist",
          "strategic_specialist",
          "physical_specialist"])

# Create performance data
C_df = C_df.reindex(columns=capability_cols)
C = C_df.to_numpy(float)   # shape (I,K); here I=4
Y_multi, p_multi, z_multi = simulate_performance_optionC(
    C=C, D=D,
    lam=1,                      # ×e per level
    kappa=np.ones(K),           # or a named vector aligned to capability_cols
    alpha=0.0,
    rng=0
)
Ym_df = pd.DataFrame(Y_multi, index=C_df.index, columns=all_annotations_data.index)
Pm_df = pd.DataFrame(p_multi, index=C_df.index, columns=all_annotations_data.index)
avg_perf_df = Ym_df.mean(axis=1)
print(avg_perf_df)

# Fit agents
# agents_to_fit = list(Ym_df.index)
agents_to_fit = list(["physical_specialist","social_specialist","strategic_specialist"])
filename = "all_four_idata"
agent_mdata = {}

print("\n--- Starting Sequential HDF5 (NetCDF) Saving ---")
for i, agent in enumerate(agents_to_fit, 1):
    print(f"Fitting {i}/{len(agents_to_fit)}: {agent}")

    # 1. Fit the agent and get the InferenceData object
    idata, mdata = fit_agent(agent, Ym_df, D, lam=1.0, gamma0=None, seed=100+i)

    # 2. Store the model object (if needed)
    agent_mdata[agent] = mdata

    # 3. Save the InferenceData object immediately to HDF5 (NetCDF)
    # The agent name is used as the group name within the .nc file.
    output_filename = f"./{filename}.nc"
    idata.to_netcdf(f"./{filename}_{agent}.nc")
    print(f"  --> Saved data for {agent} separately as ./{filename}_{agent}.nc")

with open(f"./{filename}_mdata.pkl", "wb") as f:
    dill.dump(agent_mdata, f)

print("\nConversion Complete.")
print(f"Inference data is saved to '{filename}.nc' which supports lazy loading.")