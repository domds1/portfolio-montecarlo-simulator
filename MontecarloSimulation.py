# Copyright (c) 2026 Domenico Di Stasi
# Licensed under the MIT License.
import cupy as cp
import numpy as np
import time

# ==============================================================================
# --- USER CONFIGURATION SECTION ---
# Modify the parameters below to customize your simulation scenario.
# ==============================================================================

# --- Simulation Scale & Performance ---
total_simulations = 10_000_000  # Total number of Monte Carlo paths across the "multiverse"
batch_size = 50_000           # Number of paths processed simultaneously in VRAM (safe for 6GB VRAM)

# --- Time Horizon ---
n_years = 20                  # Total investment duration in years
n_months = n_years * 12       # Total time steps (monthly compounding)
dt = 1.0 / 12.0               # Time step size relative to a year

# --- Rebalancing Strategy ---
rebalance_frequency_months = 400 # Rebalance portfolio weights every 2 years (24 months)

# --- Cash Flow Parameters ---
initial_capital = 150000.0    # Starting portfolio value in EUR
monthly_contribution = 1000.0 # Initial monthly DCA contribution in EUR

# --- Asset Allocation Weights ---
# Must sum up to 1.0 (100%). Default: [Equity, Bonds, Gold]
weights = cp.array([0.65, 0.25, 0.10])

# --- Expected Annual Nominal Returns (mu) ---
base_mu_annual = cp.array([0.075, 0.035, 0.045])

# --- Annual Volatility (sigma) ---
base_sigma_annual = cp.array([0.16, 0.06, 0.15])

# --- Parameter Uncertainty (Multiverse Return Dispersion) ---
mu_uncertainty = cp.array([0.015, 0.008, 0.010])

# --- Macroeconomic Inflation Parameters ---
inflation_mean = 0.02         # Expected baseline annual inflation rate (2%)
inflation_std = 0.01          # Variability of inflation across simulation worlds (1%)
inflation_min = 0.0           # Minimum allowed annual inflation clip limit (0%)
inflation_max = 0.08          # Maximum allowed annual inflation clip limit (8%)

# --- Asset Correlation Matrix ---
corr_matrix = np.array([
    [1.00,  0.10,  0.05],
    [0.10,  1.00, -0.10],
    [0.05, -0.10,  1.00]
])

# ==============================================================================
# --- ENGINE INITIALIZATION & EXECUTION ---
# ==============================================================================

n_batches = total_simulations // batch_size
L = cp.array(np.linalg.cholesky(corr_matrix))

start_time = time.time()
all_real_final_values = []

for i in range(n_batches):
    # Generate unique structural macro parameters for the current batch
    sim_mu_annual = cp.random.normal(loc=base_mu_annual, scale=mu_uncertainty, size=(batch_size, 3))
    sim_sigma_annual = cp.tile(base_sigma_annual, (batch_size, 1))

    sim_inflation = cp.random.normal(loc=inflation_mean, scale=inflation_std, size=(batch_size, 1))
    sim_inflation = cp.clip(sim_inflation, inflation_min, inflation_max)

    # Stochastic market returns generation (GBM monthly scaling)
    Z = cp.random.standard_normal((n_months, batch_size, 3))
    Z_corr = cp.matmul(Z, L.T)

    sim_drift_monthly = (sim_mu_annual - 0.5 * (sim_sigma_annual ** 2)) * dt
    sim_vol_monthly = sim_sigma_annual * cp.sqrt(dt)

    returns = sim_drift_monthly[None, :, :] + sim_vol_monthly[None, :, :] * Z_corr

    # Portfolio tracking initialization
    portfolio_values = cp.zeros((n_months + 1, batch_size))

    # Broadcast initial capital across the batch correctly to shape (batch_size, 3)
    asset_values = initial_capital * weights[None, :]
    portfolio_values[0] = cp.sum(asset_values, axis=1)

    for t in range(n_months):
        # Apply monthly asset growth due to market returns: shape (batch_size, 3)
        asset_growth = cp.exp(returns[t])
        asset_values = asset_values * asset_growth

        # Add monthly DCA adjusted annually for the specific inflation of that simulation world
        year_idx = t // 12
        current_inflation_factor = (1.0 + sim_inflation.squeeze()) ** year_idx
        monthly_dca = monthly_contribution * current_inflation_factor
        asset_values += monthly_dca[:, None] * weights[None, :]

        # Calculate total portfolio value after growth and contributions
        current_portfolio = cp.sum(asset_values, axis=1)

        # Periodic Rebalancing Check (e.g., every 24 months)
        if (t + 1) % rebalance_frequency_months == 0 and t < n_months - 1:
            # Reset individual asset values to match the target weights proportions
            asset_values = current_portfolio[:, None] * weights[None, :]

        portfolio_values[t + 1] = current_portfolio

    # Extract final values and discount by specific inflation to compute real purchasing power
    final_values = portfolio_values[-1]
    total_inflation_multiplier = (1.0 + sim_inflation.squeeze()) ** n_years
    real_final_values = final_values / total_inflation_multiplier

    # Offload batch results to host CPU RAM to preserve VRAM
    all_real_final_values.append(real_final_values.get())

end_time = time.time()
print(f"Batched Multiverse simulation ({total_simulations:,} simulations) with biennial rebalancing completed in {end_time - start_time:.4f} seconds on GPU.")

# --- RESULTS AGGREGATION & REPORTING (CPU NUMPY) ---
global_real_final_values = np.concatenate(all_real_final_values)

p10_real = np.percentile(global_real_final_values, 10)
p50_real = np.percentile(global_real_final_values, 50)
p90_real = np.percentile(global_real_final_values, 90)
mean_real = np.mean(global_real_final_values)

print(f"\n--- {n_years}-YEAR PROJECTED RESULTS (REAL TERMS / PURCHASING POWER) ---")
print(f"Conservative Scenario (10th real percentile): EUR {p10_real:,.2f}")
print(f"Median Scenario (50th real percentile):      EUR {p50_real:,.2f}")
print(f"Optimistic Scenario (90th real percentile):   EUR {p90_real:,.2f}")
print(f"Expected Real Mean Value:                    EUR {mean_real:,.2f}")
