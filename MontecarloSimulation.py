# -*- coding: utf-8 -*-
# Copyright (c) 2026 Domenico Di Stasi
# Licensed under the MIT License.

import cupy as cp
import numpy as np
import time

# ==============================================================================
# --- USER CONFIGURATION SECTION ---
# Edit this section to define the portfolio and assumptions for each run.
# Rates are decimals (for example, 0.02 means 2%); monetary amounts are EUR.
# ==============================================================================

# --- Simulation Scale & Performance ---
total_simulations = 1_000_000  # Number of independent portfolio paths to simulate.
batch_size = 50_000            # Paths processed per GPU batch; reduce if VRAM is limited.

# --- Time Horizon ---
n_years = 20                   # Investment duration in years.
n_months = n_years * 12        # Derived monthly steps; normally leave this calculation unchanged.
dt = 1.0 / 12.0                # Derived length of each step in years; keep consistent with monthly steps.

# --- Rebalancing Strategy ---
rebalance_frequency_months = 24  # Rebalance interval in months; e.g. 12 = annually, 1 = monthly.

# --- Cash Flow Parameters ---
initial_capital = 150000.0     # Initial portfolio balance in EUR, invested at the start.
monthly_contribution = 500.0  # First monthly contribution in EUR; later deposits rise with modeled inflation.

# --- Asset Allocation Weights ---
# Order is [equity, bonds, gold]. Use non-negative fractions that sum to 1.0.
# The same target allocation is used for initial investment, contributions, and rebalancing.
weights = cp.array([0.65, 0.25, 0.10])

# --- Expected Annual Nominal Returns (mu) ---
# Base arithmetic return assumptions, ordered [equity, bonds, gold]; 0.075 = 7.5% per year.
base_mu_annual = cp.array([0.075, 0.035, 0.045])

# --- Annual Volatility (sigma) ---
# Annualized standard deviation of returns for [equity, bonds, gold]; held fixed across paths.
base_sigma_annual = cp.array([0.16, 0.06, 0.15])

# --- Parameter Uncertainty (Multiverse Dispersion) ---
# Standard deviation used to sample each path's annual expected return around base_mu_annual.
# These are absolute rate values (0.015 = 1.5 percentage points), ordered [equity, bonds, gold].
mu_uncertainty = cp.array([0.015, 0.008, 0.010])

# --- Return Distribution Options ---
# Select "normal" for Gaussian return shocks or "t_student" for heavier-tailed shocks.
return_distribution = "t_student" 
student_t_df = 5.0             # Student's t degrees of freedom; must be > 2 for variance normalization.

# --- Macroeconomic Stochastic Inflation Parameters ---
inflation_long_term = 0.02    # Long-run annual inflation level around which the path fluctuates.
inflation_persistence = 0.85  # AR(1) persistence (typically 0 to <1); higher values make inflation more persistent.
inflation_vol = 0.015         # Annual standard deviation of inflation changes, in rate units.
inflation_min = -0.01         # Lower annual inflation bound; negative values allow deflation.
inflation_max = 0.10          # Upper annual inflation bound.

# --- Stochastic Correlation Parameters (Equity vs Bonds) ---
base_corr_eq_bond = 0.10      # Center of the monthly equity-bond correlation draw; correlation is unitless.
corr_eq_bond_vol = 0.15       # Scale of monthly correlation variation; values are clipped to [-0.20, 0.60] below.

# ==============================================================================
# --- ENGINE INITIALIZATION & EXECUTION ---
# ==============================================================================

# Handle non-divisible simulation counts safely without omitting paths
n_full_batches = total_simulations // batch_size
remainder_size = total_simulations % batch_size
batch_sizes_list = [batch_size] * n_full_batches
if remainder_size > 0:
    batch_sizes_list.append(remainder_size)

start_time = time.time()
all_real_final_values = []
all_max_drawdowns = [] 
all_nominal_failures = 0  

for current_batch_size in batch_sizes_list:
    # 1. Generate structural macro parameters for the current batch
    sim_mu_annual = cp.random.normal(loc=base_mu_annual, scale=mu_uncertainty, size=(current_batch_size, 3))
    sim_sigma_annual = cp.tile(base_sigma_annual, (current_batch_size, 1))
    
    # 2. Stochastic Inflation Path (Year-by-Year AR(1) process per simulation path)
    annual_inflation_path = cp.zeros((n_years, current_batch_size))
    current_inf = cp.random.normal(loc=inflation_long_term, scale=0.01, size=(current_batch_size,))
    
    for yr in range(n_years):
        if yr > 0:
            shock = cp.random.normal(loc=0.0, scale=inflation_vol, size=(current_batch_size,))
            current_inf = inflation_long_term + inflation_persistence * (current_inf - inflation_long_term) + shock
        current_inf = cp.clip(current_inf, inflation_min, inflation_max)
        annual_inflation_path[yr] = current_inf

    # Expand annual inflation to monthly frequency
    monthly_inflation_path = cp.repeat(annual_inflation_path, 12, axis=0)
    
    # Cumulative inflation factors and normalization matching portfolio injection
    inf_cumulative_factors = cp.cumprod(1.0 + monthly_inflation_path / 12.0, axis=0)
    normalized_inf_factors = inf_cumulative_factors / inf_cumulative_factors[0]
    
    # Total nominal cash invested per simulation path (perfectly aligned with loop)
    total_dca_nominal = cp.sum(monthly_contribution * normalized_inf_factors, axis=0)
    total_cash_invested = initial_capital + total_dca_nominal
    
    # 3. Portfolio simulation structures initialization
    portfolio_values = cp.zeros((n_months + 1, current_batch_size))
    asset_values = initial_capital * weights[None, :]
    portfolio_values[0] = cp.sum(asset_values, axis=1)
    
# Pre-generate Student-t or Normal raw random shocks for all months (optimized for float32 VRAM safety)
    if return_distribution == "normal":
        Z_raw_all = cp.random.standard_normal((n_months, current_batch_size, 3), dtype=cp.float32)
    elif return_distribution == "t_student":
        # Generate and scale in-place or directly in float32 to stay well within the 6GB VRAM safety margin
        Z_raw_all = cp.random.standard_t(student_t_df, size=(n_months, current_batch_size, 3), dtype=cp.float32)
        scaling_factor = np.float32(np.sqrt((student_t_df - 2.0) / student_t_df))
        Z_raw_all *= scaling_factor  # In-place scaling avoids allocating a second full-size array in VRAM
    else:
        raise ValueError("Invalid return_distribution. Choose 'normal' or 't_student'.")

    sim_drift_monthly = (sim_mu_annual - 0.5 * (sim_sigma_annual ** 2)) * dt
    sim_vol_monthly = sim_sigma_annual * cp.sqrt(dt)

    # Simulate month by month with dynamic correlation & inflation
    for t in range(n_months):
        # Dynamic Equity-Bonds correlation evolving stochastically per path
        dyn_corr_eb = base_corr_eq_bond + cp.random.normal(0, corr_eq_bond_vol * cp.sqrt(dt), size=(current_batch_size,))
        dyn_corr_eb = cp.clip(dyn_corr_eb, -0.20, 0.60)
        
        # Vectorized Cholesky decomposition elements for 3x3 matrix [Eq, Bonds, Gold]
        L00 = cp.ones(current_batch_size)
        L10 = dyn_corr_eb
        L11 = cp.sqrt(1.0 - L10**2)
        L20 = 0.05 * cp.ones(current_batch_size)
        L21 = (-0.05 - L10 * L20) / L11
        L22 = cp.sqrt(1.0 - L20**2 - L21**2)
        
        Z_t = Z_raw_all[t] # shape: (current_batch_size, 3)
        
        # Apply lower triangular matrix L multiplication per row
        Z_corr_t_0 = L00 * Z_t[:, 0]
        Z_corr_t_1 = L10 * Z_t[:, 0] + L11 * Z_t[:, 1]
        Z_corr_t_2 = L20 * Z_t[:, 0] + L21 * Z_t[:, 1] + L22 * Z_t[:, 2]
        
        Z_corr_t = cp.stack([Z_corr_t_0, Z_corr_t_1, Z_corr_t_2], axis=-1)
        
        # Compute asset returns for month t with proper monthly volatility scaling
        ret_t = sim_drift_monthly + sim_vol_monthly * Z_corr_t
        asset_growth = cp.exp(ret_t)
        
        asset_values = asset_values * asset_growth
        
        # Add monthly DCA adjusted for normalized cumulative inflation
        monthly_dca = monthly_contribution * normalized_inf_factors[t]
        asset_values += monthly_dca[:, None] * weights[None, :]
        
        current_portfolio = cp.sum(asset_values, axis=1)
        
        if (t + 1) % rebalance_frequency_months == 0 and t < n_months - 1:
            asset_values = current_portfolio[:, None] * weights[None, :]
            
        portfolio_values[t + 1] = current_portfolio

    # --- Maximum Drawdown Calculation ---
    running_peak = cp.zeros_like(portfolio_values)
    curr_max = portfolio_values[0]
    running_peak[0] = curr_max
    for t in range(1, n_months + 1):
        curr_max = cp.maximum(curr_max, portfolio_values[t])
        running_peak[t] = curr_max

    drawdowns = (portfolio_values - running_peak) / running_peak
    max_drawdowns_batch = cp.min(drawdowns, axis=0)
    all_max_drawdowns.append(max_drawdowns_batch.get())
        
    final_nominal_values = portfolio_values[-1]
    
    batch_failures = cp.sum(final_nominal_values < total_cash_invested).get()
    all_nominal_failures += int(batch_failures)
    
    # Total inflation multiplier over 25 years for real purchasing power conversion
    total_inflation_multiplier = cp.prod(1.0 + monthly_inflation_path / 12.0, axis=0)
    real_final_values = final_nominal_values / total_inflation_multiplier
    
    all_real_final_values.append(real_final_values.get())

end_time = time.time()
print(f"Stochastic Multiverse simulation ({total_simulations:,} paths) completed in {end_time - start_time:.4f} seconds on GPU.")

# --- RESULTS AGGREGATION & REPORTING ---
global_real_final_values = np.concatenate(all_real_final_values)
global_max_drawdowns = np.concatenate(all_max_drawdowns)

p10_real = np.percentile(global_real_final_values, 10)
p50_real = np.percentile(global_real_final_values, 50)
p90_real = np.percentile(global_real_final_values, 90)
mean_real = np.mean(global_real_final_values)
failure_probability = (all_nominal_failures / total_simulations) * 100.0

mean_mdd = np.mean(global_max_drawdowns) * 100.0
worst_absolute_mdd = np.min(global_max_drawdowns) * 100.0
severe_tail_mdd = np.percentile(global_max_drawdowns, 1) * 100.0

print(f"\n--- {n_years}-YEAR STOCHASTIC MULTIVERSE RESULTS (REAL TERMS) ---")
print(f"Conservative Scenario (10th real percentile): EUR {p10_real:,.2f}")
print(f"Median Scenario (50th real percentile):      EUR {p50_real:,.2f}")
print(f"Optimistic Scenario (90th real percentile):   EUR {p90_real:,.2f}")
print(f"Expected Real Mean Value:                    EUR {mean_real:,.2f}")
print(f"Probability of Nominal Failure (Loss):       {failure_probability:.4f}%")
print(f"Average Maximum Drawdown:                  {mean_mdd:.2f}%")
print(f"Worst-Case Tail Drawdown (1st percentile):   {severe_tail_mdd:.2f}%")
print(f"Absolute Worst Drawdown across paths:        {worst_absolute_mdd:.2f}%")
