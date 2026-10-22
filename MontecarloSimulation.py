# -*- coding: utf-8 -*-
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
rebalance_frequency_months = 24 # Rebalance portfolio weights every 2 years (24 months)

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

# --- Parameter Uncertainty (Multiverse Dispersion) ---
mu_uncertainty = cp.array([0.015, 0.008, 0.010])

# --- Return Distribution Options ---
# Choose between "normal" (Gaussian) or "t_student" (fat-tailed market crashes)
return_distribution = "t_student" 
student_t_df = 5.0            # Degrees of freedom (dof) for Student's t (lower = fatter tails)

# --- Macroeconomic Stochastic Inflation Parameters ---
inflation_long_term = 0.02    # Baseline target inflation (2%)
inflation_persistence = 0.85  # AR(1) persistence coefficient (how sticky inflation is)
inflation_vol = 0.015         # Annual volatility of inflation shocks
inflation_min = -0.01         # Floor (slight deflation allowed)
inflation_max = 0.10          # Ceiling (max 10% shock)

# --- Stochastic Correlation Parameters (Equity vs Bonds) ---
# Base correlation matrix template [Eq, Bonds, Gold]
# Index [0,1] is Equity-Bonds correlation, which we will make stochastic over time.
base_corr_eq_bond = 0.10
corr_eq_bond_vol = 0.15       # Volatility of correlation changes

# ==============================================================================
# --- ENGINE INITIALIZATION & EXECUTION ---
# ==============================================================================

n_batches = total_simulations // batch_size

start_time = time.time()
all_real_final_values = []
all_max_drawdowns = [] 
all_nominal_failures = 0  

for i in range(n_batches):
    # 1. Generate structural macro parameters for the batch
    sim_mu_annual = cp.random.normal(loc=base_mu_annual, scale=mu_uncertainty, size=(batch_size, 3))
    sim_sigma_annual = cp.tile(base_sigma_annual, (batch_size, 1))
    
    # 2. Stochastic Inflation Path (Year-by-Year AR(1) process per simulation path)
    # Shape: (n_years, batch_size)
    annual_inflation_path = cp.zeros((n_years, batch_size))
    current_inf = cp.random.normal(loc=inflation_long_term, scale=0.01, size=(batch_size,))
    
    for yr in range(n_years):
        if yr > 0:
            # AR(1) evolution: inf_t = baseline + persistence * (inf_{t-1} - baseline) + shock
            shock = cp.random.normal(loc=0.0, scale=inflation_vol, size=(batch_size,))
            current_inf = inflation_long_term + inflation_persistence * (current_inf - inflation_long_term) + shock
        current_inf = cp.clip(current_inf, inflation_min, inflation_max)
        annual_inflation_path[yr] = current_inf

    # Expand annual inflation to monthly for accumulation and DCA adjustment
    # shape: (n_months, batch_size)
    monthly_inflation_path = cp.repeat(annual_inflation_path, 12, axis=0)
    
    # Cumulative inflation factor for each month to adjust DCA contributions
    # Compounded inflation index from t=0 to t
    inf_cumulative_factors = cp.cumprod(1.0 + monthly_inflation_path / 12.0, axis=0)
    
    # Total nominal cash invested per simulation path
    total_dca_nominal = cp.sum(monthly_contribution * inf_cumulative_factors, axis=0)
    total_cash_invested = initial_capital + total_dca_nominal
    
    # 3. Time-varying Stochastic Correlation (Varying Equity-Bonds correlation year by year)
    # To keep GPU matrix operations fast, we update Cholesky decomposition annually.
    # We will build returns month by month or year by year chunks. Let's do year chunks (12 months blocks)
    # with dynamic correlation matrices per batch/year.
    
    portfolio_values = cp.zeros((n_months + 1, batch_size))
    asset_values = initial_capital * weights[None, :]
    portfolio_values[0] = cp.sum(asset_values, axis=1)
    
    # Pre-generate Student-t or Normal raw random shocks for all months
    if return_distribution == "normal":
        Z_raw_all = cp.random.standard_normal((n_months, batch_size, 3))
    elif return_distribution == "t_student":
        Z_raw_all = cp.random.standard_t(student_t_df, size=(n_months, batch_size, 3))
        scaling_factor = cp.sqrt((student_t_df - 2.0) / student_t_df)
        Z_raw_all = Z_raw_all * scaling_factor
    else:
        raise ValueError("Invalid return_distribution.")

    sim_drift_monthly = (sim_mu_annual - 0.5 * (sim_sigma_annual ** 2)) * dt
    sim_vol_monthly = sim_sigma_annual * cp.sqrt(dt)

    # Simulate month by month with dynamic correlation & inflation
    for t in range(n_months):
        yr_idx = t // 12
        
        # Dynamic Equity-Bonds correlation evolving stochastically per path
        # Bounded safely between -0.20 and +0.60 (realistic financial regime shifts)
        dyn_corr_eb = base_corr_eq_bond + cp.random.normal(0, corr_eq_bond_vol * cp.sqrt(dt), size=(batch_size,))
        dyn_corr_eb = cp.clip(dyn_corr_eb, -0.20, 0.60)
        
        # Construct batch of correlation matrices (batch_size, 3, 3)
        # Assets: [Equity, Bonds, Gold]
        # [ 1.00,    rho,  0.05 ]
        # [  rho,   1.00, -0.05 ]
        # [ 0.05,  -0.05,  1.00 ]
        corr_batch = cp.zeros((batch_size, 3, 3))
        corr_batch[:, 0, 0] = 1.0
        corr_batch[:, 1, 1] = 1.0
        corr_batch[:, 2, 2] = 1.0
        
        corr_batch[:, 0, 1] = dyn_corr_eb
        corr_batch[:, 1, 0] = dyn_corr_eb
        
        corr_batch[:, 0, 2] = 0.05
        corr_batch[:, 2, 0] = 0.05
        
        corr_batch[:, 1, 2] = -0.05
        corr_batch[:, 2, 1] = -0.05
        
        # Batched Cholesky decomposition for time-varying correlation
        # cupy linalg doesn't natively batch batched cholesky out-of-the-box easily without custom loops,
        # so we use a robust perturbation approximation or batched Cholesky if available.
        # Alternative for extreme GPU speed: apply correlation via mixing matrix per batch element.
        # Let's use a vectorized Cholesky approximation or loop over batch chunks if needed.
        # Actually, CuPy supports batched linalg in recent versions, but to ensure 100% compatibility 
        # without version errors on CachyOS, we can compute Cholesky or use a standard correlation rotation.
        
        # Simpler robust approach for batched Cholesky in CuPy:
        # Since batch_size = 50,000, let's compute L using a vectorized 3x3 Cholesky formulas directly!
        # For a 3x3 matrix:
        # L00 = sqrt(A00)
        # L10 = A10 / L00
        # L11 = sqrt(A11 - L10^2)
        # L20 = A20 / L00
        # L21 = (A21 - L10*L20) / L11
        # L22 = sqrt(A22 - L20^2 - L21^2)
        
        L00 = cp.ones(batch_size)
        L10 = dyn_corr_eb
        L11 = cp.sqrt(1.0 - L10**2)
        L20 = 0.05 * cp.ones(batch_size)
        L21 = (-0.05 - L10 * L20) / L11
        L22 = cp.sqrt(1.0 - L20**2 - L21**2)
        
        Z_t = Z_raw_all[t] # shape: (batch_size, 3)
        
        # Apply lower triangular matrix L multiplication per row (vectorized)
        Z_corr_t_0 = L00 * Z_t[:, 0]
        Z_corr_t_1 = L10 * Z_t[:, 0] + L11 * Z_t[:, 1]
        Z_corr_t_2 = L20 * Z_t[:, 0] + L21 * Z_t[:, 1] + L22 * Z_t[:, 2]
        
        Z_corr_t = cp.stack([Z_corr_t_0, Z_corr_t_1, Z_corr_t_2], axis=-1) # shape: (batch_size, 3)
        
        # Compute asset returns for month t (with proper monthly volatility scaling)
        sim_vol_monthly = sim_sigma_annual * cp.sqrt(dt)
        ret_t = sim_drift_monthly + sim_vol_monthly * Z_corr_t
        asset_growth = cp.exp(ret_t)
        
        asset_values = asset_values * asset_growth
        
        # Add monthly DCA adjusted for cumulative inflation
        monthly_dca = monthly_contribution * (inf_cumulative_factors[t] / inf_cumulative_factors[0])
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
        
    # Extract final nominal values
    final_nominal_values = portfolio_values[-1]
    
    # Count failures in this batch (Nominal final value < Total cash injected)
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
severe_tail_mdd = np.percentile(global_max_drawdowns, 1) * 100.0 # L'1% dei mondi peggiori

print(f"\n--- {n_years}-YEAR STOCHASTIC MULTIVERSE RESULTS (REAL TERMS) ---")
print(f"Conservative Scenario (10th real percentile): EUR {p10_real:,.2f}")
print(f"Median Scenario (50th real percentile):      EUR {p50_real:,.2f}")
print(f"Optimistic Scenario (90th real percentile):   EUR {p90_real:,.2f}")
print(f"Expected Real Mean Value:                    EUR {mean_real:,.2f}")
print(f"Probability of Nominal Failure (Loss):       {failure_probability:.4f}%")
print(f"Average Maximum Drawdown:                  {mean_mdd:.2f}%")
print(f"Worst-Case Tail Drawdown (1st percentile):   {severe_tail_mdd:.2f}%")
print(f"Absolute Worst Drawdown across 10M paths:    {worst_absolute_mdd:.2f}%")
