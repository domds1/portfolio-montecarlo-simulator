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

# --- Parameter Uncertainty (Multiverse Return Dispersion) ---
mu_uncertainty = cp.array([0.015, 0.008, 0.010])

# --- Return Distribution Options ---
# Choose between "normal" (Gaussian) or "t_student" (fat-tailed market crashes)
return_distribution = "t_student" 
student_t_df = 5.0            # Degrees of freedom (dof) for Student's t (lower = fatter tails)

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
all_max_drawdowns = [] # List for Maximum Drawdown tracking across batches
all_nominal_failures = 0  # Counter for failure tracking across batches

# Precompute month and year indexes for DCA inflation accumulation
months_arr = cp.arange(n_months)
year_idxs = months_arr // 12

for i in range(n_batches):
    # Generate unique structural macro parameters for the current batch
    sim_mu_annual = cp.random.normal(loc=base_mu_annual, scale=mu_uncertainty, size=(batch_size, 3))
    sim_sigma_annual = cp.tile(base_sigma_annual, (batch_size, 1))
    
    sim_inflation = cp.random.normal(loc=inflation_mean, scale=inflation_std, size=(batch_size, 1))
    sim_inflation = cp.clip(sim_inflation, inflation_min, inflation_max)
    
    # Calculate total nominal cash invested for each simulation in the batch (Initial + Inflated DCA)
    inf_factors_matrix = (1.0 + sim_inflation.squeeze()[None, :]) ** year_idxs[:, None] # shape: (n_months, batch_size)
    total_dca_nominal = cp.sum(monthly_contribution * inf_factors_matrix, axis=0) # shape: (batch_size,)
    total_cash_invested = initial_capital + total_dca_nominal
    
    # Stochastic market returns generation based on chosen distribution
    if return_distribution == "normal":
        Z = cp.random.standard_normal((n_months, batch_size, 3))
    elif return_distribution == "t_student":
        Z_raw = cp.random.standard_t(student_t_df, size=(n_months, batch_size, 3))
        scaling_factor = cp.sqrt((student_t_df - 2.0) / student_t_df)
        Z = Z_raw * scaling_factor
    else:
        raise ValueError("Invalid return_distribution. Choose 'normal' or 't_student'.")

    Z_corr = cp.matmul(Z, L.T)

    sim_drift_monthly = (sim_mu_annual - 0.5 * (sim_sigma_annual ** 2)) * dt
    sim_vol_monthly = sim_sigma_annual * cp.sqrt(dt)

    returns = sim_drift_monthly[None, :, :] + sim_vol_monthly[None, :, :] * Z_corr

    # Portfolio tracking initialization
    portfolio_values = cp.zeros((n_months + 1, batch_size))
    asset_values = initial_capital * weights[None, :]
    portfolio_values[0] = cp.sum(asset_values, axis=1)

    for t in range(n_months):
        asset_growth = cp.exp(returns[t])
        asset_values = asset_values * asset_growth

        # Add monthly DCA adjusted annually for inflation
        monthly_dca = monthly_contribution * inf_factors_matrix[t]
        asset_values += monthly_dca[:, None] * weights[None, :]
        current_portfolio = cp.sum(asset_values, axis=1)
        if (t + 1) % rebalance_frequency_months == 0 and t < n_months - 1:
            asset_values = current_portfolio[:, None] * weights[None, :]
        portfolio_values[t + 1] = current_portfolio
        
    # --- Maximum Drawdown Calculation for the batch (CuPy compatible) ---
    # CuPy doesn't support accumulate directly on maximum, so we use a cummax equivalent via maximum.reduce
    # or iterate/compute the running maximum efficiently along axis 0.
    running_peak = cp.zeros_like(portfolio_values)
    curr_max = portfolio_values[0]
    running_peak[0] = curr_max
    for t in range(1, n_months + 1):
        curr_max = cp.maximum(curr_max, portfolio_values[t])
        running_peak[t] = curr_max

    drawdowns = (portfolio_values - running_peak) / running_peak
    max_drawdowns_batch = cp.min(drawdowns, axis=0)  # Deepest drop (most negative value)
    all_max_drawdowns.append(max_drawdowns_batch.get())
        
    # Extract final nominal values
    final_nominal_values = portfolio_values[-1]
    
    # Count failures in this batch (Nominal final value < Total cash injected)
    batch_failures = cp.sum(final_nominal_values < total_cash_invested).get()
    all_nominal_failures += int(batch_failures)
    
    # Discount by specific inflation to compute real purchasing power
    total_inflation_multiplier = (1.0 + sim_inflation.squeeze()) ** n_years
    real_final_values = final_nominal_values / total_inflation_multiplier
    # Offload batch results to host CPU RAM
    all_real_final_values.append(real_final_values.get())

end_time = time.time()
print(f"Batched Multiverse simulation ({total_simulations:,} simulations) completed in {end_time - start_time:.4f} seconds on GPU.")

# --- RESULTS AGGREGATION & REPORTING (CPU NUMPY) ---
global_real_final_values = np.concatenate(all_real_final_values)
global_max_drawdowns = np.concatenate(all_max_drawdowns)

p10_real = np.percentile(global_real_final_values, 10)
p50_real = np.percentile(global_real_final_values, 50)
p90_real = np.percentile(global_real_final_values, 90)
mean_real = np.mean(global_real_final_values)
failure_probability = (all_nominal_failures / total_simulations) * 100.0
mean_mdd = np.mean(global_max_drawdowns) * 100.0 # Converted to percentage

print(f"\n--- {n_years}-YEAR PROJECTED RESULTS (REAL TERMS / PURCHASING POWER) ---")
print(f"Conservative Scenario (10th real percentile): EUR {p10_real:,.2f}")
print(f"Median Scenario (50th real percentile):      EUR {p50_real:,.2f}")
print(f"Optimistic Scenario (90th real percentile):   EUR {p90_real:,.2f}")
print(f"Expected Real Mean Value:                    EUR {mean_real:,.2f}")
print(f"Probability of Nominal Failure (Loss):       {failure_probability:.4f}%")
print(f"Average Maximum Drawdown:                  {mean_mdd:.2f}%")
