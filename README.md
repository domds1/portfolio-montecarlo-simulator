# Portfolio Monte Carlo Simulator

A GPU-accelerated Monte Carlo simulator for a monthly, three-asset portfolio. It models equity, bond, and gold returns, periodic rebalancing, recurring contributions, uncertain expected returns, stochastic inflation, and correlated market shocks. Results are aggregated across simulation paths and reported in real (inflation-adjusted) terms.

## Prerequisites

- Python 3.10 or newer.
- An NVIDIA GPU supported by CuPy, with a compatible NVIDIA driver. Install a CuPy package that matches the CUDA version supported by your system. See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for current compatibility details.
- NumPy.

The simulation allocates its random market shocks on the GPU. Required VRAM depends on the batch size, time horizon, data type, and other temporary arrays; lower `batch_size` if GPU memory is insufficient.

## Install

Create and activate a virtual environment, then install NumPy and the CuPy package matching your CUDA setup. For example, for a CUDA 12.x environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy cupy-cuda12x
```

Use the appropriate CuPy package for your installed CUDA version rather than copying the example package name blindly. On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

```bash
python MontecarloSimulation.py
```

Edit the user configuration section near the top of `MontecarloSimulation.py` to customize the simulation. The default run uses 1,000,000 paths over 20 years. `total_simulations` does not need to be an exact multiple of `batch_size`.

## Configuration

Rates and weights are decimals: `0.02` means 2%, and `0.25` means 25%. Monetary amounts are in EUR. The asset vectors all use the order **equity, bonds, gold**.

### Simulation and Time

| Parameter | Purpose | Guidance |
| --- | --- | --- |
| `total_simulations` | Number of independent portfolio paths. More paths generally make percentiles more stable, at the cost of longer runtimes. | Positive integer. |
| `batch_size` | Maximum number of paths processed together on the GPU. | Reduce this if GPU memory is insufficient; increasing it may improve throughput if VRAM allows. |
| `n_years` | Investment horizon. | Positive number of years. The script derives `n_months` as `n_years * 12`. |
| `n_months` | Number of monthly steps, derived from `n_years`. | Normally leave the calculation unchanged. |
| `dt` | Length of one monthly step in years. | Keep at `1.0 / 12.0` when simulating monthly steps. |
| `rebalance_frequency_months` | How often the portfolio is returned to the target weights. | `1` = monthly, `12` = annually, `24` = every two years. |

### Contributions and Allocation

| Parameter | Purpose | Guidance |
| --- | --- | --- |
| `initial_capital` | Starting balance invested at the beginning of the simulation. | EUR; use zero or a positive value. |
| `monthly_contribution` | First monthly deposit. Later deposits are adjusted using the modeled inflation path. | EUR; use zero or a positive value. |
| `weights` | Target allocation used for the initial investment, contributions, and rebalancing. | Three non-negative values in `[equity, bonds, gold]` order; they should sum to `1.0`. For example, `[0.60, 0.30, 0.10]` means 60%, 30%, and 10%. |

### Asset Return Assumptions

| Parameter | Purpose | Guidance |
| --- | --- | --- |
| `base_mu_annual` | Base annual expected nominal returns for equity, bonds, and gold. Each path samples its own expected returns around these values. | Three decimal rates in asset order; `0.07` means 7% per year. |
| `base_sigma_annual` | Annualized return volatility for each asset. These volatilities remain fixed across paths. | Three non-negative decimal rates in asset order; `0.15` means 15% annual volatility. |
| `mu_uncertainty` | Standard deviation used when sampling each path's expected return around `base_mu_annual`. | Three non-negative absolute rate values in asset order; `0.01` means 1 percentage point, not 1% of the base return. Set an entry to zero to disable uncertainty for that asset. |
| `return_distribution` | Distribution used for monthly asset return variation. | Choose `"normal"` for Gaussian shocks or `"t_student"` for heavier tails. |
| `student_t_df` | Degrees of freedom for Student's t returns; lower values produce heavier tails. | Used only when `return_distribution` is `"t_student"`; must be greater than 2 for the script's variance normalization. |

### Inflation and Correlation

| Parameter | Purpose | Guidance |
| --- | --- | --- |
| `inflation_long_term` | Long-run annual inflation level used as the center of the inflation process. | Decimal annual rate; `0.02` means 2%. |
| `inflation_persistence` | How strongly annual inflation carries over from the previous year in the AR(1) process. | Usually between `0` and `1`; higher values produce more persistent inflation paths. |
| `inflation_vol` | Annual standard deviation of changes in inflation. | Non-negative decimal rate; `0.01` means 1 percentage point. |
| `inflation_min`, `inflation_max` | Lower and upper bounds applied to each simulated annual inflation rate. | Decimal rates; the lower bound may be negative to allow deflation, and must be below the upper bound. |
| `base_corr_eq_bond` | Center of the monthly equity-bond correlation draws. | Unitless correlation; the script clips simulated values to `[-0.20, 0.60]`. |
| `corr_eq_bond_vol` | Scale of the monthly random variation around `base_corr_eq_bond`. | Non-negative; larger values make monthly correlations vary more. Correlations involving gold are fixed in the script. |

When changing the three-element asset vectors, update all of them in the same `[equity, bonds, gold]` order. These parameters are model assumptions, not forecasts; changing them can materially change the results.

## Model Overview

- Each path has three assets: equity, bonds, and gold. Annual expected returns are sampled per path around the configured base returns; annual volatilities are fixed at their configured values.
- Monthly asset returns use either Gaussian shocks or standardized Student's t shocks. Equity-bond correlation is sampled stochastically each month; the correlations involving gold are fixed in the script.
- Inflation follows a bounded annual AR(1) process. Monthly contributions increase with cumulative inflation, normalized so the first contribution remains equal to `monthly_contribution`.
- Contributions are invested at the configured asset weights. The portfolio is rebalanced to those weights at the configured monthly interval.

These are synthetic assumptions, not market forecasts. The script does not model fees, taxes, or other transaction costs.

## Results

The script prints the 10th, 50th, and 90th percentiles and mean of the final inflation-adjusted portfolio value, plus:

- **Probability of nominal failure:** the percentage of paths where the final nominal portfolio value is below the total nominal contributions and initial capital invested.
- **Average maximum drawdown:** the average largest peak-to-trough decline in the simulated portfolio balance. Because the balance includes ongoing contributions, deposits can reduce the measured drawdown; this is not a contribution-neutral measure of investment performance.
- **Worst-case tail drawdown:** the 1st percentile of path-level maximum drawdowns. The absolute worst drawdown is also printed.

> **Important:** Results vary between runs because no random seed is fixed. This simulator is for exploration and education; it is not a professional financial-planning tool and **does not provide financial advice**. For professional-grade tools and advice tailored to your circumstances, speak with a qualified financial professional.
