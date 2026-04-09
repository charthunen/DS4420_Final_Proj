# Movie Box Office Revenue Prediction

DS4420 Final Project — Predicting movie box office revenue using neural networks and Bayesian regression, with and without Google Trends search interest data.

## Research Questions

1. Can production metadata reliably predict a movie's log revenue?
2. Does pre-release Google Trends search interest improve prediction accuracy?
3. How do a manually-implemented MLP and a Bayesian regression model compare in performance and uncertainty quantification?

## Methods

| Method | Language | Implementation |
|--------|----------|----------------|
| Multi-Layer Perceptron (MLP) | Python | Manual (NumPy only — no Scikit-learn for modeling) |
| Bayesian Regression (MCMC) | R | Manual Metropolis-Hastings sampler |

Both methods are trained on two feature sets:
- **Traditional**: 25 production/metadata features
- **Enriched**: Traditional + Google Trends pre-release search interest

## Data

**Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) + Google Trends via `pytrends`

**After cleaning:** 2,276 movies (budget > $1,000, revenue > $0, released 2000+, valid runtime)

**Features (25 total):**
- `log_budget`, `runtime`, `popularity`, `vote_average`, `vote_count`
- `num_genres`, `num_cast`, `is_major_studio`, `num_companies`
- `is_summer`, `is_holiday`, `is_franchise`, `is_english`
- Genre indicators: action, adventure, animation, comedy, crime, drama, family, fantasy, horror, romance, science fiction, thriller
- `google_trends_interest` (enriched only) — average search interest in the 28 days before release

**Target:** `log_revenue` (log1p-transformed box office revenue)

## Repository Structure

```
├── data/
│   ├── tmdb_5000_movies.csv          # Raw TMDB movie metadata
│   ├── tmdb_5000_credits.csv         # Raw cast/crew data
│   ├── data_preperation.py           # Cleaning, feature engineering, CSV export
│   ├── fetch_trends.py               # Google Trends scraper (with caching)
│   ├── movies_clean.csv              # Cleaned dataset (traditional features)
│   └── movies_clean_enriched.csv     # Cleaned dataset + Google Trends feature
├── models/
│   ├── mlp_model.py                  # MLP: manual backprop, trains both feature sets
│   └── bayesian_model.R              # Bayesian regression: Metropolis-Hastings MCMC
└── DS4420 Literature Review.pdf
```

## Reproducing the Results

### 1. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn pytrends
```

### 2. Prepare the data

```bash
# From the project root:
python data/data_preperation.py
```

This reads `data/tmdb_5000_movies.csv` and `data/tmdb_5000_credits.csv`, engineers features, and writes `data/movies_clean.csv`. If `data/trends_cache.csv` is present, it also writes `data/movies_clean_enriched.csv`.

To fetch Google Trends data (optional, rate-limited):

```bash
python data/fetch_trends.py
# Then re-run data_preperation.py to merge the trends cache
```

### 3. Run the MLP model (Python)

```bash
python models/mlp_model.py
```

Trains the MLP on both feature sets and prints RMSE, MAE, and R² for each. Displays a training loss convergence plot.

### 4. Run the Bayesian model (R)

Open `models/bayesian_model.R` in RStudio or run:

```r
source("models/bayesian_model.R")
```

Runs Metropolis-Hastings MCMC (10,000 iterations, 2,000 burn-in, thinned by 10) on both feature sets. Outputs trace plots, ACF plots, posterior coefficient summaries with 95% credible intervals, and RMSE/MAE/R² metrics.

**Required R packages:** `dplyr`

## Model Details

### MLP (Python)
- Architecture: input → 32 hidden units (ReLU) → 1 output (linear)
- He weight initialization
- Batch gradient descent, learning rate 0.001, 500 epochs
- Loss: MSE

### Bayesian Regression (R)
- Likelihood: Gaussian
- Prior on β: Normal(0, τ²=100)
- Prior on σ²: Inverse-Gamma(a₀=2, b₀=1)
- Sampler: random-walk Metropolis-Hastings
- Outputs posterior means and 95% credible intervals for all coefficients
- Reports empirical coverage of 95% predictive credible intervals on test set

## Train/Test Split

80/20 split with `random_state=42` (Python) and `set.seed(42)` (R) for consistency across models.
