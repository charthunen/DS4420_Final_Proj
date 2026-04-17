# Movie Box Office Revenue Prediction

For our DS4420 Final Project we tried to predict movie box office revenue using two ML methods: a manually implemented MLP in Python and a Bayesian regression model with Metropolis-Hastings MCMC in R. Both models are trained on production metadata features from the TMDB 5000 dataset, and again with Google Trends pre-release search interest added as an additional feature to see if it improves the predictions.

## Data

The raw data comes from the TMDB 5000 Movie Dataset on Kaggle. After removing movies with no budget/revenue, pre-2000 releases, and invalid runtimes, we have 2,276 movies. Features include log budget, runtime, popularity, vote average/count, genre indicators, major studio flag, release timing, and franchise status. Our target is log-transformed revenue. Google Trends data was collected using the pytrends library, capturing average search interest for each movie title in the 28 days before release.

## How to Run

Run python3 data/data_preperation.py first to generate the cleaned CSVs. Optionally run python3 data/fetch_trends.py to collect Google Trends data (slow, rate-limited), then re-run data_preperation.py to merge it in. Run models/mlp_model.py for the neural network results and models/bayesian_model.R for the Bayesian results.

## Structure

data/ contains all data prep scripts and CSVs. models/ contains the MLP (Python) and Bayesian model (R).
