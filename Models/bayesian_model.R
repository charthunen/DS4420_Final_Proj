library(dplyr)

data <- read.csv("data/movies_clean.csv")
head(data)

feature_cols <- c(
  "log_budget", "runtime", "popularity", "vote_average", "vote_count",
  "num_genres", "num_cast", "is_major_studio", "num_companies",
  "is_summer", "is_holiday", "is_franchise", "is_english",
  "genre_action", "genre_adventure", "genre_animation", "genre_comedy",
  "genre_crime", "genre_drama", "genre_family", "genre_fantasy",
  "genre_horror", "genre_romance", "genre_science_fiction", "genre_thriller"
)

Phi <- as.matrix(data[, feature_cols])
y <- data$log_revenue
Phi_scaled <- scale(Phi)

# add bias column
Phi_scaled <- cbind(bias = 1, Phi_scaled)

# train/test split 

set.seed(42)
train_indices <- sample(1:nrow(Phi_scaled), size = round(nrow(Phi_scaled) * 0.8), replace = FALSE)

Phi_train <- Phi_scaled[train_indices, ]
Phi_test <- Phi_scaled[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

cat("Train:", nrow(Phi_train), "Test:", nrow(Phi_test), "\n")


log_likelihood <- function(y, X, beta, sigma2) {
  n <- length(y)
  resid <- y - X %*% beta
  -n/2 * log(2 * pi * sigma2) - sum(resid^2) / (2 * sigma2)
}

log_prior_beta <- function(beta, tau2 = 100) {
  p <- length(beta)
  -p/2 * log(2 * pi * tau2) - sum(beta^2) / (2 * tau2)
}

log_prior_sigma2 <- function(sigma2, a0 = 2, b0 = 1) {
  if (sigma2 <= 0) return(-Inf)
  a0 * log(b0) - lgamma(a0) - (a0 + 1) * log(sigma2) - b0 / sigma2
}

log_posterior <- function(y, X, beta, sigma2) {
  log_likelihood(y, X, beta, sigma2) +
    log_prior_beta(beta) +
    log_prior_sigma2(sigma2)
}


run_mcmc <- function(y, X, n_iter = 10000, burn_in = 2000,
                     proposal_sd_beta = 0.02, proposal_sd_sigma2 = 0.1) {

  p <- ncol(X)

  # initialize
  beta_cur <- as.numeric(solve(t(X) %*% X + diag(0.01, p)) %*% t(X) %*% y)
  resid <- y - X %*% beta_cur
  sigma2_cur <- as.numeric(sum(resid^2) / (length(y) - p))

  # storage
  beta_samples <- matrix(NA, n_iter, p)
  sigma2_samples <- numeric(n_iter)

  lp_cur <- log_posterior(y, X, beta_cur, sigma2_cur)

  for (i in 1:n_iter) {
    # propose new beta
    beta_prop <- beta_cur + rnorm(p, 0, proposal_sd_beta)
    lp_prop <- log_posterior(y, X, beta_prop, sigma2_cur)
    if (log(runif(1)) < lp_prop - lp_cur) {
      beta_cur <- beta_prop
      lp_cur <- lp_prop
    }

    # propose new sigma2
    ls_cur <- log(sigma2_cur)
    ls_prop <- ls_cur + rnorm(1, 0, proposal_sd_sigma2)
    s2_prop <- exp(ls_prop)
    lp_prop <- log_posterior(y, X, beta_cur, s2_prop)
    if (log(runif(1)) < (lp_prop + ls_prop) - (lp_cur + ls_cur)) {
      sigma2_cur <- s2_prop
      lp_cur <- lp_prop
    }

    beta_samples[i, ] <- beta_cur
    sigma2_samples[i] <- sigma2_cur

    if (i %% 2000 == 0) cat(sprintf("  Iteration %d/%d\n", i, n_iter))
  }

  keep <- (burn_in + 1):n_iter
  list(beta = beta_samples[keep, ], sigma2 = sigma2_samples[keep])
}

# still need to:

# posterior <- run_mcmc(y_train, Phi_train)
#
# predictions: sample from posterior predictive
# compare: traditional features vs traditional + digital signals
# metrics: RMSE, MAE, R-squared, 90% coverage, interval width
# plots: trace plots, posterior densities, predicted vs actual with credible intervals