library(dplyr)

setwd("/Users/bellafratantonio/Desktop/cs4420/DS4420_Final_Proj-main")
movies_data <- read.csv("data/movies_clean.csv")
head(movies_data)

feature_cols <- c(
  "log_budget", "runtime", "popularity", "vote_average", "vote_count",
  "num_genres", "num_cast", "is_major_studio", "num_companies",
  "is_summer", "is_holiday", "is_franchise", "is_english",
  "genre_action", "genre_adventure", "genre_animation", "genre_comedy",
  "genre_crime", "genre_drama", "genre_family", "genre_fantasy",
  "genre_horror", "genre_romance", "genre_science_fiction", "genre_thriller"
)

Phi <- as.matrix(movies_data[, feature_cols])
y <- movies_data$log_revenue
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

# run mcmc on traditional features
cat("Running MCMC traditional features...\n")
mcmc_trad <- run_mcmc(y_train, Phi_train)

# check convergence with trace plots
plot(mcmc_trad$beta[, 1], type = "l", main = "Convergence of beta bias", xlab = "Iteration", ylab = "beta")
plot(mcmc_trad$beta[, 2], type = "l", main = "Convergence of beta log_budget", xlab = "Iteration", ylab = "beta")
plot(mcmc_trad$sigma2, type = "l", main = "Convergence of sigma2", xlab = "Iteration", ylab = "sigma2")

# check ACF plots
acf(mcmc_trad$beta[, 1], main = "ACF beta bias")
acf(mcmc_trad$beta[, 2], main = "ACF beta log_budget")
acf(mcmc_trad$sigma2, main = "ACF sigma2")

# thin every 10th sample
thinned_beta   <- mcmc_trad$beta[seq(1, nrow(mcmc_trad$beta), by = 10), ]
thinned_sigma2 <- mcmc_trad$sigma2[seq(1, length(mcmc_trad$sigma2), by = 10)]

acf(thinned_beta[, 1], main = "ACF beta bias thinned")
acf(thinned_beta[, 2], main = "ACF beta log_budget thinned")

# posterior coefficient summary
coef_names  <- c("bias", feature_cols)
beta_means  <- colMeans(thinned_beta)
beta_ci     <- apply(thinned_beta, 2, quantile, c(0.025, 0.975))

coef_summary <- data.frame(
  feature = coef_names,
  mean    = beta_means,
  lower95 = beta_ci[1, ],
  upper95 = beta_ci[2, ]
)
print(coef_summary)

# posterior predictive mean for each test point
pred_matrix <- Phi_test %*% t(thinned_beta)
y_hat_trad  <- rowMeans(pred_matrix)

# 95% credible intervals
S <- nrow(thinned_beta)
pred_draws <- matrix(NA, nrow(Phi_test), S)
for (s in 1:S) {
  mu_s <- Phi_test %*% thinned_beta[s, ]
  pred_draws[, s] <- rnorm(nrow(Phi_test), mean = mu_s, sd = sqrt(thinned_sigma2[s]))
}
ci_lower <- apply(pred_draws, 1, quantile, 0.025)
ci_upper <- apply(pred_draws, 1, quantile, 0.975)
coverage  <- mean(y_test >= ci_lower & y_test <= ci_upper)

# metrics
rmse_trad <- sqrt(mean((y_hat_trad - y_test)^2))
mae_trad  <- mean(abs(y_hat_trad - y_test))
r2_trad   <- 1 - sum((y_test - y_hat_trad)^2) / sum((y_test - mean(y_test))^2)

cat(sprintf("RMSE: %.4f\n", rmse_trad))
cat(sprintf("MAE:  %.4f\n", mae_trad))
cat(sprintf("R2:   %.4f\n", r2_trad))
cat(sprintf("95pct CI coverage: %.1f%%\n", coverage * 100))

# predicted vs actual
plot(y_test, y_hat_trad,
     xlab = "actual log revenue", ylab = "predicted log revenue",
     main = "Bayesian traditional predicted vs actual",
     pch = 16, col = rgb(0, 0, 1, 0.4))
abline(0, 1, col = "red", lwd = 2)

# run mcmc on enriched features
data_enrich <- read.csv("data/movies_clean_enriched.csv")
enriched_cols <- c(feature_cols, "google_trends_interest")

Phi_e        <- as.matrix(data_enrich[, enriched_cols])
Phi_e_scaled <- cbind(bias = 1, scale(Phi_e))
Phi_e_train  <- Phi_e_scaled[train_indices, ]
Phi_e_test   <- Phi_e_scaled[-train_indices, ]

cat("Running MCMC enriched features...\n")
mcmc_enrich      <- run_mcmc(y_train, Phi_e_train)
thinned_beta_e   <- mcmc_enrich$beta[seq(1, nrow(mcmc_enrich$beta), by = 10), ]
thinned_sigma2_e <- mcmc_enrich$sigma2[seq(1, length(mcmc_enrich$sigma2), by = 10)]

y_hat_enrich <- rowMeans(Phi_e_test %*% t(thinned_beta_e))
rmse_enrich  <- sqrt(mean((y_hat_enrich - y_test)^2))
mae_enrich   <- mean(abs(y_hat_enrich - y_test))
r2_enrich    <- 1 - sum((y_test - y_hat_enrich)^2) / sum((y_test - mean(y_test))^2)

cat(sprintf("RMSE: %.4f\n", rmse_enrich))
cat(sprintf("MAE:  %.4f\n", mae_enrich))
cat(sprintf("R2:   %.4f\n", r2_enrich))

# comparison table
cat("\n--- traditional vs enriched ---\n")
cat(sprintf("RMSE: %.4f -> %.4f\n", rmse_trad, rmse_enrich))
cat(sprintf("MAE:  %.4f -> %.4f\n", mae_trad,  mae_enrich))
cat(sprintf("R2:   %.4f -> %.4f\n", r2_trad,   r2_enrich))
