// 1D Normal-Uniform mixture model with MEASUREMENT ERROR.
//
// The 1D equivalent of the 2D quad_me_trunc model, designed for
// organisms/experiments where only one experiment is available.
//
// Features (matching the 2D approach):
//   - Normal null with per-pair ME inflation: sigma_obs = sqrt(sigma^2 + se^2)
//   - Smeared uniform interaction: convolved with Gaussian ME kernel
//     P(y | z ~ Uniform(a,b)) = [Phi((b-y)/se) - Phi((a-y)/se)] / (b-a)
//   - Designed for WINSORIZED data to concentrate interaction density
//
// 2-component mixture:
//   - Null: Normal(mu, sigma) with ME-inflated variance
//   - Interaction: Uniform(min_y, max_y) smeared by measurement error
//
// Output:
//   log_pZ1[n] = log P(interaction | y_n)
//   log_lik[n] = log P(y_n | theta) for WAIC computation

data {
  int<lower=0> N;
  vector[N] y;                 // delta_prime_median (GI scores)
  vector<lower=0>[N] se;      // per-pair standard error
  real min_y;                  // lower bound for uniform (winsorized)
  real max_y;                  // upper bound for uniform (winsorized)

  // Prior hyperparameters
  real mu_mu;
  real sigma_mu;
  real a_sigma;
  real b_sigma;
  real rho_theta;              // prior mean for theta (interaction fraction)
  real kappa_theta;            // prior concentration for theta
}

transformed data {
  // Precompute smeared uniform log-densities for each observation.
  // These depend only on data and are constant across HMC iterations.
  //
  // For each point n:
  //   smeared_uniform(y_n) = [Phi((max_y - y_n)/se_n) - Phi((min_y - y_n)/se_n)] / (max_y - min_y)

  vector[N] lp_smeared_unif;
  vector[N] se_sq;
  real log_width = log(max_y - min_y);

  for (n in 1:N) {
    real lcdf_min = std_normal_lcdf((min_y - y[n]) / se[n]);
    real lcdf_max = std_normal_lcdf((max_y - y[n]) / se[n]);

    lp_smeared_unif[n] = log_diff_exp(lcdf_max, lcdf_min) - log_width;
    se_sq[n] = se[n]^2;
  }
}

parameters {
  real mu;
  real<lower=0> sigma;
  real<lower=0, upper=1> theta;  // P(interaction)
}

model {
  // Priors
  mu ~ normal(mu_mu, sigma_mu);
  sigma ~ gamma(a_sigma, b_sigma);
  theta ~ beta(rho_theta * kappa_theta, kappa_theta * (1 - rho_theta));

  real log_theta = log(theta);
  real log_1mtheta = log(1.0 - theta);
  real sigma_sq = sigma^2;

  for (n in 1:N) {
    // Null component: Normal with ME-inflated variance
    real s_tot = sqrt(sigma_sq + se_sq[n]);
    real lp_null = log_1mtheta + normal_lpdf(y[n] | mu, s_tot);

    // Interaction component: smeared uniform (precomputed)
    real lp_int = log_theta + lp_smeared_unif[n];

    target += log_sum_exp(lp_null, lp_int);
  }
}

generated quantities {
  vector[N] log_pZ1;     // log P(interaction | y)
  vector[N] log_lik;     // per-observation log-likelihood (for WAIC)

  real log_theta_gq = log(theta);
  real log_1mtheta_gq = log(1.0 - theta);
  real sigma_sq_gq = sigma^2;

  for (n in 1:N) {
    real s_tot = sqrt(sigma_sq_gq + se_sq[n]);
    real lp_null = log_1mtheta_gq + normal_lpdf(y[n] | mu, s_tot);
    real lp_int = log_theta_gq + lp_smeared_unif[n];

    real log_denom = log_sum_exp(lp_null, lp_int);

    log_pZ1[n] = lp_int - log_denom;
    log_lik[n] = log_denom;
  }
}
