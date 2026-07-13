// Joint Normal-Uniform mixture model across two experiments.
//
// For each gene pair i, a shared latent indicator Z_i determines whether
// the pair is interacting (Z_i=1, Uniform) or not (Z_i=0, Normal).
// Each experiment has its own Normal parameters (mu_k, sigma_k) and
// Uniform bounds (min_yk, max_yk), but the classification is shared.
//
// Output: log_pZ1[i] = log P(Z_i = 1 | y1_i, y2_i)

data {
  int<lower=0> N;
  vector[N] y1;          // delta_prime_median from experiment 1
  vector[N] y2;          // delta_prime_median from experiment 2
  real min_y1;
  real max_y1;
  real min_y2;
  real max_y2;

  // Prior hyperparameters (shared across experiments)
  real mu_mu;
  real sigma_mu;
  real a_sigma;
  real b_sigma;
  real rho_theta;
  real kappa_theta;
}

parameters {
  real mu1;
  real mu2;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=0, upper=1> theta;
}

model {
  mu1 ~ normal(mu_mu, sigma_mu);
  mu2 ~ normal(mu_mu, sigma_mu);
  sigma1 ~ gamma(a_sigma, b_sigma);
  sigma2 ~ gamma(a_sigma, b_sigma);
  theta ~ beta(rho_theta * kappa_theta, kappa_theta * (1 - rho_theta));

  for (n in 1:N) {
    // Joint likelihood with shared latent Z_n:
    //   P(y1,y2) = theta * U(y1) * U(y2) + (1-theta) * N(y1|mu1,s1) * N(y2|mu2,s2)
    real lp_interaction = uniform_lpdf(y1[n] | min_y1, max_y1)
                        + uniform_lpdf(y2[n] | min_y2, max_y2);
    real lp_null = normal_lpdf(y1[n] | mu1, sigma1)
                 + normal_lpdf(y2[n] | mu2, sigma2);
    target += log_mix(theta, lp_interaction, lp_null);
  }
}

generated quantities {
  vector[N] log_pZ1;
  for (n in 1:N) {
    real lp_int = uniform_lpdf(y1[n] | min_y1, max_y1)
                + uniform_lpdf(y2[n] | min_y2, max_y2)
                + log(theta);
    real lp_null = normal_lpdf(y1[n] | mu1, sigma1)
                 + normal_lpdf(y2[n] | mu2, sigma2)
                 + log(1.0 - theta);
    log_pZ1[n] = lp_int - log_sum_exp(lp_int, lp_null);
  }
}
