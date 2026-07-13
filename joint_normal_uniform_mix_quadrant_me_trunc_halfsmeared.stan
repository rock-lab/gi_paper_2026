// Joint Normal-Uniform mixture model with correlated null,
// quadrant-specific HALF-SMEARED interaction densities, and
// MEASUREMENT ERROR on the null.
//
// Compared to quad_me_trunc (v1): v1 convolves a Uniform[a,b] with the
// per-pair Gaussian ME kernel on BOTH ends of the support. Near the
// winsorized outer cap this halves the plateau density, causing
// confidently-in-quadrant pairs clipped to the edge to fail the
// 0.95 hit threshold. quad_null_me_trunc drops all smearing, which
// fixes the outer taper but creates hard-edge classification at the
// origin — pairs like (-10, 0.1) get 100% discordant when ME says
// 0.1 could easily be -0.1.
//
// This model retains origin smearing (soft quadrant boundary near 0)
// while removing outer smearing (no taper at the winsorized cap).
// For the right half-axis of y1 (support [origin_y1, max_y1], b = max-origin):
//
//   f(y1) = Phi((y1 - origin_y1) / se1) / N
//   N     = b * Phi(b/se1) + se1 * (phi(b/se1) - phi(0))
//
// Equivalently, this is the predictive density on observed y under the
// prior "true effect uniform on [0, infinity), observed with Gaussian
// ME, renormalized over the observed support [0, b]". Behavior:
//   - y near 0:  ~0.5 / N  (soft origin, same as v1)
//   - y in bulk: ~1 / N  ≈ 1/b  (matches v1 plateau)
//   - y near b:  ~1 / N  (NO taper — v1 had 0.5/N here)
//
// Only the transformed data block differs from quad_me_trunc.stan;
// parameters, model, and generated quantities blocks are identical.
//
// 5-component mixture:
//   - Null:  bivariate Normal with correlation rho (ME-inflated)
//   - Q1 (discordant):  y1 < 0, y2 > 0
//   - Q2 (alleviating): y1 > 0, y2 > 0
//   - Q3 (discordant):  y1 > 0, y2 < 0
//   - Q4 (aggravating): y1 < 0, y2 < 0

data {
  int<lower=0> N;
  vector[N] y1;
  vector[N] y2;
  vector<lower=0>[N] se1;
  vector<lower=0>[N] se2;
  real min_y1;
  real max_y1;
  real min_y2;
  real max_y2;

  real origin_y1;
  real origin_y2;

  // Prior hyperparameters
  real mu_mu;
  real sigma_mu;
  real a_sigma;
  real b_sigma;
  real rho_theta;
  real kappa_theta;
}

transformed data {
  vector[N] lp_y1_left;
  vector[N] lp_y1_right;
  vector[N] lp_y2_upper;
  vector[N] lp_y2_lower;

  vector[N] se1_sq;
  vector[N] se2_sq;

  {
    real b_y1_left  = origin_y1 - min_y1;
    real b_y1_right = max_y1 - origin_y1;
    real b_y2_upper = max_y2 - origin_y2;
    real b_y2_lower = origin_y2 - min_y2;
    real phi_0 = 1.0 / sqrt(2 * pi());  // standard normal pdf at 0

    for (n in 1:N) {
      // log-normalizer per pair per direction:
      //   log N = log(b * Phi(b/se) + se * (phi(b/se) - phi(0)))
      // N is strictly positive (monotonic in b, N(0) = 0, dN/db = Phi(b/se) > 0).
      real x_y1_left  = b_y1_left  / se1[n];
      real x_y1_right = b_y1_right / se1[n];
      real x_y2_upper = b_y2_upper / se2[n];
      real x_y2_lower = b_y2_lower / se2[n];

      real phi_y1_left  = phi_0 * exp(-0.5 * x_y1_left  * x_y1_left);
      real phi_y1_right = phi_0 * exp(-0.5 * x_y1_right * x_y1_right);
      real phi_y2_upper = phi_0 * exp(-0.5 * x_y2_upper * x_y2_upper);
      real phi_y2_lower = phi_0 * exp(-0.5 * x_y2_lower * x_y2_lower);

      real log_N_y1_left  = log(b_y1_left  * Phi(x_y1_left)
                                + se1[n] * (phi_y1_left  - phi_0));
      real log_N_y1_right = log(b_y1_right * Phi(x_y1_right)
                                + se1[n] * (phi_y1_right - phi_0));
      real log_N_y2_upper = log(b_y2_upper * Phi(x_y2_upper)
                                + se2[n] * (phi_y2_upper - phi_0));
      real log_N_y2_lower = log(b_y2_lower * Phi(x_y2_lower)
                                + se2[n] * (phi_y2_lower - phi_0));

      // Half-smeared log-density: log(Phi(signed_y / se) / N), where
      // signed_y is positive on the quadrant side (origin - y for left/lower,
      // y - origin for right/upper).
      lp_y1_left[n]  = std_normal_lcdf((origin_y1 - y1[n]) / se1[n]) - log_N_y1_left;
      lp_y1_right[n] = std_normal_lcdf((y1[n] - origin_y1) / se1[n]) - log_N_y1_right;
      lp_y2_upper[n] = std_normal_lcdf((y2[n] - origin_y2) / se2[n]) - log_N_y2_upper;
      lp_y2_lower[n] = std_normal_lcdf((origin_y2 - y2[n]) / se2[n]) - log_N_y2_lower;

      se1_sq[n] = se1[n]^2;
      se2_sq[n] = se2[n]^2;
    }
  }
}

parameters {
  real mu1;
  real mu2;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=-1, upper=1> rho;  // free parameter — tests if tight interaction bounds reduce rho
  simplex[3] mix_weights;
}

transformed parameters {
  real theta_conc = mix_weights[1] / 2.0;
  real theta_disc = mix_weights[2] / 2.0;
  real theta_null = mix_weights[3];
}

model {
  mu1 ~ normal(mu_mu, sigma_mu);
  mu2 ~ normal(mu_mu, sigma_mu);
  sigma1 ~ gamma(a_sigma, b_sigma);
  sigma2 ~ gamma(a_sigma, b_sigma);
  rho ~ normal(0, 0.3);  // weakly informative: allows data to learn, gentle pull toward 0

  mix_weights ~ dirichlet(to_vector({2.0, 2.0, 36.0}));

  real log_theta_null = log(theta_null);
  real log_theta_conc = log(theta_conc);
  real log_theta_disc = log(theta_disc);
  real sigma1_sq = sigma1^2;
  real sigma2_sq = sigma2^2;

  for (n in 1:N) {
    real s1_tot = sqrt(sigma1_sq + se1_sq[n]);
    real s2_tot = sqrt(sigma2_sq + se2_sq[n]);
    real rho_eff = rho * sigma1 * sigma2 / (s1_tot * s2_tot);

    real z1 = (y1[n] - mu1) / s1_tot;
    real z2 = (y2[n] - mu2) / s2_tot;
    real lp_null = log_theta_null
                   - log(2 * pi()) - log(s1_tot) - log(s2_tot)
                   - 0.5 * log(1 - rho_eff^2)
                   - (z1^2 - 2 * rho_eff * z1 * z2 + z2^2) / (2 * (1 - rho_eff^2));

    real lp_q1 = log_theta_disc + lp_y1_left[n]  + lp_y2_upper[n];
    real lp_q2 = log_theta_conc + lp_y1_right[n] + lp_y2_upper[n];
    real lp_q3 = log_theta_disc + lp_y1_right[n] + lp_y2_lower[n];
    real lp_q4 = log_theta_conc + lp_y1_left[n]  + lp_y2_lower[n];

    target += log_sum_exp({lp_null, lp_q1, lp_q2, lp_q3, lp_q4});
  }
}

generated quantities {
  vector[N] log_pZ1;
  vector[N] log_p_conc;
  vector[N] log_p_disc;
  vector[N] log_p_null;
  vector[N] log_p_aggr;
  vector[N] log_p_allev;
  vector[N] log_lik;  // per-observation log-likelihood (for WAIC)

  real log_theta_null_gq = log(theta_null);
  real log_theta_conc_gq = log(theta_conc);
  real log_theta_disc_gq = log(theta_disc);
  real sigma1_sq_gq = sigma1^2;
  real sigma2_sq_gq = sigma2^2;

  for (n in 1:N) {
    real s1_tot = sqrt(sigma1_sq_gq + se1_sq[n]);
    real s2_tot = sqrt(sigma2_sq_gq + se2_sq[n]);
    real rho_eff = rho * sigma1 * sigma2 / (s1_tot * s2_tot);

    real z1 = (y1[n] - mu1) / s1_tot;
    real z2 = (y2[n] - mu2) / s2_tot;
    real lp_null_n = log_theta_null_gq
                   - log(2 * pi()) - log(s1_tot) - log(s2_tot)
                   - 0.5 * log(1 - rho_eff^2)
                   - (z1^2 - 2 * rho_eff * z1 * z2 + z2^2) / (2 * (1 - rho_eff^2));

    real lp_q1 = log_theta_disc_gq + lp_y1_left[n]  + lp_y2_upper[n];
    real lp_q2 = log_theta_conc_gq + lp_y1_right[n] + lp_y2_upper[n];
    real lp_q3 = log_theta_disc_gq + lp_y1_right[n] + lp_y2_lower[n];
    real lp_q4 = log_theta_conc_gq + lp_y1_left[n]  + lp_y2_lower[n];

    real log_denom = log_sum_exp({lp_null_n, lp_q1, lp_q2, lp_q3, lp_q4});

    real lp_conc_unnorm = log_sum_exp(lp_q2, lp_q4);
    real lp_disc_unnorm = log_sum_exp(lp_q1, lp_q3);
    real lp_int_unnorm = log_sum_exp(lp_conc_unnorm, lp_disc_unnorm);

    log_pZ1[n] = lp_int_unnorm - log_denom;
    log_p_conc[n] = lp_conc_unnorm - log_denom;
    log_p_disc[n] = lp_disc_unnorm - log_denom;
    log_p_null[n] = lp_null_n - log_denom;
    log_p_aggr[n] = lp_q4 - log_denom;
    log_p_allev[n] = lp_q2 - log_denom;
    log_lik[n] = log_denom;  // log p(y_n | theta)
  }
}
