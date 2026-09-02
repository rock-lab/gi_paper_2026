// Per-guide-pair two-line (broken-stick) fitness model that predicts Y25.
// gi_scoring.py loads and compiles THIS file as its default two-line model
// (override with --stan_model / stan_model_path).

functions {
  vector get_twoline_mean(vector x, array[] int guides, vector alpha_l, vector beta_l, vector gamma, vector beta_e) {
    int N = num_elements(x);
    vector[N] mu_ii;
    for (i in 1:N) {
      // Apply 2-line model portions depending if data is before/after gamma
      if (x[i] <= gamma[guides[i]]) {
        mu_ii[i] = alpha_l[guides[i]] + (beta_l[guides[i]] * x[i]);
      } else {
        mu_ii[i] = (alpha_l[guides[i]] + beta_l[guides[i]] * gamma[guides[i]]) + (beta_e[guides[i]] * (x[i] - gamma[guides[i]]));
      }
    }
    return mu_ii;
  }
}

data {

  int<lower=0> N; // Number of data points
  int<lower=0> J; // Number of guides
  vector[N] y;    // The data: logfc values
  vector[N] x;    // The data (generations)
  array[N] int guides; // The guide for each data point

}

parameters {

  // Parameters for A+B
  real<lower=0.01, upper=100> nu_y;
  vector<lower=0.01, upper=100>[J] sigma;
  vector<lower=-10, upper=10>[J] alpha_l;
  vector<lower=-10, upper=10>[J] beta_e;
  vector<lower=-2, upper=2>[J] beta_l;
  vector<lower=0.01, upper=20>[J] gamma;


}


model {

  vector[N] mu_ii;

  // Priors and likelihood for A+B
  alpha_l ~ normal(0, 1);
  beta_e ~ normal(-0.2, 0.5);
  beta_l ~ normal(0, 0.2);
  gamma ~ normal(4, 2);
  sigma ~ normal(0.5, 1);
  nu_y ~ normal(3, 1);
  mu_ii = get_twoline_mean(x, guides, alpha_l, beta_l, gamma, beta_e);
  y ~ student_t(nu_y, mu_ii, sigma[guides]);

}

generated quantities {
  // Calculate rho, which is the difference between the average A_NT beta_e's and average NT_A beta_e's

  // Calculate Y25, which is the twoline model prediction for the 25th generation
  real Y25 = mean((alpha_l + beta_l .* gamma) + (beta_e .* (25.0 - gamma)));

}
