//
// Per-screen (1D) Normal / Uniform mixture for genetic-interaction calls.
//
// One screen's per-gene-pair GI scores y[n] are modeled as a two-component
// mixture:
//   - null component:        Normal(mu, sigma)      (no interaction => GI ~ 0)
//   - interaction component: Uniform(min_y, max_y)  over the winsorized support
//
// theta is the interaction fraction (Beta prior with mean rho_theta and
// concentration kappa_theta). The generated quantities block returns
// log_pZ1[n] = log P(interaction | y[n]) per pair, which
// run_per_screen_mixture.py averages over draws into prob_interaction_median.
//
// This is the no-measurement-error variant (the per-pair SE is ignored);
// univariate_normal_uniform_mix_me.stan is the measurement-error counterpart.
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  vector[N] y;
  real max_y;
  real min_y;

  // prior hyper parameters
  // For mu
  real mu_mu;    // 0
  real sigma_mu; // 1

  // for sigma
  real a_sigma;  // 2
  real b_sigma;  // 1/10.0

  // for theta
  real rho_theta;
  real kappa_theta;

}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real mu;
  real<lower=0> sigma;
  real<lower=0, upper=1> theta;

   

}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {

  mu ~ normal(mu_mu, sigma_mu);
  sigma ~ gamma(a_sigma, b_sigma);
  theta ~ beta(rho_theta*kappa_theta, kappa_theta*(1-rho_theta));
  for (n in 1:N) {
    target += log_mix(theta,
                      uniform_lpdf(y[n] | min_y, max_y),
                      normal_lpdf(y[n] | mu, sigma)); 
  };
  
}



generated quantities {
 
 vector[N] log_pZ1;
 for (n in 1:N) {
    // log_pZ1[n] = (uniform_lpdf(y[n] | min_y, max_y) + log(theta)) - log_mix(theta,
    //                   uniform_lpdf(y[n] | min_y, max_y),
    //                   normal_lpdf(y[n] | mu, sigma));
    //                   
    log_pZ1[n] =   (uniform_lpdf(y[n] | min_y, max_y) + log(theta)) -
                      log_sum_exp((normal_lpdf(y[n] | mu, sigma) + log(1.0-theta)),
                      (uniform_lpdf(y[n] | min_y, max_y) + log(theta)));
  };
 
  
}


