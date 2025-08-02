library(pracma)
library(gbutils)
library(rotasym)
library(parallel)

# Density function 
f_u <- function(u, kappa, chi) {
  exp(log(kappa) + kappa - log(2) + u + log1p(-exp(-2*u)) - 
        kappa/2 * (exp(chi-u) + exp(-chi + u)) + 
        log(besselI(kappa * sinh(chi) * sinh(u), nu = 0, expon.scaled = TRUE)) 
  )
}

# Define the CDF function
cdf <- function(ub_ls, kappa, chi) {
  results <- numeric(length(ub_ls))  # Initialize a vector to store results
  # Loop through all upper bounds
  for (i in seq_along(ub_ls)) {
    ub <- ub_ls[i]
    results[i] <- integrate(f_u, lower = 0, upper = ub, kappa = kappa, 
                            chi = chi)$value
  }
  return(results)
}

# create dataframe with the quantile values to interpolate from
calculate_qvals <- function(kappa, chi, upper) {
  base = seq(0, 1-1e-3, by = 0.01)
  quantile_values <- numeric(length(base))
  for (i in seq_along(base)) {
    quantile_values[i] <- gbutils::cdf2quantile(p = base[i], lower = 0, 
                                                upper = upper, cdf = cdf, 
                                                kappa = kappa, chi = chi
    ) 
  }
  df = data.frame(base, quantile_values)
  return(df)
}

# create function that estimates the quantle by interpolating 
# from the values of df
quantile_interpolation <- function(p, df) {
  return(approx(x = df$base, y = df$quantile_values, xout = p)$y)
}

r_u = function(n, quantile_values) {
  aux = runif(n, min=0, max = 0.99)
  return(quantile_interpolation(p = aux, df = quantile_values))
}


# GEN SAMPLES FUNCTION
r_hvmf = function(i, kappa, chi, theta, upper) {
  quantile_values <- calculate_qvals(kappa = kappa, chi = chi[i], upper = upper)
  u = r_u(1, quantile_values)
  e_w = r_vMF(1, mu = c(cos(theta[i]), sin(theta[i])), kappa = 
                kappa*sinh(chi[i])*sinh(u)
  )
  return(c(cosh(u), sinh(u) * e_w))
} 

# Generate the data
set.seed(1)
upper = 3
mu = c(1/sqrt(2), 1/sqrt(2))



run_simulation <- function(N, kappa) {
  t <- rnorm(1000, mean = 0, sd = 1/4)
  
  x_curve <- cosh(abs(t))
  y_curve <- sinh(abs(t)) * sign(t) * mu[1]
  z_curve <- sinh(abs(t)) * sign(t) * mu[2]
  
  m_df <- data.frame(x_curve, y_curve, z_curve)
  
  chi <- acosh(m_df[,1])
  theta <- ifelse(sign(t) >= 0, acos(m_df[,2] / sinh(chi)), 
                  2 * pi - acos(m_df[,2] / sinh(chi)))
  
  # ---- PARALLELIZE THE SAMPLING BELOW ----
  num_cores <- parallel::detectCores() - 2  # Leave 2 cores free
  cl <- parallel::makeCluster(num_cores)
  
  # Load required libraries and export variables/functions
  parallel::clusterEvalQ(cl, {
    library(rotasym)
    library(pracma)
    library(gbutils)
  })
  
  parallel::clusterExport(cl, varlist = c("f_u", "cdf", "r_u", "quantile_interpolation",
                                                                 "calculate_qvals", "mu", "r_vMF", "r_hvmf", 
                                                                 "upper", "N", "kappa", "chi"), envir = environment())
  
  # Parallel loop: each core computes one r_hvmf sample for index i
  hyp_points <- parallel::parLapply(cl, seq_along(chi), function(i) {
    r_hvmf(i, kappa, chi, theta, upper)
  })
  
  parallel::stopCluster(cl)
  
  # Combine and save
  hyp_points <- do.call(rbind, hyp_points)
  hyp_points <- cbind(t, hyp_points)
  
  # Create TypeIdata directory if it doesn't exist
  dir.create("TypeIdata", showWarnings = FALSE, recursive = TRUE)
  
  write.csv(hyp_points,
            paste0("TypeIdata/H2_type_i_N", N, "_kappa", kappa, ".csv"),
            row.names = FALSE)
}



for (N in c(50, 100, 200, 500)) {
  for (kappa in c(50, 200)) {
    run_simulation(N, kappa)
  }
}


