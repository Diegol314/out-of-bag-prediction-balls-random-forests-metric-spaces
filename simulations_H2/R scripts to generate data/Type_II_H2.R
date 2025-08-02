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

# create dataframe with the quantle values to interpolate from
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


# Function to obtain one Type II prob estimation for a given dataset
run_simulation <- function(sample, N, kappa) {
  # check if paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/TypeIdata/estimation_", 
  # estimation, "_N", N, "_kappa", kappa, ".csv") exists in destination folder

  if (file.exists(paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/TypeIIdata/samp_", 
                         sample, "_N", N, "_kappa", kappa, ".csv"))) {
    return()
  }
  
  t <- rnorm(1000, mean = 0, sd = 1/4)
  chi = NULL
  
  x_curve <- cosh(abs(t))
  y_curve <- sinh(abs(t)) * sign(t) * mu[1]
  z_curve <- sinh(abs(t)) * sign(t) * mu[2]
  
  # Create dataframe
  m_df <- data.frame(x_curve, y_curve, z_curve)
  
  # Compute chi and theta
  chi <- acosh(m_df[,1])
  theta <- ifelse(sign(t) >= 0, acos(m_df[,2] / sinh(chi)), 
                  2 * pi - acos(m_df[,2] / sinh(chi)))
  
  hyp_points <- lapply(seq_along(chi), 
                       function(i) r_hvmf(i, kappa, chi, theta, upper)
                       )
  
  # Convert list to matrix and add 't' column
  hyp_points <- do.call(rbind, hyp_points)
  hyp_points <- cbind(t, hyp_points)
  
  # Save the data
  write.csv(hyp_points, paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/TypeIIdata/samp_",
                               sample, "_N_", N, "_kappa",
                               kappa, ".csv", sep = ""))
}

# Generate the data
set.seed(1)
upper = 3
mu = c(1/sqrt(2), 1/sqrt(2))

############################################################################################################
# Generate the data
for (N in c(50, 100, 200, 500)){
  for (kappa in c(50, 200)){
    #Parallelization
    num_cores <- parallel::detectCores() - 4
    cl <- parallel::makeCluster(num_cores)
    
    # Export necessary variables and functions to the cluster
    clusterEvalQ(cl,{ 
      library(rotasym) 
      library(pracma) 
      library(gbutils) 
    })
    clusterExport(cl, c("f_u", "cdf", "r_u", "quantile_interpolation",
                        "calculate_qvals", "mu", "r_vMF", "r_hvmf", 
                        "upper", "N", "kappa"))
    
    # Run the simulations in parallel
    parLapply(cl, 1:1000, run_simulation, N = N, kappa = kappa)
    # Stop the cluster when finished
    stopCluster(cl)
  }
}

#hyp_points = matrix(0, nrow = length(chi), ncol = 3)
#for(i in seq_along(chi)) {
#  aux = runif(1, min=0, max = 0.98)
#  u = quantile_function(p = aux, kappa=kappa, chi=chi[i], upper=upper)
#  e_w = r_vMF(1, mu = c(cos(theta[1]), sin(theta[1])), kappa = kappa*sinh(chi[i])*sinh(u))
#  hyp_points[i,] = c(cosh(u), sinh(u)*e_w)
#}

#add t to the first column of hyp_points
# hyp_points = cbind(t, hyp_points)

# Convert the result into a matrix
# hyp_points <- do.call(rbind, hyp_points_parallel)
# Add 't' to the first column of hyp_points
# hyp_points = cbind(t, hyp_points)

# Stop the cluster after use
# stopCluster(cl)

#save the data

##


# Set up the cluster
# cl <- makeCluster(num_cores)
# 
# # Export necessary functions and variables to the cluster
# clusterExport(cl, c("f_u", "cdf", "quantile_function", "r_vMF", "mu", "kappa", "upper", "chi", "theta"))
# 
# hyp_points_parallel <- parLapply(cl, seq_along(chi), function(i) {
#   u = quantile_function(p = aux, kappa=kappa, chi=chi[i], upper=upper)
#   e_w = r_vMF(1, mu = c(cos(theta[1]), sin(theta[1])), kappa = kappa * sinh(chi[i]) * sinh(u))
#   return(c(cosh(u), sinh(u) * e_w))

