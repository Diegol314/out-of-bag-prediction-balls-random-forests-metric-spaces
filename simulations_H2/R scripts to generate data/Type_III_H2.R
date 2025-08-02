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
r_hvmf = function(i, kappa, chi, theta, upper, quantile_values) {
  # quantile_values <- calculate_qvals(kappa = kappa, chi = chi[i], upper = upper)
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

# Generate chi values

t <- rep(qnorm(0.25, mean = 0, sd = 1/4), 1000)

x_curve <- cosh(abs(t))
y_curve <- sinh(abs(t)) * sign(t) * mu[1]
z_curve <- sinh(abs(t)) * sign(t) * mu[2]
# Create dataframe
m_df <- data.frame(x_curve, y_curve, z_curve)
# Compute chi and theta
chi <- acosh(m_df[,1])

quantile_values_50 <- calculate_qvals(kappa = 50, chi = chi[1], upper = upper)
quantile_values_200 <- calculate_qvals(kappa = 200, chi = chi[1], upper = upper)




# # Function to run a single estimation
# run_simulation <- function(estimation, N, kappa, quantile_values_50, quantile_values_200) {
#   
#   # check if file exists in destination folder
#   if (file.exists(paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/TypeIIIdata/estimation_", 
#                          estimation, "_N", N, "_kappa", kappa,".csv"))) {
#     return()
#   }
#   t <- rep(qnorm(0.25, mean = 0, sd = 1/4), 1000)
#   
#   x_curve <- cosh(abs(t))
#   y_curve <- sinh(abs(t)) * sign(t) * mu[1]
#   z_curve <- sinh(abs(t)) * sign(t) * mu[2]
#   
#   # Create dataframe
#   m_df <- data.frame(x_curve, y_curve, z_curve)
#   
#   # Compute chi and theta
#   chi <- acosh(m_df[,1])
#   theta <- ifelse(sign(t) >= 0, acos(m_df[,2] / sinh(chi)), 
#                   2 * pi - acos(m_df[,2] / sinh(chi)))
#   
#   if (kappa == 50) {
#     quantile_values <- quantile_values_50
#   } else {
#     quantile_values <- quantile_values_200
#   }
#   
#   hyp_points <- lapply(seq_along(chi), 
#                        function(i) r_hvmf(i, kappa, chi, theta, upper, 
#                                           quantile_values)
#                        )
#   
#   # Convert list to matrix and add 't' column
#   hyp_points <- do.call(rbind, hyp_points)
#   hyp_points <- cbind(t, hyp_points)
#   
#   # Save the data
#   write.csv(hyp_points, paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/TypeIIIdata/estimation_", 
#                                estimation, "_N", N, "_kappa", kappa, ".csv"))
# }
# 
# for (N in c(50, 100, 200, 500)) {
#   for (kappa in c(50, 200)) {
#     # Export necessary variables and functions to the cluster
#     # Set up the cluster outside of parLapply
#     num_cores <- parallel::detectCores() - 4  # Use all but one core
#     cl <- parallel::makeCluster(num_cores)
#     
#     # Export necessary variables and functions to the cluster
#     clusterEvalQ(cl,{ 
#       library(rotasym) 
#       library(pracma) 
#       library(gbutils) 
#     })
#     clusterExport(cl, c("f_u", "cdf", "r_u", "quantile_interpolation",  "quantile_values_50", 
#                         "quantile_values_200" ,"mu", "r_vMF", "r_hvmf", 
#                         "upper", "N", "kappa"))
#     
#     # Run the simulations in parallel
#     parLapply(cl, 1:500, run_simulation, N = N, kappa = kappa, 
#               quantile_values_50 = quantile_values_50, 
#               quantile_values_200 = quantile_values_200
#               )
#     
#     # Stop the cluster when finished
#     stopCluster(cl)
#   }
# }


run_simulation <- function(N, kappa, quantile_values_50, quantile_values_200) {
  t <- rep(qnorm(0.25, mean = 0, sd = 1/4), 1000)
  
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
  
  if (kappa == 50) {
    quantile_values <- quantile_values_50
  } else {
    quantile_values <- quantile_values_200
  }
  
  
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
    r_hvmf(i, kappa, chi, theta, upper, quantile_values)
  })
  
  parallel::stopCluster(cl)
  
  # Combine and save
  hyp_points <- do.call(rbind, hyp_points)
  hyp_points <- cbind(t, hyp_points)
  
  write.csv(hyp_points,
            paste0("/Users/Diego/Desktop/Codigo/minimal_pyfrechet/min_pyfrechet/simulations_H2/TypeIIIdata/H2_type_iii_N", 
                   N, "_kappa", kappa, ".csv"),
            row.names = FALSE)
}



for (N in c(50, 100, 200, 500)) {
  for (kappa in c(50, 200)) {
    run_simulation(N, kappa, quantile_values_50, quantile_values_200)
  }
}


