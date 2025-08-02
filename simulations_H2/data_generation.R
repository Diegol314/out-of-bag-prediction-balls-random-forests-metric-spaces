############################################################################################################
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
# sinh(chi) * sinh(u) = 1/4*(exp(u + chi) - exp(-u + chi) - exp(u - chi) + exp(-u - chi))

# for (chi in seq(0,1, length = 20)){
#   print(integrate(f_u, lower = 0, upper = 50, kappa = 200, chi = chi)$value)
# }

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

# create function that estimates the quantle by interpolating from the values of df
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

# Set number of cores
num_cores <- detectCores() - 4  # Use all but one core

# Main loop (parallelizing over k)
for (N in c(50, 100, 200, 500)) {
  for (kappa in c(50, 200)) {
    # Export necessary functions and variables to the cluster
    cl <- makeCluster(num_cores)
    clusterExport(cl, c("f_u", "cdf", "r_u", "quantile_interpolation", "calculate_qvals", "mu", "r_vMF", "r_hvmf", "upper", "N", "kappa"))
    parLapply(cl, seq(1, 1000, 1), function(k) {
      t <- rnorm(N, mean = 0, sd = 1/4)
      x_curve <- cosh(abs(t))
      y_curve <- sinh(abs(t)) * sign(t) * mu[1]
      z_curve <- sinh(abs(t)) * sign(t) * mu[2]
      
      # Create dataframe
      m_df <- data.frame(x_curve, y_curve, z_curve)
      
      # Compute chi and theta
      chi <- acosh(m_df[,1])
      theta <- ifelse(sign(t) >= 0, acos(m_df[,2] / sinh(chi)), 
                      2 * pi - acos(m_df[,2] / sinh(chi)))
     
      # Compute hyp_points in parallel
      hyp_points_parallel <- lapply(seq_along(chi), function(i) r_hvmf(i, kappa, chi, theta, upper))
      
      # Convert list to matrix and add 't' column
      hyp_points <- do.call(rbind, hyp_points_parallel)
      hyp_points <- cbind(t, hyp_points)
      
      # Save the data
      print(paste0("Saving data for k = ", k, (k-1) %/% 25 + 1 ))
      write.csv(hyp_points, paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/data/H2_samp", 
                                   k, "_N", N, "_kappa", kappa,
                                   "_block_", (k-1) %/% 25 + 1 ,".csv"))
      
    })
    stopCluster(cl)
  }
}
# Stop the cluster after use

# Data for the plots
set.seed(2)

for (reps_t in c(200)){ 
  for (kappa in c(50, 200)){
    hyp_points = matrix(0, nrow = 1, ncol = 4)
    for (t in c(-0.5815870, -0.1686224,  0.0000000, 0.1686224,  0.5815870)){
  # for (t in c(-0.8, -0.45,  0.0000000, 0.45,  0.8)){
      # Obtain the coordinates in hyperbolic-spherical system
      x_curve = cosh(abs(t))
      y_curve = sinh(abs(t)) * sign(t) * mu[1]
      z_curve = sinh(abs(t)) * sign(t) * mu[2]
      
      # create dataframe with x_curve in first column, y_curve in second column
      # and z_curve in third column
      m_df = data.frame(x_curve, y_curve, z_curve)
      
      # chi = abs(t) (nuestro theta de los great circles)
      # theta es el mismo número, y es igual a arccos(mu[0]) = arccos(mu[1])
      chi = acosh(m_df[,1])
      theta = ifelse(sign(t)>=0, acos(m_df[,2]/sinh(chi)), 
                     2*pi - acos(m_df[,2]/sinh(chi)) )
      quantile_values = calculate_qvals(kappa=kappa, chi = chi, upper=upper)
      u = r_u(reps_t, quantile_values)
      hyp_t_points = matrix(0, nrow = length(u), ncol = 3)
      for(i in seq_along(u)) {
        e_w = r_vMF(1, mu = c(cos(theta[1]), sin(theta[1])), 
                    kappa = kappa*sinh(chi)*sinh(u[i])
                    )
        hyp_t_points[i,] = c(cosh(u[i]), sinh(u[i])*e_w)
      }
      
      #add t to the first column of hyp_points
      hyp_t_points = cbind(rep(t, reps_t), hyp_t_points)
      hyp_points = rbind(hyp_points, hyp_t_points)
      #save the data
    }
      hyp_points = hyp_points[-1,]
      write.csv(hyp_points, paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/dibujo/H2_dibujo_repst",
                                   reps_t, "_kappa", kappa, ".csv", sep = ""))
  }
}


# Data for prediction balls

set.seed(2)

for (reps_t in c(500)){
  for (kappa in c(200)){
    hyp_points = matrix(0, nrow = 1, ncol = 4)
    for (t in c(-0.15)){
      # for (t in c(-0.8, -0.45,  0.0000000, 0.45,  0.8)){
      # Obtain the coordinates in hyperbolic-spherical system
      x_curve = cosh(abs(t))
      y_curve = sinh(abs(t)) * sign(t) * mu[1]
      z_curve = sinh(abs(t)) * sign(t) * mu[2]
      
      # create dataframe with x_curve in first column, y_curve in second column
      # and z_curve in third column
      m_df = data.frame(x_curve, y_curve, z_curve)
      
      # chi = abs(t) (nuestro theta de los great circles)
      # theta es el mismo número, y es igual a arccos(mu[0]) = arccos(mu[1])
      chi = acosh(m_df[,1])
      theta = ifelse(sign(t)>=0, acos(m_df[,2]/sinh(chi)), 
                     2*pi - acos(m_df[,2]/sinh(chi)) )
      quantile_values = calculate_qvals(kappa=kappa, chi = chi, upper=upper)
      u = r_u(reps_t, quantile_values)
      hyp_t_points = matrix(0, nrow = length(u), ncol = 3)
      for(i in seq_along(u)) {
        e_w = r_vMF(1, mu = c(cos(theta[1]), sin(theta[1])), 
                    kappa = kappa*sinh(chi)*sinh(u[i])
        )
        hyp_t_points[i,] = c(cosh(u[i]), sinh(u[i])*e_w)
      }
      
      #add t to the first column of hyp_points
      hyp_t_points = cbind(rep(t, reps_t), hyp_t_points)
      hyp_points = rbind(hyp_points, hyp_t_points)
      #save the data
    }
    hyp_points = hyp_points[-1,]
    write.csv(hyp_points, paste0("/Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_H2/dibujo/H2_dibujo_pball_repst.csv", sep = ""))
  }
}








