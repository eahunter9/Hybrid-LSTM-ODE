stan_data <- function(vars_vector, type, inits = TRUE) {
  
  if(length(vars_vector) != length(type)) {
    stop("Different length sizes between 'vars_vector' & 'type' pars",
         call. = FALSE)
  }
  
  obj_list <- Map(c, type, vars_vector)
  
  data_declaration_list <- lapply(obj_list, function(var_obj) {
    
    stringr::str_glue(" array[n_obs] {var_obj[[1]]} {var_obj[[2]]};")
  })
  
  data_declaration <- paste(data_declaration_list, collapse = "\n")
  
  stan_d <- paste(
    "data {",
    "  int<lower = 1> n_obs;",
    "  int<lower = 1> n_params;",
    "  int<lower = 1> n_difeq;",
    data_declaration,
    "  real t0;",
    "  array[n_obs] real ts;", sep = "\n")
  
  if(inits) {
    stan_d <- paste(stan_d, "  vector[n_difeq] y0;", sep = "\n")
  }
  
  paste(stan_d, "}", sep = "\n")
}
