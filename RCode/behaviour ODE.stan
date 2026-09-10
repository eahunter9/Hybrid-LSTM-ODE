functions{
  vector SEIRW(real time, vector y, array[] real params) {
    vector[8] dydt;
    real lambda_parameter1;
    real SE1;
    real EI1;
    real EA1;
    real IR1;
    real AR1;
    real ID1;
    real newdeaths;
    real publicriskperception1;
    real behaviour1;
    real restrictions;
    real school;
   real move;

    
restrictions = 1;


     if(time<16){
    restrictions = 1;
 }
   else{if(time<27){
   restrictions =params[6];}

  else{if(time<120)
    {
   restrictions =params[7];}
    else{if(time<219){
      restrictions = params[8];
    }
     else{if(time < 235)
     {restrictions =params[9];}
     else{if(time < 275){restrictions = params[10];}
    else{if(time < 300){restrictions = params[11];}

     else{restrictions = params[12];}}}
     }
     }
  }
   }
  

    if(time < 12){
      school = 1;
    }else{if(time < 184){school = params[5];}
      else{if(time < 301){school = 1;}
        else{school =params[5];}}
    }
    
    
    lambda_parameter1 = restrictions*school*params[1]*(y[3]+params[2]*y[4]);
    behaviour1 = exp(-params[4]*y[8]);
    SE1 = lambda_parameter1*y[1] * behaviour1/5300000;
    EI1 = 0.2*y[2]*(1-0.25);
    EA1 =   0.2*y[2]*(0.25);
    IR1 = y[3]*0.2*(1-0.0206);
    AR1 = 0.2* y[4];
    ID1 = y[3]*0.2*(0.0206);
    newdeaths = ID1;
    publicriskperception1 = y[6] - y[8]/params[3];

    
    dydt[1] = -SE1;  //S
    dydt[2] = SE1-EI1- EA1; //E
    dydt[3] = EI1-IR1 -ID1; //I
    dydt[4] = EA1 - AR1;    //A
    dydt[5] = IR1 + AR1;  //R
    dydt[6] = ID1;        //D
    dydt[7] = newdeaths;  //Dc
    dydt[8] = publicriskperception1;  //risk

    return dydt;
  }
  
}


data {
  int<lower = 1> n_obs;
  int<lower = 1> n_params;
  int<lower = 1> n_difeq;
  int<lower = 1> n_weeks;
  
  array[(n_obs)] int Deaths;
  real t0;
  array[n_obs] int ts;
  
  

}
parameters {
  
  real<lower = 0.1, upper = 20> beta;
  
  
  real<lower = 0,upper = 1> asymptomatic_reduction;

  real<lower = 1, upper = 14> time_to_percieve_risk1;

  
  real<lower = 0, upper = 1> public_sensitivity_to_death1;

    real<lower = 0, upper = 1> school_reduction;
  real<lower = 0,upper = 1> r1;
  real<lower = 0,upper = 1> r2;
  real<lower = 0,upper = 1> r3;
  real<lower = 0,upper = 1> r4;
  real<lower = 0,upper = 1> r5;
  real<lower = 0,upper = 1> r6;
  real<lower = 0,upper = 1> r7;

}
transformed parameters{
  array[n_obs] vector[n_difeq] o; // Output from the ODE solver
  array[n_obs] real x;
  vector[n_difeq] x0;
  array[n_params] real params;
  
  // real phi;
  // phi = 1 / inv_phi;
  x0[1] = 5300000-1;
  x0[2] = 0;
  x0[3] = 1;
  x0[4] = 0;
  x0[5] = 0;
  x0[6] =0;
  x0[7] =0;
  x0[8] =0;
  
  
  
  params[1] =  beta;
  params[2] = asymptomatic_reduction;

  params[3] = time_to_percieve_risk1;

  //
    params[4] = public_sensitivity_to_death1;
    params[5] = school_reduction;


   params[6] = r1;
 params[7] = r2;
 params[8] = r3;
 params[9] = r4;
 params[10] = r5;
  params[11] = r6;
 params[12] = r7;

  
  
  o = ode_rk45(SEIRW, x0, t0, ts, params);
  x[1] =  o[1, 7]  - x0[7];
  for (i in 1:n_obs-1) {

      x[i + 1] = o[i + 1, 7] - o[i, 7] + 1e-5;
    }
    
  } 
  model {
    beta  ~ lognormal(0, 20);
    
    asymptomatic_reduction   ~ lognormal(0, 1);

    time_to_percieve_risk1   ~ lognormal(0, 1);

    
    public_sensitivity_to_death1   ~ lognormal(0, 1);
    school_reduction   ~ lognormal(0, 1);
  r1   ~ lognormal(0, 1);
  r2   ~ lognormal(0, 1);
  r3   ~ lognormal(0, 1);
 r4   ~ lognormal(0, 1);
 r5   ~ lognormal(0, 1);
  r6   ~ lognormal(0, 1);
 r7  ~ lognormal(0, 1);

    Deaths  ~ poisson(x);
    
    
    
  }
  generated quantities {
    real log_lik;
    log_lik = poisson_lpmf(Deaths|x);
    
  }
  