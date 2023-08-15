The FEM solvers here use the Ritz Galerkin approach with traditional hat shape test functions. 

Below are the five ODEs tested with their corresponding file

1. "FEM_Linear_testfunc_Ritz_Galerkin_Test_ODE"

   ODE: u_xx + u_x = -x^2

   BCs: u(0) = 0
        u_x(1) = 1

2. "FEM_Linear_testfunc_Ritz_Galerkin_Test_Autonomous_ODE"

   ODE: u_xx - u = 0

   BCs: u(0) = 0
        u_x(1) = 1

3. "FEM_linear_testfunc_RItz_Galerkin_Discontinous_Bar"

    ODE: (AEu_x +x)_x = 0
    A = 1      for 0 <= x < 0.5
    A = 0.5    for 0.5 <= x <= 1

    BCs: u(0) = 0
         u_x(1) = 1

4. "FEM_linear_testfunc_Ritz_Galerkin_Continous_Bar"

    ODE: (AEu_x +x)_x = 0
    A = 1

    BCs: u(0) = 0
         u_x(1) = 1
       
