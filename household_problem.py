import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)
def solve_hh_backwards(par,z_trans,r,w, vbeg_a_plus,vbeg_a,a,c,l):        
    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
    
            # a. prepare
            fac = (w*(1-par.tau_l)*par.z_grid[i_z] / par.phi_grid[i_fix])**(1/ par.nu) # *par.z_grid[i_z]*par.xi_grid[i_fix] i nævneren

            # b. use FOCs
            c_endo = (par.beta * vbeg_a_plus[i_fix, i_z])**(-1/par.sigma)
            l_endo = fac*(c_endo) **(-par.sigma / par.nu)

            # c. interpolation
            m_endo = c_endo + par.a_grid - w*(1-par.tau_l) * l_endo * par.z_grid[i_z]# *par.xi_grid[i_fix]
            m_exo = (1+r*(1-par.tau_r))*par.a_grid
                        
            #c = np.zeros(par.Nfix, par.Nz, par.Na))
            
            interp_1d_vec(m_endo, c_endo, m_exo, c[i_fix, i_z])
            #l = np.zeros((par.Nfix, par.Nz, par.Na))
            interp_1d_vec(m_endo , l_endo , m_exo , l[i_fix, i_z])
            income = w *(1-par.tau_l)* l[i_fix,i_z] *par.z_grid[i_z] #*par.xi_grid[i_fix]
            a[i_fix, i_z] = m_exo + income - c[i_fix,i_z]
            #print(a.shape)
            
            # d. refinement at borrowing constraint
            for i_a in range (par.Na) : 
                
                if a [i_fix, i_z, i_a ] < 0.0:

                    # i. binding constraint for a
                    a [i_fix, i_z, i_a ] = 0.0

                    # ii. solve FOC for l
                    li = l[i_fix, i_z, i_a]

                    it = 0
                    while True :

                        ci = (1+ r*(1-par.tau_r)) * par.a_grid[i_a] + w*(1-par.tau_l)* li * par.z_grid[i_z]# *par.xi_grid[i_fix]

                        error = li - fac * ci **(-par.sigma / par.nu)
                     
                        if np . abs (error) < par.tol_ell:
                            break
                        else :
                            derror = 1 - fac *(- par.sigma / par.nu)*ci**(-par.sigma / par.nu-1) * w *(1-par.tau_l) *par.z_grid[i_z] #*par.xi_grid[i_fix]
                            li = li - error / derror

                        it += 1
                        if it > par.max_iter_ell : raise ValueError ("too many iterations")

                    # iii . save
                    c[i_fix, i_z, i_a] = ci
                    #a[i_fix, i_z, i_a] = a
                    l[i_fix, i_z, i_a] = li
                    #print(a.shape)
                    
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

@nb.njit(parallel=True)
def solve_hh_backwards_new(par,z_trans,r,w, vbeg_a_plus,vbeg_a,a,c,l):        
    #a. prepare
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

    
            fac = ( w*(1-par.tau_l) / par.phi_grid[i_fix] ) **(1/ par.nu )

            # b. use FOCs
            c_endo = ( par.beta * vbeg_a_plus[i_fix, i_z]) **( -1/ par.sigma )
            l_endo = fac *( c_endo ) **( - par.sigma / par.nu )

            # c. interpolation
            m_endo = c_endo + par.a_grid - w*(1-par.tau_l) * l_endo
            m_exo = (1+ r*(1-par.tau_r) ) * par.a_grid
            #c = np . zeros ( par.Na )
            interp_1d_vec ( m_endo , c_endo , m_exo , c[i_fix, i_z] )
            #ell = np . zeros ( par.Na )
            interp_1d_vec ( m_endo , l_endo , m_exo , l[i_fix, i_z] )

            a[i_fix, i_z] = m_exo + w*(1-par.tau_l) * l[i_fix, i_z] - c[i_fix, i_z]

            # d. refinement at borrowing constraint
            for i_a in range ( par.Na ) :

                if a [i_fix, i_z, i_a ] < 0.0:

                    # i. binding constraint for a
                    a [i_fix, i_z, i_a ] = 0.0

                    # ii. solve FOC for ell
                    li = l [i_fix, i_z, i_a ]

                    it = 0
                    while True :

                        ci = (1+ r*(1-par.tau_r)) * par.a_grid [i_a ] + w*(1-par.tau_l) * li

                        error = li - fac * ci **( - par.sigma / par.nu )
                        if np . abs ( error ) < par.tol_ell :
                            break
                        else :
                            derror = 1 - fac *( - par.sigma / par.nu ) * ci **( - par.sigma / par.nu -1) * w*(1-par.tau_l)
                            li = li - error / derror

                        it += 1
                        if it > par.max_iter_ell : raise ValueError ('too many iterations')

                # iii . save
                c [i_fix,i_z, i_a ] = ci
                l [i_fix,i_z, i_a ] = li
