import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    for i in range(par.Nfix):
        ss.z_trans[i,:,:] = z_trans
        ss.Dz[i,:] = z_ergodic/par.Nfix*1.0 #DEPENDS !!!!!!!
        ss.Dbeg[i,:,0] = ss.Dz[i,:]
        ss.Dbeg[i,:,1:] = 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    
    y = (1-par.tau_l)*ss.w*par.z_grid
    c = m = (1+ss.r*(1-par.tau_r))*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r*(1-par.tau_r))*c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(KL_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss
    
    ss.KL = KL_ss
    
    # Step 1
    ss.rk = par.alpha*par.Gamma_ss*((ss.KL)**(par.alpha-1.0))
    
    # Step 2
    ss.w = (1.0-par.alpha)*par.Gamma_ss*((ss.KL)**par.alpha)
    
    #Step 3
    ss.r = ss.rk - par.delta
    
    #Step 4
    if do_print:

        print(f'guess {ss.KL = :.4f}')    
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    
    # for i in range(par.Nz):
    #    for j in range(par.Nfix):
    #    ss.l[:,i,:] = ss.l[:,i,:]*par.z_grid[i]
    #        ss.l[j,:,:] = ss.l[j,:,:]*par.xi_grid[j]
    
    ss.L = np.sum(ss.l*ss.D)

    #Step 5
    ss.B = (ss.A_hh*ss.r*par.tau_r + ss.w*par.tau_l*ss.L-par.G_ss)/ss.r 
    
    # step 6
    ss.L = ss.L_hh
    
    # step 7
    ss.K = ss.L*ss.KL
    
    ss.clearing_A = ss.K + ss.B - ss.A_hh

    return ss.clearing_A # target to hit

def find_ss(model,method='direct',do_print=False,KL_min=0.8602648066955623,KL_max=2.9208221499640707,NK=10):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,KL_min=KL_min,KL_max=KL_max,NK=NK)
    elif method == 'indirect':
        find_ss_indirect(model,do_print=do_print)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,do_print=False,KL_min=1.0,KL_max=10.0,NK=10):
    """ find steady state using direct method """

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    KL_ss_vec = np.linspace(KL_min,KL_max,NK) # trial values
    clearing_A = np.zeros(KL_ss_vec.size) # asset market errors

    for i,KL_ss in enumerate(KL_ss_vec):
        
        try:
            clearing_A[i] = obj_ss(KL_ss,model,do_print=do_print)
        except Exception as e:
            clearing_A[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_A = {clearing_A[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    KL_max = np.min(KL_ss_vec[clearing_A < 0])
    KL_min = np.max(KL_ss_vec[clearing_A > 0])

    if do_print: print(f'K in [{KL_min:12.8f},{KL_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,KL_min,KL_max,args=(model,),do_print=do_print,
        varname='KL_ss',funcname='A_hh-K-B'
    )

def find_ss_indirect(model,do_print=False):
    """ find steady state using indirect method """

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    ss.L = 1.0
    ss.r = par.r_ss_target
    ss.w = par.w_ss_target

    assert (1+ss.r)*par.beta < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    # b. stock and capital stock from household behavior
    model.solve_hh_ss(do_print=do_print) # give us ss.a and ss.c (steady state policy functions)
    model.simulate_hh_ss(do_print=do_print) # give us ss.D (steady state distribution)
    if do_print: print('')

    ss.K = ss.A_hh = np.sum(ss.a*ss.D)
    
    # c. back technology and depreciation rate
    ss.Gamma = ss.w / ((1-par.alpha)*(ss.K/ss.L)**par.alpha)
    ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1)
    par.delta = ss.rk - ss.r

    # d. remaining
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)
    ss.C = ss.Y - par.delta*ss.K
    ss.C_hh = np.sum(ss.D*ss.c)

    # e. print
    if do_print:

        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied Gamma = {ss.Gamma:6.3f}')
        print(f'Implied delta = {par.delta:6.3f}') # check is positive
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in K-A_hh = {ss.K-ss.A_hh:12.8f}') # = 0 by construction
        print(f'Discrepancy in C-C_hh = {ss.C-ss.C_hh:12.8f}\n') # != 0 due to numerical error 
