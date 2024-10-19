#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:21:27 2017
@author: michele nazareth

This algorithm solves the classification problem by considering one TT-structure associated with weighting coefficients,
by estimating each core tensor at time (sweeping procedure).
All code is written using mostly Numpy.
"""

import timeit
import dill
import pickle
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
from scipy.special import binom 


seed0 = 12345
nmax_mc = 200

# Neural network structure: generating model
stddev_nn = 2
nsamples = 10000
hiddenSize_nn = 200
nlayers = 1
stddev_noise = 0

# TT-structure:
rank_core = 12
dim_array = 3
ncomp=10

# Algorithm:
maxsweep = 50
regsweep = 4

reg_interval=np.array([2**n for n in range(-10,11,1)])
training_epochs = 50
batchsize = 100
display_step = 20

# Feature data:
feature_option = 'vand'
if feature_option=='log':
    set_feat_str = 'logarithmic'
    dim_array = 3
    
elif feature_option=='sqrt':
    set_feat_str = 'sqrt'
    dim_array = 3
    
elif feature_option=='scs':
    set_feat_str = 'Spin coherent states'
    
elif feature_option=='vand':
    set_feat_str = 'Vandermonde'
    
elif feature_option=='givenfeat':
    set_feat_str = 'given feature matrices'
    
set_data = 'random'

def tf_random_NN(samplesSize, inputSize, hiddenSize1):
    '''
    Generate a random neural network from an input (X) and output data (y).
    '''
    tf.set_random_seed(seed_mc[n_mc])
    
    def simple_nn_model():
        X = tf.random_uniform(shape=[samplesSize,inputSize], minval=-1, maxval=1)
        
        w1 = tf.random_normal(shape=[inputSize,hiddenSize1], stddev=stddev_nn)
        b1 = tf.random_normal(shape=[hiddenSize1], stddev=stddev_nn)
        layer1 = tf.add(tf.matmul(X, w1), b1)
        layer1 = tf.nn.sigmoid(layer1)
        
        w2 = tf.random_normal(shape=[hiddenSize1,1], stddev=stddev_nn)
        b2 = tf.random_normal(shape=[1], stddev=stddev_nn)
        y = tf.add(tf.matmul(layer1, w2), b2)
    
        return X, y, [w1, b1, w2, b2]

    X_input, y_input, weights = simple_nn_model()

    init = tf.global_variables_initializer()
    
    with tf.Session() as session:
        session.run(init)

        X_data, y_nn, weights_nn = session.run([X_input, y_input, weights])
        
    y_nn = y_nn.squeeze()
    y_nn += np.random.randn(y_nn.size)*stddev_noise

    data_train, data_test, target_train, target_test = train_test_split(X_data, y_nn)

    info = dict(stddev_noise=stddev_noise, stddev_nn= stddev_nn, data='random NN', coefficients=weights_nn, target=y_nn ,axes=['$x_{%d}$'%i for i in xrange(inputSize)]+['target'], training_size=data_train.shape[0], testing_size=data_test.shape[0])
        
    return data_train, data_test, target_train, target_test, info

def tt_rand_orth(n, d=None, r=2, set_dir=0):
    """
    Generate a random d-dimensional TT-vector with ranks r, 
    where n can be a number of an array of dimension d, r is a
    rank or a array of dimension d+1.
    """    
    n0 = np.asanyarray(n, dtype=np.int32)
    r0 = np.asanyarray(r, dtype=np.int32)
    if d is None:
        d = n.size
    if n0.size is 1:
        n0 = np.ones((d,), dtype=np.int32) * n0
    if r0.size is 1:
        r0 = np.ones((d + 1,), dtype=np.int32) * r0
        r0[0] = 1
        r0[d] = 1
    
    all_cores = [None]*d    
        
    if set_dir is None:
        print 'Unknown direction specified. Choose either LEFT (0) or RIGHT (1)'         
    
    if set_dir==0:
        
        for ind_c in range(d):
            vec_core = np.random.randn(r0[ind_c]*n0[ind_c]*r0[ind_c+1])
            Q, R = np.linalg.qr(vec_core.reshape(-1,r0[ind_c+1]),'reduced')
            
            if Q.shape[1]<r0[ind_c+1]:
                r0[ind_c+1] = Q.shape[1]
            
            Q = Q/np.linalg.norm(Q)
            all_cores[ind_c] = Q.reshape(r0[ind_c],n0[ind_c],r0[ind_c+1])

    else:
        
        for ind_c in range(d-1,-1,-1):
            vec_core = np.random.randn(r0[ind_c]*n0[ind_c]*r0[ind_c+1])            
            Q, R = np.linalg.qr(vec_core.reshape(-1,r0[ind_c]),'reduced')

            if Q.shape[1]<r0[ind_c]:
                r0[ind_c] = Q.shape[1]   

            Q = Q/np.linalg.norm(Q)
            all_cores[ind_c] = Q.T.reshape(r0[ind_c],n0[ind_c],r0[ind_c+1])
                
    return all_cores

def remaining_cores(set_dir, nsweep, _np1, cores_tmp, Pl_old, Pr_old, Pl_list, Pr_list):
    """
    Contraction of the left and right sides of TT-structure by including the feature transformation.
    """ 
    if _np1==0:
        Pl_k = np.ones(nsamples_train).reshape(1,-1)
        Pl_list = [None]*(ncomp-1)
        
    elif set_dir==0 or _np1==ncomp-1:
        rA2 = np.tensordot(cores_tmp[_np1-1],phi_train[_np1-1],(1,1))
        Pl_k = np.einsum('ik,ijk->jk', Pl_old, rA2)
        Pl_list[_np1-1] = Pl_k
        
    else:
        Pl_k = Pl_list[_np1-1]
        
    if nsweep==0 and _np1==0:
        Pr_list = [None]*(ncomp-1)
        _Pr = np.tensordot(cores_tmp[-1],phi_train[-1],(1,1)).squeeze()
        Pr_list[-1] = _Pr
        
        for n in range(ncomp-2,0,-1):
            rA1 = np.tensordot(cores_tmp[n],phi_train[n],(1,1))
            _Pr = np.einsum('ijk,jk->ik', rA1, _Pr)
            Pr_list[n-1] = _Pr
            
        Pr_k = Pr_list[0]
    
    elif _np1==ncomp-1:
        Pr_k = np.ones(nsamples_train).reshape(1,-1)
        Pr_list = [None]*(ncomp-1)
    
    elif set_dir==1 or _np1==0:
        rA1 = np.tensordot(cores_tmp[_np1+1],phi_train[_np1+1],(1,1))
        Pr_k = np.einsum('ijk,jk->ik', rA1, Pr_old)
        Pr_list[_np1] = Pr_k
        
    else:
        Pr_k = Pr_list[_np1]         

    return Pl_k, Pr_k, Pl_list, Pr_list

def contracting_cores(_np, cores):
    """
    Contraction of the left and right sides of TT-structure.
    """
    if _np==0:
        Gl_k=1
        
    elif _np>0:
        Gl_k = cores[0].squeeze()
        
        if _np>1:
            
            for n in range(1,_np):
                Gl_k = np.tensordot(cores[n].T, Gl_k.reshape(-1,Gl_k.shape[-1]), (-1,-1)).T
            
            Gl_k = Gl_k.reshape(-1,Gl_k.shape[-1])
    
    if _np==ncomp-1:
        Gr_k=1

    elif _np<ncomp-1:
        Gr_k = cores[ncomp-1].squeeze()
        
        if _np<ncomp-2:
            
            for n in range(ncomp-2,_np,-1):
                Gr_k = np.tensordot(cores[n], Gr_k.reshape(Gr_k.shape[0],-1), (-1,0))
            
        Gr_k = Gr_k.reshape(Gr_k.shape[0],-1).T

    return np.kron(Gl_k, Gr_k)

def core_ls(reg_previous, set_dir, nsweep, np1, np2, dim_array, delta_label, cores_tmp, Pl_old, Pr_old, Pl_list, Pr_list):
    """
    Compute estimation of the 'np1'-th core from the closed form.
    Returns: core np1 (Rnp0 x Snp1 x Rnp1)              
    """ 
    Pl, Pr, Pl_list, Pr_list = remaining_cores(set_dir, nsweep, np1, cores_tmp, Pl_old, Pr_old, Pl_list, Pr_list)
    B_k = contracting_cores(np1, cores_tmp)
    
    vec_core_mod, reg_opt = solve_ls(set_dir, reg_previous, nsweep, B_k, Pl, Pr, phi_train[np1], delta_label)    
    mat_core = vec_core_mod.reshape(phi_train[np1].shape[1],-1).T
    tens_core = np.zeros((Pl.shape[0], phi_train[np1].shape[1], Pr.shape[0]))
    
    for k in range(phi_train[np1].shape[1]): 
        tens_core[:,k,:] = mat_core[:,k].reshape(-1, Pr.shape[0])
        
    vec_core = tens_core.reshape(-1,1)
    
    if set_dir==0:
        idxQ = Pl.shape[0]*dim_array
        
        coreQ, coreR = np.linalg.qr(vec_core.reshape(idxQ,-1),'reduced')
        trunidx = coreQ.shape[1]
        
        if trunidx<rank_core:
            coreQ = coreQ[:,:trunidx]
            coreR = coreR[:trunidx,:]
            
        else:
            coreQ = coreQ[:,:rank_core]
            coreR = coreR[:rank_core,:]
        
        core_orth_tmp = coreQ.reshape(-1,dim_array,coreQ.shape[-1])        
        core_next_tmp = np.dot(coreR, cores_tmp[np2].reshape(coreR.shape[-1],-1)).reshape(coreR.shape[0],dim_array,-1)
        
    elif set_dir==1:
        idxQ = Pr.shape[0]*dim_array        
        coreQ, coreR = np.linalg.qr(vec_core.reshape(-1,idxQ).T,'reduced')
        trunidx = coreQ.shape[1]
        
        if trunidx<rank_core:
            coreQ = coreQ[:,:trunidx]
            coreR = coreR[:trunidx,:]
            
        else:
            coreQ = coreQ[:,:rank_core]
            coreR = coreR[:rank_core,:]
        
        core_orth_tmp = coreQ.T.reshape(coreQ.shape[-1],dim_array,-1)        
        core_next_tmp = np.dot(coreR, cores_tmp[np2].T.reshape(coreR.shape[-1],-1)).reshape(coreR.shape[0],dim_array,-1)

    return reg_opt, core_orth_tmp, core_next_tmp, Pl, Pr, Pl_list, Pr_list

def solve_ls(set_dir, reg_previous, nsweep, B, Pl, Pr, phi, y_input):
    '''
    Solve the problem: ||X*w - y||^2, w = (X.T*X)^⁻1 *X.T*y
    Input: If y is a vector N, y_input and y_test should have (Nx1) by reshape(-1,1).
    Output: w
    '''   
    
    def gss(a, b, tol=1e-8):
        '''
        golden section search
        to find the minimum of f on [a,b]
        f: a strictly unimodal function on [a,b]    
        '''
        gr = (np.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr 
        
        while abs(c - d) > tol:
            loss_c,_ = eval_reg(c)
            loss_d,_ = eval_reg(d)            
            
            if loss_c < loss_d:
                b = d
            
            else:
                a = c    
            
            c = b - (b - a) / gr
            d = a + (b - a) / gr            

        return (b + a) / 2

    def khatri_rao(A, B, C):
        return np.reshape(np.einsum("ik,jk,lk->ijlk", A, B, C), (A.shape[0]*B.shape[0]*C.shape[0], A.shape[1]))           
    
    y_val = y_input[:nsamples_test]
    y_train = y_input[nsamples_test:]
    
    def eval_reg(_reg):
        """
        Evaluate the solutions regarding a specific regularization factor for the regression problem.
        """        
        try: 
            theta_tmp = np.dot(np.linalg.pinv(P2_train+_reg*y_train.shape[0]*L2), Py_train)
            loss_tmp = mean_squared_error(np.dot(P_val.T,theta_tmp), y_val)+_reg*np.linalg.norm(np.dot(L,theta_tmp))**2
            
        except np.linalg.LinAlgError:
            loss_tmp = 10**10
            theta_tmp = 0  
            
        return loss_tmp, theta_tmp
    
    L = np.kron(np.eye(dim_array),B)
    B2 = np.dot(B.T, B)
    L2 = np.kron(np.eye(dim_array),B2)  

    P = khatri_rao(phi.T, Pl, Pr)
    P_val = P[:,:nsamples_test] 
    P_train = P[:,nsamples_test:] 
    P2_train = np.dot(P_train, P_train.T)
    Py_train = np.dot(P_train, y_train)
    
    if nsweep<=regsweep:        
            
        loss_tmp = np.zeros(len(reg_interval))
        theta_tmp = [None]*len(reg_interval)

        for idx_tmp, reg_tmp in enumerate(reg_interval):
            loss_tmp[idx_tmp], theta_tmp[idx_tmp] = eval_reg(reg_tmp)

        if loss_tmp.argmin()==len(reg_interval)-1:
            reg_tmp = gss(reg_interval[loss_tmp.argmin()-1], reg_interval[loss_tmp.argmin()])            

        elif loss_tmp.argmin()==0:
            reg_tmp = gss(reg_interval[loss_tmp.argmin()], reg_interval[loss_tmp.argmin()+1])

        else:
            reg_tmp = gss(reg_interval[loss_tmp.argmin()-1], reg_interval[loss_tmp.argmin()+1])         

    else:
        reg_tmp = reg_previous

    _, theta_opt = eval_reg(reg_tmp)
    
    return theta_opt, reg_tmp
    
def compute_fd(phi, cores):
    """
    Compute the decision function for each images (Ni)
    """    
    fd = np.tensordot(cores[0],phi[0],(1,1)).squeeze()
    
    for n in range(1,len(cores)):
        tens_contraction = np.tensordot(cores[n],phi[n],(1,1))
        fd = np.einsum('ij,ikj->kj', fd, tens_contraction)

    return fd.squeeze()

def eval_regr(y_true, fd_pred):
    
    return explained_variance_score(y_true, fd_pred), mean_squared_error(y_true, fd_pred)

def encode_inputs(data, feat_mat=0):
    """
    Encode features into input data (per image):
    inputs: 
        data: number of samples x number of components (Ns x Nc)
    output: phi: a list of Nc matrix elements-(Ns x dim_array)
    """      
    _nsamples, _ncomponents = data.shape
    min_value=-data.min()+10**-3

    if feature_option=='vand':
        phi_tmp = np.vander(data.reshape(-1),dim_array,increasing=True).reshape(-1,_ncomponents,dim_array)
        phi = [phi_tmp[:,n,:].squeeze() for n in range(_ncomponents)]

    elif feature_option=='log':    
        phi = [np.array([np.ones(_nsamples), data[:,n]+min_value, np.log(data[:,n]+min_value)]).T for n in range(_ncomponents)]

    elif feature_option=='sqrt':    
        phi = [np.array([np.ones(_nsamples), data[:,n]+min_value, np.sqrt(data[:,n]+min_value)]).T for n in range(_ncomponents)]

    elif feature_option=='scs':        
        phi = [(np.sqrt(binom(dim_array-1,np.arange(dim_array).reshape(-1,1)))*np.power(np.cos([np.dot(np.pi/2,data[:,n])]), dim_array-np.arange(dim_array).reshape(-1,1)-1)*np.power(np.sin([np.dot(np.pi/2,data[:,n])]),np.arange(dim_array).reshape(-1,1))).T for n in range(_ncomponents)]
        
    elif feature_option=='givenfeat':
        phi = [feat_mat[n][data[:,n]-1,:] for n in range(_ncomponents)]
    else:
        print('Set feature option.')        
    
    return phi

def sweep_estimates(y_train, cores_tmp):
    
    fd_tmp = np.zeros((nsamples_train, maxsweep))
    fd_test_tmp = np.zeros((nsamples_test, maxsweep))
    score_tmp = np.zeros((maxsweep,2))
    loss_tmp = np.zeros((maxsweep,2))
    reg_opt = np.zeros((maxsweep,ncomp-1))    
    
    Pl_k=0; Pr_k=0; Pl_list=0; Pr_list=0
    
    for nsweep in range(maxsweep):    
        set_dir = np.mod(nsweep,2)
        
        if set_dir == 0:
            seq_np = range(ncomp)
            
        else:
            seq_np = range(ncomp-1,-1,-1)
                
        count_tmp=-1
        for tmp in range(ncomp-1):
            count_tmp = count_tmp+1
            np1 = seq_np[tmp]
            np2 = seq_np[tmp+1]        
            reg_opt[nsweep,count_tmp], cores_tmp[np1], cores_tmp[np2], Pl_k, Pr_k, Pl_list, Pr_list  = core_ls(reg_opt[regsweep,:].mean(), set_dir, nsweep, np1, np2, dim_array, y_train, cores_tmp, Pl_k, Pr_k, Pl_list, Pr_list)
        
        fd_tmp[:,nsweep] = compute_fd(phi_train, cores_tmp)
        fd_test_tmp[:,nsweep] = compute_fd(phi_test, cores_tmp)
                
        score_tmp[nsweep,0], loss_tmp[nsweep,0] = eval_regr(y_train, fd_tmp[:,nsweep])
        score_tmp[nsweep,1], loss_tmp[nsweep,1] = eval_regr(y_test, fd_test_tmp[:,nsweep])        

    return reg_opt, cores_tmp, fd_tmp, fd_test_tmp, score_tmp, loss_tmp        


if __name__ == '__main__':
    
    start = timeit.default_timer()
    
    np.random.seed(seed0)
    seed_mc = np.random.randint(2**32, size=nmax_mc)
    
    score_final=[None]*nmax_mc; loss_final=[None]*nmax_mc; reg_opt=[None]*nmax_mc
    for n_mc in range(nmax_mc):        
        np.random.seed(seed_mc[n_mc])
    
        data_train, data_test, y_train, y_test, info = tf_random_NN(nsamples, ncomp, hiddenSize_nn)
            
        scaler = StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)
        
        data = np.r_[data_train, data_test]
        target = np.r_[y_train, y_test]

        if ncomp!=data.shape[1]:
            print 'problem: inconsistent number of cores.'
                    
        nsamples_train = data_train.shape[0]; nsamples_test = data_test.shape[0]

        phi_train = encode_inputs(data_train)
        phi_test = encode_inputs(data_test)
        all_cores = tt_rand_orth(dim_array,ncomp,rank_core)             
        reg_opt[n_mc], all_cores, fdecision, fdecision_test, score_final[n_mc], loss_final[n_mc] = sweep_estimates(y_train, all_cores)
        
        print 'MC: %i/%i, reg: %0.8f, loss: %f, %f, score: %f, %f' % (n_mc, nmax_mc, reg_opt[n_mc][-1,-1], loss_final[n_mc][-1,0], loss_final[n_mc][-1,1], score_final[n_mc][-1,0], score_final[n_mc][-1,1]) 
                
    stop = timeit.default_timer()
    count_time = stop - start
    
    y_train_pred = fdecision[:,-1]
    y_test_pred = fdecision_test[:,-1]
    
    score_average = np.array([np.array([score_final[idx0][idx1,:] for idx0 in range(nmax_mc)]).mean(axis=0) for idx1 in range(maxsweep)])
    loss_average = np.array([np.array([loss_final[idx0][idx1,:] for idx0 in range(nmax_mc)]).mean(axis=0) for idx1 in range(maxsweep)])
        
    print 'FINAL: count_time: %f, loss: %f, %f, score: %f, %f' % (count_time, loss_average[-1,0], loss_average[-1,1], score_average[-1,0], score_average[-1,1])

    total_coeff_NN=0
    for count in range(len(info['coefficients'])):
        total_coeff_NN += info['coefficients'][count].size
    
    total_coeff_TT = 0
    for n in range(len(all_cores)):
        total_coeff_TT += all_cores[n].size
    
    print 'total number of coefficients (NN-structure): ', total_coeff_NN
    print 'total number of coefficients (TT-structure): ', total_coeff_TT
 
    parameters = dict(nummax_mc=nmax_mc, seed0=seed0, seed_mc=seed_mc, num_samples=nsamples, num_components=ncomp)
    par_NN = dict(stddev_nn=stddev_nn, stddev_noise=stddev_noise, hiddenSize_nn=hiddenSize_nn, num_hiddenLayers_nn=nlayers, total_coeff_NN=total_coeff_NN)
    par_TT = dict(rank_core=rank_core, dim_array=dim_array, number_sweeps=maxsweep, total_coeff_TT=total_coeff_TT)
    method = dict(feature=set_feat_str, regularizer_l2=reg_opt, reg_interval_GSS=reg_interval, reg_sweep=regsweep)
    results = dict(elapsed_time=count_time, score=score_final, loss=loss_final, score_average=score_average, loss_average=loss_average)

    with open('output.pkl', 'w') as f:
        pickle.dump([parameters,par_NN,par_TT,method,results], f)

    filename = 'globalsave.pkl'
    dill.dump_session(filename)
