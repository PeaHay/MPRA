import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import scipy.stats as stats 
from scipy.special import gamma, factorial,digamma
import numdifftools as nd
from scipy.optimize import minimize 
from joblib import Parallel, delayed



##############################################################################################################
#####################################  Experimental Parameters  ##############################################

FLUORESCENCE_MAX=10**6
BINS=16
Partition=np.logspace(0,np.log10(FLUORESCENCE_MAX),BINS)  #Equally partitioning the fluorescence interval in log-space.Each entry is the upper bound for the fluoresnce in the bin
N=141694798 #Total cells sorted during the flow-cytometry step
Part_conv=np.insert(Partition,0,0) #Initialise fluorescence at zero.
Mean_expression_bins=np.array([(Part_conv[j+1]+Part_conv[j])/2 for j in range(BINS)])
Stoech=10
kappa=10
##############################################################################################################
############################################  Load Data  #####################################################

Nj=np.load('Nj_merged.npy') #FACS events in each bin ( Number of cells sorted in each bin)
Sij=np.load('Sij_merged.npy')  #Filtered Read Counts for each genetic construct (one row) in each bin (one column)
READS=np.array([ 1460332.,  2109815.,  2335533.,  3210865.,  4303324.,  5864139.,
        7490610.,  9922865., 12976416., 15188644., 19094267., 23689418.,
       23664179., 21895118., 17576043.,  5519053.])
Sij=Sij.astype(int)


##############################################################################################################
########################################  Auxiliary Values ###################################################

if np.any(READS==0):
   Enrich=Nj/(READS+0.001) 
   print('The number of reads allocated in one bin is suprisingly 0! are you sure?') 
else:
    Enrich=Nj/READS
Nihat=np.multiply(Sij,Enrich)
Nihat=np.around(Nihat)
Nihat=Nihat.astype(int)
Ni=Nihat.sum(axis=1)



##############################################################################################################
###########################################  Functions   #####################################################

# Compute Poisson intensity parameter

# Compute Poisson intensity parameter
def intensity_parameter(i,j,a,b):
    Number_construct=Ni[i]
    if Nj[j]==0:
        return(0)
    else :
        if j==BINS-1:
            probability_bin=1-stats.gamma.cdf(Part_conv[j],a,scale=kappa*b)
        else:
            probability_bin=stats.gamma.cdf(Part_conv[j+1],a,scale=kappa*b)-stats.gamma.cdf(Part_conv[j],a,scale=kappa*b) 
        return Number_construct*probability_bin*READS[j]/Nj[j]

def intensity_parameter_reparameterisation(i,j,alpha,beta):  # We enforced the positive constraint on a and b by rewriting alpha=log(a) and beta=log(b)
    Number_construct=Ni[i]
    if Nj[j]==0:
        return(0)
    else :
        if j==BINS-1:
            probability_bin=1-stats.gamma.cdf(Part_conv[j],np.exp(alpha),scale=kappa*np.exp(beta))
        else:
            probability_bin=stats.gamma.cdf(Part_conv[j+1],np.exp(alpha),scale=kappa*np.exp(beta))-stats.gamma.cdf(Part_conv[j],np.exp(alpha),scale=kappa*np.exp(beta)) 
        return Number_construct*probability_bin*READS[j]/Nj[j]


def data_transformation_bins(X):  #New representation of the data enabling the method of moments
    X=np.ceil(X)
    X=X.astype(int)
    T=np.repeat(Mean_expression_bins,X)
    return(T)

def starting_point_binned(X):   #Empirical moments enabled starting point
    X=np.ceil(X)
    X=X.astype(int)
    T=data_transformation_bins(X)
    if np.count_nonzero(X)==1:  #What if all the cells fall into one unique bin?
        j=np.where(X!=0)[0][0]
        ab=np.mean(T)/kappa
        abb=(Mean_expression_bins[j]-Mean_expression_bins[j-1])**2/(kappa**2)
    elif not np.any(T):
        return(np.array([0,0]))
    else:
        ab=np.mean(T)/kappa
        abb=np.var(T,ddof=1)/(kappa**2)
    return np.array([(ab**2)/abb,abb/ab])


def starting_point_binned_reparameterised(X):   #Empirical moments enabled starting point
    X=np.ceil(X)
    X=X.astype(int)
    T=data_transformation_bins(X)
    if np.count_nonzero(X)==1:
        j=np.where(X!=0)[0][0]
        ab=np.mean(T)/kappa
        abb=(Mean_expression_bins[j]-Mean_expression_bins[j-1])**2/(kappa**2)
    elif not np.any(T):
        return(np.array([0,0]))
    else:
        ab=np.mean(T)/kappa
        abb=np.var(T,ddof=1)/(kappa**2)
    return np.log(np.array([(ab**2)/abb,abb/ab]))



def neg_ll_reg_rep(theta,construct):
    alpha=theta[0]
    beta=theta[1]
    NL=0
    i=construct
    #if a>30 or b>20000:
        #NL=2000
    for j in range(BINS):
        intensity=intensity_parameter_reparameterisation(i,j,alpha,beta)
        if Sij[construct,j]!=0:
            intensity+=1e-300 #Avoid float error with np.log
            NL+=intensity-Sij[i,j]*np.log(intensity)
        else:
            NL+=intensity
    #NL+=((np.exp(alpha)/20)**2+(np.exp(beta)/2000)**2)*50
    #NL+=((1e-2/np.exp(alpha))**2+(5e-1/np.exp(beta))**2)*50
    return(NL)



def neg_ll(theta,construct):
    a=theta[0]
    b=theta[1]
    NL=0
    i=construct
    #if a>30 or b>20000:
        #NL=2000
    for j in range(BINS):
        intensity=intensity_parameter(i,j,a,b)
        if intensity>1e-15:
            #if Sij[construct,j]!=0:
            NL+=intensity-Sij[i,j]*np.log(intensity)
    return(NL)


def is_pos_def(X):
    return np.all(np.linalg.eigvals(X) > 0)

def ab_to_mu_sigmasquared(a,b):
    return np.array([a*b,a*b*b])

def matrix_delta(a,b):
    return np.array([[b,a],[b**2,2*a*b]])


def ML_inference_reparameterised(i):
    Dataresults=np.zeros(14)
    T=Nihat[i,:]
    if np.sum(T)!=0:     #Can we do inference? has the genetic construct been sequenced?
        Dataresults[13]=(T[0]+T[-1])/np.sum(T) #Scoring of the data- How lopsided is the read count? all on the left-right border?
        alpha,beta=starting_point_binned_reparameterised(T)
        #The four next lines provide the MOM estimates on a,b, mu and sigma
        Dataresults[8]=np.exp(alpha) #value of a
        Dataresults[9]=np.exp(beta)
        Dataresults[10]=ab_to_mu_sigmasquared(np.exp(alpha),np.exp(beta))[0] #value of mu
        Dataresults[11]=ab_to_mu_sigmasquared(np.exp(alpha),np.exp(beta))[1] #value of sigma
        if np.count_nonzero(T)==1: #is there only one bin to be considered? then naive inference
            Dataresults[12]=3 #Inference grade 3 : Naive inference
        else:  #in the remaining case, we can deploy the mle framework to imporve the mom estimation
            res=minimize(neg_ll_reg_rep,starting_point_binned_reparameterised(T),args=(i),method="Nelder-Mead")
            c,d=res.x
            Dataresults[0]=np.exp(c) #value of a
            Dataresults[1]=np.exp(d)#value of b
            Dataresults[4]=ab_to_mu_sigmasquared(np.exp(c),np.exp(d))[0] #value of a
            Dataresults[5]=ab_to_mu_sigmasquared(np.exp(c),np.exp(d))[1]
            fi = lambda x: neg_ll(x,i)
            fdd = nd.Hessian(fi) 
            hessian_ndt=fdd([np.exp(res.x[0]), np.exp(res.x[1])])
            if is_pos_def(hessian_ndt)==True:
                inv_J=np.linalg.inv(hessian_ndt)
                e,f=np.sqrt(np.diag(inv_J))
                g,h=np.sqrt(np.diag(np.matmul(np.matmul(matrix_delta(res.x[0], res.x[1]),inv_J),matrix_delta(res.x[0], res.x[1]).T)))
                Dataresults[2]=e
                Dataresults[3]=f
                Dataresults[6]=g
                Dataresults[7]=h
                Dataresults[12]=1 #Inference grade 1 : ML inference  successful
            else:
                Dataresults[12]=2 #Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
    else:
        Dataresults[12]=4   #Inference grade 4: No inference is possible
    return(Dataresults)

##############################################################################################################
###########################################  Inference   #####################################################

Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(ML_inference_reparameterised)(i)for i in range(150000))
Data_results=np.array(Data_results)


##############################################################################################################
#########################################  Save Results   ####################################################

df= pd.DataFrame(Data_results)
df.rename(columns={0: "a_MLE", 1: "b_MLE", 2: "a_std",3: "b_std", 8: "a_MOM", 9: "b_MOM", 12: "Inference_grade",4: "mu_MLE", 5: "sigma_squared_ML", 6: "mu_std",7: "sigma_std", 10: "mu_MOM", 11: "sigma_sqaured_MOM", 13: "Score"}, errors="raise",inplace=True)
df.to_csv('Cambray_results_merged.csv', index=False)
               



