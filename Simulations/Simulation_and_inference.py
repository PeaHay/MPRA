### This python script is the basis to simulate MPRA and perform inference on the data generated
### You can choose your own experimental parameters in section 1, named " #####   Experimental Parameters-Insert Values  ######## "
### It requires as input 'Taniguchi_data.csv' and outputs the results of the inference and simulation: 'data_inference.csv''data_simulation.csv'
### The ML inference step is by default the log-reparameterisation of the ML formulation, you can select other likelihoods (w/wo penalty and choosing the penalty) displayed in section 2 " ####  Functions   #### "
### by modifying the function called in the last section  (" #### Inference on MPRA data ####') :  Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(ML_inference_reparameterised)(i)for i in range(Diversity)) to select the right likelihood 



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
############################  Experimental Parameters-Insert Values  #########################################

N=1E6  #Total cells sorted during the flow-cytometry step
Diversity=1018 #Number of variants in your library
BIAS_LIBRARY=False   #False if each genetic construct is homogeneously represented accross the library. True if the ratios are to be sampled from Dirichlet(1,1,...,1)
Ratio_amplification=100  #post-flowcytometer step
BINS=12  #Number of bins in your flow-cytometer
kappa=10  #Fluorescence-protein ratio. One protein shines kappa fluorescence a.u.
FLUORESCENCE_MAX=10**5 #Maximum Fluorescence measurement of the Flow cytometer
BUDGET_READS=1e7  #Number of reads to allocate. Budget

##############################################################################################################
###########################################  Functions   #####################################################

#Two functions to augment the dataset if required. 
def sample_high_regime(p):  
    logB=(4.5) * np.random.random_sample(p) + 2.7
    a=slope*logB+intercept+(np.random.beta(2, 1.8, size=p)-0.5)*8
    return(abs(a),np.exp(logB))

def sample_low_regime(p):
    A=(4.5) * np.random.random_sample(p)+0.1
    B=0.5+(np.random.beta(3.6, 2.0, size=p))/(A/4+0.15)
    return(A,abs(B))


def sorting_protein_matrix_populate(i,j):
    #This function is necessary for step 3 of the simulation algorithm. It computes the probability of a genetic construct to fall into one bin, thus enabling to simulate the sorting matrix 
    # We're now examining the modified fluoerscence distribution where the shape parameter b has been multiplied by the fluoerscence ratio kappa.
    return(stats.gamma.cdf(Part_conv[j+1],A[i], scale=kappa*B[i])-stats.gamma.cdf(Part_conv[j],A[i], scale=kappa*B[i]))

def is_pos_def(X):
    # Test if the hessian (fisher information) is invertible
    return np.all(np.linalg.eigvals(X) > 0)

def ab_to_mu_sigmasquared(a,b):
    # Map shape (a) and scale (b) to mean (mu) and variance (sigmasquared)
    return np.array([a*b,a*b*b])

def matrix_delta(a,b):
    #Gives confidence interval for mu and sigma (delta theorem)
    return np.array([[b,a],[b**2,2*a*b]])

def data_transformation_bins(X):
    #Better representation of the raw sequencing data to facilitate the computation of the MOM estimates
    X=np.ceil(X)
    X=X.astype(int)
    T=np.repeat(Mean_expression_bins,X)
    return(T)

def starting_point_binned(X):
    #Compute empirical moments from data and return the shape and scale parameters for the gamma distribution
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



def starting_point_binned_reparameterised(X): 
    #Compute empirical moments from data and return the log reparameterisation of both shape and scale parameters of the gamma distribution 
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



def intensity_parameter_reparameterisation(i,j,alpha,beta):
    # Compute Poisson intensity parameter- We enforced the positive constraint on a and b by rewriting alpha=log(a) and beta=log(b)
    Number_construct=df2['Estimation_mixture_number'][i]
    if Nj[j]==0:
        return(0)
    else :
        if j==BINS-1:
            probability_bin=1-stats.gamma.cdf(Part_conv[j],np.exp(alpha),scale=kappa*np.exp(beta))
        else:
            probability_bin=stats.gamma.cdf(Part_conv[j+1],np.exp(alpha),scale=kappa*np.exp(beta))-stats.gamma.cdf(Part_conv[j],np.exp(alpha),scale=kappa*np.exp(beta)) 
        return Number_construct*probability_bin*READS[j]/Nj[j]


def neg_ll_reg_rep(theta,construct): 
    #Compute regularised negative likelihood with the log reparameterisation of a and b (shape and scale)
    alpha=theta[0]
    beta=theta[1]
    NL=0
    i=construct
    #if a>30 or b>20000:
        #NL=2000
    for j in range(BINS):
        intensity=intensity_parameter_reparameterisation(i,j,alpha,beta)
        if Sij[construct,j]!=0:
            if intensity>0: #Avoid float error with np.log
                NL+=intensity-Sij[i,j]*np.log(intensity)
        else:
            NL+=intensity
    return(NL)

def ML_inference_reparameterised(i):
    #inference function using the log reparameterisation for the scale and shape parameters of the gamma distribution
    Dataresults=np.zeros(14)
    T=Nijhat[i,:]
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
                if (np.exp(res.x[0])>30 or np.exp(res.x[0])<1e-3 or np.exp(res.x[1])>5000 or np.exp(res.x[1]<5e-2)) :  #is inference dubious?
                    print(i)
                    Dataresults[12]=1 #Inference grade 1 : ML inference to verify...especially if a and b are negative
            else:
                Dataresults[12]=2 #Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
    else:
        Dataresults[12]=4   #Inference grade 4: No inference is possible
    return(Dataresults)

##############################################################################################################
###################################  Load & Augment Data #####################################################


df=pd.read_csv('Taniguchi_data.csv')
df.dropna(axis=0,inplace=True)

A=np.zeros(Diversity)
B=np.zeros(Diversity)
A[:1018]=df["A_Protein"]
B[:1018]=df["B_Protein"]
for i in range(1018,Diversity):
    Reg=np.random.beta(3.6, 2.0)  #introducing asymetry: most additional constructs will be part of the low noise regime
    if Reg>0.5:
        new_construct=sample_high_regime(1)
        A[i]=new_construct[0]
        B[i]=new_construct[1]
    else:
        new_construct=sample_low_regime(1)
        A[i]=new_construct[0]
        B[i]=new_construct[1]


df2=pd.DataFrame(np.column_stack((A,B)),columns=['A_Protein','B_Protein'])
df2['Noise']=df2.apply(lambda row: 1/row["A_Protein"], axis=1)
df2['Mean']=df2.apply(lambda row: row["B_Protein"]*row["A_Protein"], axis=1)
df2['variance']=df2.apply(lambda row: row['A_Protein']*(row['B_Protein']**2), axis=1)

#a few convenient variables related to the binning for the code
Partition=np.logspace(0,np.log10(FLUORESCENCE_MAX),BINS)  #Equally partitioning the fluorescence interval in log-space.Each entry is the upper bound for the fluoresnce in the bin
Part_conv=np.insert(Partition,0,0)  #More convenient binning- it now starts at F=0 a.u.
Mean_expression_bins=np.array([(Part_conv[j+1]+Part_conv[j])/2 for j in range(BINS)])


##############################################################################################################
###################################  Simulate MPRA data #####################################################

## Sorting the cells (with amplification of 1000)

#### STEP 1 - Draw the ratio p_concentration

if BIAS_LIBRARY==True:
   params=np.ones(Diversity)
   Dir=[random.gammavariate(a,1) for a in params]
   Dir=[v/sum(Dir) for v in Dir]
   # Sample from the 30,000 simplex to get ratios 
   #p_concentration=np.ones(Diversity)/Diversity
   p_concentration=Dir   
else:
   p_concentration=[1/Diversity]*Diversity

#### STEP 2 - Draw the sample sizes= of each genetic construct

Ni=np.random.multinomial(N, p_concentration, size=1)
Ni=Ni[0]

# Are there enough cells to be reasonably confident about the inference?
Cell_sample_size_needed=50

df2['Ni_unknown']=Ni
df2['reliable_amount_of_cells_unknown']=df2.apply(lambda row: 1 if (row['Ni_unknown']> Cell_sample_size_needed) else 0, axis=1)
#df2['reliable_amount_of_cells'].sum()/Diversity

#### STEP 3 - Compute binning

## Compute ratios qji
Qij=np.fromfunction(sorting_protein_matrix_populate, (Diversity, BINS), dtype=int)
Qij[:,-1]=1-np.cumsum(Qij,axis=1)[:,-2] #Compensate for right-border effect (The flow-cytometer collects all the remaining cells, effectively 'sorting to infinity'


## Compute Nij
Nij=Qij* Ni[:, np.newaxis]  #instead of using  np.array([Dir]).T
Nij=np.floor(Nij) #Convert to Integer numbers

#### STEP 4 - Compute Nj

Nj=np.sum(Nij, axis=0)   #Number of cell sorted in each fraction
df2['Mixture_number_unknown']=np.sum(Nij,axis=1)

#### STEP 5 - PCR amplification

Nij_amplified=np.multiply(Nij,Ratio_amplification)

#### STEP 6 - Compute Reads allocation

N=np.sum(Nij)
READS=np.floor(Nj*BUDGET_READS/N) #Allocate reads with repsect to the number of cells srted in each bin

#### STEP 7 - DNA sampling

Sij=np.zeros((Diversity,BINS)) 

#Compute ratios& Multinomial sampling
for j in range(BINS):
    if np.sum(Nij_amplified,axis=0)[j]!=0:
        Concentration_vector=Nij_amplified[:,j]/np.sum(Nij_amplified,axis=0)[j]
    else:
        Concentration_vector=np.zeros(Diversity)
    Sij[:,j]=np.random.multinomial(READS[j],Concentration_vector,size=1)
    



##############################################################################################################
###################################  Inference on MPRA data ##################################################


##### Auxiliary values for inference
#Normalise read counts data
Enrich=Nj/(READS+0.01)
Nijhat=np.multiply(Sij,Enrich)

df2['Estimation_mixture_number']=np.sum(Nijhat, axis=1) 
df2['Discrepancy_ratio']=df2.apply(lambda row: (row['Estimation_mixture_number']-row['Ni_unknown'])/(row['Ni_unknown']+0.001), axis=1)
df2['Population_reliable_unknown']=df2.apply(lambda row:  0 if (row['Discrepancy_ratio']>0.2) | (row['Discrepancy_ratio']<-0.2) else 1, axis=1)
Sij=Sij.astype(int)
df2['Sequencing_depth']=np.sum(Sij,axis=1)


#Parallel computing
Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(ML_inference_reparameterised)(i)for i in range(Diversity))
Data_results=np.array(Data_results)
df3= pd.DataFrame(Data_results)
df3.rename(columns={0: "a_MLE", 1: "b_MLE", 2: "a_std",3: "b_std", 8: "a_MOM", 9: "b_MOM", 12: "Inference_grade",4: "mu_MLE", 5: "sigma_squared_MLE", 6: "mu_std",7: "sigma_squared_std", 10: "mu_MOM", 11: "sigma_squared_MOM", 13: "Score"}, errors="raise",inplace=True)
df3.to_csv('data_inference.csv', index=False)
df2.to_csv('data_simulation.csv', index=False)




