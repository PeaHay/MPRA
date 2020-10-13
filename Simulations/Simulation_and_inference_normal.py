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

N=1.5e6  #Total cells sorted during the flow-cytometry step
BIAS_LIBRARY=False   #False if each genetic construct is homogeneously represented accross the library. True if the ratios are to be sampled from Dirichlet(1,1,...,1)
Ratio_amplification=100  #post-flowcytometer step
BINS=16  #Number of bins in your flow-cytometer
FLUORESCENCE_MAX=10**6 #Maximum Fluorescence measurement of the Flow cytometer
BUDGET_READS=1e7  #Number of reads to allocate. Budget
SORTING_TO_INFINITY=False

##############################################################################################################
###################################  Load & Augment Data #####################################################


df=pd.read_csv('Library_normal.csv').iloc[:1500,:]
A=(df.iloc[:,0]).values
B=(df.iloc[:,1]).values
Diversity=len(df)

#a few convenient variables related to the binning for the code
Part_conv=np.log(np.logspace(0,np.log10(FLUORESCENCE_MAX),BINS+1))  #Equally partitioning the fluorescence interval in log-space.Each entry is the lower bound for the fluoresnce in the bin
Mean_expression_bins=np.array([(Part_conv[j+1]+Part_conv[j])/2 for j in range(BINS)])

##############################################################################################################
###########################################  Functions   #####################################################



#This function is necessary for step 3 of the simulation algorithm. It computes the probability of a genetic construct to fall into one bin, thus enabling to simulate the sorting matrix 
# We're now examining the modified fluoerscence distribution where the shape parameter b has been multiplied by the fluoerscence ratio kappa.
def sorting_protein_matrix_populate(i,j):
    return(stats.norm.cdf(Part_conv[j+1],loc=A[i], scale=B[i])-stats.norm.cdf(Part_conv[j],loc=A[i], scale=B[i]))


def intensity_parameter_reparameterisation(i,j,alpha,beta):  # We enforced the positive constraint on a and b by rewriting alpha=log(a) and beta=log(b)
    Number_construct=sum(Nijhat[i])
    if Nj[j]==0:
        return(0)
    else :
        probability_bin=stats.norm.cdf(Part_conv[j+1],loc=np.exp(alpha),scale=np.exp(beta))-stats.norm.cdf(Part_conv[j],loc=np.exp(alpha),scale=np.exp(beta))
        return Number_construct*probability_bin*READS[j]/Nj[j]


def data_transformation_bins(X):  #Better representation of the raw sequecing data to facilitate the computation of the MOM estimates
    X=np.ceil(X)
    X=X.astype(int)
    T=np.repeat(Mean_expression_bins,X)
    return(T)


def starting_point_binned_reparameterised(X):   #Compute empirical moments from data and return the log reparameterisation of both shape and scale parameters of the gamma distribution 
    X=np.ceil(X)
    X=X.astype(int)
    T=data_transformation_bins(X)
    if np.count_nonzero(X)==1:  #What if all the cells fall into one unique bin?
        j=np.where(X!=0)[0][0]
        mu=np.mean(T)
        std=(Part_conv[j+1]-Part_conv[j])/4
    elif not np.any(T):
        return(np.array([0,0]))
    else:
        mu=np.mean(T)
        std=np.std(T,ddof=1)
    if std<0:
        print("whats wrong with you?! an std is not negative",std)
    return np.log(np.array([mu,std]))



def neg_ll_rep(theta,construct):
    alpha=theta[0]
    beta=theta[1]
    NL=0
    i=construct
    for j in range(BINS):
        intensity=intensity_parameter_reparameterisation(i,j,alpha,beta)
        if Sij[construct,j]!=0:
            if intensity>0: #Avoid float error with np.log
                NL+=intensity-Sij[i,j]*np.log(intensity)
        else:
            NL+=intensity
    return(NL)



def is_pos_def(X):  # Test if the hessian (fisher information) is invertible
    return np.all(np.linalg.eigvals(X) > 0)

def ML_inference_reparameterised(i):
    Dataresults=np.zeros(8)
    T=Nijhat[i,:]
    if np.sum(T)!=0:     #Can we do inference? has the genetic construct been sequenced?
        Dataresults[7]=(T[0]+T[-1])/np.sum(T) #Scoring of the data- How lopsided is the read count? all on the left-right border?
        a,b=starting_point_binned_reparameterised(T)
        #The four next lines provide the MOM estimates on a,b, mu and sigma
        Dataresults[4]=np.exp(a) #value of mu MOM
        Dataresults[5]=np.exp(b) #Value of sigma MOM
        if np.count_nonzero(T)==1: #is there only one bin to be considered? then naive inference
            Dataresults[6]=3 #Inference grade 3 : Naive inference
        else:  #in the remaining case, we can deploy the mle framework to improve the mom estimation
            res=minimize(neg_ll_rep,starting_point_binned_reparameterised(T),args=(i),method="Nelder-Mead")
            c,d=res.x
            Dataresults[0]=np.exp(c) #value of mu, MLE
            Dataresults[1]=np.exp(d) #value of sigma squared, MLE
            fi = lambda x: neg_ll_rep(x,i)
            fdd = nd.Hessian(fi) 
            hessian_ndt=fdd([c, d])
            if is_pos_def(hessian_ndt)==True:
                inv_J=np.linalg.inv(hessian_ndt)
                e,f=np.sqrt(np.diag(np.matmul(np.matmul(np.diag((np.exp(c),np.exp(d))),inv_J),np.diag((np.exp(c),np.exp(d))))))
                Dataresults[2]=e
                Dataresults[3]=f
                Dataresults[6]=1 #Inference grade 1 : ML inference  successful
            else:
                Dataresults[6]=2 #Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
    else:
        Dataresults[6]=4   #Inference grade 4: No inference is possible
    return(Dataresults)

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

df['Ni_unknown']=Ni
df['reliable_amount_of_cells_unknown']=df.apply(lambda row: 1 if (row['Ni_unknown']> Cell_sample_size_needed) else 0, axis=1)

#### STEP 3 - Compute binning

## Compute ratios qji
Qij=np.fromfunction(sorting_protein_matrix_populate, (Diversity, BINS), dtype=int)
if SORTING_TO_INFINITY==True:
    Qij[:,-1]=1-np.cumsum(Qij,axis=1)[:,-2] #Compensate for right-border effect (The flow-cytometer collects all the remaining cells, effectively 'sorting to infinity'

## Compute Nij
Nij=Qij* Ni[:, np.newaxis]  
Nij=np.floor(Nij) #Convert to Integer numbers

#### STEP 4 - Compute Nj

Nj=np.sum(Nij, axis=0)   #Number of cell sorted in each fraction
df['Mixture_number_unknown']=np.sum(Nij,axis=1)
#### STEP 5 - PCR amplification

Nij_amplified=np.multiply(Nij,Ratio_amplification)

#### STEP 6 - Compute Reads allocation
N=sum(Nj)
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



df['Estimation_mixture_number']=np.sum(Nijhat, axis=1) 
df['Discrepancy_ratio']=df.apply(lambda row: (row['Estimation_mixture_number']-row['Ni_unknown'])/(row['Ni_unknown']+0.001), axis=1)
df['Population_reliable_unknown']=df.apply(lambda row:  0 if (row['Discrepancy_ratio']>0.2) | (row['Discrepancy_ratio']<-0.2) else 1, axis=1)
Sij=Sij.astype(int)
df['Sequencing_depth']=np.sum(Sij,axis=1)


#Parallel computing
Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(ML_inference_reparameterised)(i)for i in range(Diversity))
df_results= pd.DataFrame(Data_results)
df_results.rename(columns={0: "mu_MLE", 1: "sigma_MLE", 2: "mu_std",3: "sigma_std",4: "mu_MOM", 5: "sigma_MOM", 6: "Inference_grade",7: "Score"}, errors="raise",inplace=True)


df_results['mu_gt']=df.iloc[:,0]
df_results['sigma_gt']=df.iloc[:,1]


df_results.to_csv('ISWORK.csv', index=False)

