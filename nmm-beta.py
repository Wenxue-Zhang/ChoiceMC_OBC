from ChoiceMC_OBC import ChoiceMC
import numpy as np
from sys import argv
import matplotlib as plt
#listg=[0.,.1,.25,.5,.75,1.,1.25,1.5,1.75,2.,3.,4.]
#listg=[.75,1.,1.25]
#listg=[10]
MC_steps=40000
#T=.1
#listP=[3,5,9,17,33,65,129,257]
#P=65
m_max=5
tau=0.25
listP=[]
Pmin=3
Pmax=41
nskip=2
NP=int((Pmax-Pmin)/nskip)
for i in range(NP):
    listP.append(Pmin+nskip*i)
#list_beta=[tau*3,tau*5, tau*7]
#list_beta=[tau*11,tau*33, tau*65, tau*99, tau*129, tau*193, tau*257, tau*513]
N=2
g=1.1
Eout=open('E0_vs_beta_NMM'+'tau'+str(tau)+'g'+str(g)+'.dat','w')

ED=ChoiceMC(m_max=m_max, P=3,g=g, MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=N,PBC=False,PIGS=True)
ED.runExactDiagonalization()

for P in (listP):
    #tau=1./(P*T)#
    beta=P*tau
    T=1./beta
    rotor_chain = ChoiceMC(m_max=m_max, P=P,g=g, T=T,MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=N,PBC=False,PIGS=True)
    rotor_chain.runNMM()
    rotor_chain.runNMM_VBR(PO_DVR=False)

    Eout.write(str(beta)+' '+str(rotor_chain.E0_NMM)+' '+str(rotor_chain.E0_NMM_VBR)+' '+str(ED.E0_ED_DVR)+'\n')
Eout.close()
