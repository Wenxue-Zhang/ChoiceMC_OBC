from ChoiceMC_OBC import ChoiceMC
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
MC_steps=300
N=2
listP=[]
listP_long=[]
T=.1
Pmin=17
Pmax=41
Pmax_long=81 # 81 beads leads to a bad tau behaviour
nskip=4
NP=int((Pmax-Pmin)/nskip)
NP_long=int((Pmax_long-Pmin)/nskip)
for i in range(NP):
    listP.append(Pmin+nskip*i)
for i in range(NP_long):
    listP_long.append(Pmin+nskip*i)
m_max=5
Ngrid=int(2*m_max+1)
dphi=(2.*np.pi/Ngrid)
g=1.1
tau_list=np.zeros(len(listP),float)
tau_list_long=np.zeros(len(listP_long),float)
E0_list=np.zeros(len(listP),float)
E0_list_long=np.zeros(len(listP_long),float)

gridphi=np.zeros(Ngrid,float)
for  i in range(Ngrid): 
    gridphi[i]=dphi*i

ED=ChoiceMC(m_max=m_max, P=3,g=g, MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=N,PBC=False,PIGS=False)
ED.runExactDiagonalization()
print('gap ED =',ED.gap_ED,'gap ED DVR=',ED.gap_ED_DVR)

fig, axs = plt.subplots(2)

diff_out=open('deltaE.dat','w')
for i, P in reversed(list(enumerate(listP_long))):
    #both indices descending
    tau=1./(P*T)
    tau_list_long[i]=tau
    rotor_chain = ChoiceMC(m_max=m_max, P=P,g=g, T=T,MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=N,PBC=False,PIGS=False)
    rotor_chain.runNMM()
    E0_list_long[i]=rotor_chain.E0_NMM
for i, P in reversed(list(enumerate(listP))):
    #both indices descending
    tau=1./(P*T)
    tau_list[i]=tau
    rotor_chain = ChoiceMC(m_max=m_max, P=P,g=g, T=T,MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=N,PBC=False,PIGS=False)
    rotor_chain.runNMM()
    E0_list[i]=rotor_chain.E0_NMM


# proper fit

def func(x, a, b):
    return a *x*x + b

E0fit, pcov = curve_fit(func, tau_list, E0_list)
#print(popt,pcov)

NP_fit=100
tau_list_fit=np.zeros(NP_fit,float)
E0_list_fit_DVR=np.zeros(NP_fit,float)

dtau=1./T/Pmin/NP_fit
for i in range(NP_fit):
    tau_list_fit[i]=i*dtau
    E0_list_fit_DVR[i]=E0fit[0]*tau_list_fit[i]**2+E0fit[1]

axs[0].scatter(tau_list_long,E0_list_long,marker='o',label='DVR'+', beta='+str(1/T)+', gap='+f'{ED.gap_ED:.2f}')
axs[0].plot([0.,tau_list_long[0]],[ED.E0_ED_DVR,ED.E0_ED_DVR],'-',color='r',label='Exact, g='+str(g))
axs[0].set_ylabel('E0')
axs[0].set_xlabel(r'$\tau$')
axs[0].legend()

axs[1].scatter(tau_list,E0_list,marker='o',label='DVR'+', '+r'$\beta$='+str(1/T)+', gap='+f'{ED.gap_ED:.2f}')
axs[1].plot(tau_list_fit,E0_list_fit_DVR,'--',label='DVR-fit')
axs[1].plot([0.,tau_list[0]],[ED.E0_ED_DVR,ED.E0_ED_DVR],'-',color='r',label='Exact, g='+str(g))
axs[1].scatter([0.],[E0fit[1]],marker='*',label='extrapolated '+r'$\tau$=0')
axs[1].set_ylabel('E0')
axs[1].set_xlabel(r'$\tau$')
axs[1].legend()

fig.savefig('E0_rho_g'+str(g)+'.png')