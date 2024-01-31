from ChoiceMC_OBC import ChoiceMC
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
MC_steps=300
N=3
#listP=[33,65,77]
listP=[]
# taumin=0.166
# T=.02
# Pmin=189
# Pmax=301
T=.1
Pmin=31
Pmax=81
#T=.2
#Pmin=19
#Pmax=31
nskip=2
NP=int((Pmax-Pmin)/nskip)
for i in range(NP):
    listP.append(Pmin+nskip*i)
m_max=5
Ngrid=int(2*m_max+1)
dphi=(2.*np.pi/Ngrid)
g=1.
V0=0.
potentialField='transverse'

tau_list=np.zeros(len(listP),float)
E0_list=np.zeros(len(listP),float)
#E0_list_pair=np.zeros(len(listP),float)
#E0_list_VBR=np.zeros(len(listP),float)

gridphi=np.zeros(Ngrid,float)
for  i in range(Ngrid): 
    gridphi[i]=dphi*i

ED=ChoiceMC(m_max=m_max, P=3,g=g, MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=N,PBC=False,PIGS=False,V0=V0, potentialField=potentialField)
ED.runExactDiagonalizationN3()
print('gap ED DVR=',ED.gap_ED_N3_DVR)

fig, axs = plt.subplots(1)

for i, P in reversed(list(enumerate(listP))):
    #both indices descending
    tau=1./(P*T)
    tau_list[i]=tau
    rotor_chain = ChoiceMC(m_max=m_max, P=P,g=g, T=T,MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=N,PBC=False,PIGS=False,V0=V0, potentialField=potentialField)
    rotor_chain.runNMM_N3()
#    rotor_chain.runNMM_N3_VBR()
#    rotor_chain.runNMM_N3_pair()

    E0_list[i]=rotor_chain.E0_N3_NMM
#    E0_list_pair[i]=rotor_chain.E0_N3_pair_NMM
#    E0_list_VBR[i]=rotor_chain.E0_NMM_N3_VBR

# proper fit

def func(x, a, b):
    return a *x*x + b
def func_pair(x, a, b):
    return a *x*x*x + b

E0fit, pcov = curve_fit(func, tau_list, E0_list)
#E0fit_pair, pcov_paur = curve_fit(func_pair, tau_list, E0_list_pair)

NP_fit=100
tau_list_fit=np.zeros(NP_fit,float)
E0_list_fit=np.zeros(NP_fit,float)
#E0_list_fit_pair=np.zeros(NP_fit,float)

dtau=1./T/Pmin/NP_fit
for i in range(NP_fit):
    tau_list_fit[i]=i*dtau
    E0_list_fit[i]=E0fit[0]*tau_list_fit[i]**2+E0fit[1]
#    E0_list_fit_pair[i]=E0fit_pair[0]*tau_list_fit[i]**3+E0fit_pair[1]


#print(E0_list_VBR,E0_list_PO_DVR)
axs.scatter(tau_list,E0_list,marker='o',label='Trotter'+', beta='+str(1/T)+', gap='+f'{ED.gap_ED_N3_DVR:.2f}')
#axs.scatter(tau_list,E0_list_VBR,marker='x',label='VBR'+', beta='+str(1/T)+', gap='+f'{ED.gap_ED_N3_DVR:.2f}')
axs.plot(tau_list_fit,E0_list_fit,'--',label='Trotter-fit')
axs.plot([0.,tau_list[0]],[ED.E0_ED_N3_DVR,ED.E0_ED_N3_DVR],'-',label='Exact, g='+str(g))
axs.scatter([0.],[E0fit[1]],marker='*',label='tau=0 Trotter')
#axs.scatter([0.],[E0_list_fit_pair[1]],marker='*',label='tau=0 Pair')
axs.set_ylabel('E0')
axs.set_xlabel(r'$\tau$')
axs.legend()

plt.show()


fig.savefig('E0_rho_g'+str(g)+'.png')