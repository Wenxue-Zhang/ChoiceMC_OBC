import numpy as np
import math
#import cython
from numpy.random import default_rng
from scipy.optimize import curve_fit
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns
import os

def pot_func_Parallel(phi,V):
    pot = V * (np.cos(phi))
    #pot = V * (1. + np.cos(phi))
    return pot

def pot_func_Transverse(phi,V):
    pot = V * (1. + np.sin(phi))
    return pot

def Vij(p1,p2,g):
    #make abstraction of phi value (if possible)
    #error below , should be -3
    #V12=g*(np.cos(p1-p2)-2.*np.cos(p1)*np.cos(p2))
    V12=g*(np.cos(p1-p2)-3.*np.cos(p1)*np.cos(p2))
    return V12

def Vijphi_rel(dphi,g):
    Vij=(-g/2.)*(np.cos(dphi))
    return Vij

def VijPhi_CM(dphi,g):
    Vij=(-g*3./2.)*(np.cos(dphi))
    return Vij

def pot_matrix(size):
    V_mat=np.zeros((size,size),float)
    for m in range(size):
        for mp in range(size):
            if m==(mp+1):
                V_mat[m,mp]=.5
            if m==(mp-1):
                V_mat[m,mp]=.5
    return V_mat

def pot_matrix_cos(size):
    V_mat=np.zeros((size,size),float)
    for m in range(size):
        for mp in range(size):
            if m==(mp+1):
                V_mat[m,mp]=.5
            if m==(mp-1):
                V_mat[m,mp]=.5
    return V_mat

def pot_matrix_sin(size):
    # This is the matrix divided by i
    V_mat=np.zeros((size,size),float)
    for m in range(size):
        for mp in range(size):
            # We don't know these
            # These are the result of the math
            # {[-1+exp(-i*2pi(m-mp))]/[(m-mp)^2-1]}/2pi
            if m==(mp+1):
                V_mat[m,mp]=.5
            if m==(mp-1):
                V_mat[m,mp]=-.5
    return V_mat

def K_matrix_DVR(mmax):
    # Colbert Miller DVR, 0,2*Pi
    size=2*mmax+1
    K_mat=np.zeros((size,size),float)
    for i in range(size):
        for ip in range(size):
            di=i-ip
            if i==ip:
                K_mat[i,ip]=float(mmax*(mmax+1.)/3.)
            else:
                arg=np.pi*di/(2.*mmax+1.)
                K_mat[i,ip]=pow(-1.,di)*np.cos(arg)/(2.*np.sin(arg)**2)

    return K_mat

def gen_prob_dist(Ng,rho_phi):
    p = np.zeros((Ng, Ng, Ng),float)
    # Normalize:
    P_norm = np.zeros((Ng,Ng),float)
    for i0 in range(Ng):
        for i1 in range(Ng):
            di01=i0 - i1
            if di01 < 0:
                di01+=Ng
            for i2 in range(Ng):
                di12= i1- i2
                if di12 < 0:
                    di12 +=Ng
                p[i0,i1,i2]=rho_phi[di01]*rho_phi[di12]
                P_norm[i0,i2] += p[i0,i1,i2]
    for i0 in range(Ng):
        for i1 in range(Ng):
            for i2 in range(Ng):
                p[i0,i1,i2]=p[i0,i1,i2]/P_norm[i0,i2]
    return p

def gen_prob_dist_end(Ng,rho_phi):
    p = np.zeros((Ng, Ng),float)
    # Normalize:
    P_norm = np.zeros(Ng,float)
    for i0 in range(Ng):
        for i1 in range(Ng):
            di01=i0 - i1
            if di01 < 0:
                di01+=Ng
            p[i0,i1]=rho_phi[di01]
            P_norm[i0] += p[i0,i1]
    for i0 in range(Ng):
        for i1 in range(Ng):
            p[i0,i1]=p[i0,i1]/P_norm[i0]
    return p

def calculateOrientationalCorrelations(p1, p2):
    return np.cos(p1-p2)

# Functions for the binning error analysis
def errorpropagation(mean, data):
    ndim   = len(data)
    error = np.std(data,ddof=0)/np.sqrt(ndim)
    return error

def maxError_byBinning(mean, data, workingNdim):
    if(workingNdim<=1):
        raise Exception('Not enough points MC steps were used for the binning method, please increase the number of MC steps')
    error = np.zeros(workingNdim)
    i = 0
    error[0] = errorpropagation(mean, data)

    for i in range(1,workingNdim):
        ndim = int(len(data)/2)
        data1 = np.zeros(ndim)

        for j in range(ndim):
            data1[j] = 0.5*(data[2*j]+data[2*j+1])
        data = data1
        error[i] = errorpropagation(mean,data)
    return np.max(error)

def calculateError_byBinning(arr):
    # Finding the average and standard error using the binning method
    # This method requires 2^n data points, this truncates the data to fit this
    workingNdim  = int(math.log(len(arr))/math.log(2))
    trunc = int(len(arr)-2**workingNdim)
    mean = np.mean(arr[trunc:])
    standardError = maxError_byBinning(mean, arr[trunc:], workingNdim-6)
    return mean, standardError

def fitFuncQuadratic_E(tau, a, b):
    # Curve fitting function, this ensure that it is a quadratic centered around
    # tau = 0 that opens downwards
    return -1*abs(a*(tau**2)) + b

def fitFuncQuadratic_eiej(tau, a, b):
    # Curve fitting function, this ensure that it is a quadratic centered around
    # tau = 0 that opens upwards
    return abs(a*(tau**2)) + b

def fitFuncQuartic(tau, a, b, c):
    # Curve fitting function, this ensure that it is a quadratic centered around
    # tau = 0 that opens downwards
    return a*tau**4 + b*tau**2 + c

def extrapolate_E0(arr, fitType='quadratic'):
    # Takes an Nx2 array (tau, E0)
    # Returns the coefficients for the fitting function specified
    if fitType == 'quadratic':
        return curve_fit(fitFuncQuadratic_E, arr[:,0], arr[:,1])[0]
    elif fitType == 'quartic':
        return curve_fit(fitFuncQuartic, arr[:,0], arr[:,1])[0]
    else:
        raise Exception("Invalid fitting function type, please use quadratic or quartic")

def extrapolate_eiej(arr, fitType='quadratic'):
    # Takes an Nx2 array (tau, eiej)
    # Returns the coefficients for the fitting function specified
    if fitType == 'quadratic':
        return curve_fit(fitFuncQuadratic_eiej, arr[:,0], arr[:,1])[0]
    elif fitType == 'quartic':
        return curve_fit(fitFuncQuartic, arr[:,0], arr[:,1])[0]
    else:
        raise Exception("Invalid fitting function type, please use quadratic or quartic")

def loadResult(path):
    # Takes in a path that branches from the ChoiceMC/Results directory and returns
    # the data stored in that path
    # Ex: path = 'ED/Energy_mMax1.dat'
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    result_path = os.path.join(parent_dir,'Results')
    result_path = os.path.join(result_path, path)
    result = np.loadtxt(result_path)
    return result

class ChoiceMC(object):
    def __init__(self, m_max, P, g, MC_steps, N, Nskip=1, Nequilibrate=0, PIGS=False, T=.5, B=1., V0=0., potentialField='parallel',PBC=False):
        """
        Creates a ChoiceMC object. This object can be used to generate the density
        matrices and performed the PIMC method based on the inputs. The results
        are stored as attributes of the object.

        Example:
        PIMC = ChoiceMC(m_max=50, P=9, g=1, MC_steps=1000, N=3, PIGS=True, Nskip=100, Nequilibrate=100)
        This sets up the system using 50 grid points, 9 beads, an interaction strength of 1, 3 rotors,
        path integral ground state enabled, 100 skip steps and 100 equilibrium steps.

        Parameters
        ----------
        m_max : int
            The maximum size of the free rotor eigenstate basis used to construct
            the rotational density matrix.
        P : int
            The number of beads to use in the path-integral.
        g : float
            Interaction strength between the rotors.
        MC_steps : int
            The number of steps to use in the Monte Carlo method.
        N : int
            The number of rotors to be simulated.
        Nskip : int, optional
            The number of steps to skip when saving the trajectory. The default is 100.
        Nequilibrate : int, optional
            The number of steps to skip before the average properties are accumulated
            to allow for system equilibration. The default is 0.
        PIGS : bool, optional
            Enables path-integral ground state calculations. The default is False.
        T : float, optional
            The system temperature. The default is 1.
        B : float, optional
            The rotational constant for the rotors in energy units. The default is 1.
        V0 : float, optional
            The external potential field for the system. The default is 0.
        potentialField : string, optional
            The type of external potential field for the system. The default is transverse.

        Returns
        -------
        All inputs mentioned above are stored as attributes in the system.
        self.beta: float
            The beta value based on the system temperature.
        self.tau: float
            The tau value for the path integral method based on the beta value
            and the number of beads.
        self.Ngrid: int
            The number of steps to discretize the angle phi into.
        self.delta_phi: float
            The change in phi between the discretized phi values.
        self.potFunc: function handle
            The function handle that matches the desired potential function.
        """

        # Extracting information from kwargs for extra arguments
        self.Nskip = Nskip
        self.Nequilibrate = Nequilibrate
        self.PIGS = PIGS
        self.T = T
        self.B = B
        self.V0 = V0
        self.PBC=PBC

        # Setting which potential function will be used
        if potentialField == "transverse":
            self.potFunc = pot_func_Transverse
        elif potentialField == 'parallel':
            self.potFunc = pot_func_Parallel
        else:
            raise Exception("Unrecognized potential model, allowed models are transverse or parallel")

        self.beta = 1./self.T
        self.P = P
        if self.P <= 1:
            raise Exception("A minimum of 2 beads must be used")

        if self.PIGS:
            self.tau = self.beta/float(self.P-1)
        else:
            self.tau = self.beta/float(self.P)

        self.m_max = m_max
        self.Ngrid = 2 * m_max + 1
        self.delta_phi = 2. * np.pi / float(self.Ngrid)
        self.g = g
        self.MC_steps = MC_steps
        self.N = N

        self.createFreeRhoDVR()


        # Creating a folder to save the output data
        self.path = os.path.join(os.getcwd(), "ChoiceMC_P" + str(P) + "_N" + str(N) + "_g" + str(round(g,3)) + "_MCSteps" + str(MC_steps)+"_V" + str(V0) + "_mMax" + str(m_max))
        try:
            os.mkdir(self.path)
        except FileExistsError:
            pass

        # Throwing a warning if the center bead will not be an integer value when pigs is enabled
        if self.P % 2 == 0 and self.PIGS:
            raise Warning("PIGS is enabled and an even number of beads was input, it is recommended to use an odd number")

    def runExactDiagonalization(self):
        '''
        Performs exact diagonalization for 2 rotors and calculates the ground state
        energy and orientational correlation. Warning, this method scales with Ngrid^2
        and can become quickly intractable.

        Returns
        -------
        self.E0_ED:float
            The ground state energy calculated by exact diagonalization.
        self.eiej_ED: float
            The orientational correlation calculated by exact diagonalization.
        self.purity_ED: float
            The purity calculated by exact diagonalization.
        self.S2_ED: float
            The second Renyi entropy calculated by exact diagonalization.
        self.SvN: float
            the von Neumann entropy calculated by exact diagonalization.
        TO ADD
        -------
        Add calculations for non-PIGS simulations
        Fix calculations for external fields
        '''
        # Throwing a warning if the MC object currently being tested does not have 2 rotors
        if self.N != 2:
            raise Warning("The exact diagonalization method can only handle 2 rotors and the current MC simulation is not being performed with 2 rotors.")

        # Unpacking variables from class
        size = self.Ngrid
        size2 = self.Ngrid**2
        g = self.g
        B = self.B
        mMax = self.m_max
        # Creating the cos and sin potential matrices
        cos_mmp = pot_matrix_cos(size)
        sin_mmp = pot_matrix_sin(size)
        krondel=np.eye(size,dtype =float)

        # DVR Hamiltonian
        grid=np.zeros(size,float)
        for m1 in range(self.Ngrid):
            grid[m1]=(m1)*2.*np.pi/size
        K_DVR = K_matrix_DVR(mMax)
        #print(K_DVR)
        H_DVR = np.zeros((size2,size2), float)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m1p in range(self.Ngrid):
                    for m2p in range(self.Ngrid):
                        if m2==m2p:
                            H_DVR[m1*size + m2, m1p*size + m2p]+=B*K_DVR[m1,m1p]
                        if m1==m1p:
                            H_DVR[m1*size + m2, m1p*size + m2p]+=B*K_DVR[m2,m2p]
                        if m1==m1p and m2==m2p:
                            H_DVR[m1*size + m2, m1p*size + m2p]+=g*(np.sin(grid[m1])*np.sin(grid[m2])-2.*np.cos(grid[m1])*np.cos(grid[m2]))
        evals_DVR, evecs_DVR = np.linalg.eigh(H_DVR)
        E0_DVR = evals_DVR[0]

        # Creating the Hamiltonian
        H = np.zeros((size2,size2), float)
        H_im = np.zeros((size2,size2), float)
        dipoleX=np.zeros((size2,size2), float)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m1p in range(self.Ngrid):
                    for m2p in range(self.Ngrid):
                        if m1==m1p and m2==m2p:
                            # Kinetic contribution
                            H[m1*size + m2, m1p*size + m2p] += B * float((-mMax+m1)**2)
                            H[m1*size + m2, m1p*size + m2p] += B * float((-mMax+m2)**2)
                        # Potential contribution
                        H[m1*size + m2, m1p*size + m2p] += g * (-1*sin_mmp[m1,m1p]*sin_mmp[m2,m2p] - 2*cos_mmp[m1,m1p]*cos_mmp[m2,m2p])
                        dipoleX[m1*size + m2, m1p*size + m2p] = cos_mmp[m1,m1p]*krondel[m2,m2p]+cos_mmp[m2,m2p]*krondel[m1,m1p]
                        if self.V0!=0 and self.potFunc == pot_func_Transverse:
                            if m1==m1p:
                                H_im[m1*size + m2, m1p*size + m2p] += self.V0*(1.0 + sin_mmp[m2,m2p])
                            if m2==m2p:
                                H_im[m1*size + m2, m1p*size + m2p] += self.V0*(1.0 + sin_mmp[m1,m1p])
                        elif self.V0!=0 and self.potFunc == pot_func_Parallel:
                            if m1==m1p:
                                H[m1*size + m2, m1p*size + m2p] += self.V0*(1.0 + cos_mmp[m2,m2p])
                            if m2==m2p:
                                H[m1*size + m2, m1p*size + m2p] += self.V0*(1.0 + cos_mmp[m1,m1p])



        Hfull=H + 1j*H_im
        self.Hfull=Hfull.real
        # Finding the eigenavalues and eigenvectors
        #if self.V0!=0 and self.potFunc == pot_func_Transverse:
        if self.V0!=0:
            evals, evecs = np.linalg.eigh(H + 1j*H_im)
        else:
            evals, evecs = np.linalg.eigh(H)
        
        muX=np.dot(np.transpose(evecs),np.dot(dipoleX,evecs))
        muX2=np.dot(np.transpose(evecs),np.dot(np.dot(dipoleX,dipoleX),evecs))

        # Evaluating the observables
        E0 = evals[0]
        e1_dot_e2 = 0.
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m1p in range(self.Ngrid):
                    for m2p in range(self.Ngrid):
                        e1_dot_e2 += evecs[m1*size + m2,0]*np.conjugate(evecs[m1p*size + m2p,0])*(-1*sin_mmp[m1,m1p]*sin_mmp[m2,m2p] + cos_mmp[m1,m1p]*cos_mmp[m2,m2p])
                        # <0|m1 m2> <m1 m2|e1.e2|m1p m2p> <m1p m2p|0> = <0|e1.e2|0>
        # Finding the second renyi entropy
        # Calculate reduced density matrix rhoA
        rhoA = np.zeros((size,size), float)
        for m1 in range(self.Ngrid):
            for m1p in range(self.Ngrid):
                for m2 in range(self.Ngrid):
                    rhoA[m1,m1p] += np.real(evecs[m1*size + m2,0]*np.conjugate(evecs[m1p*size + m2,0]))
        # Diagonalize rhoA
        rhoA_E, rhoA_EV = np.linalg.eigh(rhoA)
        S2 = 0.
        SvN = 0.
        for m1 in range(size):
            SvN -= rhoA_E[m1]*np.log(abs(rhoA_E[m1]))
            S2 += (rhoA_E[m1]**2)
        self.purity_ED = S2
        S2 = -np.log(S2)

        self.E0_ED = E0
        self.evals_ED=evals
        self.muX_ED=muX
        self.muX2_ED=muX2

        self.E0_ED_DVR = E0_DVR
        self.gap_ED=evals[1]-evals[0]
        self.gap_ED_DVR=evals_DVR[1]-evals_DVR[0]

        self.eiej_ED = e1_dot_e2
        self.S2_ED = S2
        self.SvN = SvN
        print('E0_ED = ',self.E0_ED)
        print('E0_ED_DVR = ',E0_DVR)
        print('ED <ei.ej> = ',self.eiej_ED)
        print('S2_ED = ', str(self.S2_ED))
    def runExactDiagonalizationN3(self):
        '''
        Performs exact diagonalization for 2 rotors and calculates the ground state
        energy and orientational correlation. Warning, this method scales with Ngrid^2
        and can become quickly intractable.

        Returns
        -------
        self.E0_ED:float
            The ground state energy calculated by exact diagonalization.
        self.eiej_ED: float
            The orientational correlation calculated by exact diagonalization.
        self.purity_ED: float
            The purity calculated by exact diagonalization.
        self.S2_ED: float
            The second Renyi entropy calculated by exact diagonalization.
        self.SvN: float
            the von Neumann entropy calculated by exact diagonalization.
        TO ADD
        -------
        Add calculations for non-PIGS simulations
        Fix calculations for external fields
        '''
        # Throwing a warning if the MC object currently being tested does not have 2 rotors
        if self.N != 3:
            raise Warning("This exact diagonalization method can only handle 3 rotors and the current MC simulation is not being performed with 2 rotors.")

        # Unpacking variables from class
        size = self.Ngrid
        size3 = self.Ngrid**3
        g = self.g
        B = self.B
        mMax = self.m_max

        # DVR Hamiltonian
        grid=np.zeros(size,float)
        for m1 in range(self.Ngrid):
            grid[m1]=(m1)*2.*np.pi/size
        K_DVR = K_matrix_DVR(mMax)
        #print(K_DVR)
        H_DVR = np.zeros((size3,size3), float)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m3 in range(self.Ngrid):
                    for m1p in range(self.Ngrid):
                        for m2p in range(self.Ngrid):
                            for m3p in range(self.Ngrid):
                                if m2==m2p and m3==m3p:
                                    H_DVR[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]+=B*K_DVR[m1,m1p]
                                if m1==m1p and m3==m3p:
                                    H_DVR[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]+=B*K_DVR[m2,m2p]
                                if m1==m1p and m2==m2p:
                                    H_DVR[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]+=B*K_DVR[m3,m3p]    
                                if m1==m1p and m2==m2p and m3==m3p:
                                    H_DVR[(m1*size + m2)*size+m3, (m1*size + m2)*size+m3]+=g*(np.sin(grid[m1])*np.sin(grid[m2])-2.*np.cos(grid[m1])*np.cos(grid[m2]))+g*(np.sin(grid[m3])*np.sin(grid[m2])-2.*np.cos(grid[m3])*np.cos(grid[m2]))+self.potFunc(float(m1)*self.delta_phi,self.V0) + self.potFunc(float(m2)*self.delta_phi,self.V0)+self.potFunc(float(m3)*self.delta_phi,self.V0)

        evals_DVR, evecs_DVR = np.linalg.eigh(H_DVR)
        E0_DVR = evals_DVR[0]

        # Evaluating the observables
        e1_dot_e2 = 0.
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m3 in range(self.Ngrid):
                    e1_dot_e2 += (evecs_DVR[(m1*size + m2)*size+m3,0]**2)*(np.sin(grid[m1])*np.sin(grid[m2])+np.cos(grid[m1])*np.cos(grid[m2])+np.sin(grid[m3])*np.sin(grid[m2])+np.cos(grid[m3])*np.cos(grid[m2]))
                        # <0|m1 m2> <m1 m2|e1.e2|m1p m2p> <m1p m2p|0> = <0|e1.e2|0>
        # Finding the second renyi entropy
        e1_dot_e2=e1_dot_e2/2.

        self.eiej_ED_N3 = e1_dot_e2
        self.E0_ED_N3_DVR = E0_DVR
        self.gap_ED_N3_DVR=evals_DVR[1]-evals_DVR[0]

        print('E0_ED_DVR = ',E0_DVR)
        print('ED <ei.ej> = ',self.eiej_ED_N3)


    def pair_action(self):
       
        # Unpacking variables from class
        size = self.Ngrid
        size2 = self.Ngrid**2
        g = self.g
        B = self.B
        mMax = self.m_max
  
        # DVR Hamiltonian
        grid=np.zeros(size,float)
        for m1 in range(self.Ngrid):
            grid[m1]=(m1)*2.*np.pi/size
        K_DVR = K_matrix_DVR(mMax)
        #print(K_DVR)
        H_DVR = np.zeros((size2,size2), float)
        H0_DVR = np.zeros((size2,size2), float)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m1p in range(self.Ngrid):
                    for m2p in range(self.Ngrid):
                        if m2==m2p:
                            H_DVR[m1*size + m2, m1p*size + m2p]+=B*K_DVR[m1,m1p]
                            H0_DVR[m1*size + m2, m1p*size + m2p]+=B*K_DVR[m1,m1p]
                        if m1==m1p:
                            H_DVR[m1*size + m2, m1p*size + m2p]+=B*K_DVR[m2,m2p]
                            H0_DVR[m1*size + m2, m1p*size + m2p]+=B*K_DVR[m2,m2p]
                        if m1==m1p and m2==m2p:
                            H_DVR[m1*size + m2, m1p*size + m2p]+=g*(np.sin(grid[m1])*np.sin(grid[m2])-2.*np.cos(grid[m1])*np.cos(grid[m2]))
        evals_DVR, evecs_DVR = np.linalg.eigh(H_DVR)
        evals0_DVR, evecs0_DVR = np.linalg.eigh(H0_DVR)

        expH_DVR_diag=np.zeros((size2,size2),float)
        expH0_DVR_diag=np.zeros((size2,size2),float)
        for i in range(size2):
            expH_DVR_diag[i,i]=np.exp(-self.tau*evals_DVR[i])
            expH0_DVR_diag[i,i]=np.exp(-self.tau*evals0_DVR[i])
        rho_pair_DVR=np.dot(evecs_DVR,np.dot(expH_DVR_diag,np.transpose(evecs_DVR)))
        rho_pair0_DVR=np.dot(evecs0_DVR,np.dot(expH0_DVR_diag,np.transpose(evecs0_DVR)))

        count_sign=0
        for i in range(size2):
            for ip in range(size2):
                rho_pair_DVR[i,ip]=rho_pair_DVR[i,ip]/rho_pair0_DVR[i,ip]
                if rho_pair_DVR[i,ip] <0.:
                    count_sign+=1
                    rho_pair_DVR[i,ip]=np.abs(rho_pair_DVR[i,ip])
        print('negative fraction',g,self.tau,count_sign/(size2*size2))
        self.rho_pair_DVR = rho_pair_DVR

     
    def runNMM(self):
        """
        Solves for the ground state energy using the NMM method. This method can
        only be used for a two rotor system. Warning, this method scales with Ngrid^2
        and can become quickly intractable.

        Returns
        -------
        self.E0_NMM: float
            Ground state energy calculated by the NMM method
        self.eiej_NMM: float
            The orientational correlation calculated by the NMM method.
        """
        # Throwing a warning if the MC object currently being tested does not have 2 rotors
        if self.N != 2:
            raise Warning("The exact diagonalization method can only handle 2 rotors and the current MC simulation is not being performed with 2 rotors.")

        # Unpacking class attributes into variables
        size = self.Ngrid
        size2 = self.Ngrid**2
        tau = self.tau
        # Creating the 1 body free rho by the Marx method
        #self.createFreeRhoMarx()
        self.createFreeRhoDVR()
        #self.createFreeRhoSOS()
        #self.createFreeRhoPQC()
        rho_phi = self.rho_phi
        rho_free_1body = np.zeros((size,size), float)

        for i1 in range(size):
            for i1p in range(size):
                index = i1-i1p
                if index < 0:
                    index += size
                rho_free_1body[i1, i1p] = rho_phi[index]

        self.rho_free_1body=rho_free_1body
        # Creating rho potential and the 2 body free rho
        rho_potential = np.zeros((size2,size2), float)
        potential = np.zeros(size2, float)
        rho_free_2body = np.zeros((size2,size2), float)
        for i1 in range(size):
            for i2 in range(size):
                potential[i1*size+i2] += Vij(i1*self.delta_phi, i2*self.delta_phi, self.g)
                potential[i1*size+i2] += self.potFunc(float(i1)*self.delta_phi,self.V0) + self.potFunc(float(i2)*self.delta_phi,self.V0)
                rho_potential[i1*size+i2,i1*size+i2] = np.exp(-0.5*tau*potential[i1*size+i2])
                for i1p in range(size):
                    for i2p in range(size):
                        rho_free_2body[i1*size+i2,i1p*size+i2p] = rho_free_1body[i1,i1p] * rho_free_1body[i2,i2p]

        # Constructing the high temperature density matrix
        rho_tau=np.zeros((size2,size2),float)
        rho_tau = np.dot(rho_potential, np.dot(rho_free_2body, rho_potential))

        # Forming the density matrix via matrix multiplication
        rho_beta=rho_tau.copy()
        rho_beta_over2=rho_tau.copy()

        for k in range(self.P-2):
            rho_beta=(self.delta_phi**2)*np.dot(rho_beta,rho_tau)
        for k in range(int((self.P-1)/2 - 1)):
            rho_beta_over2=(self.delta_phi**2)*np.dot(rho_beta_over2,rho_tau)
        rho_beta_over2_e1_e2=rho_beta_over2.copy()
        #form rho
        psi0_nonorm=np.zeros((size2),float)

        # Forming matrices to find the orientational correlation
        for i1 in range(size):
            phi1=i1*self.delta_phi
            z1=np.cos(phi1)
            x1=np.sin(phi1)
            for i2 in range(size):
                phi2=i2*self.delta_phi
                z2=np.cos(phi2)
                x2=np.sin(phi2)
                for i1p in range(size):
                    for i2p in range(size):
                        psi0_nonorm[i1*size+i2]+=rho_beta_over2[i1*size+i2,i1p*size+i2p]
                        rho_beta_over2_e1_e2[i1*size+i2,i1p*size+i2p]=rho_beta_over2[i1*size+i2,i1p*size+i2p]*(x1*x2+z1*z2)
        rhobeta2_e1e2_rhobeta2=(self.delta_phi**2)*np.dot(rho_beta_over2,rho_beta_over2_e1_e2)

        # Finding the ground state energy and orientational correlations
        E0_nmm=0.
        rho_dot_V=np.dot(rho_beta,potential)
        Z0=0. # pigs pseudo Z
        e1_dot_e2=0.
        for i in range(size2):
            E0_nmm += rho_dot_V[i]
            for ip in range(size2):
                e1_dot_e2+=rhobeta2_e1e2_rhobeta2[i,ip]
                Z0 += rho_beta[i,ip]
        E0_nmm/=Z0

        rhoA = np.zeros((size,size), float)
        for i1 in range(size):
            for i1p in range(size):
                for i2 in range(size):
                    rhoA[i1,i1p] += (psi0_nonorm[i1*size+i2]*psi0_nonorm[i1p*size+i2]/Z0)*self.delta_phi*self.delta_phi

        #rhoA_E, rhoA_EV = np.linalg.eigh(rhoA)
        S2 = 0.
        SvN = 0.
        #for i1 in range(size):
        #    SvN -= rhoA_E[i1]*np.log(abs(rhoA_E[i1]))
        #    S2 += (rhoA_E[i1]**2)
        self.purity_ED = S2
        S2 = -np.log(S2)

        self.E0_NMM = E0_nmm
        self.S2_NMM = S2
        self.SvN_NMM = SvN
        self.eiej_NMM = e1_dot_e2/Z0
        print('E0_NMM = ', E0_nmm)
        print('Z0_NMM = ', Z0)
        print('SvN_NMM = ', SvN)
        print('S2_NMM = ', S2)
        print('NMM <e1.e2>= ', self.eiej_NMM)

    def runNMM_N3(self):
        """
        Solves for the ground state energy using the NMM method. This method can
        only be used for a two rotor system. Warning, this method scales with Ngrid^2
        and can become quickly intractable.

        Returns
        -------
        self.E0_NMM: float
            Ground state energy calculated by the NMM method
        self.eiej_NMM: float
            The orientational correlation calculated by the NMM method.
        """
        # Throwing a warning if the MC object currently being tested does not have 2 rotors
        if self.N != 3:
            raise Warning("This NMM  method can only handle 3 rotors and the current MC simulation is not being performed with 2 rotors.")

        # Unpacking class attributes into variables
        size = self.Ngrid
        size3 = self.Ngrid**3
        tau = self.tau
        # Creating the 1 body free rho by the Marx method
        #self.createFreeRhoMarx()
        self.createFreeRhoDVR()
        #self.createFreeRhoSOS()
        #self.createFreeRhoPQC()
        rho_phi = self.rho_phi
        rho_free_1body = np.zeros((size,size), float)

        for i1 in range(size):
            for i1p in range(size):
                index = i1-i1p
                if index < 0:
                    index += size
                rho_free_1body[i1, i1p] = rho_phi[index]

        self.rho_free_1body=rho_free_1body
        # Creating rho potential and the 2 body free rho
        rho_potential = np.zeros((size3,size3), float)
        potential = np.zeros(size3, float)
        rho_free_3body = np.zeros((size3,size3), float)
        for i1 in range(size):
            for i2 in range(size):
                for i3 in range(size):
                    potential[(i1*size+i2)*size+i3] += Vij(i1*self.delta_phi, i2*self.delta_phi, self.g)+Vij(i2*self.delta_phi, i3*self.delta_phi, self.g)
                    potential[(i1*size+i2)*size+i3] += self.potFunc(float(i1)*self.delta_phi,self.V0) + self.potFunc(float(i2)*self.delta_phi,self.V0)+self.potFunc(float(i3)*self.delta_phi,self.V0)
                    rho_potential[(i1*size+i2)*size+i3,(i1*size+i2)*size+i3] = np.exp(-0.5*tau*potential[(i1*size+i2)*size+i3])
                    for i1p in range(size):
                        for i2p in range(size):
                            for i3p in range(size):
                                rho_free_3body[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p] = rho_free_1body[i1,i1p] * rho_free_1body[i2,i2p]* rho_free_1body[i3,i3p]

        # Constructing the high temperature density matrix
        rho_tau=np.zeros((size3,size3),float)
        rho_tau = np.dot(rho_potential, np.dot(rho_free_3body, rho_potential))
        rho_tau=np.absolute(rho_tau)


        # Forming the density matrix via matrix multiplication
        rho_beta=rho_tau.copy()
        rho_beta_over2=rho_tau.copy()

        for k in range(self.P-2):
            rho_beta=(self.delta_phi**3)*np.dot(rho_beta,rho_tau)
        for k in range(int((self.P-1)/2 - 1)):
            rho_beta_over2=(self.delta_phi**3)*np.dot(rho_beta_over2,rho_tau)
        rho_beta_over2_e1_e2=rho_beta_over2.copy()
        #form rho
        psi0_nonorm=np.zeros((size3),float)

        # Forming matrices to find the orientational correlation
        for i1 in range(size):
            phi1=i1*self.delta_phi
            z1=np.cos(phi1)
            x1=np.sin(phi1)
            for i2 in range(size):
                phi2=i2*self.delta_phi
                z2=np.cos(phi2)
                x2=np.sin(phi2)
                for i3 in range(size):
                    phi3=i3*self.delta_phi
                    z3=np.cos(phi3)
                    x3=np.sin(phi3)
                    for i1p in range(size):
                        for i2p in range(size):
                            for i3p in range(size):
                                psi0_nonorm[(i1*size+i2)*size+i3]+=rho_beta_over2[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]
                                rho_beta_over2_e1_e2[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]=rho_beta_over2[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]*(x1*x2+z1*z2+x2*x3+z2*z3)
        rhobeta2_e1e2_rhobeta2=(self.delta_phi**3)*np.dot(rho_beta_over2,rho_beta_over2_e1_e2)

        # Finding the ground state energy and orientational correlations
        E0_nmm=0.
        rho_dot_V=np.dot(rho_beta,potential)
        Z0=0. # pigs pseudo Z
        ei_dot_ej=0.
        for i in range(size3):
            E0_nmm += rho_dot_V[i]
            for ip in range(size3):
                ei_dot_ej+=rhobeta2_e1e2_rhobeta2[i,ip]
                Z0 += rho_beta[i,ip]
        E0_nmm/=Z0

        self.E0_N3_NMM = E0_nmm
        self.eiej_N3_NMM = ei_dot_ej/2./Z0

        print('E0_NMM (Trotter) = ', E0_nmm)
        print('NMM <e1.e2>= ', self.eiej_N3_NMM)

    def runNMM_N3_pair(self):
        """
        Solves for the ground state energy using the NMM method. This method can
        only be used for a two rotor system. Warning, this method scales with Ngrid^2
        and can become quickly intractable.

        Returns
        -------
        self.E0_NMM: float
            Ground state energy calculated by the NMM method
        self.eiej_NMM: float
            The orientational correlation calculated by the NMM method.
        """
        # Throwing a warning if the MC object currently being tested does not have 2 rotors
        if self.N != 3:
            raise Warning("This NMM  method can only handle 3 rotors and the current MC simulation is not being performed with 2 rotors.")

        # Unpacking class attributes into variables
        size = self.Ngrid
        size3 = self.Ngrid**3
        tau = self.tau
        # Creating the 1 body free rho by the Marx method
        rho_phi = self.rho_phi
        rho_free_1body = np.zeros((size,size), float)

        for i1 in range(size):
            for i1p in range(size):
                index = i1-i1p
                if index < 0:
                    index += size
                rho_free_1body[i1, i1p] = rho_phi[index]

        self.rho_free_1body=rho_free_1body
        # Creating rho potential and the 2 body free rho
        rho_potential = np.zeros((size3,size3), float)
        potential = np.zeros(size3, float)
        rho_free_3body = np.zeros((size3,size3), float)
        for i1 in range(size):
            for i2 in range(size):
                for i3 in range(size):
                    potential[(i1*size+i2)*size+i3] += Vij(i1*self.delta_phi, i2*self.delta_phi, self.g)+Vij(i2*self.delta_phi, i3*self.delta_phi, self.g)
                    potential[(i1*size+i2)*size+i3] += self.potFunc(float(i1)*self.delta_phi,self.V0) + self.potFunc(float(i2)*self.delta_phi,self.V0)+self.potFunc(float(i3)*self.delta_phi,self.V0)
                    for i1p in range(size):
                        for i2p in range(size):
                            for i3p in range(size):
                                rho_free_3body[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p] = rho_free_1body[i1,i1p] * rho_free_1body[i2,i2p]* rho_free_1body[i3,i3p]

        # Constructing the high temperature density matrix
        rho_tau=np.zeros((size3,size3),float)

        self.pair_action()
        for i1 in range(size):
            for i2 in range(size):
                for i3 in range(size):
                    for i1p in range(size):
                        for i2p in range(size):
                            for i3p in range(size):
                                rho_tau[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]=rho_free_3body[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]*self.rho_pair_DVR[(i1*size+i2),(i1p*size+i2p)]*self.rho_pair_DVR[(i2*size+i3),(i2p*size+i3p)]

        # Forming the density matrix via matrix multiplication
        rho_beta=rho_tau.copy()
        rho_beta_over2=rho_tau.copy()

        for k in range(self.P-2):
            rho_beta=(self.delta_phi**3)*np.dot(rho_beta,rho_tau)
        for k in range(int((self.P-1)/2 - 1)):
            rho_beta_over2=(self.delta_phi**3)*np.dot(rho_beta_over2,rho_tau)
        rho_beta_over2_e1_e2=rho_beta_over2.copy()
        #form rho
        psi0_nonorm=np.zeros((size3),float)

        # Forming matrices to find the orientational correlation
        for i1 in range(size):
            phi1=i1*self.delta_phi
            z1=np.cos(phi1)
            x1=np.sin(phi1)
            for i2 in range(size):
                phi2=i2*self.delta_phi
                z2=np.cos(phi2)
                x2=np.sin(phi2)
                for i3 in range(size):
                    phi3=i3*self.delta_phi
                    z3=np.cos(phi3)
                    x3=np.sin(phi3)
                    for i1p in range(size):
                        for i2p in range(size):
                            for i3p in range(size):
                                psi0_nonorm[(i1*size+i2)*size+i3]+=rho_beta_over2[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]
                                rho_beta_over2_e1_e2[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]=rho_beta_over2[(i1*size+i2)*size+i3,(i1p*size+i2p)*size+i3p]*(x1*x2+z1*z2+x2*x3+z2*z3)
        rhobeta2_e1e2_rhobeta2=(self.delta_phi**3)*np.dot(rho_beta_over2,rho_beta_over2_e1_e2)

        # Finding the ground state energy and orientational correlations
        E0_nmm=0.
        rho_dot_V=np.dot(rho_beta,potential)
        Z0=0. # pigs pseudo Z
        ei_dot_ej=0.
        for i in range(size3):
            E0_nmm += rho_dot_V[i]
            for ip in range(size3):
                ei_dot_ej+=rhobeta2_e1e2_rhobeta2[i,ip]
                Z0 += rho_beta[i,ip]
        E0_nmm/=Z0

        self.E0_N3_pair_NMM = E0_nmm
        self.eiej_N3_pair_NMM = ei_dot_ej/2./Z0

        print('E0_NMM (pair) = ', E0_nmm)
        print('NMM <e1.e2>= ', self.eiej_N3_pair_NMM)

    def runNMM_N3_VBR(self):
        """
        Solves for the ground state energy using the NMM method. This method can
        only be used for a two rotor system. Warning, this method scales with Ngrid^2
        and can become quickly intractable.

        Returns
        -------
        self.E0_NMM: float
            Ground state energy calculated by the NMM method
        self.eiej_NMM: float
            The orientational correlation calculated by the NMM method.
        """
        # Throwing a warning if the MC object currently being tested does not have 2 rotors
        if self.N != 3:
            raise Warning("This NMM  method can only handle 3 rotors and the current MC simulation is not being performed with 2 rotors.")

        # Unpacking class attributes into variables
        # Unpacking class attributes into variables
        size = self.Ngrid
        size2 = self.Ngrid**2
        size3 = self.Ngrid**3
        tau = self.tau
        mMax = self.m_max
        B = self.B
        g = self.g

        rho_free_1body = np.zeros((size,size), float)
        # Creating the cos and sin potential matrices
        cos_mmp = pot_matrix_cos(size)
        sin_mmp = pot_matrix_sin(size)
        rho_free_1body = np.zeros((size,size), float)
        for i1 in range(size):
            rho_free_1body[i1, i1] = np.exp(-tau*B * float((-mMax+i1)**2))

        self.rho_free_1body=rho_free_1body

        V2_VBR = np.zeros((size2,size2), float)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m1p in range(self.Ngrid):
                    for m2p in range(self.Ngrid):
                        V2_VBR[m1*size + m2, m1p*size + m2p] += g * (-1.*sin_mmp[m1,m1p]*sin_mmp[m2,m2p] - 2.*cos_mmp[m1,m1p]*cos_mmp[m2,m2p])

        evalsV2, evecsV2 = np.linalg.eigh(V2_VBR)
        expV2_VBR_diag=np.zeros((size2,size2),float)
        for i in range(size2):
            expV2_VBR_diag[i,i]=np.exp(-.5*self.tau*evalsV2[i])
        rho2_potential=np.dot(evecsV2,np.dot(expV2_VBR_diag,np.transpose(evecsV2)))


        V_VBR = np.zeros((size3,size3), float)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m3 in range(self.Ngrid):
                    for m1p in range(self.Ngrid):
                        for m2p in range(self.Ngrid):
                            for m3p in range(self.Ngrid):
                                if (m3==m3p):
                                    V_VBR[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p] += g * (-1.*sin_mmp[m1,m1p]*sin_mmp[m2,m2p] - 2.*cos_mmp[m1,m1p]*cos_mmp[m2,m2p])
                                if (m1==m1p):
                                    V_VBR[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p] += g * (-1.*sin_mmp[m3,m3p]*sin_mmp[m2,m2p] - 2.*cos_mmp[m3,m3p]*cos_mmp[m2,m2p])

        evalsV, evecsV = np.linalg.eigh(V_VBR)
        expV_VBR_diag=np.zeros((size3,size3),float)
        for i in range(size3):
            expV_VBR_diag[i,i]=np.exp(-.5*self.tau*evalsV[i])
        rho_potential=np.dot(evecsV,np.dot(expV_VBR_diag,np.transpose(evecsV)))

        #print(rho_potential)

        # rho FK potential
        FK=True
        if (FK==True):
            rho_potential12=np.zeros((size3,size3),float)
            rho_potential23=np.zeros((size3,size3),float)
            for m1 in range(self.Ngrid):
                for m2 in range(self.Ngrid):
                    for m3 in range(self.Ngrid):
                        for m1p in range(self.Ngrid):
                            for m2p in range(self.Ngrid):
                                for m3p in range(self.Ngrid):
                                    if (m3==m3p):
                                        rho_potential12[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]=rho2_potential[(m1*size + m2), (m1p*size + m2p)]
                                    if (m1==m1p):
                                        rho_potential23[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]=rho2_potential[(m2*size + m3), (m2p*size + m3p)]
            # intermdiate beads here
            rho_potential=np.dot(rho_potential12,rho_potential23)
        # get rid of negatives, make stoquastic! no difference if set to zero and/or absolute
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m3 in range(self.Ngrid):
                    for m1p in range(self.Ngrid):
                        for m2p in range(self.Ngrid):
                            for m3p in range(self.Ngrid):
                                #print('FK',rho_potential[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p],rho_potential[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]-rho_potential_bak[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p])
                                if rho_potential[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]<-1.e-15:
                                    
                                    #print(rho_potential[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p])
                                    rho_potential[(m1*size + m2)*size+m3, (m1p*size + m2p)*size+m3p]=0.
        rho_potential=np.absolute(rho_potential)

        # Creating the 2 body free rho
        rho_free_3body = np.zeros((size3,size3), float)
        for i1 in range(size):
            for i2 in range(size):
                for i3 in range(size):
                    rho_free_3body[(i1*size+i2)*size+i3,(i1*size+i2)*size+i3] = rho_free_1body[i1,i1] * rho_free_1body[i2,i2]* rho_free_1body[i3,i3]

        # Constructing the high temperature density matrix
        rho_tau=np.zeros((size3,size3),float)
        rho_tau = np.dot(rho_potential, np.dot(rho_free_3body, rho_potential))

        # Forming the density matrix via matrix multiplication
        rho_beta=rho_tau.copy()
        rho_beta_over2=rho_tau.copy()

        for k in range(self.P-2):
            rho_beta=np.dot(rho_beta,rho_tau)
        for k in range(int((self.P-1)/2 - 1)):
            rho_beta_over2=np.dot(rho_beta_over2,rho_tau)

        rho_dot_V=np.dot(rho_beta,V_VBR)

        Z0=0.
        E0_nmm=0.

        index_psit=((mMax*size+mMax)*size+mMax)

        E0_nmm = rho_dot_V[index_psit,index_psit]
        Z0 = rho_beta[index_psit,index_psit]
        E0_nmm/=(Z0)

        self.E0_NMM_N3_VBR = E0_nmm
        print('E0_NMM_N3_VBR = ', self.E0_NMM_N3_VBR)

    def runNMM_VBR(self,PO_DVR=True):
        """
        Solves for the ground state energy using the NMM method. This method can
        only be used for a two rotor system. Warning, this method scales with Ngrid^2
        and can become quickly intractable.

        Returns
        -------
        self.E0_NMM: float
            Ground state energy calculated by the NMM method
        self.eiej_NMM: float
            The orientational correlation calculated by the NMM method.
        """
        # Throwing a warning if the MC object currently being tested does not have 2 rotors
        if self.N != 2:
            raise Warning("The exact diagonalization method can only handle 2 rotors and the current MC simulation is not being performed with 2 rotors.")

        # Unpacking class attributes into variables
        size = self.Ngrid
        size2 = self.Ngrid**2
        tau = self.tau
        mMax = self.m_max
        B = self.B
        g = self.g
     # Creating the cos and sin potential matrices
        cos_mmp = pot_matrix_cos(size)
        sin_mmp = pot_matrix_sin(size)
        rho_free_1body = np.zeros((size,size), float)
        for i1 in range(size):
            rho_free_1body[i1, i1] = np.exp(-.5*tau*B * float((-mMax+i1)**2))

        self.rho_free_1body_VBR=rho_free_1body

        V_VBR = np.zeros((size2,size2), float)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
                for m1p in range(self.Ngrid):
                    for m2p in range(self.Ngrid):
                        V_VBR[m1*size + m2, m1p*size + m2p] = g * (-1.*sin_mmp[m1,m1p]*sin_mmp[m2,m2p] - 2.*cos_mmp[m1,m1p]*cos_mmp[m2,m2p])
        self.V=V_VBR
        evalsV, evecsV = np.linalg.eigh(V_VBR)
        expV_VBR_diag=np.zeros((size2,size2),float)
        for i in range(size2):
            expV_VBR_diag[i,i]=np.exp(-self.tau*evalsV[i])
        rho_potential=np.dot(evecsV,np.dot(expV_VBR_diag,np.transpose(evecsV)))
        #rho_potential=expm(-tau*V_VBR)
        for m1 in range(self.Ngrid):
            for m2 in range(self.Ngrid):
              for m1p in range(self.Ngrid):
                  for m2p in range(self.Ngrid):
                    if rho_potential[m1*size + m2, m1p*size + m2p]<0.:
                        rho_potential[m1*size + m2, m1p*size + m2p]=0.
                        #print('negative',rho_potential[m1*size + m2, m1p*size + m2p])
        self.rho_potential=rho_potential
        # get rid of negatives, make stoquastic! no difference if set to zero and/or absolute
       # rho_potential=np.absolute(rho_potential)

        # Creating the 2 body free rho
        rho_free_2body = np.zeros((size2,size2), float)
        for i1 in range(size):
            for i2 in range(size):
                rho_free_2body[i1*size+i2,i1*size+i2] = rho_free_1body[i1,i1] * rho_free_1body[i2,i2]

        # Constructing the high temperature density matrix
        rho_tau=np.zeros((size2,size2),float)
        rho_tau_12=np.zeros((size2,size2,size2),float)
        rho_tau = np.dot(rho_free_2body, np.dot(rho_potential, rho_free_2body))

        # DMO-DVR
        if PO_DVR==True:
            rho_tau = np.dot(expV_VBR_diag,np.dot(np.dot(np.transpose(evecsV),np.dot(rho_free_2body,evecsV)),expV_VBR_diag))
            V_VBR=np.dot(np.transpose(evecsV),np.dot(V_VBR,evecsV))
        rho_beta=rho_tau.copy()
        rho_betaminus1=rho_tau.copy()
        rho_beta_over2=rho_tau.copy()

        for k in range(self.P-2):
            rho_beta=np.dot(rho_beta,rho_tau)
        for k in range(self.P-3):
            rho_betaminus1=np.dot(rho_betaminus1,rho_tau)
        for k in range(int((self.P-1)/2 - 1)):
            rho_beta_over2=np.dot(rho_beta_over2,rho_tau)

        rho_dot_V=np.dot(rho_beta,V_VBR)


        Z0=0.
        E0_nmm=0.
        index_psit=(mMax*size+mMax)

        rho_weight=np.dot(rho_potential, rho_free_2body)
        rho_weightV=np.dot(rho_weight,V_VBR)
        estimatorV=np.zeros(size2,float)
        for i in range(size2):
            if rho_weight[i,index_psit] != 0.:
                estimatorV[i]=rho_weightV[i,index_psit]/rho_weight[i,index_psit]

        E0_mixed=0.
        prob=np.zeros(size2,float)

        norm=0.
        for i in range(size2):
            E0_mixed+=rho_betaminus1[index_psit,i]*estimatorV[i]*rho_tau[i,index_psit]
            prob[i]=rho_betaminus1[index_psit,i]*rho_tau[i,index_psit]
            norm+=prob[i]
        for i in range(size2):
            prob[i]/=norm

        indexmap=np.zeros((size2,2),int)
        for i1 in range(size):
            for i2 in range(size):
                indexmap[i1*size+i2,0]=i1
                indexmap[i1*size+i2,1]=i2

        for k1 in range(size2):
            for k2 in range(size2):
                for k3 in range(size2):
                    rho_tau_12[k1,k2,k3]=rho_tau[k1,k2]*rho_tau[k2,k3]
        norm12=np.zeros((size2,size2),float)
        for k1 in range(size2):
            for k2 in range(size2):
                for k3 in range(size2):
                    norm12[k1,k3]+=rho_tau_12[k1,k2,k3]
        for k1 in range(size2):
            for k2 in range(size2):
                for k3 in range(size2):
                    rho_tau_12[k1,k2,k3]/=norm12[k1,k3]

        self.rho_tau_VBR=rho_tau_12

        self.E0_N2_prob_VBR=prob
        self.E0_N2_estim_VBR=estimatorV
        self.E0_N2_map_VBR=indexmap


        if PO_DVR==True:
            for i in range(size2):
                for ip in range(size2):
                    E0_nmm += rho_dot_V[i,ip]*evecsV[index_psit,ip]*evecsV[index_psit,i]
                    Z0 += rho_beta[i,ip]*evecsV[index_psit,ip]*evecsV[index_psit,i]
        else:
            E0_nmm = rho_dot_V[index_psit,index_psit]
            Z0 = rho_beta[index_psit,index_psit]

        E0_nmm/=(Z0)
        E0_mixed/=Z0

        self.E0_NMM_VBR = E0_nmm
        self.E0_mixed_VBR = E0_mixed
        #self.eiej_NMM = e1_dot_e2/Z0
        print('E0_NMM_VBR = ', self.E0_NMM_VBR)
        #print('NMM <e1.e2>= ', self.eiej_NMM)

    def createFreeRhoMarx(self):
        """
        Creates the free density matrix using the Marx method.
        This function will overwrite the self.rho_phi attribute of the object
        used during the Monte Carlo method. The Marx method is most accurate, and
        is used as the default in runMC.

        Returns
        -------
        self.free_rho_marx: numpy array
            Nx2 numpy array with the phi value in the 1st column and free density
            matrix values in the 2nd column
        self.rho_phi: numpy array
            Nx1 numpy array with density matrix values, used in the runMC function
        """
        # Creating the free rotor density matrix using the MARX method, most accurate
        rho_phi=np.zeros(self.Ngrid,float)
        rho_marx_out=open(os.path.join(self.path,'rhofree_marx'),'w')
        self.free_rho_marx = np.zeros((self.Ngrid, 2),float)
        for i in range(self.Ngrid):
            dphi = float(i) * self.delta_phi
            integral = 0.
            for m in range(self.m_max):
                integral += np.exp(-1./(4.*self.tau*self.B)*(dphi+2.*np.pi*float(m))**2)
            for m in range(1,self.m_max):
                integral+=np.exp(-1./(4.*self.tau*self.B)*(dphi+2.*np.pi*float(-m))**2)
            integral*=np.sqrt(1./(4.*np.pi*self.B*self.tau))
            rho_phi[i]=integral
            rho_marx_out.write(str(dphi)+' '+str(integral)+'\n')
            self.free_rho_marx[i,:] = [dphi, integral]
        rho_marx_out.close()

        # Overwrites the current rho_phi to match the marx method
        self.rho_phi = rho_phi
        return self.free_rho_marx.copy()
    def createFreeRhoDVR(self):
        """
        Creates the free density matrix using the DVR.
        This function will overwrite the self.rho_phi attribute of the object
        used during the Monte Carlo method. 

        Returns
        -------
        self.free_rho_marx: numpy array
            Nx2 numpy array with the phi value in the 1st column and free density
            matrix values in the 2nd column
        self.rho_phi: numpy array
            Nx1 numpy array with density matrix values, used in the runMC function
        """
        mMax=self.m_max
        K_DVR = K_matrix_DVR(mMax)
        size=self.Ngrid
        evalsK, evecsK = np.linalg.eigh(K_DVR)
        expK_DVR_diag=np.zeros((size,size),float)
        #physics on our side
        evalsK[0]=0.
        for i in range(size):
            expK_DVR_diag[i,i]=np.exp(-self.tau*evalsK[i])
        rho_phiDVR=np.dot(evecsK,np.dot(expK_DVR_diag,np.transpose(evecsK)))
        rho_phi=np.zeros(self.Ngrid,float)
        rho_min=0.
        for i in range(size):
            if rho_phiDVR[0,i]<rho_min:
                rho_min=rho_phiDVR[0,i]
        for i in range(size):
            rho_phi[i]=np.abs(rho_phiDVR[0,i])
        #    rho_phi[i]=(rho_phiDVR[0,i])

        self.rho_phi = rho_phi
        self.K_DVR=K_DVR
        return self.rho_phi.copy()
    def createFreeRhoSOS(self):
        """
        Creates the free density matrix using the SOS method.
        This function will overwrite the self.rho_phi attribute of the object
        used during the Monte Carlo method.

        Returns
        -------
        self.free_rho_sos: numpy array
            Nx2 numpy array with the phi value in the 1st column and free density
            matrix values in the 2nd column
        self.rho_phi: numpy array
            Nx1 numpy array with density matrix values, used in the runMC function
        """
        # Creating the free rotor density matrix using the SOS method
        rho_phi=np.zeros(self.Ngrid,float)
        rhofree_sos_out = open(os.path.join(self.path,'rhofree_sos'),'w')
        self.free_rho_sos = np.zeros((self.Ngrid, 2),float)
        for i in range(self.Ngrid):
            dphi = float(i) * self.delta_phi
            integral = 0.
            for m in range(1,self.m_max):
                integral += (2. * np.cos(float(m) * dphi)) * np.exp(-self.tau * self.B * m**2)
            integral = integral / (2.*np.pi)
            integral = integral + 1./(2.*np.pi)
            rho_phi[i]=np.fabs(integral)
            rhofree_sos_out.write(str(dphi)+' '+str(rho_phi[i])+'\n')
            self.free_rho_sos[i,:] = [dphi, rho_phi[i]]
        rhofree_sos_out.close()

        # Overwrites the current rho_phi to match the sos method
        self.rho_phi = rho_phi
        return self.free_rho_sos.copy()

    def createFreeRhoPQC(self):
        """
        Creates the free density matrix using the PQC method.
        This function will overwrite the self.rho_phi attribute of the object
        used during the Monte Carlo method.

        Returns
        -------
        self.free_rho_pqc: numpy array
            Nx2 numpy array with the phi value in the 1st column and free density
            matrix values in the 2nd column
        self.rho_phi: numpy array
            Nx1 numpy array with density matrix values, used in the runMC function
        """
        # Creating the free rotor density matrix using the PQC method
        rho_phi_pqc=np.zeros(self.Ngrid,float)
        rho_pqc_out=open(os.path.join(self.path,'rhofree_pqc'),'w')

        self.free_rho_pqc = np.zeros((self.Ngrid, 2),float)
        for i in range(self.Ngrid):
            dphi=float(i) * self.delta_phi
            rho_phi_pqc[i]=np.sqrt(1./(4.*np.pi*self.B*self.tau))*np.exp(-1./(2.*self.tau*self.B)*(1.-np.cos(dphi)))
            rho_pqc_out.write(str(dphi)+' '+str(rho_phi_pqc[i])+'\n')
            self.free_rho_pqc[i,:] = [dphi, rho_phi_pqc[i]]
        rho_pqc_out.close()

        # Overwrites the current rho_phi to match the pqc method
        self.rho_phi = rho_phi_pqc
        return self.free_rho_pqc.copy()
    

    def createRhoSOS(self):
        """
        Performs the calculation of the density matrix using the SOS method.
        Stores the values outlined below in the object and also returns the
        final density matrix. This function will overwrite the self.rho_phi
        attribute of the object used during the Monte Carlo method.

        Returns
        -------
        self.rho_sos: numpy array
            Nx2 numpy array with the phi value in the 1st column and density
            matrix values in the 2nd column
        self.rho_phi: numpy array
            Nx1 numpy array with density matrix values, used in the runMC function
        self.Z_sos: float
            Partition function calculated by the SOS method
        self.A_sos: float
            Helmholtz energy calculated by the SOS method
        self.E0_sos: float
            Ground state energy calculated by the SOS method
        self.E0_PIGS_sos: float
            Ground state energy calculated using PIGS and the SOS method
        """
        # 1 body Hamiltonian
        V = self.V0 * pot_matrix(2*self.m_max+1)
        H = V.copy()

        for m in range(self.Ngrid):
            m_value = -self.m_max+m
            H[m,m] = self.B * float(m_value**2) + self.V0 # constant potential term on diagonal
        evals, evecs = np.linalg.eigh(H)

        rho_mmp=np.zeros((self.Ngrid,self.Ngrid), float)

        Z_exact = 0.  #sum over state method
        for m in range(self.Ngrid):
            Z_exact += np.exp(-self.beta * evals[m])
            for mp in range(self.Ngrid):
                for n in range(self.Ngrid):
                    rho_mmp[m,mp]+=np.exp(-self.beta * evals[n]) * evecs[m,n] * evecs[mp,n]

        self.Z_sos = Z_exact
        self.A_sos = -(1./self.beta)*np.log(Z_exact)
        self.E0_sos = evals[0]
        if self.PIGS == True:
            Z_exact_pigs=rho_mmp[self.m_max,self.m_max]
            rho_dot_V_mmp=np.dot(rho_mmp,H)
            E0_pigs_sos=rho_dot_V_mmp[self.m_max,self.m_max]
            self.E0_PIGS_sos = E0_pigs_sos/Z_exact_pigs

        print('Z (sos) = ', self.Z_sos)
        print('A (sos) = ', self.A_sos)
        print('E0 (sos) =', self.E0_sos)
        if self.PIGS == True:
            print('E0 (pigs sos) =', self.E0_PIGS_sos)
        print(' ')

        # <phi|m><m|n> exp(-beta E n) <n|m'><m'|phi>
        rho_sos_out=open(os.path.join(self.path,'rho_sos'),'w')

        #built basis
        psi_m_phi=np.zeros((self.Ngrid,self.Ngrid),float)
        for i in range(self.Ngrid):
            for m in range(self.Ngrid):
                m_value =- self.m_max+m
                psi_m_phi[i,m] = np.cos(i * self.delta_phi*m_value) / np.sqrt(2.*np.pi)

        psi_phi=np.zeros((self.Ngrid,self.Ngrid),float)
        for i in range(self.Ngrid):
            for n in range(self.Ngrid):
                for m in range(self.Ngrid):
                    psi_phi[i,n] += evecs[m,n] * psi_m_phi[i,m]

        self.rho_sos = np.zeros((self.Ngrid, 3),float)
        for i in range(self.Ngrid):
            rho_exact=0.
            for n in range(self.Ngrid):
                rho_exact+=np.exp(-self.beta*evals[n])*(psi_phi[i,n]**2)
            rho_exact/=(Z_exact)
            rho_sos_out.write(str(i * self.delta_phi)+ ' '+str(rho_exact)+' '+str(psi_phi[i,0]**2)+'\n')
            self.rho_sos[i,:] = [i * self.delta_phi, rho_exact, psi_phi[i,0]**2]
        rho_sos_out.close()
        self.rho_phi = self.rho_sos[:,1]
        return self.rho_sos.copy()

    def createRhoNMM(self):
        """
        Creates the density and free density matrices by the NMM method. This
        function will overwrite the self.rho_phi attribute of the object used
        during the Monte Carlo method.

        Returns
        -------
        self.potential: numpy array
            Nx2 array containing the potential for the rotors
        self.rho_nmm: numpy array
            Nx2 numpy array with the phi value in the 1st column and density
            matrix values in the 2nd column
        self.free_rho_nmm: numpy array
            Nx2 numpy array with the phi value in the 1st column and free density
            matrix values in the 2nd column
        self.rho_phi: numpy array
            Nx1 numpy array with density matrix values, used in the runMC function
        self.Z_nmm:float
            Partition function calculated by the NMM method
        self.E0_nmm: float
            Ground state energy calculated by the NMM method
        """
        rho_free=np.zeros((self.Ngrid,self.Ngrid),float)
        rho_potential=np.zeros(self.Ngrid,float)
        potential=np.zeros(self.Ngrid,float)
        for i in range(self.Ngrid):
            potential[i] = self.potFunc(float(i)*self.delta_phi,self.V0)
            rho_potential[i]=np.exp(-(self.tau/2.)*potential[i])
            for ip in range(self.Ngrid):
                integral=0.
                dphi=float(i-ip)*self.delta_phi
                for m in range(self.m_max):
                    integral+=np.exp(-1./(4.*self.tau*self.B)*(dphi+2.*np.pi*float(m))**2)
                for m in range(1,self.m_max):
                    integral+=np.exp(-1./(4.*self.tau*self.B)*(dphi+2.*np.pi*float(-m))**2)
                integral*=np.sqrt(1./(4.*np.pi*self.B*self.tau))
                rho_free[i,ip]=integral

        self.potential = np.zeros((self.Ngrid, 2),float)
        #output potential to a file
        potential_out=open(os.path.join(self.path,'V'),'w')
        for i in range(self.Ngrid):
                potential_out.write(str(float(i)*self.delta_phi)+' '+str(potential[i])+'\n')
                self.potential[i,:] = [float(i)*self.delta_phi, potential[i]]
        potential_out.close()

        # construct the high temperature density matrix
        rho_tau=np.zeros((self.Ngrid,self.Ngrid),float)
        for i1 in range(self.Ngrid):
                for i2 in range(self.Ngrid):
                        rho_tau[i1,i2]=rho_potential[i1]*rho_free[i1,i2]*rho_potential[i2]

        # form the density matrix via matrix multiplication
        rho_beta=rho_tau.copy()
        for k in range(self.P-1):
            rho_beta=self.delta_phi*np.dot(rho_beta,rho_tau)

        E0_nmm=0.
        rho_dot_V=np.dot(rho_beta,potential)
        Z0=0. # pigs pseudo Z
        rho_nmm_out=open(os.path.join(self.path,'rho_nmm'),'w')
        Z_nmm=rho_beta.trace()*self.delta_phi # thermal Z
        Z_free_nmm = rho_free.trace()*self.delta_phi

        self.rho_nmm = np.zeros((self.Ngrid, 2), float)
        self.free_rho_nmm = np.zeros((self.Ngrid, 2), float)
        for i in range(self.Ngrid):
            E0_nmm += rho_dot_V[i]
            rho_nmm_out.write(str(i*self.delta_phi)+ ' '+str(rho_beta[i,i]/Z_nmm)+'\n')
            self.rho_nmm[i,:] = [i*self.delta_phi, rho_beta[i,i]/Z_nmm]
            self.free_rho_nmm[i,:] = [i*self.delta_phi, rho_free[i,i]/Z_free_nmm]
            for ip in range(self.Ngrid):
                Z0 += rho_beta[i,ip]
        self.rho_phi = self.rho_nmm[:,1]
        rho_nmm_out.close()
        E0_nmm/=Z0

        print('Z (tau) = ',Z_nmm)
        print('E0 (tau) = ',E0_nmm)
        self.Z_nmm = Z_nmm
        self.E0_nmm = E0_nmm
        E0_vs_tau_out=open('Evst','a')
        E0_vs_tau_out.write(str(self.tau)+' '+str(E0_nmm)+'\n')
        E0_vs_tau_out.close()
        print(' ')

    def createRhoVij(self):
        """
        Creates the rhoVij matrix for the nearest neighbour interactions of the system

        Returns
        -------
        self.rhoVij: numpy array
            NxN array containing the probability density based on the interaction
            potential between nearest neighbours
        self.rhoV: numpy array
            Nx1 array containing the probability density based on the on-site interactions
        self.rhoVij_half: numpy array
            NxN array containing the probability density based on the interaction potential
            using tau/2 between nearest neighbours, only applicable to PIGS
        self.rhoV_half: numpy array
            Nx1 array containing the probability density based on the on-site interactions
            using tau/2, only applicable to PIGS

        """
        # potential rho
        self.rhoV = np.zeros((self.Ngrid),float)
        for i_new in range(self.Ngrid):
            self.rhoV[i_new] = np.exp(-self.tau * (self.potFunc(float(i_new)*self.delta_phi, self.V0)))
        # rho pair
        self.rhoVij = np.zeros((self.Ngrid,self.Ngrid),float)
        for i in range(self.Ngrid):
            for j in range(self.Ngrid):
                self.rhoVij[i,j] = np.exp(-self.tau * (Vij(i*self.delta_phi, j*self.delta_phi, self.g)))

        # Creating the rhoV for the end beads, where they only have tau/2 interactions
        if self.PIGS:
            self.rhoV_half = np.sqrt(self.rhoV)
            self.rhoVij_half = np.sqrt(self.rhoVij)

    def runMC(self, averagePotential = True, averageEnergy = True, orientationalCorrelations = True, initialState='random'):
        """
        Performs the monte carlo integration to simulate the system.

        Parameters
        ----------
        averagePotential : bool, optional
            Enables the tracking and calculation of the average potential.
            The default is True.
        averageEnergy : bool, optional
            Enables the tracking and calculation of the average energy.
            The default is True.
        orientationalCorrelations : bool, optional
            Enables the tracking and calculation of the orientational correllations.
            The default is True.
        initialState : string, optional
            Selects the distribution of the initial state of the system. The allowed
            options are random, ordered_pi or ordered_0.
            The default is random.

        Returns
        -------
        self.V_MC: float
            The resultant average system potential.
        self.V_stdError_MC: float
            The resultant standard error in the system potential.
        self.E_MC: float
            The resultant average system energy.
        self.E_stdError_MC: float
            The resultant standard error in the system energy.
        self.eiej_MC: float
            The resultant average system orientational correlation.
        self.eiej_stdError_MC: float
            The resultant standard error in the system orientational correlation.
        self.histo_N: dict
            Dictionary of Nx5 numpy arrays containing the following histograms
            for each individual rotor:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_total: numpy array
            Nx5 array for the entire systemcontaining:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_initial: numpy array
            Nx2 array containing:
                1st column: The angle phi
                2nd column: Initial overall histogram

        Outputs
        -------
        histo_A_P_N: Nx2 txt file
            A saved version of the self.histo outlined above.
        traj_A: Nx3 dat file
            The phi values of the left, right and middle beads in columns 1, 2
            and 3 respectively, with each row corresponding to a specific rotor
            during a specific MC step.

        """

        # Creating histograms to store each rotor's distributions
        histoLeft_N = {}
        histoRight_N = {}
        histoMiddle_N = {}
        histoPIMC_N = {}
        for n in range(self.N):
            histoLeft_N.update({n: np.zeros(self.Ngrid,float)})
            histoRight_N.update({n: np.zeros(self.Ngrid,float)})
            histoMiddle_N.update({n: np.zeros(self.Ngrid,float)})
            histoPIMC_N.update({n: np.zeros(self.Ngrid,float)})

        # Creating a histogram that stores the initial distribution
        histo_initial=np.zeros(self.Ngrid,float)

        if not hasattr(self, 'rho_phi'):
            #self.createFreeRhoMarx()
            self.createFreeRhoDVR()

        if not hasattr(self, 'rhoVij'):
            self.createRhoVij()

        if not self.PIGS:
            averageEnergy = False

        p_dist=gen_prob_dist(self.Ngrid, self.rho_phi)
        p_dist_end = gen_prob_dist_end(self.Ngrid, self.rho_phi) if self.PIGS == True else None

        self.p_dist=p_dist

        #path_phi=np.zeros((self.N,self.P),int) ## i  N => number of beads
        path_phi=np.zeros((self.N,self.P),dtype=np.uint8) ## i  N => number of beads

        # Initial conditions
        if initialState == 'random':
            # Rotors have random angles
            for i in range(self.N):
                for p in range(self.P):
                    path_phi[i,p]=np.random.randint(self.Ngrid)
                    histo_initial[path_phi[i,p]]+=1. #why adding 1 on the randome index
        elif initialState == 'ordered_0':
            # All rotors have angle of 0
            histo_initial[0] += 1.
        elif initialState == 'ordered_pi':
            # All rotors have angle of pi
            path_phi += int(self.Ngrid/2)
            histo_initial[int(self.Ngrid/2)] += 1.
        else:
            raise Exception("An invalid selection was made for the initial conditions, please use random, ordered_0 or ordered_pi")

        traj_out=open(os.path.join(self.path,'traj_A.dat'),'w')
        log_out=open(os.path.join(self.path,'MC.log'),'w')

        circles_out=open(os.path.join(self.path,'circles.dat'),'w')
        circles_grid_out=open(os.path.join(self.path,'circles_grid.dat'),'w')

        for i in range(self.N):
            for p in range(self.P):
                xcm=3+i*3
                ycm=3+p*3
                circles_out.write(str(xcm)+' '+str(ycm)+' 1'+'\n')
                for a in range(self.Ngrid):
                    phi=a*self.delta_phi
                    x=np.cos(phi)+xcm
                    y=np.sin(phi)+ycm
                    circles_grid_out.write(str(x)+' '+str(y)+' .2'+'\n')

        # recommanded numpy random number initialization
        rng = default_rng()

        #print('start MC')

        # Initializing the estimators, the method currently employed is only for pigs
        V_arr = np.zeros(self.MC_steps,float) if averagePotential == True else None
        E_arr = np.zeros(self.MC_steps,float) if averageEnergy == True else None
        eiej_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        zizj_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        polarization_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        chi_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        Vchi_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None

        P_middle = int((self.P-1)/2)
        prob_full=np.zeros(self.Ngrid,float)

        for n in range(self.MC_steps):
            for i in range(self.N):
                for p in range(self.P):
                    #print(n,i,p)
                    p_minus=p-1
                    p_plus=p+1
                    if (p_minus<0):
                        p_minus+=self.P
                    if (p_plus>=self.P):
                        p_plus-=self.P

                    # kinetic action, links between beads
                    if self.PIGS==True:
                        # This uses a split open path of beads for PIGS
                        if p==0:
                            # Special case for leftmost bead
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist_end[ip,path_phi[i,p_plus]]
                        elif p==(self.P-1):
                            # Special case for rightmost bead
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist_end[path_phi[i,p_minus],ip]
                        elif (p!=0 and p!= (self.P-1)):
                            # All other beads
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]
                    else:
                        # Regular kinetic interactions between beads, periodic conditions
                        for ip in range(self.Ngrid):
                            prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]

                    # Local on site interaction with the potential field
                    if self.V0 != 0.:
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ip in range(self.Ngrid):
                                prob_full[ip]*=self.rhoV_half[ip]
                        else:
                            for ip in range(self.Ngrid):
                                prob_full[ip]*=self.rhoV[ip]

                    # NN interactions and PBC(periodic boundary conditions)
                    if (i<(self.N-1)):
                        # Interaction with the rotor to the right
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[i+1,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[i+1,p]]
                    if (i>0):
                        # Interaction with the rotor to the left
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[i-1,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[i-1,p]]
                    if (i==0) and (self.N>2) and self.PBC==True:
                        # Periodic BC for the leftmost rotor, doesn't apply to the 2 rotor system
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[self.N-1,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[self.N-1,p]]
                    if (i==(self.N-1)) and (self.N>2) and self.PBC==True:
                        # Periodic BC for the rightmost rotor, doesn't apply to the 2 rotor system
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[0,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[0,p]]
                    # Normalize
                    norm_pro=0.
                    for ir in range(len(prob_full)):
                        norm_pro+=prob_full[ir]
                    for ir in range(len(prob_full)):
                        prob_full[ir]/=norm_pro
                    #index=rng.choice(self.Ngrid,1, p=prob_full,shuffle=False)
                    index=rng.choice(self.Ngrid,1, p=prob_full)
                    # Rejection free sampling
                    path_phi[i,p] = index

                    histoPIMC_N[i][path_phi[i,p]]+=1.

                # End of bead loop

                # Adding to the histogram counts.
                histoLeft_N[i][path_phi[i,0]]+=1.
                histoRight_N[i][path_phi[i,self.P-1]]+=1.
                histoMiddle_N[i][path_phi[i,P_middle]]+=1.

        # End of rotor loop
            # all beads
            if (n%self.Nskip==0):
                for i in range(self.N):
                    for p in range(self.P):
                        xcm=3+i*3
                        ycm=3+p*3
                        phi=path_phi[i,p]*self.delta_phi
                        x=np.cos(phi)+xcm
                        y=np.sin(phi)+ycm
                        traj_out.write(str(x)+' '+str(y)+' '+str(xcm)+' '+str(ycm)+ '\n')

            traj_out.write('\n')
            traj_out.write('\n')

            # Updating the estimators, these only look at the interactions to the left to avoid
            # double counting and to ensure that the interactions being added are from the current MC Step
            if n >= self.Nequilibrate and averagePotential == True and orientationalCorrelations == True:
                # External field
                Vlocal=0.
                Vchi=np.zeros(self.P,float)
                for i in range(self.N):
                    if self.PIGS==True:
                        Vlocal+= self.potFunc(float(path_phi[i,P_middle])*self.delta_phi,self.V0)
                        #V_arr[n] += self.potFunc(float(path_phi[i,P_middle])*self.delta_phi,self.V0)
                    else:
                        for p in range(self.P):
                            Vlocal += self.potFunc(float(path_phi[i,p])*self.delta_phi,self.V0)
                    # Nearest neighbour interactions
                    if (i>0):
                        # Only look at left neighbour to avoid double counting
                        if self.PIGS==True:
                            p=P_middle
                            Vlocal += Vij(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                Vlocal += Vij(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi, self.g)
                    if (i==(self.N-1)) and (self.N>2 and self.PBC==True):
                        # Periodic BCs
                        if self.PIGS==True:
                            p=P_middle
                            Vlocal += Vij(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                Vlocal += Vij(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
                    for p in range(self.P):
                        if p==0 or p==(self.P-1):
                            if (i>0):
                                Vchi[p] += .5*Vij(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi, self.g)
                        else:
                            if (i>0):
                                Vchi[p] += Vij(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi, self.g)

                V_arr[n]+=Vlocal
                eiej_local=0.
                for i in range(self.N):
                    if self.PBC==True:
                        if (i>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_local += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_local += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)
                        if (i==(self.N-1)) and (self.N>2):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_local += calculateOrientationalCorrelations(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[i,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_local += calculateOrientationalCorrelations(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[i,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                    else:
                        if self.PIGS==True:
                            p=P_middle
                            polarization_arr[n] += np.cos(path_phi[i,p]*self.delta_phi)

                        if (i>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_local += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_local += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)
                eiej_arr[n]+=eiej_local
                for p in range(self.P):
                    chi_arr[n]+=eiej_local*Vchi[p]
                    Vchi_arr[n]+=Vchi[p]
            # Updating the ground state energy estimators for PIGS
            if n >= self.Nequilibrate and averageEnergy == True and self.PIGS==True:
                for i in range(self.N):
                # External field
                    E_arr[n] += self.potFunc(float(path_phi[i,0])*self.delta_phi,self.V0)
                    E_arr[n] += self.potFunc(float(path_phi[i,self.P-1])*self.delta_phi,self.V0)
                # Nearest neighbour interactions, only looks at left neighbour to avoid double counting
                    if (i>0):
                        E_arr[n] += Vij(path_phi[i-1,0]*self.delta_phi, path_phi[i,0]*self.delta_phi, self.g)
                        E_arr[n] += Vij(path_phi[i-1,self.P-1]*self.delta_phi, path_phi[i,self.P-1]*self.delta_phi, self.g)
                        if self.PBC==True:
                            if (i==(self.N-1)) and (self.N>2):
                                E_arr[n] += Vij(path_phi[i,0]*self.delta_phi, path_phi[0,0]*self.delta_phi, self.g)
                                E_arr[n] += Vij(path_phi[i,self.P-1]*self.delta_phi, path_phi[0,self.P-1]*self.delta_phi, self.g)
                #print(E_arr[n]/2)

        traj_out.close()
        circles_out.close()
        circles_grid_out.close()

        if averagePotential == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            meanV, stdErrV = calculateError_byBinning(V_arr[self.Nequilibrate:])
            if not self.PIGS:
                meanV /= self.P
                stdErrV /= self.P
            log_out.write('<V> = '+str(meanV)+'\n')
            log_out.write('V_SE = '+str(stdErrV)+'\n')
            self.V_MC = meanV
            self.V_stdError_MC = stdErrV
        if averageEnergy == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            # Need to divide by two according to the PIGS formula
            E_arr/=2
            self.E_arr=E_arr
            meanE, stdErrE = calculateError_byBinning(E_arr[self.Nequilibrate:])
            log_out.write('E0 = '+str(meanE)+'\n')
            log_out.write('E0_SE = '+str(stdErrE)+'\n')
            self.E_MC = meanE
            self.E_stdError_MC = stdErrE
        if orientationalCorrelations == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1
            meaneiej, stdErreiej = calculateError_byBinning(eiej_arr[self.Nequilibrate:]/Ninteractions)
            meanchi, stdErrchi = calculateError_byBinning(chi_arr[self.Nequilibrate:]/Ninteractions)
            meanVchi, stdErrVchi = calculateError_byBinning(Vchi_arr[self.Nequilibrate:])
            meanzizj, stdErrzizj = calculateError_byBinning(zizj_arr[self.Nequilibrate:]/Ninteractions)
            meanpolarization, stdErrpolarization = calculateError_byBinning(polarization_arr[self.Nequilibrate:]/Ninteractions)
            if not self.PIGS:
                meaneiej /= self.P
                meanzizj /= self.P
                meanpolarization/= self.P
                stdErreiej /= self.P
                stdErrzizj /= self.P
                stdErrpolarization/= self.P
            log_out.write('<ei.ej> = '+str(meaneiej)+'\n')
            log_out.write('ei.ej_SE = '+str(stdErreiej)+'\n')
            log_out.write('<zi zj> = '+str(meanzizj)+'\n')
            log_out.write('zi zj_SE = '+str(stdErrzizj)+'\n')
            self.eiej_arr=eiej_arr
            self.polarization_arr=polarization_arr
            self.chi_arr=chi_arr
            self.Vchi_arr=Vchi_arr
            self.eiej_MC = meaneiej
            self.eiej_stdError_MC = stdErreiej
            self.chi_MC=meanchi
            self.chi_stdError_MC=stdErrchi
            self.Vchi_MC=meanVchi
            self.Vchi_stdError_MC=stdErrVchi
            self.polarization_MC=meanpolarization
            self.polarization_stdError_MC=stdErrpolarization

        # Creating arrays to store the overall system's distribution
        histoPIMC_total=np.zeros(self.Ngrid,float)
        histoMiddle_total=np.zeros(self.Ngrid,float)
        histoLeft_total=np.zeros(self.Ngrid,float)
        histoRight_total=np.zeros(self.Ngrid,float)

        # Saving the individual rotor distributions and accumulating the total distributions
        self.histo_N = {}
        for n in range(self.N):
            histoPIMC_total += histoPIMC_N[n]
            histoMiddle_total += histoMiddle_N[n]
            histoLeft_total += histoLeft_N[n]
            histoRight_total += histoRight_N[n]

            histoN_arr = np.zeros((self.Ngrid,5))
            histoN_out = open(os.path.join(self.path,'histo_N'+str(n)),'w')
            for i in range(self.Ngrid):
                histoN_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi) + ' ' +
                                 str(histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoRight_N[n][i]/(self.MC_steps)/self.delta_phi) +'\n')
                histoN_arr[i,:] = [i*self.delta_phi,
                                   histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi,
                                   histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoRight_N[n][i]/(self.MC_steps)/self.delta_phi]
            self.histo_N.update({n: histoN_arr})
            histoN_out.close()

        # Saving the overall and initial distributions
        self.histo_total = np.zeros((self.Ngrid,5))
        self.histo_initial = np.zeros((self.Ngrid,2))
        histo_out = open(os.path.join(self.path,'histo_test_total'),'w')
        histo_init_out = open(os.path.join(self.path,'histo_initial'),'w')
        for i in range(self.Ngrid):
            histo_out.write(str(i*self.delta_phi) + ' ' +
                            str(histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi) + ' ' +
                            str(histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi) + '\n')
            self.histo_total[i,:] = [i*self.delta_phi,
                                    histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi,
                                    histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi]
            histo_init_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histo_initial[i]/(self.N*self.P)/self.delta_phi)+'\n')
            self.histo_initial[i,:] = [i*self.delta_phi,
                                       histo_initial[i]/(self.N*self.P)/self.delta_phi]
        histo_out.close()
        log_out.close()

    def run_MH_MC(self, averagePotential = True, averageEnergy = True, orientationalCorrelations = True, initialState='random'):
        """
        Performs the monte carlo integration to simulate the system.

        Parameters
        ----------
        averagePotential : bool, optional
            Enables the tracking and calculation of the average potential.
            The default is True.
        averageEnergy : bool, optional
            Enables the tracking and calculation of the average energy.
            The default is True.
        orientationalCorrelations : bool, optional
            Enables the tracking and calculation of the orientational correllations.
            The default is True.
        initialState : string, optional
            Selects the distribution of the initial state of the system. The allowed
            options are random, ordered_pi or ordered_0.
            The default is random.

        Returns
        -------
        self.V_MC: float
            The resultant average system potential.
        self.V_stdError_MC: float
            The resultant standard error in the system potential.
        self.E_MC: float
            The resultant average system energy.
        self.E_stdError_MC: float
            The resultant standard error in the system energy.
        self.eiej_MC: float
            The resultant average system orientational correlation.
        self.eiej_stdError_MC: float
            The resultant standard error in the system orientational correlation.
        self.histo_N: dict
            Dictionary of Nx5 numpy arrays containing the following histograms
            for each individual rotor:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_total: numpy array
            Nx5 array for the entire systemcontaining:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_initial: numpy array
            Nx2 array containing:
                1st column: The angle phi
                2nd column: Initial overall histogram

        Outputs
        -------
        histo_A_P_N: Nx2 txt file
            A saved version of the self.histo outlined above.
        traj_A: Nx3 dat file
            The phi values of the left, right and middle beads in columns 1, 2
            and 3 respectively, with each row corresponding to a specific rotor
            during a specific MC step.

        """

        # Creating histograms to store each rotor's distributions
        histoLeft_N = {}
        histoRight_N = {}
        histoMiddle_N = {}
        histoPIMC_N = {}
        for n in range(self.N):
            histoLeft_N.update({n: np.zeros(self.Ngrid,float)})
            histoRight_N.update({n: np.zeros(self.Ngrid,float)})
            histoMiddle_N.update({n: np.zeros(self.Ngrid,float)})
            histoPIMC_N.update({n: np.zeros(self.Ngrid,float)})

        # Creating a histogram that stores the initial distribution
        histo_initial=np.zeros(self.Ngrid,float)

        #if not hasattr(self, 'rho_phi'):
        self.createFreeRhoMarx()

        if not hasattr(self, 'rhoVij'):
            self.createRhoVij()

        if not self.PIGS:
            averageEnergy = False

        path_phi=np.zeros((self.N,self.P),float) ## i  N => number of beads

        # recommanded numpy random number initialization
        rng = default_rng()
        # Initial conditions
        if initialState == 'random':
            # Rotors have random angles
            for i in range(self.N):
                for p in range(self.P):
                    path_phi[i,p]=rng.random()*(2.*np.pi)
                    histo_initial[path_phi[i,p]]+=1. #why adding 1 on the randome index
        elif initialState == 'ordered_0':
            # All rotors have angle of 0
            histo_initial[0] += 1.
        elif initialState == 'ordered_pi':
            # All rotors have angle of pi
            path_phi += np.pi
            histo_initial[int(self.Ngrid/2)] += 1.
        else:
            raise Exception("An invalid selection was made for the initial conditions, please use random, ordered_0 or ordered_pi")

        traj_out=open(os.path.join(self.path,'traj_A.dat'),'w')
        log_out=open(os.path.join(self.path,'MC.log'),'w')

        phi_step=3.
        two_pi=2.*np.pi

        #print('start MC')

        # Initializing the estimators, the method currently employed is only for pigs
        V_arr = np.zeros(self.MC_steps,float) if averagePotential == True else None
        E_arr = np.zeros(self.MC_steps,float) if averageEnergy == True else None
        eiej_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        zizj_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        polarization_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None

        P_middle = int((self.P-1)/2)

        Naccept=0
        ncount=0
        for n in range(self.MC_steps):
            for i in range(self.N):
                for p in range(self.P):
                    #print(n,i,p)
                    p_minus=p-1
                    p_plus=p+1
                    if (p_minus<0):
                        p_minus+=self.P
                    if (p_plus>=self.P):
                        p_plus-=self.P

                    old_phi=path_phi[i,p]
                    index_old=int(old_phi/self.delta_phi)
                    if index_old >=self.Ngrid:
                        index_old=0

                    new_phi=old_phi+phi_step*(rng.random()-.5)
                    if new_phi > two_pi:
                        new_phi=new_phi-two_pi
                    if new_phi < 0:
                        new_phi=new_phi+two_pi 
                    
                    index_new=int(new_phi/self.delta_phi)
                    if index_new >=self.Ngrid:
                        index_new=0

                    # kinetic action, links between beads
                    if self.PIGS==True:
                        # This uses a split open path of beads for PIGS
                        if p==0:
                            index_old_plus=int((old_phi-path_phi[i,p_plus])/self.delta_phi)
                            if index_old_plus < 0:
                                index_old_plus+=self.Ngrid
                            index_new_plus=int((new_phi-path_phi[i,p_plus])/self.delta_phi)
                            if index_new_plus < 0:
                                index_new_plus+=self.Ngrid
                            rho_old=self.rho_phi[index_old_plus]
                            rho_new=self.rho_phi[index_new_plus]

                        elif p==(self.P-1):
                            # Special case for rightmost bead
                            index_old_minus=int((old_phi-path_phi[i,p_minus])/self.delta_phi)
                            if index_old_minus < 0:
                                index_old_minus+=self.Ngrid
                            index_new_minus=int((new_phi-path_phi[i,p_minus])/self.delta_phi)
                            if index_new_minus < 0:
                                index_new_minus+=self.Ngrid
                            rho_old=self.rho_phi[index_old_minus]
                            rho_new=self.rho_phi[index_new_minus]
                        elif (p!=0 and p!= (self.P-1)):
                            # All other beads
                            index_old_plus=int((old_phi-path_phi[i,p_plus])/self.delta_phi)
                            if index_old_plus < 0:
                                index_old_plus+=self.Ngrid
                            index_new_plus=int((new_phi-path_phi[i,p_plus])/self.delta_phi)
                            if index_new_plus < 0:
                                index_new_plus+=self.Ngrid
                            rho_old=self.rho_phi[index_old_plus]
                            rho_new=self.rho_phi[index_new_plus]
                            index_old_minus=int((old_phi-path_phi[i,p_minus])/self.delta_phi)
                            if index_old_minus < 0:
                                index_old_minus+=self.Ngrid
                            index_new_minus=int((new_phi-path_phi[i,p_minus])/self.delta_phi)
                            if index_new_minus < 0:
                                index_new_minus+=self.Ngrid
                            rho_old*=self.rho_phi[index_old_minus]
                            rho_new*=self.rho_phi[index_new_minus]

                    else:
                        # Regular kinetic interactions between beads, periodic conditions
                        index_old_plus=int((old_phi-path_phi[i,p_plus])/self.delta_phi)
                        if index_old_plus < 0:
                            index_old_plus+=self.Ngrid
                        index_new_plus=int((new_phi-path_phi[i,p_plus])/self.delta_phi)
                        if index_new_plus < 0:
                            index_new_plus+=self.Ngrid
                        rho_old=self.rho_phi[index_old_plus]
                        rho_new=self.rho_phi[index_new_plus]
                        index_old_minus=int((old_phi-path_phi[i,p_minus])/self.delta_phi)
                        if index_old_minus < 0:
                            index_old_minus+=self.Ngrid
                        index_new_minus=int((new_phi-path_phi[i,p_minus])/self.delta_phi)
                        if index_new_minus < 0:
                            index_new_minus+=self.Ngrid
                        rho_old=self.rho_phi[index_old_minus]
                        rho_new=self.rho_phi[index_new_minus]

                    # Local on site interaction with the potential field
                    if self.V0 != 0.:
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            rho_old*=self.rhoV_half[index_old]
                            rho_new*=self.rhoV_half[index_new]
                        else:
                            rho_old*=self.rhoV[index_old]
                            rho_new*=self.rhoV[index_new]

                    # NN interactions and PBC(periodic boundary conditions)
                    if (i<(self.N-1)):
                        index_i_plus_1=int(path_phi[i+1,p]/self.delta_phi)
                        if index_i_plus_1 >=self.Ngrid:
                            index_i_plus_1=0
                        # Interaction with the rotor to the right
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            rho_old*=self.rhoVij_half[index_old,index_i_plus_1]
                            rho_new*=self.rhoVij_half[index_new,index_i_plus_1]
                        else:
                            rho_old*=self.rhoVij[index_old,index_i_plus_1]
                            rho_new*=self.rhoVij[index_new,index_i_plus_1]
                    if (i>0):
                        index_i_minus_1=int(path_phi[i-1,p]/self.delta_phi)
                        if index_i_minus_1 >=self.Ngrid:
                            index_i_minus_1=0
                        # Interaction with the rotor to the left
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            rho_old*=self.rhoVij_half[index_old,index_i_minus_1]
                            rho_new*=self.rhoVij_half[index_new,index_i_minus_1]
                        else:
                            rho_old*=self.rhoVij[index_old,index_i_minus_1]
                            rho_new*=self.rhoVij[index_new,index_i_minus_1]
                    #PBC not implemented
                    # if (i==0) and (self.N>2) and self.PBC==True:
                    #     # Periodic BC for the leftmost rotor, doesn't apply to the 2 rotor system
                    #     if (p==0 or p==(self.P-1)) and self.PIGS:
                    #         # Half interaction at the end beads
                    #         for ir in range(len(prob_full)):
                    #             prob_full[ir]*=self.rhoVij_half[ir,path_phi[self.N-1,p]]
                    #     else:
                    #         for ir in range(len(prob_full)):
                    #             prob_full[ir]*=self.rhoVij[ir,path_phi[self.N-1,p]]
                    # if (i==(self.N-1)) and (self.N>2) and self.PBC==True:
                    #     # Periodic BC for the rightmost rotor, doesn't apply to the 2 rotor system
                    #     if (p==0 or p==(self.P-1)) and self.PIGS:
                    #         # Half interaction at the end beads
                    #         for ir in range(len(prob_full)):
                    #             prob_full[ir]*=self.rhoVij_half[ir,path_phi[0,p]]
                    #     else:
                    #         for ir in range(len(prob_full)):
                    #             prob_full[ir]*=self.rhoVij[ir,path_phi[0,p]]
                    # Normalize
                   
                    ratio=rho_new/rho_old

                    index_old
                    if ratio >1.:
                        path_phi[i,p]=new_phi
                        Naccept+=1
                        index_old=index_new
                    else:
                        if ratio >rng.random():
                            path_phi[i,p]=new_phi
                            Naccept+=1
                            index_old=index_new
                    # Adding to the histogram counts.
                    if (p==0):
                        histoLeft_N[i][index_old]+=1.
                    if (p==(self.P-1)):
                        histoRight_N[i][index_old]+=1.
                    if (p==P_middle):
                        histoMiddle_N[i][index_old]+=1.
                    #histoPIMC_N[i][index_old]+=1.
                # End of bead loop
        # End of rotor loop
            if (n%self.Nskip==0):
                ncount+=1
                traj_out.write(str(ncount)+' ')
                for i in range(self.N):
                    traj_out.write(str(path_phi[i,0])+' ')
                    traj_out.write(str(path_phi[i,self.P-1])+' ')
                    traj_out.write(str(path_phi[i,P_middle])+' ') #middle bead
                traj_out.write('\n')

            # Updating the estimators, these only look at the interactions to the left to avoid
            # double counting and to ensure that the interactions being added are from the current MC Step
            if n >= self.Nequilibrate and averagePotential == True:
                # External field
                for i in range(self.N):
                    if self.PIGS==True:
                        V_arr[n] += self.potFunc(float(path_phi[i,P_middle]),self.V0)
                    else:
                        for p in range(self.P):
                            V_arr[n] += self.potFunc(float(path_phi[i,p]),self.V0)
                    # Nearest neighbour interactions
                    if (i>0):
                        # Only look at left neighbour to avoid double counting
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[i-1,p], path_phi[i,p], self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[i-1,p], path_phi[i,p], self.g)
                    if (i==(self.N-1)) and (self.N>2 and self.PBC==True):
                        # Periodic BCs
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[i,p], path_phi[0,p], self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[i,p], path_phi[0,p], self.g)
            if n >= self.Nequilibrate and orientationalCorrelations == True:
                for i in range(self.N):
                    if self.PBC==True:
                        if (i>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p], path_phi[i,p])
                                zizj_arr[n] += np.cos(path_phi[i-1,p])*np.cos(path_phi[i,p])
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p], path_phi[i,p])
                                    zizj_arr[n] += np.cos(path_phi[i-1,p])*np.cos(path_phi[i,p])
                        if (i==(self.N-1)) and (self.N>2):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i,p], path_phi[0,p])
                                zizj_arr[n] += np.cos(path_phi[i,p])*np.cos(path_phi[0,p])
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i,p], path_phi[0,p])
                                    zizj_arr[n] += np.cos(path_phi[i,p])*np.cos(path_phi[0,p])
                    else:
                        if self.PIGS==True:
                            p=P_middle
                            polarization_arr[n] += np.cos(path_phi[i,p])

                        if (i>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p], path_phi[i,p])
                                zizj_arr[n] += np.cos(path_phi[i-1,p])*np.cos(path_phi[i,p])
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p], path_phi[i,p])
                                    zizj_arr[n] += np.cos(path_phi[i-1,p])*np.cos(path_phi[i,p])

            # Updating the ground state energy estimators for PIGS
            if n >= self.Nequilibrate and averageEnergy == True and self.PIGS==True:
                for i in range(self.N):
                # External field
                    E_arr[n] += self.potFunc(float(path_phi[i,0]),self.V0)
                    E_arr[n] += self.potFunc(float(path_phi[i,self.P-1]),self.V0)
                # Nearest neighbour interactions, only looks at left neighbour to avoid double counting
                    if (i>0):
                        E_arr[n] += Vij(path_phi[i-1,0], path_phi[i,0], self.g)
                        E_arr[n] += Vij(path_phi[i-1,self.P-1], path_phi[i,self.P-1], self.g)
                        if self.PBC==True:
                            if (i==(self.N-1)) and (self.N>2):
                                E_arr[n] += Vij(path_phi[i,0], path_phi[0,0], self.g)
                                E_arr[n] += Vij(path_phi[i,self.P-1], path_phi[0,self.P-1], self.g)
                #print(E_arr[n]/2)

        traj_out.close()

        print("Acceptance ratio:",Naccept/(self.P*self.N*self.MC_steps))

        if averagePotential == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            meanV, stdErrV = calculateError_byBinning(V_arr[self.Nequilibrate:])
            if not self.PIGS:
                meanV /= self.P
                stdErrV /= self.P
            log_out.write('<V> = '+str(meanV)+'\n')
            log_out.write('V_SE = '+str(stdErrV)+'\n')
            self.V_MC = meanV
            self.V_stdError_MC = stdErrV
        if averageEnergy == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            # Need to divide by two according to the PIGS formula
            E_arr/=2
            self.E_arr=E_arr
            meanE, stdErrE = calculateError_byBinning(E_arr[self.Nequilibrate:])
            log_out.write('E0 = '+str(meanE)+'\n')
            log_out.write('E0_SE = '+str(stdErrE)+'\n')
            self.E_MC = meanE
            self.E_stdError_MC = stdErrE
        if orientationalCorrelations == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1
            meaneiej, stdErreiej = calculateError_byBinning(eiej_arr[self.Nequilibrate:]/Ninteractions)
            meanzizj, stdErrzizj = calculateError_byBinning(zizj_arr[self.Nequilibrate:]/Ninteractions)
            meanpolarization, stdErrpolarization = calculateError_byBinning(polarization_arr[self.Nequilibrate:]/Ninteractions)
            if not self.PIGS:
                meaneiej /= self.P
                meanzizj /= self.P
                meanpolarization/= self.P
                stdErreiej /= self.P
                stdErrzizj /= self.P
                stdErrpolarization/= self.P
            log_out.write('<ei.ej> = '+str(meaneiej)+'\n')
            log_out.write('ei.ej_SE = '+str(stdErreiej)+'\n')
            log_out.write('<zi zj> = '+str(meanzizj)+'\n')
            log_out.write('zi zj_SE = '+str(stdErrzizj)+'\n')
            self.eiej_arr=eiej_arr
            self.eiej_MC = meaneiej
            self.polarization_arr=polarization_arr
            self.eiej_stdError_MC = stdErreiej
            self.polarization_MC=meanpolarization
            self.polarization_stdError_MC=stdErrpolarization

        # Creating arrays to store the overall system's distribution
        histoPIMC_total=np.zeros(self.Ngrid,float)
        histoMiddle_total=np.zeros(self.Ngrid,float)
        histoLeft_total=np.zeros(self.Ngrid,float)
        histoRight_total=np.zeros(self.Ngrid,float)

        # Saving the individual rotor distributions and accumulating the total distributions
        self.histo_N = {}
        for n in range(self.N):
            histoPIMC_total += histoPIMC_N[n]
            histoMiddle_total += histoMiddle_N[n]
            histoLeft_total += histoLeft_N[n]
            histoRight_total += histoRight_N[n]

            histoN_arr = np.zeros((self.Ngrid,5))
            histoN_out = open(os.path.join(self.path,'histo_N'+str(n)),'w')
            for i in range(self.Ngrid):
                histoN_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi) + ' ' +
                                 str(histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoRight_N[n][i]/(self.MC_steps)/self.delta_phi) +'\n')
                histoN_arr[i,:] = [i*self.delta_phi,
                                   histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi,
                                   histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoRight_N[n][i]/(self.MC_steps)/self.delta_phi]
            self.histo_N.update({n: histoN_arr})
            histoN_out.close()

        # Saving the overall and initial distributions
        self.histo_total = np.zeros((self.Ngrid,5))
        self.histo_initial = np.zeros((self.Ngrid,2))
        histo_out = open(os.path.join(self.path,'histo_test_total'),'w')
        histo_init_out = open(os.path.join(self.path,'histo_initial'),'w')
        for i in range(self.Ngrid):
            histo_out.write(str(i*self.delta_phi) + ' ' +
                            str(histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi) + ' ' +
                            str(histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi) + '\n')
            self.histo_total[i,:] = [i*self.delta_phi,
                                    histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi,
                                    histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi]
            histo_init_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histo_initial[i]/(self.N*self.P)/self.delta_phi)+'\n')
            self.histo_initial[i,:] = [i*self.delta_phi,
                                       histo_initial[i]/(self.N*self.P)/self.delta_phi]
        histo_out.close()
        log_out.close()

    
    def runMC2x1(self, averagePotential = True, averageEnergy = True, orientationalCorrelations = True, initialState='random'):
        """
        Performs the monte carlo integration to simulate the system.

        Parameters
        ----------
        averagePotential : bool, optional
            Enables the tracking and calculation of the average potential.
            The default is True.
        averageEnergy : bool, optional
            Enables the tracking and calculation of the average energy.
            The default is True.
        orientationalCorrelations : bool, optional
            Enables the tracking and calculation of the orientational correllations.
            The default is True.
        initialState : string, optional
            Selects the distribution of the initial state of the system. The allowed
            options are random, ordered_pi or ordered_0.
            The default is random.

        Returns
        -------
        self.V_MC: float
            The resultant average system potential.
        self.V_stdError_MC: float
            The resultant standard error in the system potential.
        self.E_MC: float
            The resultant average system energy.
        self.E_stdError_MC: float
            The resultant standard error in the system energy.
        self.eiej_MC: float
            The resultant average system orientational correlation.
        self.eiej_stdError_MC: float
            The resultant standard error in the system orientational correlation.
        self.histo_N: dict
            Dictionary of Nx5 numpy arrays containing the following histograms
            for each individual rotor:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_total: numpy array
            Nx5 array for the entire systemcontaining:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_initial: numpy array
            Nx2 array containing:
                1st column: The angle phi
                2nd column: Initial overall histogram

        Outputs
        -------
        histo_A_P_N: Nx2 txt file
            A saved version of the self.histo outlined above.
        traj_A: Nx3 dat file
            The phi values of the left, right and middle beads in columns 1, 2
            and 3 respectively, with each row corresponding to a specific rotor
            during a specific MC step.

        """

        # Creating histograms to store each rotor's distributions
        histoLeft_N = {}
        histoRight_N = {}
        histoMiddle_N = {}
        histoPIMC_N = {}
        for n in range(self.N):
            histoLeft_N.update({n: np.zeros(self.Ngrid,float)})
            histoRight_N.update({n: np.zeros(self.Ngrid,float)})
            histoMiddle_N.update({n: np.zeros(self.Ngrid,float)})
            histoPIMC_N.update({n: np.zeros(self.Ngrid,float)})

        # Creating a histogram that stores the initial distribution
        histo_initial=np.zeros(self.Ngrid,float)

        if not hasattr(self, 'rho_phi'):
            self.createFreeRhoMarx()

        if not hasattr(self, 'rhoVij'):
            self.createRhoVij()

        if not self.PIGS:
            averageEnergy = False

        p_dist=gen_prob_dist(self.Ngrid, self.rho_phi)
        p_dist_end = gen_prob_dist_end(self.Ngrid, self.rho_phi) if self.PIGS == True else None

        self.p_dist=p_dist

        path_phi=np.zeros((self.N,self.P),int) ## i  N => number of beads

        # Initial conditions
        if initialState == 'random':
            # Rotors have random angles
            for i in range(self.N):
                for p in range(self.P):
                    path_phi[i,p]=np.random.randint(self.Ngrid)
                    histo_initial[path_phi[i,p]]+=1. #why adding 1 on the randome index
        elif initialState == 'ordered_0':
            # All rotors have angle of 0
            histo_initial[0] += 1.
        elif initialState == 'ordered_pi':
            # All rotors have angle of pi
            path_phi += int(self.Ngrid/2)
            histo_initial[int(self.Ngrid/2)] += 1.
        else:
            raise Exception("An invalid selection was made for the initial conditions, please use random, ordered_0 or ordered_pi")

        log_out=open(os.path.join(self.path,'MC.log'),'w')
        traj_out=open('traj.out','w')

        # recommanded numpy random number initialization
        rng = default_rng()

        #print('start MC')

        # Initializing the estimators, the method currently employed is only for pigs
        V_arr = np.zeros(self.MC_steps,float) if averagePotential == True else None
        E_arr = np.zeros(self.MC_steps,float) if averageEnergy == True else None
        eiej_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        zizj_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        polarization_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None

        P_middle = int((self.P-1)/2)
        prob_full=np.zeros((self.Ngrid*self.Ngrid),float)
        index_map=np.zeros((self.Ngrid*self.Ngrid,2),int)
        for i1 in range(self.Ngrid):
            for i2 in range(self.Ngrid):
                i12=i1*self.Ngrid+i2
                index_map[i12,0]=i1
                index_map[i12,1]=i2
        
        for n in range(self.MC_steps):
            for i in range(0,self.N,2):
                for p in range(self.P):
                    p_minus=p-1
                    p_plus=p+1
                    if (p_minus<0):
                        p_minus+=self.P
                    if (p_plus>=self.P):
                        p_plus-=self.P

                    # kinetic action, links between beads
                    if self.PIGS==True:
                        # This uses a split open path of beads for PIGS
                        if p==0:
                            # Special case for leftmost bead
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]=p_dist_end[i1,path_phi[i,p_plus]]*p_dist_end[i2,path_phi[i+1,p_plus]]
                        elif p==(self.P-1):
                            # Special case for rightmost bead
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]=p_dist_end[i1,path_phi[i,p_minus]]*p_dist_end[i2,path_phi[i+1,p_minus]]
                        elif (p!=0 and p!= (self.P-1)):
                            # All other beads
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]=p_dist[path_phi[i,p_minus],i1,path_phi[i,p_plus]]*p_dist[path_phi[i+1,p_minus],i2,path_phi[i+1,p_plus]]
                    else:
                        # Regular kinetic interactions between beads, periodic conditions
                        for i1 in range(self.Ngrid):
                            for i2 in range(self.Ngrid):
                                prob_full[i1*self.Ngrid+i2]=p_dist[path_phi[i,p_minus],i1,path_phi[i,p_plus]]*p_dist[path_phi[i+1,p_minus],i2,path_phi[i+1,p_plus]]

                    # Local on site interaction with the potential field
                    if self.V0 != 0.:
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]*=self.rhoV_half[i1]*self.rhoV_half[i2]
                        else:
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]*=self.rhoV[i1]*self.rhoV[i2]

                    # NN interactions and PBC(periodic boundary conditions)

                    if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                        for i1 in range(self.Ngrid):
                            for i2 in range(self.Ngrid):
                                prob_full[i1*self.Ngrid+i2]*=self.rhoVij_half[i1,i2]
                    else:
                        for i1 in range(self.Ngrid):
                            for i2 in range(self.Ngrid):
                                prob_full[i1*self.Ngrid+i2]*=self.rhoVij[i1,i2]
                    if (i<(self.N-2) and (self.N >2)):
                        # Interaction with the rotor to the right
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]*=self.rhoVij_half[i2,path_phi[i+2,p]]
                        else:
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]*=self.rhoVij[i2,path_phi[i+2,p]]
                    if (i>1):
                        # Interaction with the rotor to the left
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]*=self.rhoVij_half[i1,path_phi[i-1,p]]
                        else:
                            for i1 in range(self.Ngrid):
                                for i2 in range(self.Ngrid):
                                    prob_full[i1*self.Ngrid+i2]*=self.rhoVij[i1,path_phi[i-1,p]]
                    # PBC not yet implemented
                    # Normalize
                    norm_pro=0.
                    for i1 in range(self.Ngrid):
                        for i2 in range(self.Ngrid):                        
                            norm_pro+=prob_full[i1*self.Ngrid+i2]
                    for i1 in range(self.Ngrid):
                        for i2 in range(self.Ngrid): 
                            prob_full[i1*self.Ngrid+i2]/=norm_pro
                    index=rng.choice(self.Ngrid*self.Ngrid,1, p=prob_full)[0]
                    i1=index_map[index,0]
                    i2=index_map[index,1]
                    # Rejection free sampling
                    path_phi[i,p] = i1
                    path_phi[i+1,p] = i2

                    traj_out.write(str(n)+' ')

                    for j in range(self.N):
                        traj_out.write(str(path_phi[j,p])+' ')
                    traj_out.write('\n')
                    traj_out.flush()

                    histoPIMC_N[i][path_phi[i,p]]+=1.
                    histoPIMC_N[i+1][path_phi[i+1,p]]+=1.

                # End of bead loop

                # Adding to the histogram counts.
                histoLeft_N[i][path_phi[i,0]]+=1.
                histoLeft_N[i+1][path_phi[i+1,0]]+=1.
                histoRight_N[i][path_phi[i,self.P-1]]+=1.
                histoRight_N[i+1][path_phi[i+1,self.P-1]]+=1.
                histoMiddle_N[i][path_phi[i,P_middle]]+=1.
                histoMiddle_N[i+1][path_phi[i+1,P_middle]]+=1.
            # End of rotor loop
            # Updating the estimators, these only look at the interactions to the left to avoid
            # double counting and to ensure that the interactions being added are from the current MC Step
            if n >= self.Nequilibrate and averagePotential == True:
                # External field
                for ii in range(self.N):
                    if self.PIGS==True:
                        V_arr[n] += self.potFunc(float(path_phi[ii,P_middle])*self.delta_phi,self.V0)
                    else:
                        for p in range(self.P):
                            V_arr[n] += self.potFunc(float(path_phi[ii,p])*self.delta_phi,self.V0)
                    # Nearest neighbour interactions
                    if (ii>0):
                        # Only look at left neighbour to avoid double counting
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi, self.g)
                    if (ii==(self.N-1)) and (self.N>2 and self.PBC==True):
                        # Periodic BCs
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
            if n >= self.Nequilibrate and orientationalCorrelations == True:
                for ii in range(self.N):
                    if self.PBC==True:
                        if (ii>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)
                        if (ii==(self.N-1)) and (self.N>2):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[ii,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[ii,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                    else:
                        if self.PIGS==True:
                            p=P_middle
                            polarization_arr[n] += np.cos(path_phi[ii,p]*self.delta_phi)

                        if (ii>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)

            # Updating the ground state energy estimators for PIGS
            if n >= self.Nequilibrate and averageEnergy == True and self.PIGS==True:
                for ii in range(self.N):
                # External field
                    E_arr[n] += self.potFunc(float(path_phi[ii,0])*self.delta_phi,self.V0)
                    E_arr[n] += self.potFunc(float(path_phi[ii,self.P-1])*self.delta_phi,self.V0)
                # Nearest neighbour interactions, only looks at left neighbour to avoid double counting
                    if (ii>0):
                        E_arr[n] += Vij(path_phi[ii-1,0]*self.delta_phi, path_phi[ii,0]*self.delta_phi, self.g)
                        E_arr[n] += Vij(path_phi[ii-1,self.P-1]*self.delta_phi, path_phi[ii,self.P-1]*self.delta_phi, self.g)
                        if self.PBC==True:
                            if (ii==(self.N-1)) and (self.N>2):
                                E_arr[n] += Vij(path_phi[ii,0]*self.delta_phi, path_phi[0,0]*self.delta_phi, self.g)
                                E_arr[n] += Vij(path_phi[ii,self.P-1]*self.delta_phi, path_phi[0,self.P-1]*self.delta_phi, self.g)
                #print(E_arr[n]/2)
        print('Mc done')
        if averagePotential == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            meanV, stdErrV = calculateError_byBinning(V_arr[self.Nequilibrate:])
            if not self.PIGS:
                meanV /= self.P
                stdErrV /= self.P
            log_out.write('<V> = '+str(meanV)+'\n')
            log_out.write('V_SE = '+str(stdErrV)+'\n')
            self.V_MC = meanV
            self.V_stdError_MC = stdErrV
        if averageEnergy == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            # Need to divide by two according to the PIGS formula
            E_arr/=2
            meanE, stdErrE = calculateError_byBinning(E_arr[self.Nequilibrate:])
            log_out.write('E0 = '+str(meanE)+'\n')
            log_out.write('E0_SE = '+str(stdErrE)+'\n')
            self.E_MC = meanE
            self.E_stdError_MC = stdErrE
        if orientationalCorrelations == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1
            meaneiej, stdErreiej = calculateError_byBinning(eiej_arr[self.Nequilibrate:]/Ninteractions)
            meanzizj, stdErrzizj = calculateError_byBinning(zizj_arr[self.Nequilibrate:]/Ninteractions)
            meanpolarization, stdErrpolarization = calculateError_byBinning(polarization_arr[self.Nequilibrate:]/Ninteractions)
            if not self.PIGS:
                meaneiej /= self.P
                meanzizj /= self.P
                meanpolarization/= self.P
                stdErreiej /= self.P
                stdErrzizj /= self.P
                stdErrpolarization/= self.P
            log_out.write('<ei.ej> = '+str(meaneiej)+'\n')
            log_out.write('ei.ej_SE = '+str(stdErreiej)+'\n')
            log_out.write('<zi zj> = '+str(meanzizj)+'\n')
            log_out.write('zi zj_SE = '+str(stdErrzizj)+'\n')
            self.eiej_MC = meaneiej
            self.eiej_stdError_MC = stdErreiej
            self.polarization_MC=meanpolarization
            self.polarization_stdError_MC=stdErrpolarization

        # Creating arrays to store the overall system's distribution
        histoPIMC_total=np.zeros(self.Ngrid,float)
        histoMiddle_total=np.zeros(self.Ngrid,float)
        histoLeft_total=np.zeros(self.Ngrid,float)
        histoRight_total=np.zeros(self.Ngrid,float)

        # Saving the individual rotor distributions and accumulating the total distributions
        self.histo_N = {}
        for n in range(self.N):
            histoPIMC_total += histoPIMC_N[n]
            histoMiddle_total += histoMiddle_N[n]
            histoLeft_total += histoLeft_N[n]
            histoRight_total += histoRight_N[n]

            histoN_arr = np.zeros((self.Ngrid,5))
            histoN_out = open(os.path.join(self.path,'histo_N'+str(n)),'w')
            for i in range(self.Ngrid):
                histoN_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi) + ' ' +
                                 str(histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoRight_N[n][i]/(self.MC_steps)/self.delta_phi) +'\n')
                histoN_arr[i,:] = [i*self.delta_phi,
                                   histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi,
                                   histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoRight_N[n][i]/(self.MC_steps)/self.delta_phi]
            self.histo_N.update({n: histoN_arr})
            histoN_out.close()

        # Saving the overall and initial distributions
        self.histo_total = np.zeros((self.Ngrid,5))
        self.histo_initial = np.zeros((self.Ngrid,2))
        histo_out = open(os.path.join(self.path,'histo_test_total'),'w')
        histo_init_out = open(os.path.join(self.path,'histo_initial'),'w')
        for i in range(self.Ngrid):
            histo_out.write(str(i*self.delta_phi) + ' ' +
                            str(histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi) + ' ' +
                            str(histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi) + '\n')
            self.histo_total[i,:] = [i*self.delta_phi,
                                    histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi,
                                    histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi]
            histo_init_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histo_initial[i]/(self.N*self.P)/self.delta_phi)+'\n')
            self.histo_initial[i,:] = [i*self.delta_phi,
                                       histo_initial[i]/(self.N*self.P)/self.delta_phi]
        histo_out.close()
        log_out.close()
    def runMC5x1(self, averagePotential = True, averageEnergy = True, orientationalCorrelations = True, initialState='random'):
        """
        Performs the monte carlo integration to simulate the system.

        Parameters
        ----------
        averagePotential : bool, optional
            Enables the tracking and calculation of the average potential.
            The default is True.
        averageEnergy : bool, optional
            Enables the tracking and calculation of the average energy.
            The default is True.
        orientationalCorrelations : bool, optional
            Enables the tracking and calculation of the orientational correllations.
            The default is True.
        initialState : string, optional
            Selects the distribution of the initial state of the system. The allowed
            options are random, ordered_pi or ordered_0.
            The default is random.

        Returns
        -------
        self.V_MC: float
            The resultant average system potential.
        self.V_stdError_MC: float
            The resultant standard error in the system potential.
        self.E_MC: float
            The resultant average system energy.
        self.E_stdError_MC: float
            The resultant standard error in the system energy.
        self.eiej_MC: float
            The resultant average system orientational correlation.
        self.eiej_stdError_MC: float
            The resultant standard error in the system orientational correlation.
        self.histo_N: dict
            Dictionary of Nx5 numpy arrays containing the following histograms
            for each individual rotor:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_total: numpy array
            Nx5 array for the entire systemcontaining:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_initial: numpy array
            Nx2 array containing:
                1st column: The angle phi
                2nd column: Initial overall histogram

        Outputs
        -------
        histo_A_P_N: Nx2 txt file
            A saved version of the self.histo outlined above.
        traj_A: Nx3 dat file
            The phi values of the left, right and middle beads in columns 1, 2
            and 3 respectively, with each row corresponding to a specific rotor
            during a specific MC step.

        """

        # Creating histograms to store each rotor's distributions
        histoLeft_N = {}
        histoRight_N = {}
        histoMiddle_N = {}
        histoPIMC_N = {}
        for n in range(self.N):
            histoLeft_N.update({n: np.zeros(self.Ngrid,float)})
            histoRight_N.update({n: np.zeros(self.Ngrid,float)})
            histoMiddle_N.update({n: np.zeros(self.Ngrid,float)})
            histoPIMC_N.update({n: np.zeros(self.Ngrid,float)})

        # Creating a histogram that stores the initial distribution
        histo_initial=np.zeros(self.Ngrid,float)

        if not hasattr(self, 'rho_phi'):
            self.createFreeRhoMarx()

        if not hasattr(self, 'rhoVij'):
            self.createRhoVij()

        if not self.PIGS:
            averageEnergy = False

        p_dist=gen_prob_dist(self.Ngrid, self.rho_phi)
        p_dist_end = gen_prob_dist_end(self.Ngrid, self.rho_phi) if self.PIGS == True else None

        self.p_dist=p_dist

        Ng=self.Ngrid

        logp_dist_end=np.zeros((Ng,Ng),float)
        logrhoVij_half=np.zeros((Ng,Ng),float)
        logrhoVij=np.zeros((Ng,Ng),float)
        logp_dist=np.zeros((Ng,Ng,Ng),float)

        for i1 in range(Ng):
            for i2 in range(Ng):
                logp_dist_end[i1,i2]=np.log(p_dist_end[i1,i2])
                logrhoVij_half[i1,i2]=np.log(self.rhoVij_half[i1,i2])
                logrhoVij[i1,i2]=np.log(self.rhoVij_half[i1,i2])
                for i3 in range(Ng):
                    logp_dist[i1,i2,i3]=np.log(p_dist[i1,i2,i3])

        path_phi=np.zeros((self.N,self.P),int) ## i  N => number of beads

        # Initial conditions
        if initialState == 'random':
            # Rotors have random angles
            for i in range(self.N):
                for p in range(self.P):
                    path_phi[i,p]=np.random.randint(self.Ngrid)
                    histo_initial[path_phi[i,p]]+=1. #why adding 1 on the randome index
        elif initialState == 'ordered_0':
            # All rotors have angle of 0
            histo_initial[0] += 1.
        elif initialState == 'ordered_pi':
            # All rotors have angle of pi
            path_phi += int(self.Ngrid/2)
            histo_initial[int(self.Ngrid/2)] += 1.
        else:
            raise Exception("An invalid selection was made for the initial conditions, please use random, ordered_0 or ordered_pi")

        log_out=open(os.path.join(self.path,'MC.log'),'w')
        traj_out=open('traj.out','w')

        # recommanded numpy random number initialization
        rng = default_rng()

        # Initializing the estimators, the method currently employed is only for pigs
        V_arr = np.zeros(self.MC_steps,float) if averagePotential == True else None
        E_arr = np.zeros(self.MC_steps,float) if averageEnergy == True else None
        eiej_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        zizj_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        polarization_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None

        P_middle = int((self.P-1)/2)

        Ng5=Ng**5
        prob_full=np.ones(Ng5,float)
        index_map=np.zeros((Ng5,5),int)
        rev_index_map=np.zeros((Ng,Ng,Ng,Ng,Ng),int)
        logrhoVij5_half=np.zeros((Ng5),float)
        logrhoVij5=np.zeros(Ng5,float)
        logrhoV5_half=np.zeros(Ng5,float)
        logrhoV5=np.zeros(Ng5,float)

        for i1 in range(Ng):
            for i2 in range(Ng):
                for i3 in range(Ng):
                    for i4 in range(Ng):
                        for i5 in range(Ng):
                            i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                            index_map[i15,0]=i1
                            index_map[i15,1]=i2
                            index_map[i15,2]=i3
                            index_map[i15,3]=i4
                            index_map[i15,4]=i5
                            rev_index_map[i1,i2,i3,i4,i5]=i15
                            logrhoV5_half[i15]=np.log(self.rhoV_half[i1]*self.rhoV_half[i2]*self.rhoV_half[i3]*self.rhoV_half[i4]*self.rhoV_half[i5])
                            logrhoV5[i15]=np.log(self.rhoV[i1]*self.rhoV[i2]*self.rhoV[i3]*self.rhoV[i4]*self.rhoV[i5])
                            logrhoVij5_half[i15]=np.log(self.rhoVij_half[i1,i2]*self.rhoVij_half[i2,i3]*self.rhoVij_half[i3,i4]*self.rhoVij_half[i4,i5])
                            logrhoVij5[i15]=np.log(self.rhoVij[i1,i2]*self.rhoVij[i2,i3]*self.rhoVij[i3,i4]*self.rhoVij[i4,i5])
        print('start MC')
        for n in range(self.MC_steps):
            print(n)
            for i in range(0,self.N,5):
                for p in range(self.P):
                    #print(n,i,p)
                    p_minus=p-1
                    p_plus=p+1
                    if (p_minus<0):
                        p_minus+=self.P
                    if (p_plus>=self.P):
                        p_plus-=self.P
                    # kinetic action, links between beads
                    if self.PIGS==True:
                        # This uses a split open path of beads for PIGS
                        if p==0:
                            # Special case for leftmost bead
                            for i1 in range(Ng):
                                factor1=logp_dist_end[i1,path_phi[i,p_plus]]
                                for i2 in range(Ng):
                                    factor2=factor1+logp_dist_end[i2,path_phi[i+1,p_plus]]
                                    for i3 in range(Ng):
                                        factor3=factor2+logp_dist_end[i3,path_phi[i+2,p_plus]]
                                        for i4 in range(Ng):
                                            factor4=factor3+logp_dist_end[i4,path_phi[i+3,p_plus]]
                                            for i5 in range(Ng):
                                                factor5=factor4+logp_dist_end[i5,path_phi[i+4,p_plus]]
                                                #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                                i15=rev_index_map[i1,i2,i3,i4,i5]
                                                prob_full[i15]=factor5
                        elif p==(self.P-1):
                            # Special case for rightmost bead
                            for i1 in range(Ng):
                                factor1=logp_dist_end[i1,path_phi[i,p_minus]]
                                for i2 in range(Ng):
                                    factor2=factor1+logp_dist_end[i2,path_phi[i+1,p_minus]]
                                    for i3 in range(Ng):
                                        factor3=factor2+logp_dist_end[i3,path_phi[i+2,p_minus]]
                                        for i4 in range(Ng):
                                            factor4=factor3+logp_dist_end[i4,path_phi[i+3,p_minus]]
                                            for i5 in range(Ng):
                                                factor5=factor4+logp_dist_end[i5,path_phi[i+4,p_minus]]
                                                #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                                i15=rev_index_map[i1,i2,i3,i4,i5]
                                                prob_full[i15]=factor5
                        elif (p!=0 and p!= (self.P-1)):
                            # All other beads
                             for i1 in range(Ng):
                                factor1=logp_dist[path_phi[i,p_minus],i1,path_phi[i,p_plus]]
                                for i2 in range(Ng):
                                    factor2=factor1+logp_dist[path_phi[i+1,p_minus],i2,path_phi[i+1,p_plus]]
                                    for i3 in range(Ng):
                                        factor3=factor2+logp_dist[path_phi[i+2,p_minus],i3,path_phi[i+2,p_plus]]
                                        for i4 in range(Ng):
                                            factor4=factor3+logp_dist[path_phi[i+3,p_minus],i4,path_phi[i+3,p_plus]]
                                            for i5 in range(Ng):
                                                factor5=factor4+logp_dist[path_phi[i+4,p_minus],i5,path_phi[i+4,p_plus]]
                                                #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                                i15=rev_index_map[i1,i2,i3,i4,i5]
                                                prob_full[i15]=factor5
                    else:
                        # Regular kinetic interactions between beads, periodic conditions
                        for i1 in range(Ng):
                            factor1=logp_dist[path_phi[i,p_minus],i1,path_phi[i,p_plus]]
                            for i2 in range(Ng):
                                factor2=factor1+logp_dist[path_phi[i+1,p_minus],i2,path_phi[i+1,p_plus]]
                                for i3 in range(Ng):
                                    factor3=factor2+logp_dist[path_phi[i+2,p_minus],i3,path_phi[i+2,p_plus]]
                                    for i4 in range(Ng):
                                        factor4=factor3+logp_dist[path_phi[i+3,p_minus],i4,path_phi[i+3,p_plus]]
                                        for i5 in range(Ng):
                                            factor5=factor4+logp_dist[path_phi[i+4,p_minus],i5,path_phi[i+4,p_plus]]
                                            #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                            i15=rev_index_map[i1,i2,i3,i4,i5]
                                            prob_full[i15]=factor5
                    # Local on site interaction with the potential field
                    if self.V0 != 0.:
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for i15 in range(Ng5):
                                prob_full[i15]+=logrhoV5_half[i15]
                        else:
                            for i15 in range(Ng5):
                                prob_full[i15]+=logrhoV5[i15]
                    # NN interactions and PBC(periodic boundary conditions)
                    if (p==0 or p==(self.P-1)) and self.PIGS:
                        # Half interaction at the end beads
                        for i15 in range(Ng5):
                            prob_full[i15]+=logrhoVij5_half[i15]
                    else:
                        for i15 in range(Ng5):
                           prob_full[i15]+=logrhoVij5_half[i15]
                    if (i<(self.N-5) and self.N>5):
                        # Interaction with the rotor to the right
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for i1 in range(Ng):
                                for i2 in range(Ng):
                                    for i3 in range(Ng):
                                        for i4 in range(Ng):
                                            for i5 in range(Ng):
                                                #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                                i15=rev_index_map[i1,i2,i3,i4,i5]
                                                prob_full[i15]+=logrhoVij_half[i5,path_phi[i+5,p]]
                        else:
                            for i1 in range(Ng):
                                for i2 in range(Ng):
                                    for i3 in range(Ng):
                                        for i4 in range(Ng):
                                            for i5 in range(Ng):
                                                #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                                i15=rev_index_map[i1,i2,i3,i4,i5]
                                                prob_full[i15]+=logrhoVij[i5,path_phi[i+5,p]]
                    if (i>4):
                        # Interaction with the rotor to the left
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for i1 in range(Ng):
                                for i2 in range(Ng):
                                    for i3 in range(Ng):
                                        for i4 in range(Ng):
                                            for i5 in range(Ng):
                                                #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                                i15=rev_index_map[i1,i2,i3,i4,i5]
                                                prob_full[i15]+=logrhoVij_half[i1,path_phi[i-1,p]]
                        else:
                            for i1 in range(Ng):
                                for i2 in range(Ng):
                                    for i3 in range(Ng):
                                        for i4 in range(Ng):
                                            for i5 in range(Ng):
                                                #i15=(((i1*Ng+i2)*Ng+i3)*Ng+i4)*Ng+i5
                                                i15=rev_index_map[i1,i2,i3,i4,i5]
                                                prob_full[i15]+=logrhoVij[i1,path_phi[i-1,p]]
                                                # blas
                    # PBC not yet implemented
                    # Normalize
                    norm_pro=0.
                    for i15 in range(Ng5):
                        norm_pro+=np.exp(prob_full[i15])

                    for i15 in range(Ng5):
                        prob_full[i15]=np.exp(prob_full[i15])/norm_pro

                    index=rng.choice(Ng5,1, p=prob_full)[0]
                    #index=1
                    i1=index_map[index,0]
                    i2=index_map[index,1]
                    i3=index_map[index,2]
                    i4=index_map[index,3]
                    i5=index_map[index,4]
                    # Rejection free sampling
                    path_phi[i,p] = i1
                    path_phi[i+1,p] = i2
                    path_phi[i+2,p] = i3
                    path_phi[i+3,p] = i4
                    path_phi[i+4,p] = i5

                    traj_out.write(str(n)+' ')
                    for j in range(self.N):
                        traj_out.write(str(path_phi[j,p])+' ')
                    traj_out.write('\n')
                    traj_out.flush()

                    # histoPIMC_N[i][path_phi[i,p]]+=1.
                    # histoPIMC_N[i+1][path_phi[i+1,p]]+=1.
                    # histoPIMC_N[i+2][path_phi[i+2,p]]+=1.
                    # histoPIMC_N[i+3][path_phi[i+3,p]]+=1.
                    # histoPIMC_N[i+4][path_phi[i+4,p]]+=1.

                # End of bead loop

                # Adding to the histogram counts.
                # histoLeft_N[i][path_phi[i,0]]+=1.
                # histoLeft_N[i+1][path_phi[i+1,0]]+=1.
                # histoLeft_N[i+2][path_phi[i+2,0]]+=1.
                # histoLeft_N[i+3][path_phi[i+3,0]]+=1.
                # histoLeft_N[i+4][path_phi[i+4,0]]+=1.
                # histoRight_N[i][path_phi[i,self.P-1]]+=1.
                # histoRight_N[i+1][path_phi[i+1,self.P-1]]+=1.
                # histoRight_N[i+2][path_phi[i+2,self.P-1]]+=1.
                # histoRight_N[i+3][path_phi[i+3,self.P-1]]+=1.
                # histoRight_N[i+4][path_phi[i+4,self.P-1]]+=1.
                # histoMiddle_N[i][path_phi[i,P_middle]]+=1.
                # histoMiddle_N[i+1][path_phi[i+1,P_middle]]+=1.
                # histoMiddle_N[i+2][path_phi[i+2,P_middle]]+=1.
                # histoMiddle_N[i+3][path_phi[i+3,P_middle]]+=1.
                # histoMiddle_N[i+4][path_phi[i+4,P_middle]]+=1.
            # End of rotor loop
            # Updating the estimators, these only look at the interactions to the left to avoid
            # double counting and to ensure that the interactions being added are from the current MC Step
            if n >= self.Nequilibrate and averagePotential == True:
                # External field
                for ii in range(self.N):
                    if self.PIGS==True:
                        V_arr[n] += self.potFunc(float(path_phi[ii,P_middle])*self.delta_phi,self.V0)
                    else:
                        for p in range(self.P):
                            V_arr[n] += self.potFunc(float(path_phi[ii,p])*self.delta_phi,self.V0)
                    # Nearest neighbour interactions
                    if (ii>0):
                        # Only look at left neighbour to avoid double counting
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi, self.g)
                    if (ii==(self.N-1)) and (self.N>2 and self.PBC==True):
                        # Periodic BCs
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
            if n >= self.Nequilibrate and orientationalCorrelations == True:
                for ii in range(self.N):
                    if self.PBC==True:
                        if (ii>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)
                        if (ii==(self.N-1)) and (self.N>2):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[ii,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[ii,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                    else:
                        if self.PIGS==True:
                            p=P_middle
                            polarization_arr[n] += np.cos(path_phi[ii,p]*self.delta_phi)

                        if (ii>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[ii-1,p]*self.delta_phi, path_phi[ii,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[ii-1,p]*self.delta_phi)*np.cos(path_phi[ii,p]*self.delta_phi)

            # Updating the ground state energy estimators for PIGS
            if n >= self.Nequilibrate and averageEnergy == True and self.PIGS==True:
                for ii in range(self.N):
                # External field
                    E_arr[n] += self.potFunc(float(path_phi[ii,0])*self.delta_phi,self.V0)
                    E_arr[n] += self.potFunc(float(path_phi[ii,self.P-1])*self.delta_phi,self.V0)
                # Nearest neighbour interactions, only looks at left neighbour to avoid double counting
                    if (ii>0):
                        E_arr[n] += Vij(path_phi[ii-1,0]*self.delta_phi, path_phi[ii,0]*self.delta_phi, self.g)
                        E_arr[n] += Vij(path_phi[ii-1,self.P-1]*self.delta_phi, path_phi[ii,self.P-1]*self.delta_phi, self.g)
                        if self.PBC==True:
                            if (ii==(self.N-1)) and (self.N>2):
                                E_arr[n] += Vij(path_phi[ii,0]*self.delta_phi, path_phi[0,0]*self.delta_phi, self.g)
                                E_arr[n] += Vij(path_phi[ii,self.P-1]*self.delta_phi, path_phi[0,self.P-1]*self.delta_phi, self.g)
                #print(E_arr[n]/2)

        if averagePotential == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            meanV, stdErrV = calculateError_byBinning(V_arr[self.Nequilibrate:])
            if not self.PIGS:
                meanV /= self.P
                stdErrV /= self.P
            log_out.write('<V> = '+str(meanV)+'\n')
            log_out.write('V_SE = '+str(stdErrV)+'\n')
            self.V_MC = meanV
            self.V_stdError_MC = stdErrV
        if averageEnergy == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            # Need to divide by two according to the PIGS formula
            E_arr/=2
            meanE, stdErrE = calculateError_byBinning(E_arr[self.Nequilibrate:])
            log_out.write('E0 = '+str(meanE)+'\n')
            log_out.write('E0_SE = '+str(stdErrE)+'\n')
            self.E_MC = meanE
            self.E_stdError_MC = stdErrE
        if orientationalCorrelations == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1
            meaneiej, stdErreiej = calculateError_byBinning(eiej_arr[self.Nequilibrate:]/Ninteractions)
            meanzizj, stdErrzizj = calculateError_byBinning(zizj_arr[self.Nequilibrate:]/Ninteractions)
            meanpolarization, stdErrpolarization = calculateError_byBinning(polarization_arr[self.Nequilibrate:]/self.N)
            if not self.PIGS:
                meaneiej /= self.P
                meanzizj /= self.P
                meanpolarization/= self.P
                stdErreiej /= self.P
                stdErrzizj /= self.P
                stdErrpolarization/= self.P
            log_out.write('<ei.ej> = '+str(meaneiej)+'\n')
            log_out.write('ei.ej_SE = '+str(stdErreiej)+'\n')
            log_out.write('<zi zj> = '+str(meanzizj)+'\n')
            log_out.write('zi zj_SE = '+str(stdErrzizj)+'\n')
            self.eiej_MC = meaneiej
            self.eiej_stdError_MC = stdErreiej
            self.polarization_MC=meanpolarization
            self.polarization_stdError_MC=stdErrpolarization

        # Creating arrays to store the overall system's distribution
        histoPIMC_total=np.zeros(self.Ngrid,float)
        histoMiddle_total=np.zeros(self.Ngrid,float)
        histoLeft_total=np.zeros(self.Ngrid,float)
        histoRight_total=np.zeros(self.Ngrid,float)

        # Saving the individual rotor distributions and accumulating the total distributions
        self.histo_N = {}
        for n in range(self.N):
            histoPIMC_total += histoPIMC_N[n]
            histoMiddle_total += histoMiddle_N[n]
            histoLeft_total += histoLeft_N[n]
            histoRight_total += histoRight_N[n]

            histoN_arr = np.zeros((self.Ngrid,5))
            histoN_out = open(os.path.join(self.path,'histo_N'+str(n)),'w')
            for i in range(self.Ngrid):
                histoN_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi) + ' ' +
                                 str(histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoRight_N[n][i]/(self.MC_steps)/self.delta_phi) +'\n')
                histoN_arr[i,:] = [i*self.delta_phi,
                                   histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi,
                                   histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoRight_N[n][i]/(self.MC_steps)/self.delta_phi]
            self.histo_N.update({n: histoN_arr})
            histoN_out.close()

        # Saving the overall and initial distributions
        self.histo_total = np.zeros((self.Ngrid,5))
        self.histo_initial = np.zeros((self.Ngrid,2))
        histo_out = open(os.path.join(self.path,'histo_test_total'),'w')
        histo_init_out = open(os.path.join(self.path,'histo_initial'),'w')
        for i in range(self.Ngrid):
            histo_out.write(str(i*self.delta_phi) + ' ' +
                            str(histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi) + ' ' +
                            str(histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi) + '\n')
            self.histo_total[i,:] = [i*self.delta_phi,
                                    histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi,
                                    histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi]
            histo_init_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histo_initial[i]/(self.N*self.P)/self.delta_phi)+'\n')
            self.histo_initial[i,:] = [i*self.delta_phi,
                                       histo_initial[i]/(self.N*self.P)/self.delta_phi]
        histo_out.close()
        log_out.close()
    def runMCnew(self, averagePotential = True, averageEnergy = True, orientationalCorrelations = True, initialState='random'):    
        """
        Performs the monte carlo integration to simulate the system.

        Parameters
        ----------
        averagePotential : bool, optional
            Enables the tracking and calculation of the average potential.
            The default is True.
        averageEnergy : bool, optional
            Enables the tracking and calculation of the average energy.
            The default is True.
        orientationalCorrelations : bool, optional
            Enables the tracking and calculation of the orientational correllations.
            The default is True.
        initialState : string, optional
            Selects the distribution of the initial state of the system. The allowed
            options are random, ordered_pi or ordered_0.
            The default is random.

        Returns
        -------
        self.V_MC: float
            The resultant average system potential.
        self.V_stdError_MC: float
            The resultant standard error in the system potential.
        self.E_MC: float
            The resultant average system energy.
        self.E_stdError_MC: float
            The resultant standard error in the system energy.
        self.eiej_MC: float
            The resultant average system orientational correlation.
        self.eiej_stdError_MC: float
            The resultant standard error in the system orientational correlation.
        self.histo_N: dict
            Dictionary of Nx5 numpy arrays containing the following histograms
            for each individual rotor:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_total: numpy array
            Nx5 array for the entire systemcontaining:
                1st column: The angle phi
                2nd column: PIMC Histogram
                3rd column: Middle bead histogram
                4th column: Left bead histogram
                5th column: Right bead histogram
        self.histo_initial: numpy array
            Nx2 array containing:
                1st column: The angle phi
                2nd column: Initial overall histogram

        Outputs
        -------
        histo_A_P_N: Nx2 txt file
            A saved version of the self.histo outlined above.
        traj_A: Nx3 dat file
            The phi values of the left, right and middle beads in columns 1, 2
            and 3 respectively, with each row corresponding to a specific rotor
            during a specific MC step.

        """

        # Creating histograms to store each rotor's distributions
        histoLeft_N = {}
        histoRight_N = {}
        histoMiddle_N = {}
        histoPIMC_N = {}
        for n in range(self.N):
            histoLeft_N.update({n: np.zeros(self.Ngrid,float)})
            histoRight_N.update({n: np.zeros(self.Ngrid,float)})
            histoMiddle_N.update({n: np.zeros(self.Ngrid,float)})
            histoPIMC_N.update({n: np.zeros(self.Ngrid,float)})

        # Creating a histogram that stores the initial distribution
        histo_initial=np.zeros(self.Ngrid,float)

        if not hasattr(self, 'rho_phi'):
            self.createFreeRhoMarx()

        if not hasattr(self, 'rhoVij'):
            self.createRhoVij()

        if not self.PIGS:
            averageEnergy = False

        p_dist=gen_prob_dist(self.Ngrid, self.rho_phi)
        p_dist_end = gen_prob_dist_end(self.Ngrid, self.rho_phi) if self.PIGS == True else None

        self.p_dist=p_dist

        path_phi=np.zeros((self.N,self.P),int) ## i  N => number of beads

        # Initial conditions
        if initialState == 'random':
            # Rotors have random angles
            for i in range(self.N):
                for p in range(self.P):
                    path_phi[i,p]=np.random.randint(self.Ngrid)
                    histo_initial[path_phi[i,p]]+=1. #why adding 1 on the randome index
        elif initialState == 'ordered_0':
            # All rotors have angle of 0
            histo_initial[0] += 1.
        elif initialState == 'ordered_pi':
            # All rotors have angle of pi
            path_phi += int(self.Ngrid/2)
            histo_initial[int(self.Ngrid/2)] += 1.
        else:
            raise Exception("An invalid selection was made for the initial conditions, please use random, ordered_0 or ordered_pi")

        traj_out=open(os.path.join(self.path,'traj_A.dat'),'w')
        log_out=open(os.path.join(self.path,'MC.log'),'w')


        # recommanded numpy random number initialization
        rng = default_rng()

        #print('start MC')

        # Initializing the estimators, the method currently employed is only for pigs
        V_arr = np.zeros(self.MC_steps,float) if averagePotential == True else None
        E_arr = np.zeros(self.MC_steps,float) if averageEnergy == True else None
        eiej_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None
        zizj_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None

        polarization_arr = np.zeros(self.MC_steps,float) if orientationalCorrelations == True else None

        P_middle = int((self.P-1)/2)
        prob_full=np.zeros(self.Ngrid,float)

        listN=[]
        listPe=[]
        listPo=[]
        for n in range(self.N):
            listN.append(n)        
        for n in range(self.N):
            listN.append(n)
  
        for pe in range(int(self.P/2)+1):
            listPe.append(int(2*pe))
        for po in range(int(self.P/2)):
            listPo.append(int(2*po+1))

        for n in range(self.MC_steps):
            for indexi,i in enumerate(listN):
                if indexi<self.N:
                    if i%2==0:
                        listP=listPe
                    else:
                        listP=listPo
                else:
                    if i%2==0:
                        listP=listPo
                    else:
                        listP=listPe

                for p in listP:
                    p_minus=p-1
                    p_plus=p+1
                    if (p_minus<0):
                        p_minus+=self.P
                    if (p_plus>=self.P):
                        p_plus-=self.P

                    # kinetic action, links between beads
                    if self.PIGS==True:
                        # This uses a split open path of beads for PIGS
                        if p==0:
                            # Special case for leftmost bead
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist_end[ip,path_phi[i,p_plus]]
                        elif p==(self.P-1):
                            # Special case for rightmost bead
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist_end[path_phi[i,p_minus],ip]
                        elif (p!=0 and p!= (self.P-1)):
                            # All other beads
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]
                    else:
                        # Regular kinetic interactions between beads, periodic conditions
                        for ip in range(self.Ngrid):
                            prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]

                    # Local on site interaction with the potential field
                    if self.V0 != 0.:
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ip in range(self.Ngrid):
                                prob_full[ip]*=self.rhoV_half[ip]
                        else:
                            for ip in range(self.Ngrid):
                                prob_full[ip]*=self.rhoV[ip]

                    # NN interactions and PBC(periodic boundary conditions)
                    if (i<(self.N-1)):
                        # Interaction with the rotor to the right
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[i+1,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[i+1,p]]
                    if (i>0):
                        # Interaction with the rotor to the left
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[i-1,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[i-1,p]]
                    if (i==0) and (self.N>2) and self.PBC==True:
                        # Periodic BC for the leftmost rotor, doesn't apply to the 2 rotor system
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[self.N-1,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[self.N-1,p]]
                    if (i==(self.N-1)) and (self.N>2) and self.PBC==True:
                        # Periodic BC for the rightmost rotor, doesn't apply to the 2 rotor system
                        if (p==0 or p==(self.P-1)) and self.PIGS:
                            # Half interaction at the end beads
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij_half[ir,path_phi[0,p]]
                        else:
                            for ir in range(len(prob_full)):
                                prob_full[ir]*=self.rhoVij[ir,path_phi[0,p]]
                    # Normalize
                    norm_pro=0.
                    for ir in range(len(prob_full)):
                        norm_pro+=prob_full[ir]
                    for ir in range(len(prob_full)):
                        prob_full[ir]/=norm_pro
                    index=rng.choice(self.Ngrid,1, p=prob_full)
                    # Rejection free sampling
                    path_phi[i,p] = index

                    histoPIMC_N[i][path_phi[i,p]]+=1.

                # End of bead loop

                # Adding to the histogram counts.
                histoLeft_N[i][path_phi[i,0]]+=1.
                histoRight_N[i][path_phi[i,self.P-1]]+=1.
                histoMiddle_N[i][path_phi[i,P_middle]]+=1.

        # End of rotor loop

            if (n%self.Nskip==0):
                for i in range(self.N):
                    traj_out.write(str(path_phi[i,0]*self.delta_phi)+' ')
                    traj_out.write(str(path_phi[i,self.P-1]*self.delta_phi)+' ')
                    traj_out.write(str(path_phi[i,P_middle]*self.delta_phi)+' ') #middle bead
                traj_out.write('\n')

            # Updating the estimators, these only look at the interactions to the left to avoid
            # double counting and to ensure that the interactions being added are from the current MC Step
            if n >= self.Nequilibrate and averagePotential == True:
                # External field
                for i in range(self.N):
                    if self.PIGS==True:
                        V_arr[n] += self.potFunc(float(path_phi[i,P_middle])*self.delta_phi,self.V0)
                    else:
                        for p in range(self.P):
                            V_arr[n] += self.potFunc(float(path_phi[i,p])*self.delta_phi,self.V0)
                    # Nearest neighbour interactions
                    if (i>0):
                        # Only look at left neighbour to avoid double counting
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi, self.g)
                    if (i==(self.N-1)) and (self.N>2 and self.PBC==True):
                        # Periodic BCs
                        if self.PIGS==True:
                            p=P_middle
                            V_arr[n] += Vij(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
                        else:
                            for p in range(self.P):
                                V_arr[n] += Vij(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi, self.g)
            if n >= self.Nequilibrate and orientationalCorrelations == True:
                for i in range(self.N):
                    if self.PBC==True:
                        if (i>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)
                        if (i==(self.N-1)) and (self.N>2):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[i,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i,p]*self.delta_phi, path_phi[0,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[i,p]*self.delta_phi)*np.cos(path_phi[0,p]*self.delta_phi)
                    else:
                        if self.PIGS==True:
                            polarization_arr[n] += np.cos(path_phi[i,p]*self.delta_phi)

                        if (i>0):
                            if self.PIGS==True:
                                p=P_middle
                                eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)
                            else:
                                for p in range(self.P):
                                    eiej_arr[n] += calculateOrientationalCorrelations(path_phi[i-1,p]*self.delta_phi, path_phi[i,p]*self.delta_phi)
                                    zizj_arr[n] += np.cos(path_phi[i-1,p]*self.delta_phi)*np.cos(path_phi[i,p]*self.delta_phi)

            # Updating the ground state energy estimators for PIGS
            if n >= self.Nequilibrate and averageEnergy == True and self.PIGS==True:
                for i in range(self.N):
                # External field
                    E_arr[n] += self.potFunc(float(path_phi[i,0])*self.delta_phi,self.V0)
                    E_arr[n] += self.potFunc(float(path_phi[i,self.P-1])*self.delta_phi,self.V0)
                # Nearest neighbour interactions, only looks at left neighbour to avoid double counting
                    if (i>0):
                        E_arr[n] += Vij(path_phi[i-1,0]*self.delta_phi, path_phi[i,0]*self.delta_phi, self.g)
                        E_arr[n] += Vij(path_phi[i-1,self.P-1]*self.delta_phi, path_phi[i,self.P-1]*self.delta_phi, self.g)
                        if self.PBC==True:
                            if (i==(self.N-1)) and (self.N>2):
                                E_arr[n] += Vij(path_phi[i,0]*self.delta_phi, path_phi[0,0]*self.delta_phi, self.g)
                                E_arr[n] += Vij(path_phi[i,self.P-1]*self.delta_phi, path_phi[0,self.P-1]*self.delta_phi, self.g)
                #print(E_arr[n]/2)

        traj_out.close()

        if averagePotential == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            meanV, stdErrV = calculateError_byBinning(V_arr[self.Nequilibrate:])
            if not self.PIGS:
                meanV /= self.P
                stdErrV /= self.P
            log_out.write('<V> = '+str(meanV)+'\n')
            log_out.write('V_SE = '+str(stdErrV)+'\n')
            self.V_MC = meanV
            self.V_stdError_MC = stdErrV
        if averageEnergy == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1

            # Need to divide by two according to the PIGS formula
            E_arr/=2
            meanE, stdErrE = calculateError_byBinning(E_arr[self.Nequilibrate:])
            log_out.write('E0 = '+str(meanE)+'\n')
            log_out.write('E0_SE = '+str(stdErrE)+'\n')
            self.E_MC = meanE
            self.E_stdError_MC = stdErrE
        if orientationalCorrelations == True:
            if self.PBC==True:
                Ninteractions = self.N if self.N>2 else 1
            else:
                Ninteractions = (self.N-1) if self.N>2 else 1
            meaneiej, stdErreiej = calculateError_byBinning(eiej_arr[self.Nequilibrate:]/Ninteractions)
            meanzizj, stdErrzizj = calculateError_byBinning(zizj_arr[self.Nequilibrate:]/Ninteractions)
            meanpolarization, stdErrpolarization = calculateError_byBinning(polarization_arr[self.Nequilibrate:]/self.N)

            if not self.PIGS:
                meaneiej /= self.P
                meanzizj /= self.P
                meanpolarization/= self.P
                stdErreiej /= self.P
                stdErrzizj /= self.P
                stdErrpolarization/= self.P
            log_out.write('<ei.ej> = '+str(meaneiej)+'\n')
            log_out.write('ei.ej_SE = '+str(stdErreiej)+'\n')
            log_out.write('<zi zj> = '+str(meanzizj)+'\n')
            log_out.write('zi zj_SE = '+str(stdErrzizj)+'\n')
            self.eiej_MC = meaneiej
            self.eiej_stdError_MC = stdErreiej
            self.polarization_MC=meanpolarization
            self.polarization_stdError_MC=stdErrpolarization

        # Creating arrays to store the overall system's distribution
        histoPIMC_total=np.zeros(self.Ngrid,float)
        histoMiddle_total=np.zeros(self.Ngrid,float)
        histoLeft_total=np.zeros(self.Ngrid,float)
        histoRight_total=np.zeros(self.Ngrid,float)

        # Saving the individual rotor distributions and accumulating the total distributions
        self.histo_N = {}
        for n in range(self.N):
            histoPIMC_total += histoPIMC_N[n]
            histoMiddle_total += histoMiddle_N[n]
            histoLeft_total += histoLeft_N[n]
            histoRight_total += histoRight_N[n]

            histoN_arr = np.zeros((self.Ngrid,5))
            histoN_out = open(os.path.join(self.path,'histo_N'+str(n)),'w')
            for i in range(self.Ngrid):
                histoN_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi) + ' ' +
                                 str(histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi) + ' ' +
                                 str(histoRight_N[n][i]/(self.MC_steps)/self.delta_phi) +'\n')
                histoN_arr[i,:] = [i*self.delta_phi,
                                   histoPIMC_N[n][i]/(self.MC_steps*self.P)/self.delta_phi,
                                   histoMiddle_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoLeft_N[n][i]/(self.MC_steps)/self.delta_phi,
                                   histoRight_N[n][i]/(self.MC_steps)/self.delta_phi]
            self.histo_N.update({n: histoN_arr})
            histoN_out.close()

        # Saving the overall and initial distributions
        self.histo_total = np.zeros((self.Ngrid,5))
        self.histo_initial = np.zeros((self.Ngrid,2))
        histo_out = open(os.path.join(self.path,'histo_test_total'),'w')
        histo_init_out = open(os.path.join(self.path,'histo_initial'),'w')
        for i in range(self.Ngrid):
            histo_out.write(str(i*self.delta_phi) + ' ' +
                            str(histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi) + ' ' +
                            str(histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi) + ' ' +
                            str(histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi) + '\n')
            self.histo_total[i,:] = [i*self.delta_phi,
                                    histoPIMC_total[i]/(self.MC_steps*self.N*self.P)/self.delta_phi,
                                    histoMiddle_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoLeft_total[i]/(self.MC_steps*self.N)/self.delta_phi,
                                    histoRight_total[i]/(self.MC_steps*self.N)/self.delta_phi]
            histo_init_out.write(str(i*self.delta_phi) + ' ' +
                                 str(histo_initial[i]/(self.N*self.P)/self.delta_phi)+'\n')
            self.histo_initial[i,:] = [i*self.delta_phi,
                                       histo_initial[i]/(self.N*self.P)/self.delta_phi]
        histo_out.close()
        log_out.close()

    def runMCReplica(self, ratioTrick=False, initialState='random'):
        """
        Performs the monte carlo integration to simulate the system with entanglement considered. This employs
        the replica trick and the extended ensemble to do so.

        Parameters
        ----------
        ratioTrick : bool, optional
            Enables the ratio trick.
            The default is False.
        initialState : string, optional
            Selects the distribution of the initial state of the system. The allowed
            options are random, ordered_pi or ordered_0.
            The default is random.

        Returns
        -------
        self.S2_MC: float
            The resultant second Renyi entropy.
        self.S2_stdError: float
            The resultant standard error in the second Renyi entropy.
        self.purity_MC: float
            The total acceptance ratio for the swap/unswap configufrations, equivalent
            to the purity of the system.
        self.purity_stdError_MC: float
            The standard error in the acceptance ratio for the swap/unswap configufrations,
            equivalent to the purity of the system.
        self.AR_MC_arr: np.array
            An N//2+1 x 1 array containing the acceptance ratios of each of the
            partitions. Only applicable if the ratio trick is enabled.
        self.AR_stdError_MC_arr : np.array
            An N//2+1 x 1 array containing the standard error in the acceptance ratios
            of each of the partitions. Only applicable if the ratio trick is enabled.

        """

        if (self.N//2 != self.N/2):
            raise Exception("An even number of rotors must be used for partitioning")

        if not hasattr(self, 'rho_phi'):
            self.createFreeRhoMarx()

        if not hasattr(self, 'rhoVij'):
            self.createRhoVij()

        if not self.PIGS:
            raise Exception("PIGS must be enabled to run runMCReplica, please create a choiceMC object with this enabled")

        p_dist=gen_prob_dist(self.Ngrid, self.rho_phi)
        p_dist_end=gen_prob_dist_end(self.Ngrid, self.rho_phi)

        # Index of the partition in the chain
        NPartition_max = self.N // 2

        # Creating arrays to store the indeces of the first rotor in the B partition
        if ratioTrick:
            N_partitions = np.zeros((NPartition_max,2), int)
            for i, partition in enumerate(N_partitions):
                N_partitions[i,:] = [i, i+1]
        else:
            N_partitions = np.array([[0, NPartition_max]])

        # The indeces of the beads in the middle and to the left of the middle of the chain
        P_middle = int((self.P-1)/2)
        P_midLeft = P_middle - 1

        purity_arr = np.zeros(np.shape(N_partitions)[0], float)
        purity_err_arr = np.zeros(np.shape(N_partitions)[0], float)

        for i_partition, N_partition in enumerate(N_partitions):

            path_phi=np.zeros((self.N,self.P),int) ## i  N => number of beads

            # Initial conditions
            if initialState == 'random':
                # Rotors have random angles
                for i in range(self.N):
                    for p in range(self.P):
                        path_phi[i,p]=np.random.randint(self.Ngrid)
            elif initialState == 'ordered_0':
                # All rotors have angle of 0
                path_phi += 0.
            elif initialState == 'ordered_pi':
                # All rotors have angle of pi
                path_phi += int(self.Ngrid/2)
            else:
                raise Exception("An invalid selection was made for the initial conditions, please use random, ordered_0 or ordered_pi")

            # Creating a replica of the original path
            path_phi_replica = path_phi.copy()

            # recommanded numpy random number initialization
            rng = default_rng()

            if ratioTrick:
                print('start MC for partitions ' + str(N_partition[0]) + " to " + str(N_partition[1]))
            else:
                print('start MC')

            # Counter for the number of MC steps in the swapped and unswapped configuration
            N_swapped = 0
            rSwapped_arr = np.zeros(self.MC_steps,float)
            N_unswapped = 0
            rUnswapped_arr = np.zeros(self.MC_steps,float)
            swapped = False

            prob_full=np.zeros(self.Ngrid,float)
            prob_full_replica=np.zeros(self.Ngrid,float)


            for n in range(self.MC_steps):
                # Entanglement estimators
                N_swapped += swapped
                N_unswapped += (not swapped)

                # As the rotors are looped through, the only ones that can partake in the swapped/unswapped
                # configuration are the rotors in the "A" partition
                for i in range(self.N):
                    for p in range(self.P):
                        p_minus=p-1
                        p_plus=p+1
                        if (p_minus<0):
                            p_minus+=self.P
                        if (p_plus>=self.P):
                            p_plus-=self.P

                        # kinetic action
                        if p==0:
                            # Special case on the left end of the chain
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist_end[ip,path_phi[i,p_plus]]
                                prob_full_replica[ip]=p_dist_end[ip,path_phi_replica[i,p_plus]]
                        elif p==(self.P-1):
                            # Special case on the right end of the chain
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist_end[path_phi[i,p_minus],ip]
                                prob_full_replica[ip]=p_dist_end[path_phi_replica[i,p_minus],ip]
                        elif (p==P_midLeft) and ((swapped and i < N_partition[1]) or (not swapped and i < N_partition[0])):
                                # Special case for extended ensemble for the bead to the left of the middle
                                # This only applies to the "A" partition
                                for ip in range(self.Ngrid):
                                    prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi_replica[i,p_plus]]
                                    prob_full_replica[ip]=p_dist[path_phi_replica[i,p_minus],ip,path_phi[i,p_plus]]
                        elif (p==P_middle) and ((swapped and i < N_partition[1]) or (not swapped and i < N_partition[0])):
                            # Special case for extended ensemble on the middle bead
                            # This only applies to the "A" partition
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist[path_phi_replica[i,p_minus],ip,path_phi[i,p_plus]]
                                prob_full_replica[ip]=p_dist[path_phi[i,p_minus],ip,path_phi_replica[i,p_plus]]
                        elif (p!=0 and p!=(self.P-1)):
                            # Interactions for non-swapped and non-end beads
                            for ip in range(self.Ngrid):
                                prob_full[ip]=p_dist[path_phi[i,p_minus],ip,path_phi[i,p_plus]]
                                prob_full_replica[ip]=p_dist[path_phi_replica[i,p_minus],ip,path_phi_replica[i,p_plus]]

                        # Local on site interaction with the external potential field
                        if self.V0 != 0.:
                            if (p==0 or p==(self.P-1)):
                                # Half interaction at the end beads
                                for ip in range(self.Ngrid):
                                    prob_full[ip]*=self.rhoV_half[ip]
                                    prob_full_replica[ip]*=self.rhoV_half[ip]
                            else:
                                for ip in range(self.Ngrid):
                                    prob_full[ip]*=self.rhoV[ip]
                                    prob_full_replica[ip]*=self.rhoV[ip]

                        # NN interactions and PBC(periodic boundary conditions)
                        if (i<(self.N-1)):
                            # Interaction with right neighbour
                            if (p==P_middle) and ((swapped and i == (N_partition[1]-1)) or (not swapped and i == (N_partition[0]-1))):
                                # Swaps the right interaction for the middle bead of the rotor at the partition on the A side
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[i+1,p]]
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi_replica[i+1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi[i+1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[i+1,p]]
                            elif (p==0 or p==(self.P-1)):
                                # Half interaction at the end beads
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[i+1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[i+1,p]]
                            else:
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij[ir,path_phi[i+1,p]]
                                    prob_full_replica[ir]*=self.rhoVij[ir,path_phi_replica[i+1,p]]
                        if (i>0):
                            # Interaction with left neighbour
                            if (p==P_middle) and ((swapped and i == (N_partition[1])) or (not swapped and i == (N_partition[0]))):
                                # Swaps the left interaction for the middle bead of the rotor at the partition on the B side
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[i-1,p]]
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi_replica[i-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi[i-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[i-1,p]]
                            elif (p==0 or p==(self.P-1)):
                                # Half interaction at the end beads
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[i-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[i-1,p]]
                            else:
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij[ir,path_phi[i-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij[ir,path_phi_replica[i-1,p]]
                        #if (i==0) and (self.N>2):
                        if (i==0) and (self.N>2) and self.PBC==True:

                            # Periodic BC for the leftmost rotor, only applies for more than 2 rotors
                            if (p==P_middle) and (swapped or (not swapped and N_partition[0]>0)):
                                # Swaps the left interaction for the middle bead of the leftmost rotor
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[self.N-1,p]]
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi_replica[self.N-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi[self.N-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[self.N-1,p]]
                            elif (p==0 or p==(self.P-1)):
                                # Half interaction at the end beads
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[self.N-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[self.N-1,p]]
                            else:
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij[ir,path_phi[self.N-1,p]]
                                    prob_full_replica[ir]*=self.rhoVij[ir,path_phi_replica[self.N-1,p]]
                        if (i==(self.N-1)) and (self.N>2) and self.PBC==True:

                            # Periodic BC for the rightmost rotor, only applies for more than 2 rotors
                            if (p==P_middle) and (swapped or (not swapped and N_partition[0]>0)):
                                # Swaps the right interaction for the middle bead of the rightmost rotor
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[0,p]]
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi_replica[0,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi[0,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[0,p]]
                            elif (p==0 or p==(self.P-1)):
                                # Half interaction at the end beads
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij_half[ir,path_phi[0,p]]
                                    prob_full_replica[ir]*=self.rhoVij_half[ir,path_phi_replica[0,p]]
                            else:
                                for ir in range(len(prob_full)):
                                    prob_full[ir]*=self.rhoVij[ir,path_phi[0,p]]
                                    prob_full_replica[ir]*=self.rhoVij[ir,path_phi_replica[0,p]]

                        # Normalize
                        norm_pro=0.
                        norm_pro_replica=0.
                        for ir in range(len(prob_full)):
                            norm_pro+=prob_full[ir]
                            norm_pro_replica+=prob_full_replica[ir]
                        for ir in range(len(prob_full)):
                            prob_full[ir]/=norm_pro
                            prob_full_replica[ir]/=norm_pro_replica
                        index=rng.choice(self.Ngrid,1, p=prob_full)
                        index_replica=rng.choice(self.Ngrid,1, p=prob_full_replica)

                        # Rejection free sampling
                        path_phi[i,p] = index
                        path_phi_replica[i,p] = index_replica

                    # End of bead loop
                # End of rotor loop

                # Metropolis critereon

                # The interaction with the external potential field is ignored, as this will
                # be the same for both the swapped and unswapped ensembles
                # Any unchanged interactions (kinetic or potential) between the swapped and
                # unswapped configurations are not calculated

                rhoUnswapped = 1.
                rhoSwapped = 1.

                # Kinetic contribution from the swapped interactions in the A partition
                ##########################################################
                # This needs to be double checked, should this be multiplying the middle and midleft bead or is this double counting?

                for i_diff in range(N_partition[1]-N_partition[0]):
                    # This looks only at kinetic links that are different, if the ratio trick is employed this loops once
                    i = i_diff + N_partition[0]
                    # Middle bead
                    rhoUnswapped *= p_dist[path_phi[i,P_midLeft],path_phi[i,P_middle],path_phi[i,P_middle+1]]
                    rhoUnswapped *= p_dist[path_phi_replica[i,P_midLeft],path_phi_replica[i,P_middle],path_phi_replica[i,P_middle+1]]
                    # Bead to the left of the middle
                    rhoUnswapped *= p_dist[path_phi[i,P_midLeft-1],path_phi[i,P_midLeft],path_phi[i,P_middle]]
                    rhoUnswapped *= p_dist[path_phi_replica[i,P_midLeft-1],path_phi_replica[i,P_midLeft],path_phi_replica[i,P_middle]]

                    # Middle bead
                    rhoSwapped *= p_dist[path_phi_replica[i,P_midLeft],path_phi[i,P_middle],path_phi[i,P_middle+1]]
                    rhoSwapped *= p_dist[path_phi[i,P_midLeft],path_phi_replica[i,P_middle],path_phi_replica[i,P_middle+1]]
                    # Bead to the left of the middle
                    rhoSwapped *= p_dist[path_phi[i,P_midLeft-1],path_phi[i,P_midLeft],path_phi_replica[i,P_middle]]
                    rhoSwapped *= p_dist[path_phi_replica[i,P_midLeft-1],path_phi_replica[i,P_midLeft],path_phi[i,P_middle]]

                # Potential contribution, this only impacts the middle bead interactions where the two partitions differ
                i0 = N_partition[0] - 1
                if i0<0:
                    i0 += self.N
                i1 = N_partition[0]
                # This only applies for more than 2 rotors, it encompases the PBCs for N_partition=[0,x]
                if (self.N>2):
                    rhoUnswapped *= self.rhoVij_half[path_phi[i0,P_middle],path_phi[i1,P_middle]]
                    rhoUnswapped *= self.rhoVij_half[path_phi_replica[i0,P_middle],path_phi_replica[i1,P_middle]]
                    rhoSwapped *= self.rhoVij_half[path_phi_replica[i0,P_middle],path_phi[i1,P_middle]]
                    rhoSwapped *= self.rhoVij_half[path_phi[i0,P_middle],path_phi_replica[i1,P_middle]]

                j0 = N_partition[1] - 1
                j1 = N_partition[1]
                rhoUnswapped *= self.rhoVij_half[path_phi[j0,P_middle],path_phi[j1,P_middle]]
                rhoUnswapped *= self.rhoVij_half[path_phi_replica[j0,P_middle],path_phi_replica[j1,P_middle]]
                rhoSwapped *= self.rhoVij_half[path_phi_replica[j0,P_middle],path_phi[j1,P_middle]]
                rhoSwapped *= self.rhoVij_half[path_phi[j0,P_middle],path_phi_replica[j1,P_middle]]

                # Determing if we should be sampling the swapped or unswapped distribution
                if swapped:
                    ratio = rhoUnswapped/rhoSwapped
                    if ratio > 1:
                        swapped = False
                    elif ratio > np.random.uniform():
                        swapped = False
                elif not swapped:
                    ratio = rhoSwapped/rhoUnswapped
                    if ratio > 1:
                        swapped = True
                    elif ratio > np.random.uniform():
                        swapped = True

                rSwapped_arr[n] = N_swapped / (n+1)
                rUnswapped_arr[n] = N_unswapped / (n+1)
            # End of MC Step loop

            # Finding the average and standard error in the purity using the binning method
            meanSwapped, errSwapped = calculateError_byBinning(rSwapped_arr)
            meanUnswapped, errUnswapped = calculateError_byBinning(rUnswapped_arr)
            purity_arr[i_partition] = meanSwapped/meanUnswapped
            purity_err_arr[i_partition] = abs(purity_arr[i_partition]) * np.sqrt((errUnswapped/meanUnswapped)**2 + (errSwapped/meanSwapped)**2)
        # End of partition loop

        # Calculating the average and standard error in the entropy
        if (np.shape(N_partitions)[0] > 1):
            purity = np.prod(purity_arr, axis=0)
            entropy = -np.log(purity)
            err_purity = abs(purity)*np.sqrt(np.sum(np.square(np.divide(purity_err_arr, purity_arr))))
            err_entropy = abs(err_purity)/purity
        else:
            purity = purity_arr[0]
            entropy = -np.log(purity)
            err_purity = purity_err_arr[0]
            err_entropy = abs(err_purity)/purity

        self.S2_MC = entropy
        self.S2_stdError_MC = err_entropy
        self.purity_MC = purity
        self.purity_stdError_MC = err_purity
        if ratioTrick:
            self.AR_MC_arr = purity_arr
            self.AR_stdError_MC_arr = purity_err_arr

        print('S2 = ', str(self.S2_MC))
        print('Purity = ', str(self.purity_MC))

    def plotRho(self, rhoList):
        """
        Plots and saves the specified density matrices.

        Parameters
        ----------
        rhoList : List
            A list containing the names of the density matrices to be plotted.
            Allowed choices: rho_sos, free_rho_sos, free_rho_pqc, free_rho_marx, rho_nmm, free_rho_nmm.

        Outputs
        -------
        DensityMatrices.png: png file
            Image of the plot of the density matrices.
        """

        fig, (ax, ax_table) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [5, 1]}, figsize=(8,5))
        for rho in rhoList:
            if hasattr(self, rho):
                exec("ax.plot(self." + rho + "[:,0],self." + rho + "[:,1], label='" + rho + "')")
            else:
                raise Warning(rho + " has not been generated, make sure that the name is correct and that it has been generated")

        ax.set_xlabel('Phi')
        ax.set_ylabel('Density Matrix')
        ax.set_title('Density Matrices')
        ax.legend()
        text = np.array([["Temperature", str(round(self.T,3)) + " K"],
                        ["Number of\nGrid Points", str(self.Ngrid)],
                        ["Number of\nBeads", str(self.P)],
                        ["Number of\nMC Steps", str(self.MC_steps)],
                        ["Number of\nRotors", str(self.N)],
                        ["Interaction\nStrength", str(round(self.g,3))],
                        ["Rotational\nConstant", str(round(self.B,3))],
                        ["Skip Steps", str(self.Nskip)],
                        ["Equilibration\nSteps", str(self.Nequilibrate)],
                        ["PIGS", str(self.PIGS)],
                        ["Potential (V0)", str(round(self.V0,3))]])
        ax_table.axis("off")
        table = ax_table.table(cellText=text,
                               loc='center',
                               colWidths=[1.5, 1],
                               cellLoc='center')
        table.set_fontsize(9)
        table.scale(1,2.0)
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, "DensityMatrices.png"))
        plt.close('all')

    def plotHisto_total(self, *args):
        """
        Plots and saves the specified histograms for the overall system.

        Parameters
        ----------
        *args : String
            Keywords to determine which histograms to plot, for the overall
            PIMC, left, middle and right beads and the initial conditions.
            Allowed choices: PIMC, left, middle, right, initial.

        Outputs
        -------
        labels_Histograms_total.png: png file
            Image of the plots of the histograms specified.
        """
        if not hasattr(self, "histo_total"):
            raise Warning("The histograms do not exist, please execute ChoiceMC.runMC() or ChoiceMC.runMCReplica")
            return

        histo_dict = {"PIMC" : 1,
                      "middle" : 2,
                      "left" : 3,
                      "right" : 4,
                      "initial": 5}

        for hist in args:
            if hist not in histo_dict:
                raise Exception("An invalid argument was input, please use PIMC, left, middle, right or initial")

        fig_label = ""
        fig, (ax, ax_table) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [5, 1]}, figsize=(8,5))
        for hist in args:
            if hist == 'initial':
                exec("ax.plot(self.histo_initial[:,0],self.histo_initial[:,1], label='" + hist + "')")
            else:
                exec("ax.plot(self.histo_total[:,0],self.histo_total[:," + str(histo_dict[hist]) + "], label='" + hist + "')")
            fig_label += hist + "_"
        ax.set_xlabel('Phi')
        ax.set_ylabel('Probability Density')
        ax.set_title('Histograms')
        ax.legend()
        text = np.array([["Temperature", str(round(self.T,3)) + " K"],
                        ["Number of\nGrid Points", str(self.Ngrid)],
                        ["Number of\nBeads", str(self.P)],
                        ["Number of\nMC Steps", str(self.MC_steps)],
                        ["Number of\nRotors", str(self.N)],
                        ["Interaction\nStrength", str(round(self.g,3))],
                        ["Rotational\nConstant", str(round(self.B,3))],
                        ["Skip Steps", str(self.Nskip)],
                        ["Equilibration\nSteps", str(self.Nequilibrate)],
                        ["PIGS", str(self.PIGS)],
                        ["Potential (V0)", str(round(self.V0,3))]])
        ax_table.axis("off")
        table = ax_table.table(cellText=text,
                               loc='center',
                               colWidths=[1.5, 1],
                               cellLoc='center')
        table.set_fontsize(9)
        table.scale(1,2.0)
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, fig_label + "Histograms_total.png"))
        plt.close('all')

    def plotHisto_N(self, *args):
        """
        Plots and saves the specified histograms for each individual rotor.

        Parameters
        ----------
        *args : String
            Keywords to determine which histograms to plot, for the overall
            PIMC, left, middle and right beads and the initial conditions.
            Allowed choices: PIMC, left, middle, right.

        Outputs
        -------
        Histogram_label_N.png: png file
            Image of the plots of the histograms for each individual rotor of the
            specified bead.
        """
        if not hasattr(self, "histo_N"):
            raise Warning("The histograms do not exist, please execute ChoiceMC.runMC() or ChoiceMC.runMCReplica")
            return

        histo_dict = {"PIMC" : 1,
                      "middle" : 2,
                      "left" : 3,
                      "right" : 4}

        for hist in args:
            if hist not in histo_dict:
                raise Exception("An invalid argument was input, please use PIMC, left, middle or right")

        for hist in args:
            fig, (ax, ax_table) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [5, 1]}, figsize=(8,5))
            for n in range(self.N):
                exec("ax.plot(self.histo_N["+str(n)+"][:,0],self.histo_N["+str(n)+"][:," + str(histo_dict[hist]) + "], label='" + str(n) + "')")
            ax.set_xlabel('Phi')
            ax.set_ylabel('Probability Density')
            ax.set_title('Histograms')
            ax.legend()
            text = np.array([["Temperature", str(round(self.T,3)) + " K"],
                            ["Number of\nGrid Points", str(self.Ngrid)],
                            ["Number of\nBeads", str(self.P)],
                            ["Number of\nMC Steps", str(self.MC_steps)],
                            ["Number of\nRotors", str(self.N)],
                            ["Interaction\nStrength", str(round(self.g,3))],
                            ["Rotational\nConstant", str(round(self.B,3))],
                            ["Skip Steps", str(self.Nskip)],
                            ["Equilibration\nSteps", str(self.Nequilibrate)],
                            ["PIGS", str(self.PIGS)],
                            ["Potential (V0)", str(round(self.V0,3))]])
            ax_table.axis("off")
            table = ax_table.table(cellText=text,
                                   loc='center',
                                   colWidths=[1.5, 1],
                                   cellLoc='center')
            table.set_fontsize(9)
            table.scale(1,2.0)
            fig.tight_layout()
            fig.savefig(os.path.join(self.path, "Histogram_" + hist + "_N.png"))
        plt.close('all')
