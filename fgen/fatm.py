#from utils import *
from numpy import sin,cos,pi,exp
import numpy as np
#import var as v


def recentre(im,A,B):
    """Applique une transformation affine pour que les valeurs de im soient entre A et B (A < B)"""
    M, m = np.max(im), np.min(im)
    return (im-m)/(M-m)*(B-A) + A
    
def expo(l=1):
    # Fonction d'auto correlation
    c = lambda r : np.exp(-l*1e1*r)
    gam = np.abs(np.fft.fft2(c(v.r)))**2
    return gam/np.max(gam)

def geo(x, b=1):
    # Fonction d'auto correlation
    c = lambda r : 1/(r)**(b*1.2)  
    gam = np.abs(np.fft.fft2(c(x)))**2
    return gam/np.max(gam)
    
def phi_atm(Gam, alea):
    # FFT d'un bruit blanc gaussien,
    B = np.fft.fft2(alea)
    # Convolution des deux fonctions (par mult de la fft)
    AtmF = B*Gam
    #On recupere la fonction associee a la fft associee
    Atm = np.real(np.fft.ifft2(AtmF))
    eps1, eps2 = 0.04*np.random.random(2)-0.02
    Atm = recentre(Atm,-0.1+eps1,0.1+eps2)
    return (Atm - np.mean(Atm))
   
def genere_atm(Gam, aleas, nt):
    """Serie temporelle du bruit athmospherique"""
    return np.array([ phi_atm(Gam, aleas[i]) for i in range(nt)])

def phi_corr(Gam, alea):
    # FFT d'un bruit blanc gaussien,
    B = np.fft.fft2(alea)
    # Convolution des deux fonctions (par mult de la fft)
    AtmF = B*Gam
    #On recupere la fonction associee a la fft associee
    Atm = np.real(np.fft.ifft2(AtmF))
    eps1, eps2 = 0.05*np.random.random(), 0.05*np.random.random()
    return recentre(Atm,eps1,1-eps2)

def genere_corr(Gam, aleas, nt):
    """Serie temporelle de la correlation"""
    return np.array([ phi_corr(Gam, aleas[i]) for i in range(nt)])


#Tests
# aff(phi_corr(geo(1),recentre(np.random.normal(0.5,0.2,(v.nx,v.ny)),0,1)),fig=1)

# A = np.random.gamma(0.1,1,(v.nx,v.ny))

# aff(phi_atm(expo(),A), color = True,fig = 1 )
# plt.tight_layout(pad=0, rect=[-0.2,-0.05,1.01,1.01])
# aff(phi_atm(geo(),A), color = True, fig = 2)
# plt.tight_layout(pad=0, rect=[-0.2,-0.05,1.01,1.01])
# plt.clf()
# plt.hist(recentre(flat(np.random.normal(0.5,0.2,(v.nx,v.ny))),0,1), bins=1000)
# plt.show()

# x = np.linspace(0,1,100000)
# plt.plot(x, 1/(x)**(1.2) )
# plt.axis([0,1,0,10])
# plt.tight_layout()

# A = np.random.gamma(0.1,1,(v.nx,v.ny))
# plt.figure()
# plt.imshow(phi_atm(geo(),A),cmap = pylab.get_cmap("jet"))
# plt.colorbar().ax.tick_params(labelsize=14)
# plt.axis('off')
# plt.show() 
# name='atm_example1.png'
# print(name)
# plt.savefig(name)

