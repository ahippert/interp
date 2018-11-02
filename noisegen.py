import numpy as np

# function taken from R. Prebet 
def recentre(im, A, B):
    """Applique une transformation affine pour que les valeurs de im soient entre A et B (A < B)"""
    M, m = np.max(im), np.min(im)
    return (im-m)/(M-m)*(B-A) + A

def aff_transform(val, a, b, c, d):
    return (val-a)/(b-a)*(d-c) + c

def geo(r, b):
    global gam
    corr = 1/(r)**(b)
    gam = np.abs(np.fft.fft2(corr))**2
    return gam/np.max(gam)

def exp(r, b):
    corr = np.exp(-b*r)
    gam = np.abs(np.fft.fft2(corr))**2
    return gam/np.max(gam)

def gen_noise(corr, alea):
    filt = corr*np.fft.fft2(alea)
    inv = np.real(np.fft.ifft2(filt))
    eps1, eps2 = 0.04*np.random.random(2)-0.02
    inv = recentre(inv,-0.4+eps1,0.4+eps2) # amp du bruit ici
    return (inv - np.mean(inv))   

def gen_noise2(corr, alea, coeff=0.05):
    filt = corr*np.fft.fft2(alea)
    inv = np.real(np.fft.ifft2(filt))
    eps1, eps2 = coeff*np.random.random(), coeff*np.random.random()
    return recentre(inv,eps1,1-eps2)
    
def gen_noise_series(corr, alea, time):
    return np.array([gen_noise(corr, alea[i]) for i in range(0, time)])

def gen_noise_series2(corr, alea, time):
    return np.array([gen_noise(corr[i], alea) for i in range(0, time)])

def temporal_noise():
    return None
