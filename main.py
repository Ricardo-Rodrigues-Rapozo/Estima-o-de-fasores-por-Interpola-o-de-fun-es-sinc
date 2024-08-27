import numpy as np
import matplotlib
##matplotlib.use('TKAgg')  # Experimente outros bacKends se necessário
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import sinc
import sinais

"""
Teste em relação ao artigo:

Dynamic Harmonic Synchrophasor Estimator Based on Sinc Interpolation Functions
"""


## ----------------------------------------------------------------------------------------------------------------------------------------------------
##     Parâmetros do teste 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
f0 = 50.0    # Frequência fundamental em Hz
Fs = 10000   # Frequencia de amostragem 
N0 = Fs/f0   #número de amostras por ciclo do sinal, ou seja, quantas amostras são capturadas durante um ciclo completo do sinal de frequência nominal f0
Ns = 60*N0   #número total de amostras do sinal
Ts = 1/Fs    # Período de amostragem em segundos
T0 = 1/f0
Tw = 3/f0
Nw = int(N0*(Tw/T0) + 1)
K = 1        # Número de amostras ao redor da amostra central
B_h = 0.575  # Frequência de amostragem para as ordens harmônicas
Frep = 50    ## frames/s   
hmax = 13    ## maior harmonico 
hmag = 0.1   ## magnitude dos harmonicos 
SNR  =  60   ## relação sinal ruido
f1 = 50    ## frequencia off nominal 

## ---------------------------------------------------------------------------------------------------------------------------------------------------
##      Gerando o sinal
## ---------------------------------------------------------------------------------------------------------------------------------------------------
x, X, f, ROCOF = sinais.signal_frequency(f0, Ns, f1, Fs, Frep, hmax, hmag, SNR)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Plotando o sinal gerado 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

plt.figure()
plt.plot(x)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.title('Sinal de Teste')
plt.show(block = False)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Analise espectral do sinal original 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
X = np.fft.fft(x) ## Calculo da fft
freq = np.arange(0,Fs/Ns+Fs/2,Fs/Ns) ##
mag = abs(X[0:1+len(X)//2]) ##
plt.figure()
plt.stem(mag*2/Ns)
plt.grid()
plt.show(block = False)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Calculo das matrizes phi_real e phi_imaginario 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
phi_real = sinc.phi_real(Nw, B_h, K, f0, hmax, Ts)
phi_im = sinc.phi_im(Nw, B_h, K, f0, hmax, Ts)

plt.figure()
plt.plot(abs(phi_real[:,0]))
plt.plot(abs(phi_real[:,1]))
plt.plot(abs(phi_real[:,2]))
plt.show(block = False)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Juntando as matrizes phi's em uma unica matriz coluna e em seguida 
##    fazendo a psdeudo inversa para calcular os valores estimados de p^
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
phi = sinc.add_column(phi_real,phi_im)
p0 = np.zeros((int(Ns//Nw),26)) + 1j*np.zeros((int(Ns//Nw),26))
p1 = np.zeros((int(Ns//Nw),26)) + 1j*np.zeros((int(Ns//Nw),26))
p_less_1 = np.zeros((int(Ns//Nw),26)) + 1j*np.zeros((int(Ns//Nw),26))
for nn in range (int(Ns//Nw)):## 0 ate 19
    s = x[nn*Nw:(nn+1)*Nw] 
    p_est = np.array([sinc.pseudo_inversa(phi) @ s ])
    p_est = np.reshape(p_est,(78,1))
    p0[nn,:] = sinc.harm_est(p_est)
    p1[nn,:] = sinc.harm_est_p1(p_est)
    p_less_1[nn,:] = sinc.harm_est_p_less_1(p_est)
    


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Calculo do Ph'0  
##    
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

ph0 = p0
    
 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Calculo do Ph'1
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
   
    
ph1 = 2*B_h*(p1 - p_less_1 )
    


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Calculo do Ph''2
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

ph2 = 4*(B_h**2)*(2*p1 + 2*p_less_1 - (p0*(np.pi**2))/3)

print(ph2.shape)

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Erro de Frequencia (FE)
## -------------------------------------------------------------------------- --------------------------------------------------------------------------



## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    ROCOF
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

ROCOF1 =  (1/(2*np.pi)) * (np.imag(ph2)*np.imag(p0))/(np.abs(ph0)**2)
ROCOF2 = -(1/np.pi)*(np.real(ph1)*np.real(ph0)*np.imag(ph1)*np.imag(ph0))/(np.abs(ph0)**4)
ROCOF = ROCOF1 + ROCOF2
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Graficos de Magnitude e Fase respectivamente do fasor  
##    
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
#mag
plt.figure()
plt.subplot(311)
plt.stem(abs(p0[0,:]))
plt.subplot(312)
plt.stem(abs(p0[1,:]))
plt.subplot(313)
plt.stem(abs(p0[2,:]))
plt.show(block = False)
# Fase
plt.figure()
plt.subplot(311)
plt.plot(np.unwrap(np.angle(p0[0,:])))
plt.subplot(312)
plt.plot(np.unwrap(np.angle(p0[1,:])))
plt.subplot(313)
plt.plot(np.unwrap(np.angle(p0[2,:])))
plt.show(block = False)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Graficos de Magnitude e Fase respectivamente do ROCOF  
##    
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
#mag
plt.figure()
plt.subplot(311)
plt.stem(abs(ROCOF[0,:]))
plt.subplot(312)
plt.stem(abs(ROCOF[1,:]))
plt.subplot(313)
plt.stem(abs(ROCOF[2,:]))
plt.show(block = True)
# Fase
plt.figure()
plt.subplot(311)
plt.plot(np.unwrap(np.angle(ROCOF[0,:])))
plt.subplot(312)
plt.plot(np.unwrap(np.angle(ROCOF[1,:])))
plt.subplot(313)
plt.plot(np.unwrap(np.angle(ROCOF[2,:])))
plt.show(block = True)