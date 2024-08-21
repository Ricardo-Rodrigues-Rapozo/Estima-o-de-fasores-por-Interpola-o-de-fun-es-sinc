import numpy as np
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
f0 = 50.0  # Frequência fundamental em Hz
Fs = 10100 # Frequencia de amostragem 
N0 = Fs/f0  #número de amostras por ciclo do sinal, ou seja, quantas amostras são capturadas durante um ciclo completo do sinal de frequência nominal f0
Ns = 50*N0
Ts = 1/Fs  # Período de amostragem em segundos
N = 100 # Representa o número de amostras de cada lado do ponto central t = 0 
Nw = 2*N +1 # Tem que ser impar e corresponde ao numero de amostras na janela de obs Tw.
k = 2  # Número de amostras ao redor da amostra central
B_h = 0.575  # Frequência de amostragem para as ordens harmônicas
tf = np.arange(-Nw//2, Nw//2 + 1) * Ts ## numero de amostras * o periodo de amostragem  deve estar entre +-Tw/2 (vetor de tempo do filtro)
t = np.arange(Ns)/Fs ## vetor de tempo do sinal 
Frep = 50 ## frames/s   
hmax = 13 ## maior harmonico 
hmag = 0.1 ## magnitude dos harmonicos 
SNR  =  60 ## relação sinal ruido 
T0 = 1/f0    ## periodo do sinal 
Tw = Nw*T0/N0 ## janela de observação deve ser menor que k/Bh
Nppc = Fs/f0 ## numero de pontos por ciclo 
f1 = 50.5 ## frequencia off nominal 

## ---------------------------------------------------------------------------------------------------------------------------------------------------
##      Gerando o sinal
## ---------------------------------------------------------------------------------------------------------------------------------------------------
x, X, f, ROCOF = sinais.signal_frequency(f0, Ns, f1, Fs, Frep, hmax, hmag, SNR)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Plotando o sinal gerado 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
x = np.sqrt(2)*x
# plt.figure()
# plt.plot(x)
# plt.xlabel('Time (s)')
# plt.ylabel('Magnitude')
# plt.title('Sinal de Teste')
# plt.show()


## ------------certo ---
phi_real = sinc.phi_real(tf, B_h, k, f0, hmax)
#print(phi_real.shape)
phi_im = sinc.phi_im(tf, B_h, k, f0, hmax)
#print(phi_im.shape)

#
#print(sinc.plus_column(phi_real,phi_im).shape)
#print(sinc.pseudo_inversa(sinc.plus_column(phi_real,phi_im)).shape) 
#print(t.reshape((202,50)).shape) # vetor de tempo em 50 janelas
p_est =  (2/np.sqrt(2))*sinc.pseudo_inversa(sinc.plus_column(phi_real,phi_im)) @ x.reshape((202,50))
print(type(p_est))


