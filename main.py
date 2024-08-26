import numpy as np
import matplotlib
##matplotlib.use('TkAgg')  # Experimente outros backends se necessário
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
Fs = 10100   # Frequencia de amostragem 
N0 = Fs/f0   #número de amostras por ciclo do sinal, ou seja, quantas amostras são capturadas durante um ciclo completo do sinal de frequência nominal f0
Ns = 50*N0   #número total de amostras do sinal
Ts = 1/Fs    # Período de amostragem em segundos
N = 100      # Representa o número de amostras de cada lado do ponto central t = 0 
Nw = 2*N +1  # Tem que ser impar e corresponde ao numero de amostras na janela de obs Tw.
k = 1        # Número de amostras ao redor da amostra central
B_h = 0.575  # Frequência de amostragem para as ordens harmônicas
tf = np.arange(-Nw//2, Nw//2 + 1) * Ts ## numero de amostras * o periodo de amostragem  deve estar entre +-Tw/2 (vetor de tempo do filtro)
t = np.arange(Ns)/Fs ## vetor de tempo do sinal 
Frep = 50    ## frames/s   
hmax = 13    ## maior harmonico 
hmag = 0.1   ## magnitude dos harmonicos 
SNR  =  60   ## relação sinal ruido
T0 = 1/f0    ## periodo do sinal 
Tw = Nw*T0/N0## janela de observação deve ser menor que k/Bh
Nppc = Fs/f0 ## numero de pontos por ciclo 
f1 = 50.5    ## frequencia off nominal 

## ---------------------------------------------------------------------------------------------------------------------------------------------------
##      Gerando o sinal
## ---------------------------------------------------------------------------------------------------------------------------------------------------
x, X, f, ROCOF = sinais.signal_frequency(f0, Ns, f1, Fs, Frep, hmax, hmag, SNR)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Plotando o sinal gerado 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
x = np.sqrt(2)*x
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
phi_real = sinc.phi_real(tf, B_h, k, f0, hmax)
phi_im = sinc.phi_im(tf, B_h, k, f0, hmax)
#print(phi_real.shape)
#print(phi_im.shape)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Juntando as matrizes phi's em uma unica matriz coluna e em seguida 
##    fazendo a psdeudo inversa para calcular os valores estimados de p^
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
p = sinc.add_column(phi_real,phi_im)
#print(p.shape)
# pinv  = sinc.pseudo_inversa(p)
# print(pinv.shape)
p_est = np.array([sinc.pseudo_inversa(sinc.add_column(phi_real,phi_im)) @ np.reshape(x,(202,50)) ])
# #print((np.array(p_est)).shape)
print(type(p_est))


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Ajustando o tamanho da matriz p_est
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
p_est = np.sqrt(2) * np.reshape(p_est,(78,50))
# print(p_est.shape)
# print(p_est.shape)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##   Fazendo a interpolação dos valores p0 do vetor p^
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
P = sinc.harm_est(p_est)
plt.figure()
plt.stem(P)
plt.show()

