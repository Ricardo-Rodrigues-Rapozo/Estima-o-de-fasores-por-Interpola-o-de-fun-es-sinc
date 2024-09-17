import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import lfilter, freqz
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
T0 = 1/f0    # Periodo do sinal
Tw = 3/f0    # Janela de Observação
Nw = int(N0*(Tw/T0) + 1) # Numero de amostras correspontes ai intervalo de obs(Tw)
K = 1        # Número de amostras ao redor da amostra central
B_h = 0.575  # Frequência de amostragem para as ordens harmônicas
Frep = 50    ## frames/s   
hmax = 13   ## maior harmonico 
hmag = 0.1   ## magnitude dos harmonicos 
SNR  = 6000000   ## relação sinal ruido
f1 = 45  ## frequencia off nominal 
num_Tw = int(Ns//Nw) #representa a quantidade de janelas de observação Tw que cabem no total de amostras NsNs coletadas.
num_harm = 5 ## numero de harmonico que desejo aferir 
vectK = 13 ## pega o k_0
harm_Fasor_Ref = 5

## ---------------------------------------------------------------------------------------------------------------------------------------------------
##      Gerando o sinal
## ---------------------------------------------------------------------------------------------------------------------------------------------------

x, X, f, ROCOF = sinais.signal_frequency(f1, Ns, f0, Fs, Frep, hmax, hmag, SNR)

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Plotando o sinal gerado 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

## ----------------------------------------------------------------------------------------------------------------------------------------------------
##     Calculo das matrizes phi_real e phi_imaginario 
## ----------------------------------------------------------------------------------------------------------------------------------------------------

phi_real = sinc.phi_real(Nw, B_h, K, f0, hmax, Ts)
phi_im = sinc.phi_im(Nw, B_h, K, f0, hmax, Ts)
#print(phi_real.shape)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Juntando as matrizes phi's em uma unica matriz coluna e em seguida 
##    fazendo a psdeudo inversa para calcular os valores estimados de p^
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

phi = sinc.add_column(phi_real,phi_im)
#print(phi.shape)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Implementação como Banco de Filtros
##    
## -------------------------------------------------------------------------- --------------------------------------------------------------------------


gkh = sinc.pseudo_inversa(phi) ## [resposta ao impulso, Nw] ## 78 pq para cada 13 harmonicos existem -k a k nesse caso 3 para o lado negativo e positivo 
#print(gkh.shape)
coeff1 = gkh[vectK,:] ## pegando a parte k_0 referente ao fasor 
coeff2 = np.flip(coeff1) ## invertendo o vetor como na equação 
coeff3 = np.roll(coeff2,Nw-1) ## rolando o vetor por um fator de Nw -1  
print(gkh.shape,"shape ")
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Plot dos coeficientes do banco de filtros
##    
## -------------------------------------------------------------------------- --------------------------------------------------------------------------


# plt.figure()
# plt.title('Coeficientes do Filtro ')
# ax1 = plt.subplot(211)
# plt.stem(np.real(coeff1))
# plt.stem(np.real(coeff2),'r')
# plt.stem(np.real(coeff3),'k')
# plt.grid()
# ax2 = plt.subplot(212, sharex = ax1)
# plt.stem(np.imag(coeff1))
# plt.stem(np.imag(coeff2),'r')
# plt.stem(np.imag(coeff3),'k')
# plt.grid()

# plt.show(block=False)

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Estimação 
##    
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

omega,h = freqz(coeff3,1,4096) ## Frequências normalizadas (em radianos por amostra) nas quais a resposta foi calculada. O intervalo é de 0 a π
f = omega*Fs/(2*np.pi) # frequencia em hz 
wM1_mag = abs(h) # 
wM1_ang = np.unwrap(np.angle(h)) # fase em radianos 

plt.figure()
ax1 = plt.subplot(211)
plt.plot(f,wM1_mag, label='Magnitude do filtro ')
plt.grid()
plt.legend()  # Adiciona legenda para magnitude
ax2 = plt.subplot(212, sharex = ax1)
plt.plot(f,wM1_ang*180/np.pi, label='Fase em graus do filtro ')
plt.grid()
plt.legend()  # Adiciona legenda para magnitude
plt.show(block=True)

phasor = lfilter(coeff3,1,x)# sinal filtrado 

# phasor = phasor*np.exp(1j*2*np.pi*f0*(Nw/2)/Fs)
Xmag = (2/np.sqrt(2))*abs(phasor) # magnitude do phasor 
Xpha = np.unwrap(np.angle(phasor)) - 2*np.pi*(f0/Fs)*np.arange(len(phasor)) # rad 

Dphi = sinc.AcertaFase(Fs, f0,f1, omega, h, f, phasor,num_harm)

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Comparação entre os estimados e a referencia 
##    
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
l = Xpha*180/np.pi - Dphi*180/np.pi
l1 = np.unwrap(np.angle(X[harm_Fasor_Ref,:]))*180/np.pi
print(Xmag.shape,X.shape)
plt.figure()
plt.subplot(211)
plt.plot(Xmag,'b',label = "Mag do sinal estimado")
plt.plot(abs(X[harm_Fasor_Ref,:]),'r',label="Mag harmonico de ref")# X[harmonico , amostras ]
plt.legend()  # Adiciona legenda para magnitude


plt.subplot(212)
plt.plot(Xpha*180/np.pi - Dphi*180/np.pi,'b', label = 'Estimação da fase corrigida')
plt.plot(np.unwrap(np.angle(X[harm_Fasor_Ref,:]))*180/np.pi ,'r',label = 'Fase de referencia ')
plt.legend()  # Adiciona legenda para magnitude
plt.show(block=True)
plt.figure()
plt.plot(l - l1)
plt.show()

##     phi50
## phi50*180/np.pi = -896.39
## em 50 no grafico tem -895.6

##    Dphi = phi - phi 50 formula
## phi = -844.660
## Dphi1*180/np.pi = 53.514
## Dphi1*180/np.pi - phi50*180/np.pi = 949.913
