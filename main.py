import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import lfilter
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
hmax = 1   ## maior harmonico 
hmag = 0.1   ## magnitude dos harmonicos 
SNR  = 6000000   ## relação sinal ruido
f1 = 51  ## frequencia off nominal 
num_Tw = int(Ns//Nw) #representa a quantidade de janelas de observação Tw que cabem no total de amostras NsNs coletadas.

#print(N0)
## ---------------------------------------------------------------------------------------------------------------------------------------------------
##      Gerando o sinal
## ---------------------------------------------------------------------------------------------------------------------------------------------------

x, X, f, ROCOF = sinais.signal_frequency(f1, Ns, f0, Fs, Frep, hmax, hmag, SNR)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Plotando o sinal gerado 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

# plt.figure()
# plt.plot(x)
# plt.xlabel('Time (s)')
# plt.ylabel('Magnitude')
# plt.title('Sinal de Teste')
# plt.show(block = False)

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Analise espectral do sinal original 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# freq = np.arange(0,Fs/Ns+Fs/2,Fs/Ns) ##
# mag = abs(X[0:1+len(X)//2]) ##
# plt.figure()
# plt.stem(mag*2/Ns)
# plt.grid()
# plt.show(block = False)


## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##     Calculo das matrizes phi_real e phi_imaginario 
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

phi_real = sinc.phi_real(Nw, B_h, K, f0, hmax, Ts)
phi_im = sinc.phi_im(Nw, B_h, K, f0, hmax, Ts)
# print(phi_real.shape)
# plt.figure()
# plt.plot(abs(phi_real[:,0]))
# plt.plot(abs(phi_real[:,1]))
# plt.plot(abs(phi_real[:,2]))
# plt.show()

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Juntando as matrizes phi's em uma unica matriz coluna e em seguida 
##    fazendo a psdeudo inversa para calcular os valores estimados de p^
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

phi = sinc.add_column(phi_real,phi_im)
#print(phi.shape)
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##        Pegando as amostras p-1  p0   p1
## -------------------------------------------------------------------------- --------------------------------------------------------------------------

# p0 = np.zeros((num_Tw,26)) + 1j*np.zeros((num_Tw,26))
# p1 = np.zeros((num_Tw,26)) + 1j*np.zeros((num_Tw,26))
# p_less_1 = np.zeros((num_Tw,26)) + 1j*np.zeros((num_Tw,26))
# #print(num_Tw)
# for nn in range (num_Tw): ##0 ate 18 isso por conta do 601 e não 600(19)
#     s = x[nn*Nw:(nn+1)*Nw] #Nw = 601 - janelando x com tamanho Nw
#     p_est = np.array([sinc.pseudo_inversa(phi) @ s ])
#     p_est = np.reshape(p_est,(78,1))
#     p0[nn,:] = sinc.harm_est(p_est)
#     p1[nn,:] = sinc.harm_est_p1(p_est)
#     p_less_1[nn,:] = sinc.harm_est_p_less_1(p_est)
    

# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# ##    Calculo do Ph'0  
# ##    NOSSO FASOR É O ph0
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# index = np.arange(83,19*601,601)
# print(index)
# Xref = X[:,index].T

# ph0 = p0[:,0:13]
# print(p0[0,:])

# plt.figure()
# plt.subplot(211)
# plt.stem(2*abs(ph0[0,:]))
# plt.stem(abs(Xref[0,:]),'r')
# plt.subplot(212)
# plt.stem((np.angle(ph0[0,:]) )*180/np.pi)
# plt.stem(np.angle(Xref[0,:])*180/np.pi,'r')
# plt.show(block=False)

# plt.figure()
# plt.subplot(211)
# plt.stem(2*abs(ph0[1,:]))
# plt.stem(abs(Xref[1,:]),'r')
# plt.subplot(212)
# plt.stem((np.angle(ph0[1,:]) )*180/np.pi) # + np.array([1,0,1,0,1,0,1,0,1,0,1,0,1])*np.pi
# plt.stem(np.angle(Xref[1,:])*180/np.pi,'r')
# plt.show(block=False)

# Implementação como Banco de Filtros
#-----------------------------------------------------------------------------------------
gkh = sinc.pseudo_inversa(phi)
#print(gkh.shape)

coeff = gkh[1,:] ## 

coeff = np.flip(coeff)## inverteru a matriz 
coeff = np.roll(coeff,-Nw+1) 

phasor = lfilter(coeff,1,x)## faz  o filtro com os coeficientes de gkh

phasor = phasor*np.exp(-1j*2*np.pi*f0*(Nw/2)/Fs)
Xmag = (2/np.sqrt(2))*abs(phasor)
Xpha = np.unwrap(np.angle(phasor)) - 2*np.pi*(f0/Fs)*np.arange(len(phasor))

plt.figure()
plt.subplot(211)
plt.plot(Xmag)
plt.plot(abs(X[0,:]),'r')
plt.subplot(212)
plt.plot(Xpha*180/np.pi)
plt.plot(np.unwrap(np.angle(X[0,:]))*180/np.pi,'r')
plt.show(block=False)

plt.figure()
plt.plot((Xpha - np.unwrap(np.angle(X[0,:])))*180/np.pi)
plt.show(block=True)


#print(ph0.shape)     
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Calculo do Ph'1
## -------------------------------------------------------------------------- --------------------------------------------------------------------------   
    
# ph1 = 2*B_h*(p1 - p_less_1 )
    
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# ##    Calculo do Ph''2
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------

# ph2 = 4*(B_h**2)*(2*p1 + 2*p_less_1 - (p0*(np.pi**2))/3)
# #print(ph2.shape)

# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# ##    Erro de Frequencia (FE)
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# harm = np.arange(1,hmax+1)
# Fe = harm*f0 + (np.imag(ph1[:,0:13])*np.imag(ph0[:,0:13]))/(np.abs(ph0[:,0:13]))**2

# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# ##    ROCOF
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------

# ROCOF1 =  (1/(2*np.pi)) * (np.imag(ph2[:,0:13])*np.imag(p0[:,0:13]))/(np.abs(ph0[:,0:13])**2)
# ROCOF2 = -(1/np.pi)*(np.real(ph1[:,0:13])*np.real(ph0[:,0:13])*np.imag(ph1[:,0:13])*np.imag(ph0[:,0:13]))/(np.abs(ph0[:,0:13])**4)
# ROCOF = ROCOF1 + ROCOF2

# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# ##    Tempo de latencia 
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------

# T_lat = (Nw - 1)/2*Fs

# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# ##    TVE 
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------

# #print(X.shape) #[num harmonicos, numeros de pontos do sinal]
# #print(ph0.shape)
# ph01 = ph0[:,0:13]
# ph02 = ph0[:,13:26]
# #print(ph02.shape)
# ph0_plus = ph0[:,0:13]
# print(ph0_plus.shape)
# Xref = sinc.t0(X,hmax)
# print(Xref.shape)
# TVE = sinc.TVE(ph0_plus.T,Xref) # de 0 a 19 janelas e 13 harmonicos
# #print(TVE.shape)
# plt.figure()
# plt.stem(TVE[:,0])
# plt.show(block = False)

# ## -------------------  ------------------------------------------------------- --------------------------------------------------------------------------
# ##    Graficos de Magnitude e Fase respectivamente do fasor      
# ## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# #mag
# plt.figure()
# plt.subplot(311)
# plt.stem(2*abs(ph0_plus[0,:]))
# plt.stem(abs(Xref[:,0]),'r')
# plt.subplot(312)
# plt.stem(2*abs(ph0_plus[1,:]))
# plt.stem(abs(Xref[:,1]),'r')
# plt.subplot(313)
# plt.stem(2*abs(ph0_plus[2,:]))
# plt.stem(abs(Xref[:,2]),'r')
# plt.show(block = False)
# # Fase
# plt.figure()
# plt.subplot(311)
# plt.stem(np.angle(ph0_plus[0,:]))
# plt.stem(np.angle(Xref[:,0]),  'r')
# plt.subplot(312)
# plt.stem(np.angle(ph0_plus[1,:]))
# plt.stem(np.angle(Xref[:,1]),'r')

# plt.subplot(313)

# plt.stem((np.angle(ph0_plus[2,:])))
# plt.stem(np.angle(Xref[:,2]),'r')

# plt.show(block = True)
# print(X.shape)

# plt.figure()
# plt.plot(np.angle(X[0,:]))
# plt.show()
# -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Graficos de Magnitude e Fase respectivamente do ROCOF  
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
#mag
# plt.figure()
# plt.subplot(311)
# plt.stem(abs(ROCOF[0,:]))
# plt.subplot(312)
# plt.stem(abs(ROCOF[1,:]))
# plt.subplot(313)
# plt.stem(abs(ROCOF[2,:]))
# plt.show(block = False)
# # Fase
# plt.figure()
# plt.subplot(311)
# plt.plot(np.unwrap(np.angle(ROCOF[0,:])))
# plt.subplot(312)
# plt.plot(np.unwrap(np.angle(ROCOF[1,:])))
# plt.subplot(313)
# plt.plot(np.unwrap(np.angle(ROCOF[2,:])))
# plt.show(block = False)

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    Plot da FE
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# print(Fe.shape)
# plt.figure()
# plt.stem(Fe[0,0:13])
# plt.show()

## -------------------------------------------------------------------------- --------------------------------------------------------------------------
##    PLot da ROCOF
## -------------------------------------------------------------------------- --------------------------------------------------------------------------
# print(ROCOF.shape)
# plt.figure()
# plt.stem(ROCOF[0,0:13])
# plt.show()

# 