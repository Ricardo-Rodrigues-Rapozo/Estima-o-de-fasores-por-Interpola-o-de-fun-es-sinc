import numpy as np 
import matplotlib.pyplot as plt 
import sinais 
from scipy.signal import freqz, lfilter


def phi_real(Nw, B_h, K, f0, H, Ts):
    """Calcula a parte real da função phi com base nos parâmetros fornecidos.

    Args:
        Nw(float): Numero de amostras correspontes ai intervalo de obs(Tw)
        B_h (float): Largura de banda associada ao harmônico.
        K (int): Ordem da interpolação
        f0 (float): Frequência fundamental.
        H (int): Ordem do harmônico.
        Ts (float): Tempo de amostragem 
    Returns:
        float: Parte real da função phi. Retorna uma matriz com dimensões 
        de (Nw,H(2k+1))  
    """
    
    nTs = np.arange(-Nw//2, Nw//2) * Ts ## numero de amostras * o periodo de amostragem  deve estar entre +-Tw/2 (vetor de tempo do filtro)
    ind = 0
    phik_h = np.zeros((Nw,H*(2*K+1))) + 1j*np.zeros((Nw,H*(2*K+1)))
    
    for h in range(H):
        for k in np.arange(-K,K+1):
            if(k == 0):
                k = 0.00001    
            phik_h[:,ind] = (np.sin(np.pi*(2*B_h*nTs-k))/(np.pi*(2*B_h*nTs-k)))*np.exp(1j * (2*np.pi*(h+1)*f0*nTs))
            ind = ind+1
    
    return phik_h



def phi_im(Nw, B_h, K, f0, H, Ts):
    """Calcula a parte imaginária da função phi com base nos parâmetros fornecidos.

    Args:
        t (float): Instante de tempo.
        B_h (float): Largura de banda associada ao harmônico.
        k (int): Índice da amostra.
        f0 (float): Frequência fundamental.
        h (int): Ordem do harmônico.

    Returns:
        float: Parte imaginária da função phi. Retorna uma matriz com dimensões 
        de (Nw,H(2k+1))
    """
    
    nTs = np.arange(-Nw//2, Nw//2) * Ts ## numero de amostras * o periodo de amostragem  deve estar entre +-Tw/2 (vetor de tempo do filtro)
    
    ind = 0
    phik_h = np.zeros((len(nTs),H*(2*K+1))) + 1j*np.zeros((len(nTs),H*(2*K+1)))
    
    for h in range(H):
        for k in np.arange(-K,K+1):
            if(k == 0):
                k = 0.00001    
            phik_h[:,ind] = (np.sin(np.pi*(2*B_h*nTs-k))/(np.pi*(2*B_h*nTs-k)))*np.exp(-1j*(2*np.pi*(h+1)*f0*nTs))
            ind = ind+1
    return phik_h



def add_column(m1,m2):
    """
    A func np.c_ basicamente cria duas colunas com os arrays informados
    entao m vai ser a uniao horizontal de m1 e m2
    Args:
    m1,m2(ndarray): matrizes de tamanhos hxc  e hxc

    return: 
    m(ndarray): Matriz com tamanho hx(c+c)
    """

    m = np.c_[m1, m2]
    return m 


def pseudo_inversa(m):
    """Realiza a pseudo inversa de uma matriz nao quadrada lxc 
    e retorna a matriz inversa de tamanho cxl

    Args : 
    m(ndarray): Matriz 

    return:
    m1(ndarray): Matriz invertida
    
    """

    m1 = np.linalg.pinv(m)
    return m1




def TVE(X, Xr):
    """_
Calcula o Total Vector Error (TVE) entre dois vetores de sinais complexos.

    O TVE é uma métrica utilizada para quantificar a diferença entre um vetor de
    sinal medido ou calculado (X) e um vetor de sinal de referência (Xr). 
    É expresso como uma porcentagem, que representa a magnitude da diferença
    entre os vetores complexos, considerando tanto a parte real quanto a imaginária.

    Args:
    -----------
    X : array-like
        Vetor de valores complexos representando o sinal medido ou calculado.
        
    Xr : array-like
        Vetor de valores complexos representando o sinal de referência.

    Return:
    --------
    TVE : array-like
        Vetor de erros totais de vetores (TVE), expresso como uma porcentagem.
        Cada elemento do vetor corresponde ao TVE calculado para o par 
        de valores complexos em X e Xr.
    """
    X_re = np.real(X)
    X_im = np.imag(X)

    Xr_re = np.real(Xr)
    Xr_im = np.imag(Xr)

    TVE = 100*np.sqrt(((X_re - Xr_re)**2 + (X_im - Xr_im)**2)/(Xr_re**2 + Xr_im**2))

    return TVE



def wrap_to_pi(theta):
    """
    Wraps an angle in radians to the range [-pi, pi].
    
    Parameters:
    - theta: The angle in radians to be wrapped.
    
    Returns:
    - The angle wrapped to the range [-pi, pi].
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi

def AcertaFase(Fs, f0, f_off, omega, h, f, phasor):
    """_summary_

    Args:
        Fs (_type_): _description_
        f0 (_type_): _description_
        f_off (_type_): _description_
        omega (_type_): _description_
        h (_type_): _description_
        f (_type_): _description_
        phasor (_type_): _description_

    Returns:
        _type_: _description_
    """
    delta_f = -(f0 - f_off)
    f_est = f_off
    wM1_mag = abs(h) ## calcula a magnitude da resposta em frequência do filtro 
    wM1_ang = np.unwrap(np.angle(h)) # angulo em rad

    idx = np.abs(f - f_est).argmin()
    idx1 = np.abs(f - 45).argmin() # retorna o indice em que o v
    idx2 = np.abs(f - 55).argmin()
    phi = wM1_ang[idx]

    phi1 = wM1_ang[idx1]  # Fase em idx1 Hz
    phi2 = wM1_ang[idx2] # Fase em idx2 Hz
    f1 = f[idx1] # Frequência correspondente a idx1 (~45 Hz)
    f2 = f[idx2] # Frequência correspondente a idx1 (~55 Hz)

    m = (phi2-phi1)/(f2-f1) ## delta y/ delta x

    phi50 = m*(50-f1) + phi1
    phi55 = m*(55-f1) + phi1
  
    Dphi = (delta_f/5)*(phi55-phi50)
    print(type(Dphi) ,(phi50))
    Dphi = wrap_to_pi(Dphi - phi50)
    plt.figure()
    plt.suptitle('Correção de Fase')
    ax1 = plt.subplot(211)
    plt.plot(f,wM1_mag)
    plt.plot(f[idx],wM1_mag[idx],'mo')  
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    #plt.xlim([44,56])
    ax2 = plt.subplot(212, sharex = ax1)
    plt.plot(f,wM1_ang*180/np.pi)
    plt.plot(f[idx],phi*180/np.pi,'mo') 
    plt.plot(f1,phi1*180/np.pi,'rx')  
    plt.plot(f2,phi2*180/np.pi,'rx')  
    plt.plot(50,phi50*180/np.pi,'go')   
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Fase em °')
    plt.show(block=False)
    return Dphi


