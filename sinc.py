import numpy as np 
import matplotlib.pyplot as plt 
import sinais 


def dadosVect(x):
    """Pega os dados de magnitude e fase do
       sinal passado como argumento

    Args:
        x (np.ndarray ): sinal 

    Returns:
        ah (np.ndarray): vetor com as magnitudes de x
        ph (np.ndarray): vetor com as fases de x 
    """
    ah = np.abs(x)
    ph = np.angle(x)
    return ah,ph
    
    




def fasor_harmonico(t,a_h,theta_h):
    """ Sincrofasor harmonico na posição hth

    Args:
        t (type): description
        a_h (type): description
        theta_h (type): description

    Returns:
        list : 
    """
    
    for i in range(len(t)):
        #ph = a_h[i]*np.exp(1j * theta_h)
        ph = a_h*np.exp(1j *theta_h)
    return ph



            
def phi_real(t, B_h, k, f0, h):
    """Calcula a parte real da função phi com base nos parâmetros fornecidos.

    Args:
        t (float): Instante de tempo.
        B_h (float): Largura de banda associada ao harmônico.
        k (int): Índice da amostra.
        f0 (float): Frequência fundamental.
        h (int): Ordem do harmônico.

    Returns:
        float: Parte real da função phi.
    """
    ind = 0
    phik_h = np.zeros((len(t),h*(2*k+1))) + 1j*np.zeros((len(t),h*(2*k+1)))
    
    for H in range(h):
        for kk in np.arange(-k,k+1):
            phik_h[:,ind] = (np.sin(np.pi*(2*B_h*t-k))/(np.pi*(2*B_h*t-k)))*np.exp(1j * (2 * np.pi * (H+1) * f0 * t))
            ind = ind+1
    
    # for K in range(k):
    #     for H in range(h):
    #         for i in range(len(t)):
    #             term = np.exp(1j * (2 * np.pi * H * f0 * t[i]))
    #             phi_re = (np.sin(np.pi * (B_h * t[i] - K)) / (np.pi * (2 * B_h * t - K))) * term
    return phik_h



def phi_im(t, B_h, k, f0, h):
    """Calcula a parte imaginária da função phi com base nos parâmetros fornecidos.

    Args:
        t (float): Instante de tempo.
        B_h (float): Largura de banda associada ao harmônico.
        k (int): Índice da amostra.
        f0 (float): Frequência fundamental.
        h (int): Ordem do harmônico.

    Returns:
        float: Parte imaginária da função phi.
    """
    term = np.exp(-1j * (2 * np.pi * h * f0 * t))
    phi_im = (np.sin(np.pi * (B_h * t - k)) / (np.pi * (2 * B_h * t - k))) * term
    return np.imag(phi_im)



def Filter_bank(phi_real, phi_im, phk):
    """Aplica o banco de filtros em relação aos vetores fornecidos.

    Args:
        phi_real (np.array): Vetor real (R).
        phi_im (np.array): Vetor imaginário (I).
        phk (np.array): Vetor de amostras do sinal (pK).

    Returns:
        np.array: Resultado da operação de filtro.
    """
    # Converte os vetores em colunas (vetores coluna)
    ar1 = np.array([phi_real]).T
    ar2 = np.array([phi_im]).T
    ar3 = np.array([phk]).T
    ar4 = np.array([np.conj(phk)]).T
    
    # Aplica a expressão √2/2 * [ R I ] [ pK ; p*K ]
    result = (np.sqrt(2)/2) * (ar1 * ar3 + ar2 * ar4)
    
    return result