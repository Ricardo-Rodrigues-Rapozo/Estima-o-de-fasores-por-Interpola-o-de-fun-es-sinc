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
    
    ind = 0
    phik_h = np.zeros((len(t),h*(2*k+1))) + 1j*np.zeros((len(t),h*(2*k+1)))
    
    for H in range(h):
        for kk in np.arange(-k,k+1):
            phik_h[:,ind] = (np.sin(np.pi*(2*B_h*t-k))/(np.pi*(2*B_h*t-k)))*np.exp(-1j * (2 * np.pi * (H+1) * f0 * t))
            ind = ind+1
    
    # for K in range(k):
    #     for H in range(h):
    #         for i in range(len(t)):
    #             term = np.exp(1j * (2 * np.pi * H * f0 * t[i]))
    #             phi_re = (np.sin(np.pi * (B_h * t[i] - K)) / (np.pi * (2 * B_h * t - K))) * term
    return phik_h



def plus_column(m1,m2):
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
