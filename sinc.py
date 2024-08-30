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

def harm_est(m):
    """Recebe uma matriz de estimador de harmonicos no dominio da frequencia
    e faz a interpolação desses valores de acordo com K. Pega os valores de uma 
    matriz do tipo p_(-1),p_(0),p_(1), p_(-1),p_(0),p_(1),p_(-1),p_(0),p_(1)...
    e pega os valores referentes a p_(0) que são referentes ao fasor.
    
    Args:

    returns:
        P(): VALOR DA MAGNITUDE DOS FASORES 
    
    """
    # m = abs(m)
    P = np.zeros(26) + 1j*np.zeros(26)
    cont = 1
    con = 0
    for i in range(len(m)):
        if i == cont:
            P[con] = m[i]
            con = con +1
            cont = cont + 3
    return P


def harm_est_p1(m):
    """Recebe uma matriz de estimador de harmonicos no dominio da frequencia
    e faz a interpolação desses valores de acordo com K. Pega os valores de uma 
    matriz do tipo p_(-1),p_(0),p_(1), p_(-1),p_(0),p_(1),p_(-1),p_(0),p_(1)...
    e pega os valores referentes a p_(0) que são referentes ao fasor.
    
    Args:

    returns:
        P(): VALOR DA MAGNITUDE DOS FASORES 
    
    """
    # m = abs(m)
    P = np.zeros(26) + 1j*np.zeros(26)
    cont = 2
    con = 0
    for i in range(len(m)):
        if i == cont:
            P[con] = m[i]
            con = con +1
            cont = cont + 3
    return P



def harm_est_p_less_1(m):
    """Recebe uma matriz de estimador de harmonicos no dominio da frequencia
    e faz a interpolação desses valores de acordo com K. Pega os valores de uma 
    matriz do tipo p_(-1),p_(0),p_(1), p_(-1),p_(0),p_(1),p_(-1),p_(0),p_(1)...
    e pega os valores referentes a p_(0) que são referentes ao fasor.
    
    Args:

    returns:
        P(): VALOR DA MAGNITUDE DOS FASORES 
    
    """
    # m = abs(m)
    P = np.zeros(26) + 1j*np.zeros(26)
    cont = 0
    con = 0
    for i in range(len(m)):
        if i == cont:
            P[con] = m[i]
            con = con +1
            cont = cont + 3
    return P    



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


def t0(X,hmax):
    """_summary_
        É necessario que que no X[numero de harm, numero total de pontos ] fique com 
        [numero de harm, cada amostra dna posição 300 de cada janela(são 20 janelas)]
    Args:
        X (_type_): Sinal no dominio da frequencia 
        Nw (_type_): Numero de amostras por janela do sinal
        num_TW (_type_): numero de janelas no sinal 
    return:
        something
    """

    ans = np.arange(300,12000,600)
    ans = np.arange(300,11700 , 600)
    ind = 0
    i = 0
    S = np.zeros((13,19)) + 1j*np.zeros((13,19)) ## Tem que ser 19 pois são 19 janelas
    for h in range(hmax):
        for i, value in enumerate(ans):
            S[h,i] = X[h,value]
    return S


