import numpy as np
from scipy.signal import freqz, lfilter
from matplotlib import pyplot as plt


def signal_frequency(f1, N, f0, Fs, Frep, hmax, hmag, SNR):
    """ Gera sinais para o teste Signal Frequency segundo a norma IEC/IEEE 60255-118
    Parameters:
    -----------
        f1 (float): Frequência do sinal
        N (integer): Número de pontos do sinal gerado
        f0 (float): Frequência nominal
        Fs (float): Frequência de Amostragem
        Frep (float): Frequência de Reporte

    Returns:
    --------
        x (array ): Amplitude do sinal
        X (complex array): Frequência do sinal
        f (array): Número de pontos do sinal gerado
        ROCOF (array): Frequência nominal
    """

    status = True
    if Frep < 10:
        if abs(f1-f0) > 2:
            raise TypeError('Frequencia do sinal fora do intervalo estipulado pela norma')
    elif Frep < 25:
        if abs(f1-f0) > Frep/5:
            raise TypeError('Frequencia do sinal fora do intervalo estipulado pela norma')
    else:
        if Frep >= 25:
            if abs(f1-f0) > 5:
                raise TypeError('Frequencia do sinal fora do intervalo estipulado pela norma') 
    
    if(status == False):
        return 0
    else:
        t = np.arange(N)/Fs
        var_ruido = (1/2)*(10**(-SNR/10))
        ruido = np.sqrt(var_ruido)*np.random.randn(len(t))


        phi = 0#np.random.uniform(-np.pi,np.pi)

        x = np.cos(2*np.pi*f1*t + phi)

        X = np.zeros((hmax,len(t))) + 1j*np.zeros((hmax,len(t)))
        X[0,:] = (1/np.sqrt(2))*np.exp(1j*(2*np.pi*(f1-f0)*t + phi))

        for hh in range(2,hmax+1):
            phi = 0#np.random.uniform(-np.pi,np.pi)
            x = x + hmag*np.cos(2*np.pi*hh*f1*t + phi)
            
            X[hh-1,:] = (hmag/np.sqrt(2))*np.exp(1j*(2*np.pi*hh*(f1-f0)*t + phi))
        
        x = x + ruido

        f = f1*np.ones(len(x))
        ROCOF = np.zeros(len(x))

        return (x, X, f, ROCOF)
    




def estima_fundamental(x, wM, f0, Fs):   
    """
    Estima a componente fundamental de um sinal, corrigindo a magnitude, fase, 
    e estimando a frequência e ROCOF (Rate of Change of Frequency).

    Parâmetros:
    -----------
    x (array_like): Sinal de entrada a ser analisado.
    wM (array_like): Janela de filtragem utilizada para suavizar o sinal.
    f0 (float): Frequência nominal do sinal.
    Fs (float): Frequência de amostragem do sinal.

    Retorna:
    --------
    X (array_like): Componente fundamental do sinal corrigida em magnitude e fase.
    f_est (array_like): Estimativa da frequência do sinal ao longo do tempo.
    ROCOF (array_like): Taxa de variação da frequência (Rate of Change of Frequency) ao longo do tempo.
    """
    N = len(wM)
    n = np.arange(N) #np.arange(-(N-1)/2,1+(N-1)/2,1)
    wM1 = wM*np.exp(1j*2*np.pi*f0*n/Fs)
    X = lfilter(wM1,1,x)
    X = X*np.exp(1j*2*np.pi*f0*(N/2)/Fs)
    # Correção da Magnitude e da Fase
    #---------------------------------------------------
    Xabs = (2/np.sqrt(2))*np.abs(X)
    Xpha = np.unwrap(np.angle(X)) - 2*np.pi*(f0/Fs)*np.arange(len(X)) - np.pi
    # Estimação Frequência e ROCOF
    #---------------------------------------------------
    coeff = np.array([0.5, 0, -0.5])
    delta_f = lfilter(coeff, 1, Xpha)*(Fs/(2*np.pi))
    f_est = f0 + delta_f
    ROCOF = lfilter(coeff, 1, f_est)*(Fs)
    # Correção da Fase
    #---------------------------------------------------
    omega,h = freqz(wM1,1,4096)
    f = omega*Fs/(2*np.pi)
    wM1_mag = abs(h)
    wM1_ang = np.unwrap(np.angle(h))
    idx = np.abs(f - f_est[-1]).argmin()
    phi = wM1_ang[idx]
    idx1 = np.abs(f - 45).argmin()
    idx2 = np.abs(f - 55).argmin()
    phi1 = wM1_ang[idx1]
    phi2 = wM1_ang[idx2]
    f1 = f[idx1]
    f2 = f[idx2]
    m = (phi2-phi1)/(f2-f1)
    phi50 = m*(50-f1) + phi1
    phi55 = m*(55-f1) + phi1
    Dphi = (delta_f/5)*(phi55-phi50)
    Xpha = wrapToPi(Xpha - Dphi)
    X = Xabs*np.cos(Xpha) + 1j*Xabs*np.sin(Xpha)
#### ---------- plots ----------------------
    plt.figure()
    plt.suptitle('Correção de Fase')
    ax1 = plt.subplot(211)
    plt.plot(f,wM1_mag)
    plt.plot(f[idx],wM1_mag[idx],'mo')  
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    #plt.xlim([44,56])
    ax2 = plt.subplot(212, sharex = ax1)
    plt.plot(f,wM1_ang)
    plt.plot(f[idx],phi,'mo') 
    plt.plot(f1,phi1,'rx')  
    plt.plot(f2,phi2,'rx')  
    plt.plot(50,phi50,'go')   
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Fase em °')
    #plt.show(block=False)
    plt.show()
    return X, f_est, ROCOF



def wrapToPi(x):
    """Essa função faz a "ajuste" dos ângulos para o intervalo [−π,π].
       Em outras palavras, ela normaliza os ângulos para que estejam dentro desse intervalo.

    Args:
        x (numpy.ndarray): description
    Returns:
       xwrap(): Retorna os ângulos ajustados.
    """
    xwrap = np.remainder(x, 2 * np.pi) #Calcula o resto da divisão de cada valor em x por 2π, o que garante que os valores estejam no intervalo 0-2pi 
    mask = np.abs(xwrap) > np.pi #Cria uma máscara para identificar valores que estão fora do intervalo 
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask]) ## ajusta os valores que estao fora do intervalo (subtraindo 2pi se necessario)
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap


def estima_harmonicos(x, wM, f0, f1, Fs, hmax):
    """
            Estima os componentes harmônicos de um sinal de entrada.

            Parameters:
            -----------
            x  (numpy.ndarray): Sinal de entrada (sinal de tempo discreto) sobre o qual os harmônicos serão estimados. Deve ser um vetor unidimensional.
            
            wM (numpy.ndarray): Janela de ponderação (window) aplicada ao sinal antes de realizar a filtragem. Deve ser um vetor de tamanho N, onde N é o número de amostras.
            
            f0 (float): Frequência fundamental do sinal (em Hz). Usada para ajustar a fase do sinal.
            
            f1 (float): Frequência estimada para a primeira harmônica do sinal (em Hz). Usada para ajustar a frequência dos harmônicos.
            
            Fs (float): Frequência de amostragem do sinal (em Hz). Usada para converter índices de amostra em tempo ou frequência.
            
            hmax (int): Número máximo de harmônicos a serem estimados. A função calcula os harmônicos do 2º até o hmax-ésimo.

            Returns:
            --------
            X (numpy.ndarray): Matriz complexa contendo os harmônicos estimados. A matriz tem dimensões (hmax-1, len(x)), onde cada linha corresponde a um harmônico diferente (do 2º até o hmax-ésimo) e cada coluna corresponde a um ponto de tempo.

            Description:
            ------------
            Esta função estima os componentes harmônicos de um sinal de entrada x até o hmax-ésimo harmônico. A estimativa é realizada aplicando uma janela de ponderação wM ajustada para cada harmônico, seguida de uma filtragem do sinal. 

            O sinal resultante é então corrigido em termos de magnitude e fase para garantir uma representação precisa dos harmônicos.

            A função retorna uma matriz complexa X, onde cada linha representa um harmônico diferente e cada coluna representa o valor do harmônico em um determinado ponto no tempo.

            Steps:
            ------
            1. Inicializa uma matriz X para armazenar as estimativas dos harmônicos.
            
            2. Para cada harmônico hh do 2º até o hmax-ésimo:
                - Ajusta a janela wM para a frequência específica do harmônico hh.
                - Filtra o sinal x usando a janela ajustada.
                - Aplica correções de magnitude e fase ao sinal filtrado.
                - Armazena o resultado na matriz X.
            
            3. Retorna a matriz X contendo os harmônicos estimados.
            """
    N = len(wM)
    n = np.arange(N) 

    X = np.zeros((hmax-1,len(x))) + 1j*np.zeros((hmax-1,len(x)))

    for hh in range(2,hmax+1):
        wmH = wM*np.exp(1j*2*np.pi*hh*f1*n/Fs)
        Xh = lfilter(wmH,1,x)

        Xh = Xh*np.exp(1j*2*np.pi*f0*(N/2)/Fs)
        
        # Correção da Magnitude e da Fase
        #---------------------------------------------------
        Xabs = (2/np.sqrt(2))*np.abs(Xh)
    
        Xpha = np.unwrap(np.angle(Xh)) - 2*np.pi*(hh*f0/Fs)*np.arange(len(Xh)) - np.pi
        
        Xh = Xabs*np.cos(Xpha) + 1j*Xabs*np.sin(Xpha)
        
        X[hh-2,:] = Xh

    return X