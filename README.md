# Syncrophasor Project
* Autor: Ricardo Rodrigues Rapozo
* Autor : Mayk Alves Lamim
* [LinkedIn](www.linkedin.com/in/ricardo-rodrigues-rapozo-569560227)  
## Descrição do Projeto

Este projeto busca validar o método de estimação de fasores do artigo "Dynamic Harmonic Synchrophasor Estimator Based on Sinc Interpolation Functions", implementando a estimação em Python.


## Ambiente de Desenvolvimento

- **Sistema Operacional:** Ubuntu 22.04.4 LTS (Codename: Jammy)
- **Versão do Python:** Python 3.10.12
- **Dependências Python:**
  - contourpy==1.2.1
  - cycler==0.12.1
  - fonttools==4.53.1
  - kiwisolver==1.4.5
  - matplotlib==3.9.2
  - numpy==2.1.0
  - packaging==24.1
  - pillow==10.4.0
  - pyparsing==3.1.2
  - python-dateutil==2.9.0.post0
  - scipy==1.14.1
  - six==1.16.0

## Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/syncrophasor.git
   cd syncrophasor

##
## O p_est é uma estimativa dos coeficientes de Fourier para os diferentes componentes harmonicos. Ou seja, estão no dominio da frequencia e representam amplitude e fase de cada componente harmonico

## Explicação da função harm_est no modulo sinc.
 No caso o vetor de fasores estimados tem 78 linhas e 50 colunas ([78 50]), ele tem esse shape por conta do uso do reshape utilizado apenas para que fosse possivel fazer a operação com matrizes.
 O importante aqui é perceber que em relação as linhas tem-se uma divisão do espectro que é simetrico em relação a 0, então temos a divisão de 78 por 2 = 39.
 Então no tocante ao lado positivo do espectro temos 39 amostras, sendo que como K = 1, o k vai de -1 ate 1, que dão 3 dados para cada amostra, então a cada 3 amostras temos dados de uma amostra que representa algo para o 
 fasor, no caso do fasor propriamente dito as amostras p0 são utilizadas. Por exemplo, suponha que o vetor tenha o seguinte formato p_estimado = [p[0] p[1] p[2] p[3] p[4] p[5] p[6] ... p[38]]
 e em relação aos k's que ditos anteriormente, é o mesmo que dizer que p[0],p[1],p[2],...p[38] são referentes a p_(k,h), em que k representa a iteração da interpolação e o h a ordem harmonica, por exemplo fazendo para o primeiro harmonico, que é a 
 fundamental, p_(-1,1) = p[0], p_(0,1) = p[1], p_(1,1) = p[2], então e para o segundo harmonico seria a intepolação novamente de k indo de -1 a 1 com h igual 2 indo das amostras p[3] ate a amostra p[5].
 No artigo é dito que para pegar os valores do fasor, basta usar as amostras p_(0,h), sendo que as outras são usadas para outros calculos como o erro da frequencia(RFE) e a ROCOF.
