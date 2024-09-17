# def AcertaFase(Fs, f0, f_off, omega, h, f, phasor,ha,hb):
#     """_summary_

#     Args: h*f_off
#         Fs (_type_): _description_
#         f0 (_type_): _description_
#         f_off (_type_): _description_
#         omega (_type_): _description_
#         h (_type_): _description_
#         f (_type_): _description_
#         phasor (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     f_est = f_off # 45
#     fmais = f_est + 5
#     fmenos = f_est - 5
#     delta_f = -(f0 - f_est) ## -5
#     wM1_mag = abs(h) ## calcula a magnitude da resposta em frequência do filtro 
#     wM1_ang = np.unwrap(np.angle(h)) # angulo em rad
#     idx = np.abs(f - f_est).argmin() 
#     idx1 = np.abs(f - f_est).argmin() # retorna o indice no array f em que o val é mais procimo de 45 
#     idx2 = np.abs(f - 55).argmin()
#     phi = wM1_ang[idx] ##no array;´´ de angs(rads) na posição em que f é o mais proximo de f_est
#     phi1 = wM1_ang[idx1]  # Fase em idx1 
#     phi2 = wM1_ang[idx2]  # Fase em idx2
#     fatual = f[idx]
#     f1 = f[idx1] # Frequência correspondente a idx1 (~45 Hz)
#     f2 = f[idx2] # Frequência correspondente a idx1 (~55 Hz)
#     m = (phi2-phi1)/(f2-f1) ## delta y/ delta x
#     phi50 = m*(50-f1) + phi1
#     phi55 = m*(55-f1) + phi1
#     Dphi1 = (delta_f/5)*(phi55-phi50)
#     Dphi =  wrap_to_pi(Dphi1) + wrap_to_pi(phi50)## DIferença entre o angulo de 55 e 50 
#     print(wrap_to_pi(phi50)*180/np.pi)
#     plt.figure()
#     # plt.suptitle('Correção de Fase')
#     # ax1 = plt.subplot(211)
#     # plt.plot(f,wM1_mag)
#     # plt.plot(f[idx],wM1_mag[idx],'mo')  
#     # plt.xlabel('Frequência (Hz)')
#     # plt.ylabel('Magnitude')
#     # #plt.xlim([44,56])
#     ax2 = plt.subplot(212)
#     plt.plot(f,wM1_ang*180/np.pi)
#     plt.plot(f[idx],phi*180/np.pi,'mo') 
#     plt.plot(f1,phi1*180/np.pi,'rx')  
#     plt.plot(f2,phi2*180/np.pi,'rx')  
#     plt.plot(50,phi50*180/np.pi,'go')   
#     plt.xlabel('Frequência (Hz)')
#     plt.ylabel('Fase em ° here ')
#     plt.show(block=True)
#     return Dphi