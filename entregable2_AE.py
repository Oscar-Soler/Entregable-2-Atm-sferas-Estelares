#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Bienvenid@.
Este es un código desarrollado para resolver el entregable 2 de la asignatura Atmósferas Estelares
del master de Astrofísica de la Universidad de La Laguna, impartida por el profesor Artemio Herrero Davó.

El objetivo de dicho entregable es realizar un cálculo de poblaciones y opacidades a partir de 
unos modelos de atmósfera del programa MARCS.

Para el correcto funcionamiento de este código es necesario contar con un directorio que contenga los archivos:
    - t4000.dat
    - t6000.dat
    - t8000.dat
    - Ai.txt 
Los tres primeros corresponden con los modelos de atmósfera de MARCS analizados, 
mientras que el último es un archivo de texto que contiene únicamente las líneas que 
hacen referencia a las abundancias químicas logarítmicas de los modelos.

El código está compuesto por dos funciones principales para el cálculo de poblaciones y opacidades,
junto con todas las líneas empleadas para realizar los plots que se encuentran en el informe correspondiente.

También se encuentran algunas líneas que permitieron extraer los datos de interés en formato de tabla de Latex.

Cualquier duda o problema técnico que pueda tener o error que pueda encontrar, no dude en escribirnos:
    - Oscar: alu0101327365@ull.edu.es
    - Carlota: alu0101341309@ull.edu.es
'''

# Importación de paquetes empleados
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io.ascii import read

# Preferencias para el dibujo de las figuras
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['lines.linewidth'] = 2

matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['xtick.minor.size'] = 4
matplotlib.rcParams['ytick.minor.size'] = 4
matplotlib.rcParams['ytick.minor.visible'] = True  
matplotlib.rcParams['xtick.minor.visible'] = True   

matplotlib.rcParams['axes.edgecolor'] = 'black'   
matplotlib.rcParams['axes.linewidth'] = 2


a = input('Quien eres? Carlota, Oscar u Otro? \n Si eres otro, introduce aquí el path del directorio donde tienes los archivos t4000.dat, t6000.dat y t8000.dat:  ')

if a == 'C':
    os.chdir('/Users/carlota/Desktop/Atmosferas_estelares/entregable2')
elif a=='O':
    os.chdir('C:/Users/osole/OneDrive/Documentos/1_Astro/Atmosferas/Entregable_2/')
else:
    os.chdir(a)

#%% Importación de tablas
t4 = read('t4000.dat', data_start = 25, names=['k', 'lgTauR', 'lgTau5', 'Depth', 'T', 'Pe', 'Pg', 'Prad', 'Pturb'])
t6 = read('t6000.dat', data_start = 25, names=['k', 'lgTauR', 'lgTau5', 'Depth', 'T', 'Pe', 'Pg', 'Prad', 'Pturb'])
t8 = read('t8000.dat', data_start = 25, names=['k', 'lgTauR', 'lgTau5', 'Depth', 'T', 'Pe', 'Pg', 'Prad', 'Pturb'])

file_path = "Ai.txt"

Ai = []
with open(file_path, 'r') as file:
    for line in file:
        values_list = line.split()
        Ai += ([float(i) for i in values_list])

#Ai es un array que contiene todas las abundancias químicas logarítmicas excepto la del hidrógeno
Ai = np.array(Ai[1:])
# se empleará para verificar la población total de hidrógeno obtenida

#%% función que realiza plots según las variables solicitadas
def plots(x_var, y_var, model, t, num_f=0, f=0, logy = False, title = 'Titulo'):
    
    if f==0: #si no existe figura previa, se crea
        plt.close(num_f)
        fig, ax = plt.subplots(1,1, figsize=(10,6), num=num_f)
        
    else: # si ya existe una figura previa, se dibuja en esa
        fig = f
        ax = fig.gca()
    colors = plt.cm.inferno_r(np.linspace(0, 1, 3+2)) #mapa de color empleado
    for i, mod in zip(range(3), ['4000 K', '6000 K', '8000 K']):
        if model == mod:
            k=i+1
    ax.plot(t[x_var], t[y_var], label=model, color = colors[k])
    
    # selección específica de label según las variables ploteadas
    if x_var == 'lgTauR':
        ax.set_xlabel('$\log \\tau_{\\rm Ross}$')
    elif x_var == 'Depth':
        ax.set_xlabel('$r$ [cm]')
    else: ax.set_xlabel(x_var)
    
    if y_var == 'lgTauR':
        ax.set_ylabel('$\log \\tau_{\\rm Ross}$')
    elif y_var == 'T':
        ax.set_ylabel('$T$ [K]')
    elif y_var == 'Pe':
        ax.set_ylabel('$P_{\\rm e}$ [$\\rm dyn/cm^2$]')
    elif y_var == 'Pe/Pg':
        ax.set_ylabel('$P_{\\rm e}/P_{\\rm g}$')
    elif y_var == 'Prad/Pg':
        ax.set_ylabel('$P_{\\rm rad}/P_{\\rm g}$')                  
    else: ax.set_ylabel(y_var)
    
    if logy == True:
        ax.set_yscale('log')
        
    if title != 'Titulo':
        ax.set_title(title)
    
    ax.grid(alpha=1)
    ax.grid(which='minor', alpha = .3)
    ax.legend()

    return fig

#%% Profundidad óptica de Rosseland frente a profundidad geométrica
fig1 = plots('Depth', 'lgTauR', '4000 K', t4, num_f=1)
plots('Depth', 'lgTauR', '6000 K', t6, f=fig1)
plots('Depth', 'lgTauR', '8000 K', t8, f=fig1, title = 'Profundidad óptica de Rosseland\n frente a profundidad geométrica')
plt.tight_layout()
plt.show()

#%% Temperatura frente profundidad óptica de Rosseland
fig2 = plots('lgTauR', 'T', '4000 K', t4, num_f=2)
plots('lgTauR', 'T', '6000 K', t6, f=fig2)
plots('lgTauR', 'T', '8000 K', t8, f=fig2, logy = True, title = 'Temperatura frente profundidad óptica de Rosseland')
plt.tight_layout()
plt.show()

#%% Presión electrónica frente a profundidad óptica de Rosseland
fig3 = plots('lgTauR', 'Pe', '4000 K', t4, num_f=3)
plots('lgTauR', 'Pe', '6000 K', t6, f=fig3)
plots('lgTauR', 'Pe', '8000 K', t8, f=fig3, logy = True, title = 'Presión electrónica frente a \n profundidad óptica de Rosseland')
plt.tight_layout()
plt.show()

#%% Creación de columnas con cocientes
def div(num, denom, t):
    t[num+'/'+denom] = t[num]/t[denom]

for i in [t4,t6,t8]:
    div('Pe', 'Pg', i)
    div('Prad', 'Pg', i)
#%% Cociente entre presión electrónica y del gas\n frente a profundidad óptica de Rosseland
fig4= plots('lgTauR', 'Pe/Pg', '4000 K', t4, num_f=4)
plots('lgTauR', 'Pe/Pg', '6000 K', t6, f=fig4)
plots('lgTauR', 'Pe/Pg', '8000 K', t8, f=fig4, logy = True, title='Cociente entre presión electrónica y del gas\n frente a profundidad óptica de Rosseland')
plt.tight_layout()
plt.show()

#%%  Cociente entre presión de radiación y del gas frente a profundidad óptica de Rosseland
fig5= plots('lgTauR', 'Prad/Pg', '4000 K', t4, num_f=5)
plots('lgTauR', 'Prad/Pg', '6000 K', t6, f=fig5)
plots('lgTauR', 'Prad/Pg', '8000 K', t8, f=fig5, logy = True, title='Cociente entre presión de radiación y del gas\n frente a profundidad óptica de Rosseland')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

#%% POBLACIONES

# Calculo de poblaciones
def popu(saha = False, boltzmann = False):
        #test = False, test2 = False, testNHII = False):
        # saha = True: cálculo de poblaciones de iones
        # boltzmann = True: cálculo de poblaciones de estados excitados
    
    t = [t6, t8]  # modelos de atmósfera a analizar  
# -----------------------------------------------------------------------------------------------------
    # Saha:
    u = [1,2,1] #uH-, uHI, uHII
    chi = [0.765, 13.6] #eV (de ionizacion)
    
    # Definición de matrices que serán rellenadas con los datos
    # el valor N...[i][j] corresponderá con el modelo de temperatura 6000K (i=0) o 8000K (i=1)
    # y a una profundidad óptica tau_Ross = 1 (j=0) o 10 (j=1)
    
    Ne = np.zeros([2,2]) #población de electrones
    # Resultados de ecuación de Saha:
    NH_NHI, NHINHII = np.zeros([2,2]), np.zeros([2,2]) # cocientes de poblaciones de iones de H
    # poblaciones de iones de H:
    NH_, NHI, NHII = np.zeros([2,2]), np.zeros([2,2]), np.zeros([2,2])
    # verificación usando Ai de la población total de H
    NHtest = np.zeros([2,2])
    
    # constantes:
    #cte de boltzmann en diferentes unidades según el caso en el que se emplee
    kb_cgs = 1.380649E-16 #erg/K
    kb = 8.6173E-5 #ev/K
    # Uso de las abundancias químicas logarítmicas
    A = 1-np.sum(10**(Ai-12))
    #valores de tau_rosseland de interés
    tauR_values = [1, 10]
    
    # Ahora se procede a identificar los índices de los modelos que corresponden con los tau_rosseland 
    # especificados a cada temperatura
    indexes = np.zeros([2,2], dtype=int)
    i = 0
    for t_i in t:
        j=0
        for tauR in tauR_values:
            # [i] indica el modelo de temperatura
            # [j] indica la profundidad óptica
            indexes[i][j] = int(np.argmin(np.abs(t_i['lgTauR']-np.log10(tauR))))
            j+=1
        i+=1

    for i in range(2):
        for j in range(2):
            # se recorren todas las temperaturas y profundidades ópticas
            # población de electrones
            Ne[i][j] = t[i]['Pe'][indexes[i][j]] / (kb_cgs*t[i]['T'][indexes[i][j]])
            # Ecuación de Saha
            NH_NHI[i][j] = 2.07E-16*Ne[i][j]*u[0]/u[1]*t[i]['T'][indexes[i][j]]**(-3/2)*np.exp(chi[0]/(kb*t[i]['T'][indexes[i][j]]))
            NHINHII[i][j] = 2.07E-16*Ne[i][j]*u[1]/u[2]*t[i]['T'][indexes[i][j]]**(-3/2)*np.exp(chi[1]/(kb*t[i]['T'][indexes[i][j]]))
            # población total de H meidante abundancias químicas logarítmicas
            NHtest[i][j] = (t[i]['Pg'][indexes[i][j]]-t[i]['Pe'][indexes[i][j]]) / (kb_cgs*t[i]['T'][indexes[i][j]]*A)
    # Resolución del sistema de ecuaciones
    NH_ = NH_NHI*NHINHII*Ne / (1-NH_NHI*NHINHII)
    NHI = NH_ / NH_NHI
    NHII = NHI / NHINHII
      
    
    if saha == True:
        return Ne, NH_, NHI, NHII, NH_NHI, NHINHII, NHtest
    
    elif boltzmann == True:
        chi = [0,10.206, 12.095]     # chi1, chi2, chi3 [eV] ; energias de excitacion de los niveles
        g = [2,8,18]                 # g(n) = 2n*2 pesos estadísticos

        n1, n2, n3 = np.zeros([2,2]), np.zeros([2,2]), np.zeros([2,2])

        for i in range(2):
            for j in range(2):
                # Ecuación de boltzmann a falta de multiplicar por N_jk
                n1[i][j] = (g[0]*np.exp(-chi[0]/(kb*t[i]['T'][indexes[i][j]]))) / u[1]
                n2[i][j] = (g[1]*np.exp(-chi[1]/(kb*t[i]['T'][indexes[i][j]]))) / u[1]
                n3[i][j] = (g[2]*np.exp(-chi[2]/(kb*t[i]['T'][indexes[i][j]]))) / u[1]

        n1 = n1 * NHI
        n2 = n2 * NHI
        n3 = n3 * NHI
    
        return n1, n2, n3
    
    else: 
        return Ne, NH_, NHI, NHII, NH


# el valor N...[i][j] corresponderá con el modelo de temperatura 6000K (i=0) o 8000K (i=1)
# y a una profundidad óptica tau_Ross = 1 (j=0) o 10 (j=1)
Ne, NH_, NHI, NHII, NH_NHI, NHINHII, NH = popu(saha = True)
n1, n2, n3 = popu(boltzmann=True)


#%% OPACIDADES

# [lambda] = Amstrong
# dtype(N_saha, n_boltz) = list

# esta función para el cálculo de las opacidades requiere de un array que contenga las longitudes de 
# onda que se quieren cubrir (np.linspace(500,20000,20000) por ejemplo);
# y de dos listas que contengan las poblaciones calculadas por Saha y Boltzmann en el apartado anterior.
# El orden de las variables en dichas listas es esencial para el funcionamiento del código

def opac(lamb, N_saha, n_boltz, HI=False, H_=False,  es=False, cs=False):
    # HI = True para cálculo de opacidades de HI
    # idem para H_, es (electron scattering)
    # si cs = True se devolverán las secciones eficaces en lugar de las opacidades
    
    n_lamb = len(lamb)
    
    # constantes empleadas
    R = 1.0968E5                # 1/cm    
    kb_cgs = 1.380649E-16       # erg/K
    h = 6.62607015E-34 * 1E7    # erg*s
    c = 299792458*1E2           # cm/s
    
    # modelos de amtósfera y valores de profundidad óptica bajo estudio
    t = [t6, t8]
    tauR_values = [1, 10]
    
    # longitud de onda en cm y frecuencia en s^-1
    lamb_cgs = lamb * 1E-8 #cm
    nu = c/lamb_cgs # s^-1
    
    # misma búsqueda de índices correspondientes a cada tau_R que en la función anterior
    indexes = np.zeros([2,2], dtype=int)
    i = 0
    for t_i in t:
        j=0
        for tauR in tauR_values:
            indexes[i][j] = int(np.argmin(np.abs(t_i['lgTauR']-np.log10(tauR))))
            j+=1
        i+=1
            
    # Calculo de opacidades en HI
    if HI == True:
    # definición de variables:
    # g... = gaunt factor
    # sig... = sección eficaz [cm^2]
    # k... = coeficiente de opacidad [cm^-1]
    
        # free - free:
        gff, sigff, kff = np.zeros([2,2, n_lamb]), np.zeros([2,2, n_lamb]), np.zeros([2,2, n_lamb])
        
        # bound - free:
        gbf, sigbf, kbf = np.zeros([3,2,2,n_lamb]), np.zeros([3,2,2,n_lamb]), np.zeros([3,2,2,n_lamb])
        

        for i in range(2):      #Temperaturas
            for j in range(2):  #TausR
                T = t[i]['T'][indexes[i][j]]
                ne = N_saha[0]
                nk = N_saha[3]
                
                
        # ----------------------------------------------------------------------------
        # free - free
                gff[i][j] = 1+ (0.3456/(lamb_cgs*R)**(1/3))*((lamb_cgs*kb_cgs*T)/(h*c) + (1/2))
                sigff[i][j] = 3.7E8*(1**2 *gff[i][j])/(T**(1/2)*nu**3)
                
                kff[i][j] = sigff[i][j]*ne[i][j]*nk[i][j]*(1-np.exp((-h*nu)/(kb_cgs*T)))
        

        # ----------------------------------------------------------------------------
        # bound - free
                # se hace un bucle extra en torno a los diferentes estados excitados del HI
                for k in range(3):
                    n = k+1
                    ni = n_boltz[k]
                    
                    lamb0 = n**2/(R*1E-8)
                    lamb_ok = lamb <= lamb0
                    
                    gbf[k][i][j] = 1- (0.3456/(lamb_cgs*lamb_ok*R)**(1/3))*((lamb_cgs*lamb_ok*R)/(n**2) - (1/2))
                    sigbf[k][i][j] = 2.815E29*1**4 / (n**5 * nu**3) *gbf[k][i][j]
                    
                    kbf[k][i][j] = sigbf[k][i][j]*ni[i][j]*(1-np.exp((-h*nu)/(kb_cgs*T)))
                    
        
        # ----------------------------------------------------------------------------
        # bound - bound
            # No depende de T; se pide solo para tauR=1
            # Ly_alpha, Ly_beta, H_alpha

        lamb_bb = [1215.66, 1025.83, 6562.8] # AA
        nu_bb = c/(np.array(lamb_bb)*10**(-8)) # s^-1   
        
        # Constantes a emplear
        e = 1.602E-19 * 3E9    #statC; carga electron
        m = 9.1094E-28         #g; masa electron
        
        f = [0.0, 0.0, 0.6407]
        f[0] = 2**5 * 0.717/(3**(3/2)*np.pi*1**5*2**3) * ((1/1**2)-(1/2**2))**(-3)
        f[1] = 2**5 * 0.765/(3**(3/2)*np.pi*1**5*3**3) * ((1/1**2)-(1/3**2))**(-3)
        
        sigbb, kbb = np.zeros(3), np.zeros([2,3])
        
        for i in range(3):
            sigbb[i] = np.pi*e**2*f[i] / (m*c)
        
        for i in range(2): #Temperaturas
            T = t[i]['T'][indexes[i][0]]
            
            for j in range(3): #Diferentes niveles de ocupación: diferentes lineas
                kbb[i][j] = sigbb[j]*n_boltz[j][i][0]*(1-np.exp((-h*nu_bb[j])/(kb_cgs*T)))
                
        # aparecen algunos valores erróneos al dividir por 0, los hacemos 0.
        kff[np.isinf(kff)], kbf[np.isinf(kbf)], kbb[np.isinf(kbb)] = 0, 0, 0


        if cs == True:
            sigff[np.isinf(sigff)], sigbf[np.isinf(sigbf)], sigbb[np.isinf(sigbb)] = 0, 0, 0
            return sigff, sigbf, sigbb
        else:
            return kff, kbf, kbb

    # Calculo de opacidades en H-           
    if H_ == True:
        
        # ----------------------------------------------------------------------------
        # free - free
        
        f0 = - 2.2763 - 1.6850*np.log10(lamb) + 0.76661*np.log10(lamb)**2 - 0.053346*np.log10(lamb)**3
        f1 = + 15.2827 - 9.2846*np.log10(lamb) + 1.99381*np.log10(lamb)**2 - 0.142631*np.log10(lamb)**3
        f2 = - 197.789 + 190.266*np.log10(lamb) -67.9775*np.log10(lamb)**2 + 10.6913*np.log10(lamb)**3 - 0.625151*np.log10(lamb)**4
        
        sigff_H_, kff_H_ = np.zeros([2,2, n_lamb]), np.zeros([2,2, n_lamb])
        
        for i in range(2): #Temperaturas
            for j in range(2): #TauR
                T = t[i]['T'][indexes[i][j]]
                theta = 5040/T
                
                sigff_H_[i][j] = 1E-26 * 10**(f0 + f1*np.log10(theta) + f2*np.log10(theta)**2)
                kff_H_[i][j] = t[i]['Pe'][indexes[i][j]]*sigff_H_[i][j]*N_saha[1][i][j]


        # ----------------------------------------------------------------------------
        # bound - free

        a0 = 1.99654
        a1 = -1.18267E-5
        a2 = 2.64243E-6
        a3 = -4.40524E-10
        a4 = 3.23992E-14
        a5 = -1.39568E-18
        a6 = 2.78701E-23

        sigbf_H_ = (a0 + a1*lamb + a2*lamb**2 + a3*lamb**3 + a4*lamb**4 + a5*lamb**5 + a6*lamb**6)*1E-18
        sigbf_H_[sigbf_H_ < 0] = 0
        kbf_H_ = np.zeros([2,2, n_lamb])

        for i in range(2): #Temperatura
            for j in range(2): #TauR para Pe
                T = t[i]['T'][indexes[i][j]]
                theta = 5040/T

                kbf_H_[i][j] = 4.158E-10 * sigbf_H_ * t[i]['Pe'][indexes[i][j]] * theta**(5/2) * 10**(0.754*theta)
                
        if cs == True:
            sigff_H_[np.isinf(sigff_H_)], sigbf_H_[np.isinf(sigbf_H_)] = 0, 0
            return sigff_H_, sigbf_H_
        else:
            kbf_H_[np.isinf(kbf_H_)] = 0
            return kff_H_, kbf_H_
    
    
    # Calculo de opacidades para electron scattering          
    if es == True:
        siges = 6.25E-25
        kes = N_saha[0]*siges

        if cs == True:
            return siges
        else:
            return kes

#%%    
# longitudes de onda bajo estudio. se escogen 19501 valores para que el paso sea de 1 AA
lamb_all = np.linspace(500,20000, 19501)

N_s = [Ne, NH_, NHI, NHII, NH_NHI, NHINHII, NH]
n_b = [n1, n2, n3]

# HI
HIkff, HIkbf, HIkbb = opac(lamb_all, N_s, n_b, HI=True)
# H_
H_kff, H_kbf = opac(lamb_all, N_s, n_b, H_=True)
# electrones
esk = opac(lamb_all, N_s, n_b, es=True)

# Secciones eficaces:
HIcsff, HIcsbf, HIcsbb = opac(lamb_all, N_s, n_b, HI=True, cs=True)
H_csff, H_csbf = opac(lamb_all, N_s, n_b, H_=True, cs=True)
escs = opac(lamb_all, N_s, n_b, es=True, cs=True)

#%% Tabla opacidades a 911+-1A, 3646+-1, 8203+-1
ind_1 = np.argmin(abs(lamb_all-911))
ind_2 = np.argmin(abs(lamb_all-3646))
ind_3 = np.argmin(abs(lamb_all-8205.6))

# los siguientes bucles, ejecutados de uno en uno, permiten crear las filas de las tablas en formato latex

for k, name in zip([H_kff, HIkff, H_kbf], ['H_kff', 'HIkff', 'H_kbf']):
    string = ''
    string+=str(name+' & ')
    for i in [ind_1, ind_2, ind_3]:
        string+='%.2e & '%k[0][0][i-1]
        string+='%.2e & '%k[0][0][i+1]
    print(string)
    
for j, name in zip(range(3), ['n=1', '2', '3']):
    string = ''
    string+=str(name+' & ')
    for i in [ind_1, ind_2, ind_3]:
        string+='%.2e & '%HIkbf[j][0][0][i-1]
        string+='%.2e & '%HIkbf[j][0][0][i+1]
    print(string)
    
for k, name in zip([H_kff, HIkff, H_kbf], ['H_kff', 'HIkff', 'H_kbf']):
    string = ''
    string+=str(name+' & ')
    for i in [ind_1, ind_2, ind_3]:
        string+='%.2e & '%k[1][0][i-1]
        string+='%.2e & '%k[1][0][i+1]
    print(string)
    
for j, name in zip(range(3), ['n=1', '2', '3']):
    string = ''
    string+=str(name+' & ')
    for i in [ind_1, ind_2, ind_3]:
        string+='%.2e & '%HIkbf[j][1][0][i-2]
        string+='%.2e & '%HIkbf[j][1][0][i+2]
    print(string)

for i in range(2):
    string = ''
    for j in range(3):
        string+='%.2e & '%HIkbb[i][j]
    print(string)


#%% PLOT OPACIDADES
# Plot de opacidades totales

fig, ax = plt.subplots(1,2, figsize=(12,8), sharey=True)

ind_16290 = np.argmin(abs(lamb_all-16290)) #correspondiente a donde se hace cero H_kbf
for i in range(2):
    for j in range(2):
        H_kbf[i][j][ind_16290:] = 0 

for i in range(2): # bucle querecorre las diferentes temperaturas
    T = ['T = 6000K', 'T = 8000K']
    
    ax[i].plot(lamb_all, HIkff[i][0], ls='dotted', label='HI ; ff')
    
    
    for j in range(3): #bucle que recorre los diferentes estados excitados de HI
        ax[i].plot(lamb_all, HIkbf[j][i][0], ls='dotted', label='HI ; bf n=%.i' % (j+1))

    ax[i].plot(lamb_all, H_kff[i][0], ls='dashed', label='H- ; ff')
    ax[i].plot(lamb_all, H_kbf[i][0], ls='dashed', label='H- ; bf')

    ax[i].axhline(esk[i][0], ls='-.', label='e- scattering')

    k_tot = HIkff[i][0] + HIkbf[0][i][0] + HIkbf[1][i][0] + HIkbf[2][i][0] + H_kff[i][0] + H_kbf[i][0] 
    k_tot += + esk[i][0]
    ax[i].plot(lamb_all, k_tot, 'k', label='$\\kappa_{tot}$', zorder = 0)
    
    ax[i].set_xlabel('$\lambda$ [$\AA$]')
    ax[i].set_ylabel('$\\kappa\ [\\rm cm^{-1}]$')

    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    
    ax[i].set_title(T[i])
    ax[i].grid(alpha=1)
    ax[i].grid(which='minor', alpha = .3)
    ax[i].set_ylim(1E-25)
ax[0].legend(loc = 'lower left')
plt.tight_layout()

#%% PLOT SECCIONES EFICACES

fig, ax = plt.subplots(2,2, figsize=(12,8))

for i in range(2):
    T = ['T = 6000K', 'T = 8000K']
    sigma = ['$\sigma_{ff}$', '$\sigma_{bf}$']
    
    ax[0][0].plot(lamb_all, HIcsff[i][0], ls='dotted', label='HI ; ff ; %s' % T[i])
    ax[1][0].plot(lamb_all, H_csff[i][0], ls='dashed', label='H- ; ff ; %s' % T[i])

    if i==0:
        for j in range(3):
            ax[0][1].plot(lamb_all, HIcsbf[j][0][0], ls='dotted', label='HI ; bf n=%.i ; %s, %s' % ((j+1), T[0], T[1]))

    if i == 1:    
        ax[0][1].plot(lamb_all, H_csbf, ls='dashed', label='H- ; bf')

for i in range(2):
    for j in range(2):
        ax[i][j].set_xlabel('$\lambda$ [$\AA$]')
        ax[i][j].set_ylabel('$\sigma$ [$cm^2$]')
        if (j,i)!=(1,1):
            ax[i][j].set_title(sigma[j])
            ax[i][j].legend(loc='upper left')
            ax[i][j].grid(alpha=1)
            ax[i][j].grid(which='minor', alpha = .3)
            ax[i][j].set_xscale('log')
            ax[i][j].set_yscale('log')
ax[0][1].set_xlim(0,16500)
ax[0][1].set_ylim(1E-18)
ax[0][1].legend(loc='lower right')
plt.tight_layout()

# Las siguientes líneas obtienen una configuración de una columna con dos filas y la otra columna entera
ax[0][1].set_position([0.57, 0.105, 0.4, 0.845])
ax[1, 1].axis('off')


#%% Función simple para exportar otras tablas de latex
def table(array, exp = True, name=''):
    string = ''
    if name !='':
        string+=str(name+' & ')
    for i in range(2):
        for j in range(2):
            if exp == False:
                string+='%.2f '%array[i][j]
            if exp == True:
                string+='%.2e '%array[i][j]
            if i==1 and j==1:
                string+='\\ '
            else: string+='& '
    print(string)

for N, name in zip([NH_, NHI, NHII, Ne, n1, n2, n3],
                   ['$N_{\rm H^-}$', '$N_{\rm HI}$', '$N_{\rm HII}$', '$N_e', '$N_{\rm HI},\ n=1$', '$N_{\rm HI},\ n=2$', '$N_{\rm HI},\ n=2$']):
    table(N, name = name)

for N, name in zip([(NH-(NH_+NHI+ NHII))/NH],
                   ['$N_{\rm H^-} + N_{\rm HI} + N_{\rm HII}$', '$N_{\rm H}$']):
    table(N, name = name)

for N, name in zip([(Ne-(NHII-NH_))/Ne],
                   ['$N_e$', '$N_{\rm HII} + N_{\rm H^-}$']):
    table(N, name = name)
    
for N, name in zip([(n1+n2+n3-NHI)/NHI],
                   ['$N_e$', '$N_{\rm HII} + N_{\rm H^-}$']):
    table(N, name = name)