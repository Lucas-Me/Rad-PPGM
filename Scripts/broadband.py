# ===================================================================
# 				Script com funções  mais complexas
# 			relacionadas ao cálculo da emissividade broadband
# 		considerando apenas o vapor d'água e suas bandas de absorção


# Referencias utilizadas
#	[1] Parametrization of Infrared Radiative Transfer in Cloudy Atmospheres
# 		(Liou, 1981)
#	[2] The computation of infra-red cooling rate in planetary atmospheres
#		(Rodgers and Walshaw, 1966)
#	[3] Infrared continuum absorption by atmospheric water vapor 
#      in the 8-12 µm window (Roberts et al., 1976)
#	[4] Influence of Cirrus Clouds on the Infrared Cooling Rate in the
# 		Troposphere and Lower Stratosphere (Rowe & Liou, 1978)
		

# Made by: Lucas da Silva Menezes
# 31/07/2023
# ===================================================================

# IMPORTS
import numpy as np

# IMPORTS LOCAIS
from Scripts.calc import *

# CONSTANTES
# ----------------------------------------------------------
# constantes
a = np.array(
	[-2.93, 1.43, 9.59, 14.3, 15.2, 19.0, 21.7, 24.1],
	) * 1e-3 # (deg^-1)

b = np.array(
	[2.01, -13, -41.8, -23.7, -30.1, -44.6, -53.2, -40.3],
	) * 1e-6 # (deg^-2)

# k / delta (g^-1 cm²)
C1 = np.array([7210.3, 6024.8, 1614.1, 139.03, 21.64, 2.919, 0.3856, 0.0715])

# pi * alpha_0 / delta (adimensional)
C2 = np.array([0.182, 0.094, 0.081, 0.080, 0.068, 0.060, 0.059, 0.067])

# a' e b' para cada interalo de numero de onda (TABLE 3. de [2]]
a_ = np.array(
	[-2.68, 2.03, 9.08, 15.1, 16.2, 18.6, 23.1, 26.2],
	) * 1e-3 # [▲K^-1]
b_ = np.array(
	[1.57, -10.3, -38.1, -54.1, -38.1, -62.6, -74.7, -74.1],
	) * 1e-6 # [▲K^-2]

# Respectivos intervalos de numero de onda
intervalos = np.array([
	[40, 160],
	[160, 280],
	[280, 380],
	[380, 500],
	[500, 600],
	[600, 720],
	[720, 800],
	[800, 900]]) # [cm^-1]

# Constantes A' e B' segundo a equação 23 de [1].
dvi = intervalos[:, 1] - intervalos[:, 0]
dv = intervalos[-1, -1] - intervalos[0, 0] # cm^-1
v_mean = np.mean(intervalos, axis = 1)
#
A_ = np.sum(dvi * a_) / dv
B_ = np.sum(dvi * b_) / dv


# BANDA ROTACIONAL DO VAPOR D'AGUA (de 0 a 1000 cm^-1)
# ----------------------------------------------------------

def path_length_rot(T, p, u):
	'''
	Calculado o path length do vapor d'água de acordo com a equação (22) de [1].
	considerando a banda rotacional.
	
	Parameters
	----------
	T : Array[float]
		Temperatura [K]
	p : Array[float]
		Pressão atmosférica [hPa]
	u: Array[float]
		Path Length [g / cm²]
	
	Returns
	-------
	u_ajustado : Array[float]
		Path length corrigido para esta banda de absorção [g / cm²]
	'''

	# Temperatura inicial considerada na equação (22) de [1]
	T0 = 260 # [K]
	p0 = 1013 # [hPa]

	# Função a ser integrada
	Tmean = (T[1:] + T[:-1]) / 2
	pmean = (p[1:] + p[:-1]) / 2
	# y = p[1:] / p[:-1] * np.exp(A_ * (Tmean - T0) + B_ * (Tmean - T0) ** 2)
	y = pmean / p0 * np.exp(A_ * (Tmean - T0) + B_ * (Tmean - T0) ** 2)
	integrate = np.diff(u) * y

	# Inicializa um array vazio para armazenar os resultados
	u_ajustado = np.zeros(u.shape, dtype = np.float64)
	for i in range(integrate.shape[0]):
		u_ajustado[i + 1] = np.sum(integrate[:i + 1])

	return u_ajustado


def transmitance_rot(T, u):
	'''
	Calcula a transmitância devido a banda rotacional do H2O.
	Retorna uma lista, com a transmitância em cada intervalo segundo Rodgers & Walshaw (1966)

	Parameters
	----------
	T : array[float]
		Temperatura [K]
	u: array[float]
		Path length corrigido para esta banda de absorção [g / cm²]
	
	Returns
	-------
	intervalos : Array[float], shape[N, 2]
		Intervalos de número de onda [cm^-1] considerada no calculo da transmitancia
	transmitance: Array[float], shape[N]
		Transmitancia em cada intervalo de numero de onda [adimensional]
	'''

	# Calculos
	# ------------------------------------------------------------
	T0 = 260 # K
	
	n = a.shape[0]
	du = np.abs(np.diff(u))
	Tmean = (T[1:] + T[:-1]) / 2

	transmitance = np.ones(n)
	# Se du = 0, transmitancia é 1.
	if du.shape[0] == 0 or np.sum(du) == 0:
		return (intervalos, np.ones(n))
	
	for i in range(du):
	# Curtis-Godson approximation
	exp1 = np.exp(a[] * (Tmean - T0) + b.reshape((n, 1))  * (Tmean - T0) ** 2)
	exp2 = np.exp(a_[] * (Tmean - T0) + b_.reshape((n, 1))  * (Tmean - T0) ** 2)


	# Termo da transmitanica difusa
	termo_raiz = 1 + 1.66 * C1 / C2 * np.matmul(exp1 ** 2 / exp2, du)
	expoente = 1.66 * C1 * np.matmul(exp1, du) * np.power(termo_raiz, -0.5)
	
	# Calculo da transmitancia difusa
	transmitance = np.exp(-expoente)
	
	return (intervalos, transmitance)


# BANDA 10µm DO VAPOR D'AGUA (de 8 a 12 µm)
# ----------------------------------------------------------
def path_length_cont(T, u, e):
	'''
	Calcula o Path Length corrigido para esta banda de absorção [8 - 12 µm].
	Banda continuum.

	Parameters
	----------
	T : Array[float]
		A temperatura do ar [K]
	u : Array[float]
		Path Length [g / cm²]
	e : Array[float]
		Pressão parcial do vapor d'agua [Pa]

	Returns
	-------
	u_ajustado : Array[float]
		Path length corrigido para esta banda de absorção [g / cm²]
	'''
		
	# Definindo variaveis
	# --------------------------------------------
	# Cosntantes
	e0 = 1906.51 # 14.3 torr para Pa, tomado para T = 294K
	T0 = 294 # [K]

	# funcao a ser integrada
	y = (e / e0) * np.exp(- 1800 * (T - T0) / (T * T0) )

	# Integrando da superfície até o topo, metodo trapezoidal
	# ---------------------------------------------
	integrate = np.diff(u) * (y[1:] + y[:-1]) / 2

	# Inicializa um array vazio para armazenar os resultados
	u_ajustado = np.zeros(u.shape, dtype = np.float64)
	for i in range(integrate.shape[0]):
		u_ajustado[i + 1] = np.sum(integrate[:i + 1])

	return u_ajustado


def transmitance_cont(T, u, p, e):
	'''
	Calcula a transmitancia do H20 para banda de 10 µm.
	Referencias: [3] e [4]
 
	Parameters
	----------
	T : Array[float]
		A temperatura do ar [K]
	u : Array[float]
		Path Length [g / cm²]
	p : Array[float]
		Pressão atmosférica [hPa]
	e : Array[float]
		Pressão parcial do vapor d'agua [Pa]

	Returns
	-------
	intervalos : Array[float], shape[N, 2]
		Intervalos de número de onda [cm^-1] considerada no calculo da transmitancia
	transmitance: Array[float], shape[N]
		Transmitancia em cada intervalo de numero de onda [adimensional]
	'''

	# Intervalos de numero de onda, segundo [4]
	intervalos = np.array([
		[800, 900],
		[900, 1000], 
		[1000, 1200]]) # [cm^-1]

	# Numero de onda medio entre cada intervalo
	v = np.mean(intervalos, axis = 1)

	# Se apenas um valor, consdira-se u = 0, logo Transmitancia = 1.
	if T.shape[0] <= 1:
		return (intervalos, np.ones(v.shape[0]))

	# propriedades para o calculo do coeficiente de extincao (k)
	# ----------------------------------------
	# massa molar da agua [g / mol]
	mv = 18

	# Numero de moleculas por mol [molecules / mol]
	avogadro = 6.022 * 1e23

	# Self-broadening absorption coefficient
	# ---------------------------------------
	# Constantes
	beta = 8.3 * 1e-3 # [cm]
	a = 1.25 * 1e-22  # mol^-1 cm^2 atm^-1 
	b = 2.34 * 1e-19 # mol^-1 cm^2 atm^-1

	# converte a unidade de a e b para [g^-1 cm^2 atm^-1]
	a = a / (mv / avogadro)
	b = b / (mv / avogadro)

	# Calcula k
	k = a + b * np.exp(- beta * v) # [g^-1 cm^2 atm^-1]

	# Correcao de temperatura e pressao
	gama = 0.005
	correcao_T = np.exp(1800 * (1 / T - 1 / 296)) 
	correcao_P = (e + gama * (p * 1e2 - e)) * 9.86923 * 1e-6 # [atm]
	
	# k corrigido (sigma)
	sigma = np.matmul(
		k.reshape((3, 1)),
		(correcao_T * correcao_P).reshape((1, e.shape[0]))
	)  # em [g^-1 cm^2]

	# Calculo da transmitancia
	# ----------------------------------
	expoente = np.matmul(
		 (sigma[:, 1:] + sigma[:, :-1]) / 2,
		 np.abs(np.diff(u))
	)
	transmitance = np.exp(-expoente)

	# retorna os intervalos e as suas respectivas transmitancias
	return intervalos, transmitance


# BANDA 6.3µm DO VAPOR D'AGUA (de 1200 a 2200 cm^-1)
# ----------------------------------------------------------
def path_length_vib(u, p):
	'''
	Calcula o Path Length corrigido para esta banda de absorção 6.3µm.
	Referência: [1]

	Parameters
	----------
	u : Array[float]
			Path Length [g / cm²]
	p : Array[float]
		Pressão atmosférica [hPa]

	Returns
	-------
	u_ajustado : Array[float]
		Path length corrigido para esta banda de absorção [g / cm²]
	'''

	# Preparativos
	du = np.diff(u)
	y = p / 1013 # adimensional

	# Integral metodo trapezoidal
	integrate = du * (y[1:] + y[:-1]) / 2

	# Resultados
	u_ajustado = np.zeros(u.shape, dtype = np.float64)
	for i in range(integrate.shape[0]):
		u_ajustado[i + 1] = np.sum(integrate[:i + 1])

	return u_ajustado


def transmitance_vib(u):
	'''
	Calcula a transmissividade do H20 para banda de 6.3 µm.
	Referencias: [1] e [4]
 
	Parameters
	----------
	u : float
		Path Length [g / cm²]

	Returns
	-------
	intervalos : Array[float], shape[N, 2]
		Intervalos de número de onda [cm^-1] considerada no calculo da transmitancia
	transmitance: Array[float], shape[N]
		Transmitancia em cada intervalo de numero de onda [adimensional]
	'''

	# Intervalos de numero de onda
	intervalos = np.array([
		[1200, 1350],
		[1350, 1450],
		[1450, 1550],
		[1550, 1650],
		[1650, 1750],
		[1750, 1850],
		[1850, 1950],
		[1950, 2050],
		[2050, 2200],
	]) # [cm^-1]

	# Garante que u é positivo, afinal so depende do caminho
	u = np.abs(u)

	# Constantes para cada intervalo
	# ------------------------------

	# S0 / Delta  [cm² g^-1]
	Rw1 = np.array([12.65, 134.4, 632.9, 331.2, 434.1, 136.0, 35.7, 9.015, 1.529])

	# Pi * Alpha0 / delta [adimensional]
	Rw2 = np.array([0.089, 0.230, 0.320 ,0.296, 0.452, 0.359, 0.165, 0.104, 0.116])

	# Calculando a transmitancia
	termo_raiz = np.sqrt(1 + 1.66 * u * Rw1 / Rw2)
	transmitance = np.exp( - 1.66 * Rw1 * u / termo_raiz)

	# Resultado final
	return (intervalos, transmitance)
