# ===================================================================
# 				Script com funções  mais complexas
# 			relacionadas ao cálculo da emissivdade broadband
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
from calc import *

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
Rw1 = np.array([7210.3, 6024.8, 1614.1, 139.03, 21.64, 2.919, 0.3856, 0.0715])

# pi * alpha_0 / delta (adimensional)
Rw2 = np.array([0.182, 0.094, 0.081, 0.080, 0.068, 0.060, 0.059, 0.067])

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
dv = 860 # cm^-1
v_mean = (intervalos[:, 1] + intervalos[:, 0]) / 2
#
A_ = np.sum(dvi * a_) / dv
B_ = np.sum(dvi * b_) / dv


# BANDA ROTACIONAL DO VAPOR D'AGUA (de 0 a 1000 cm^-1)
# ----------------------------------------------------------

def reduced_path_length_rot(T, p, u):
	'''
	Calculado o path length do vapor d'água de acordo com a equação (22) de [1].
	considerando a banda rotacional.

	Existem duas formas de calcular qual seria a pressão do ar em T = T0 (260 K)
	1. Processo Isobárica, onde o termo p / p0 = T / T0
	2. Processo Adiabática, onde o termo p / p0 = (T / T0) ** (Cp / R)

	Esta função considera um processo adiabático.

	Entrada: Array

		T: Temperatura [K]
		p: Pressão atmosférica [hPa]
		u: Path Length [g/cm² ou Kg/m²]
		
	Saída: Array - mesma unidade de u
	'''

	# Temperatura inicial considerada na equação (22) de [1]
	T0 = 260 # [K]

	# Constantes do ar seco
	Cp = 1005 # [J * Kg^-1 * K^-1]
	R = 287 # [J * Kg^-1 * K^-1]

	# Função a ser integrada
	y = (T / T0) ** (Cp / R) * np.exp(A_ * (T - T0) + B_ * (T - T0) ** 2)
	# y = p / p[0] * np.exp(A_ * (T - T0) + B_ * (T - T0) ** 2)

	# SE O ARRAY U NÃO ESTIVER EM ORDEM ASCENDENTE
	# É PRECISO ORGANIZA-LO ANTES DE INTEGRAR
	ind = np.argsort(u)
	y_sorted = y[ind]
	u_sorted = u[ind]

	# Inicializa um array vazio para armazenar os resultados
	resultados = np.zeros(u.shape, dtype = np.float64)

	# Loop para integração de ydu, de 0 até cada nível "u"
	for i in range(u.shape[0]):

		resultados[i] = integral(y_sorted[:i + 1], u_sorted[:i + 1])

	# Organiza os resultados na ordem do array original "u"
	ind1 = np.argsort(ind)

	return resultados[ind1]


def transmitance_rotational_H2O(T, u):
	'''
	Calcula a transmitância devido a banda rotacional do H2O.
	Retorna uma lista, com a transmitância em cada intervalo segundo Rodgers & Walshaw (1966)

	Entrada: Float

		u = Path length Corrigido para banda rotacional (22) de [1] [g / cm²]
		T = Temperatura [K]

	Saída: tuple(intervalos[Array2D], transmitance)
	'''

	# Calculos
	# ------------------------------------------------------------
	T0 = 260 # K
	S_S0 = np.exp(a * (T - T0) + b * (T - T0) ** 2) # S / S0

	# Transmitancia Difusa
	termo_raiz = 1 + 1.66 * S_S0 * Rw2 / Rw1 * u
	transmitance = 1.66 * S_S0 * Rw1 * u / np.power(termo_raiz, 0.5)
	transmitance = np.exp(-transmitance)
	
	return (intervalos, transmitance)


# BANDA 10µm DO VAPOR D'AGUA (de 8 a 12 µm)
# ----------------------------------------------------------
def path_length_10µm(T, u):
	'''
	Calcula o Path Length corrigido para esta banda de absorção [8 - 12 µm]

	Existem duas formas de calcular qual seria a pressão do gás absorvedor em T = T0 (2)
	1. Processo Isobárica, onde o termo p / p0 = T / T0
	2. Processo Adiabática, onde o termo p / po = (T / T0) ** (Cp / R)

	Esta função considera um processo adiabático.

	Entrada: Array

		T = Temperatura [K]
		u = Path Length do H20  [g / cm²]

	Saída: Array [g / cm²]
	'''
		
	# Definindo variaveis
	# --------------------------------------------
	T0 = 296 # K
	R = 461.5 # R do vapor d´agua [J * Kg^-1 * K^-1]
	Cv = 1410 # Cv do vapor d'água  [J * Kg^-1 * K^-1]
	Cp = Cv + R # Cp do vapor d'agua [J * Kg^-1 * K^-1]
	
	# Integrando da superfície até o topo
	# ---------------------------------------------

	# funcao a ser integrada
	y = ( (T / T0) ** (Cp / R) ) * np.exp(- 1800 / (T0 * T) * (T - T0))

	# SE O ARRAY U NÃO ESTIVER EM ORDEM ASCENDENTE
	# É PRECISO ORGANIZA-LO ANTES DE INTEGRAR
	ind = np.argsort(u)
	y_sorted = y[ind]
	u_sorted = u[ind]

	# Inicializa um array vazio para armazenar os resultados
	resultados = np.zeros(u.shape, dtype = np.float64)

	for i in range(resultados.shape[0]):

		# Nivel i
		resultados[i] = integral(y_sorted[:i + 1], u_sorted[:i + 1])

	return resultados


def transmitance_10µm(T, p, u):
	'''
	Calcula a transmitancia do H20 para banda de 10 µm.
	Referencias: [3] e [4]
 
	Entrada: Float

		T = Temperatura [K] 
		p = Pressão Atmosférica [hPa]
		u = Path Length do H20 - Corrigido para P e T nesta banda [g / cm²]

	Saída: tuple(intervalos[Array2D], transmitance)
	'''

	# Intervalos de numero de onda, segundo [4]
	intervalos = np.array([
		[900, 1000],
		[1000, 1100], 
		[1100, 1200]]) # [cm^-1]

	v = np.mean(intervalos, axis = 1)

	# Constantes
	a = 4.2 # [cm² (g atm^-1)]
	b = 5588 # [cm² (g atm^-1)]
	beta = 7.87 * 1e-3 # [cm]

	# K(v , T = 296 K)
	k = a + b * np.exp(- beta * v) # [cm² (g atm)^-1]

	# correcao de temperatura
	k = k * np.exp(1800 * (1 / T - 1 / 296))

	# Conversão de hPa para atm
	p0 = 1013.25 # 1 atm [Pa]
	new_p = p / p0
	k = k * new_p

	# retorna ja com a conversao em metros
	return intervalos, np.exp(- k * u ) 


# BANDA 6.3µm DO VAPOR D'AGUA (de 1200 a 2200 cm^-1)
# ----------------------------------------------------------
def path_length_6µm(p, u):
	'''
	Calcula o Path Length corrigido para esta banda de absorção 6.3µm.
	Referência: [1]

	Entrada: Array

		p = Pressão Atmosférica [hPa]
		u = Path Length do H20  [g / cm²]

	Saída: Array [g / cm²]
	'''

	# Constante
	p0 =  1013 # [hPA]
	y = p / p0 # adimensional

	# Resultados
	new_u = np.zeros(u.shape)

	# loop para integral da sup até cada nível
	for i in range(u.shape[0]):
		
		# Nível i
		new_u[i] = integral(y[:i + 1], u[:i + 1])

	return new_u


def transmitance_6µm(T, u):
	'''
	Calcula a transmissividade do H20 para banda de 6.3 µm.
	Referencias: [1] e [4]
 
	Entrada: Float

		T = Temperatura [K] 
		u = Path Length do H20 - Corrigido para P nesta banda [g / cm²]

	Saída: tuple(intervalos[Array2D], transmitance)
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


# EMISSIVIDADE BROADBAND, CONSIDERANDO TODAS AS BANDAS
# ----------------------------------------------------------

def emissivity(T, p, u_rot, u_10, u_6, band = 'all'):
	'''
	Calcula a emissividade broadband dada uma temperatura e path length,
	através da integração ao longo do espectro IR.

	Entrada: Float

		T : Temperatura [K]
		p : Pressão Atmosférica [hPa]
		u_rot: Path length corrigido para a banda rotacional [g / cm²]
		u_10: Path length corrigido para a banda continuum [g / cm²]
		u_6: Path length corrigido para a banda vibrational [g / cm²]
		band: Banda do espectro a ser considerada ['all', 'rotational', 'continuum', 'vibrational']

	Saída: Float
	'''

	# Caso 1: Todas as bandas (rotational, continuum, vibrational-rotational)
	if band == 'all':
		v1, tau1 = transmitance_rotational_H2O(T, u_rot)
		v2, tau2 = transmitance_10μm(T, p, u_10)
		v3, tau3 = transmitance_6μm(T, u_6)

		intervalos = np.concatenate((v1, v2, v3)) # Intervalos de numero de onda
		transmitance = np.concatenate((tau1, tau2, tau3)) # Transmitancia em cada intervalo
	
	# Caso 2:
	elif band == 'rotational':
		intervalos, transmitance = transmitance_rotational_H2O(T, u_rot)

	# Caso 3: 
	elif band == 'continuum':
		intervalos, transmitance = transmitance_10μm(T, p, u_10)

	# Caso 4:
	elif band == 'vibrational':
		intervalos, transmitance = transmitance_6μm(T, u_6)
	
	else: # Lança um erro
		raise KeyError
	
	# Lei de Stefan-Boltzmann
	steboltz = stefan_boltzmann(T) # J * m^-2 * s^-1

	# transforam em comprimento de onda [cm] e converte para metros
	lambda_ = (1 / intervalos) * 1e-2
	mean_lambda = np.mean(lambda_, axis = 1) # [m]
	dlambda = np.abs(lambda_[:, 1] - lambda_[:, 0]) # [m]

	# Calcula a emissividade de cada intervalo e contabiliza
	emissivity = np.nansum(np.pi * planck(mean_lambda, T) * (1 - transmitance) * dlambda)

	# Resultado final
	# -------------------------------------------------
	emissivity = emissivity / steboltz

	return emissivity