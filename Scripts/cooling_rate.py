# ===================================================================
# 				Script com funções  mais complexas
# 			relacionadas ao do cooling rate considernado
#				uma atmosfera com nuvens e sem nuvens.


# Referencias utilizadas
#	[1] Parametrization of Infrared Radiative Transfer in Cloudy Atmospheres
# 		(Liou, 1981)
		

# Made by: Lucas da Silva Menezes
# 02/08/2023
# ===================================================================


# IMPORTS
# -------------------------------------
import numpy as np

# import local
import scripts.broadband as broadband
import scripts.calc as calc


# COOLING RATE SEM NUVENS
# -------------------------------------

def no_clouds(T, u, q, p, ur, band = 'all'):
	'''
	Calcula o cooling rate, considerando apenas o vapor d'água e
	uma atmosfera sem nuvens, a partir da equação (5) de [1].
	
	Para a abordagem do plano-paralelo IR, u = 0 na superficie, de modo que no topo
	u = u1.

	Entrada: Array[float], Array[float], Array[float], Array[float]

		T: Temperatura [K]
		u: Path Length [g / cm²]
		q: Razão de mistura do ar úmido [adimensional - Kg/Kg]
		p: Pressão atmosférica [hPa]
		ur: Umidade relativa [adimensional]

	Saída: Array[float] [K / day]
	'''

	# Preparações
	# ---------------------------
	Cp = 1005 # J * Kg^-1 K^-1

	u = u[0] - u # Inverte-se para uso com a abordagem plano-paralelo no IR

	# Calculo do u corrigido para cada banda
	u_rot = broadband.reduced_path_length_rot(T, p, u) # [g / cm²]
	u_10 = broadband.path_length_10μm(T, u, ur) # [g / cm²]
	u_6 = broadband.path_length_6μm(p, u) # [g / cm²]
	# print(u_rot)
	# print(u_10)
	# print(u_6)

	# Termo 1: Sigma T^4 * dEf(u1 - u, T1)/du | u1  = u[0]
    # -------------------------------------------
	# emissividade em cada nivel
	Ef = [broadband.emissivity(
		T[-1], # T no topo
		p[-1], # p no topo
		ur[-1],
		u_rot[-1] - u_rot[i], # Indice -1 equivale ao path length de toda a coluna atmosferica
		u_10[-1] - u_10[i],
		u_6[-1] - u_6[i],
		band = band
		) for i in range(u_rot.shape[0])]
	
	# Derivada simples (u está em ordem crescente)
	du = np.diff(u) #  [g / cm²]
	dEf = np.diff(np.array(Ef))
	dEf1du = dEf / du # [( g / cm²)^-1]

	# Resultado do termo 1
	termo1 = -1 * calc.stefan_boltzmann(T[-1]) * dEf1du
	termo1 = np.append(termo1, np.nan) # [J * m^-2 * s^-1 * ( g / cm²)^-1]
	termo1 = termo1 / 10 # corrigindo para [J * s^-1 * Kg^-1]

	# Termo 2: Integral
    # -------------------------------------------

	# inicializa o termo
	termo2 = np.full(u.shape, fill_value=np.nan, dtype = np.float64)

	# termos medios
	u_mean = (u[:-1] + u[1:]) / 2
	T_mean = np.interp(u_mean, u, T)
	p_mean = np.interp(u_mean, u, p)
	u_rot_mean = np.interp(u_mean, u, u_rot)
	u_10_mean = np.interp(u_mean, u, u_10)
	u_6_mean = np.interp(u_mean, u, u_6)

	# d(Sigma * T(u')^4)/du'
	du_ = np.diff(u)
	dsigma_t_4_du = np.diff(calc.stefan_boltzmann(T)) / du_

	# loop para cada nível da sondagem
	for i in range(termo2.shape[0] - 1): # ultimo nao incluso
		
		# Função a ser integrada
		dEfdu = np.full(du_.shape[0], fill_value=np.nan)

		# loop para  a integral, da superfície até o topo da sondagem
		for j in range(du_.shape[0]):
			# emissividade em cada nivel
			Ef2 = broadband.emissivity(
				T_mean[j],
				p_mean[j],
				ur[j],
				np.abs(u_rot[i + 1] - u_rot_mean[j]),
				np.abs(u_10[i + 1] - u_10_mean[j]),
				np.abs(u_6[i + 1] - u_6_mean[j]),
				band = band
				)
			
			Ef1 = broadband.emissivity(
				T_mean[j],
				p_mean[j],
				ur[j],
				np.abs(u_rot[i] - u_rot_mean[j]),
				np.abs(u_10[i] - u_10_mean[j]),
				np.abs(u_6[i] - u_6_mean[j]),
				band = band
			)

			# Derivada simples (u está em ordem crescente)
			dEf = Ef2 - Ef1
			dEfdu[j] = dEf / du_[i]

		# Função a ser integrada
		f = dEfdu * dsigma_t_4_du
		termo2[i] = np.nansum(f * du_) / 10 # Conserta a unidade

	# Resultado final
	cooling_rate = -1 * (termo1 + termo2) * q / Cp # [K/s]

	# Inverte o array e converte a unidade para [K / day]
	cooling_rate = cooling_rate * 3600 * 24

	return cooling_rate