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
import broadband
import calc


# COOLING RATE SEM NUVENS
# -------------------------------------
	
def no_clouds(T, u, q, p, band = 'all'):
	'''
	Calcula o cooling rate, considerando apenas o vapor d'água e
	uma atmosfera sem nuvens, a partir da equação (5) de [1].

	Entrada: Array[float], Array[float], Array[float], Array[float]

		T: Temperatura [K]
		u: Path Length [g / cm²]
		q: Razão de mistura do ar úmido [adimensional - Kg/Kg]
		p: Pressão atmosférica [hPa]

	Saída: Array[float]
	'''

	# Preparações
	# ---------------------------

	# Termo 1: Sigma T^4 * dEf(u1 - u, T1)/du
    # -------------------------------------------
	T1 = T[-1] # No topo

	# Calculo do u corrigido para banda rotacional
	# interpolacao linear de T
	u_rot = broadband.reduced_path_length_rot(T, p, u) # [g / cm²]
	u_10 = broadband.path_length_10μm(T, u) # [g / cm²]
	u_6 = broadband.path_length_6μm(p, u) # [g / cm²]

	# emissividade em cada nivel
	Ef = [broadband.emissivity(
		T1,
		p[-1],
		u_rot[-1] - u_rot[i],
		u_10[-1] - u_10[i],
		u_6[-1] - u_6[i],
		band = band
		) for i in range(u_rot.shape[0])]
	
	# Derivada simples (u está em ordem crescente)
	du = np.diff(u) #  [g / cm²]
	dEf = np.diff(np.array(Ef))
	dEf1du = dEf / du # [( g / cm²)^-1]

	# Resultado do termo 1
	termo1 = -1 * calc.stefan_boltzmann(T1) * dEf1du
	termo1 = np.insert(termo1,-1, np.nan) # [J * m^-2 * s^-1 * ( g / cm²)^-1]
	termo1 = termo1 / 10 # corrigindo para [J * s^-1 * Kg^-1]

	# Termo 2: Integral
    # -------------------------------------------

	# inicializa o termo
	termo2 = np.full(u.shape, fill_value=np.nan, dtype = np.float64)

	# termos medios
	u_mean = (u[:-1] + u[1:]) / 2
	T_mean = np.interp(u_mean, u, T)
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
				p[j],
				np.abs(u_rot[i + 1] - u_rot_mean[j]),
				np.abs(u_10[i + 1] - u_10_mean[j]),
				np.abs(u_6[i + 1] - u_6_mean[j]),
				band = band
				)
			
			Ef1 = broadband.emissivity(
				T_mean[j],
				p[j],
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
	Cp = 1005 # J * Kg^-1 K^-1
	cooling_rate = -1 * (termo1 + termo2) * q / Cp # [K/s]

	# Retorna o resultado em K / dia
	return cooling_rate * 3600 * 24

	
