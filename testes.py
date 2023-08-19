# =============================
# script para testes

# Made by: Lucas da Silva Menezes
# 31/07/2023
# =============================

# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt

# import locais
from scripts.calc import *
import scripts.cooling_rate as cooling_rate


def read_profile(file_path):
	# lendo o arquivo
	df = pd.read_csv(
		file_path,
		sep = ',',
		converters = dict(
			pres = lambda x: pd.to_numeric(x, errors = 'coerce'),
			air_density = lambda x: pd.to_numeric(x, errors = 'coerce'),
			water_density = lambda x: pd.to_numeric(x, errors = 'coerce'),
			ozone_density = lambda x: pd.to_numeric(x, errors = 'coerce')
		)
	)

	# COLUNA | UNIDADE | DESCRICAO
	# hght | km | altitude
	# pres | mb | Pressão atmosférica
	# temp | K | Temperatura do ar
	# air_density | g / m³ | Densidade da parcela de ar úmido
	# water_density | g / m³ | Densidade do vapor d'água na parcela
	# ozone_density | g / m³ | Densidade do ozonio na parcela

	# Calcula a razao de mistura | Adimensional
	df['mixr'] = df['water_density'] / (df["air_density"] - df['water_density'])

	# Converte a altitude de [km] para [m]
	df['hght'] *= 1e3

	# Calcula o path length do vapor d'agua [g / cm²]
	df['u'] = path_length(df['water_density'].values, df['hght'].values)
	df['u'] = df['u'] * 1e-4 # converte de g / m² para g / cm²

	# Calcula a umidade relativa
	# -------------------------------

	# Densidade do vapor d'agua na parcela SATURADA
	df['relh'] = relative_humidty_from_density(
		Qv = df['water_density'] * 1e-3, # converte de [g/m³] para [Kg/m³]
		T =  df['temp']
	)
	
	return df


def figura(df):
	fig, ax  = plt.subplots(figsize = (5, 9))

	ax.plot(df['cr_nc_all'], df['hght'] * 1e-3, linestyle = 'solid', color = 'k', label = 'total')
	ax.plot(df['cr_nc_rot'], df['hght'] * 1e-3, linestyle =  'dashed', label = 'H20 Rotational')
	ax.plot(df['cr_nc_cont'], df['hght'] * 1e-3, linestyle = 'dashdot', label = 'H20 Continuum')
	ax.plot(df['cr_nc_vib'], df['hght'] * 1e-3, linestyle = 'dotted', label = 'H20 Vibrational')

	# Eixo Y
	ax.set_ylim(0, 15)
	ax.set_yticks([0, 5, 10, 15])

	# Eixo X
	left, right = ax.get_xlim()
	ax.set_xticks(np.arange(-20, 20, 1))
	ax.set_xlim(-7, 2)

	# Textos
	ax.set_ylabel("Altitude (Km)")
	ax.set_xlabel("Cooling rate (K / day)")

	# Legenda
	plt.legend()

	plt.show()


if __name__ == '__main__':
	df = read_profile(r".\Dados\tropical.csv")

	df['cr_nc_cont'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values,
		p = df['pres'].values,
		Qv = df['water_density'].values * 1e-3, # [kg / m^3]
		band = 'continuum' # Banda continuum (10µm)
	) # [K / day]

	df['cr_nc_rot'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values,
		p = df['pres'].values,
		Qv = df['water_density'].values * 1e-3, # [kg / m^3]
		band = 'rotational' # Banda rotacional (0 a 1000 cm^-1)
	) # [K / day]


	df['cr_nc_vib'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values,
		p = df['pres'].values,
		Qv = df['water_density'].values * 1e-3, # [kg / m^3]
		band = 'vibrational' # Banda vibrational-rotational (6.3µm)
	) # [K / day]

	df['cr_nc_all'] = cooling_rate.no_clouds(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values,
		p = df['pres'].values,
		Qv = df['water_density'].values * 1e-3, # [kg / m^3]
		band = 'all' # Todas as bandas
	) # [K / day]

	# print(df)
	figura(df)