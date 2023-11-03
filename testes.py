# =============================
# script para testes

# Made by: Lucas da Silva Menezes
# 31/07/2023
# =============================

# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp

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
	
	return df


def figura(df):
	fig, ax  = plt.subplots(figsize = (5, 6))

	ax.plot(df['cr_nc_all'], df['hght'] * 1e-3, linestyle = 'solid', color = 'k', label = 'total')
	ax.plot(df['cr_nc_rot'], df['hght'] * 1e-3, linestyle =  'dashed', label = 'H20 Rotational')
	ax.plot(df['cr_nc_cont'], df['hght'] * 1e-3, linestyle = 'dashdot', label = 'H20 Continuum')
	ax.plot(df['cr_nc_vib'], df['hght'] * 1e-3, linestyle = 'dotted', label = 'H20 Vibrational')

	# Eixo Y
	ax.set_ylim(0, 15)
	ax.set_yticks([0, 5, 10, 15])

	# Eixo X
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

	modelo = cooling_rate.CoolingRate(
		T = df['temp'].values,
		u = df['u'].values,
		q = df['mixr'].values,
		p = df['pres'].values,
		Qv = df['water_density'].values * 1e-3 # [kg / m^3]
	) # [K / day]

	df['cr_nc_rot'] = modelo._clear_sky(band = 'rot')
	df['cr_nc_cont'] = modelo._clear_sky(band = 'cont')
	df['cr_nc_vib'] = modelo._clear_sky(band = 'vib')
	df['cr_nc_all'] = modelo._clear_sky(band = 'all')

	# # print(df)
	figura(df)