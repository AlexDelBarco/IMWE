### IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp 
import scipy.stats as s
from scipy.special import gamma

from IPython import get_ipython
#get_ipython().magic('reset -f') # clear all variables
#plt.close("all") # close all figures



### READ DATA
headers = ['timestamp', 'wind_speed', 'wind_direction_67.5m', 'wind_direction_70m']
sprog = pd.read_csv("sprog.tsv", sep='\t', header=None, names=headers)

### DATA CLEANING
sprog = sprog.apply(pd.to_numeric, errors='coerce')

sprog.replace(999, np.nan, inplace=True)
sprog['wind_direction'] = sprog['wind_direction_67.5m'].combine_first(sprog['wind_direction_70m'])
sprog[sprog['wind_direction_67.5m'].isna() & sprog['wind_direction_70m'].notna()]
sprog[sprog['wind_direction_67.5m'].isna()]

sprog.replace(99.99, np.nan, inplace=True)

#Remove NaN rows
sprog = sprog.dropna(subset=['wind_speed'])
sprog = sprog.dropna(subset=['wind_direction'])

sprog['timestamp'] = pd.to_datetime(sprog['timestamp'], format='%Y%m%d%H%M')


################ PART 2 : EXTREME WIND SPEED ###################
mean_WS = sprog['wind_speed'].mean()
std_WS = sprog['wind_speed'].std()
var_WS = sprog['wind_speed'].var()

print(f"Mean Wind Speed: {mean_WS:.2f} m/s")
print(f"Standard Deviation: {std_WS:.2f} m/s")
print(f"Variance: {var_WS:.2f} m/s²\n")


# Define constants
T = 365*24       # hours in a year
z0_w = 0.02e-2   #in m, roughness over water 
z0_l = 3e-2
z_hub = 112
kappa = 0.4

gam = 0.5772 # Euler-Mascheroni constant CAREFUL : has the same name as gamma function in python
T = 50
T_0 = 1 # Reference time

print('PWM')
############# PWM METHOD ###############
windspeed = sprog['wind_speed']
N = len(windspeed)
sorted_speeds = np.sort(windspeed)

# Calculate the PWM moments
b0 = mean_WS          # b0 is the mean of the sorted data
b1 = np.sum((np.arange(1, N+1) - 1) * sorted_speeds) / (N * (N-1))

alpha = 2*b1- b0*np.log(2)
beta = b0 - gam * alpha
U50 = alpha * np.log(T/T_0) + beta  #Maybe try T/t_0 = 50

print(f"PWM Method:")
print(f"α with PWM: {alpha}")
print(f"β with PWM: {beta}")
print(f"Estimated U50 : {U50:.2f} m/s")
print("\n")

print('Gumbells')
########### GUMBELL'S FITTING METHOD #############
annual_maxima = sprog.resample('Y', on='timestamp')['wind_speed'].max() # Take the maxima of each year 
sorted_winds = np.sort(annual_maxima)
N_g = len(sorted_winds)

ranks = np.arange(1, N_g + 1)
Fi = ranks / (N_g + 1)
y = -np.log(-np.log(Fi))


# Linear regression to find alpha and beta 
a, b = np.polyfit(sorted_winds, y, 1)
alpha_g = 1/a
beta_g = -alpha_g*b

U50_g = alpha_g * np.log(T/T_0) + beta_g

print(f"Gumbel Fitting Method:")
print(f"α: {alpha_g}")
print(f"β: {beta_g}")
print(f"Estimated U50 : {U50_g:.2f} m/s")
print("\n")

print('Weibull paraeters')
########### WEIBULL PARAMETER METHOD #############
# I used my old code to re-calculate each weibull parameters 
def weibull_pdf(U, A, k):
    U = np.maximum(U, 1e-6)  # Avoid division by zero
    pdf=(k/U)*(U/A)**k *np.exp(-(U / A)**k)
    return pdf

data_frac = 1 + (var_WS/(mean_WS**2))

def weibull_frac(k):
    gamma_1 = sp.gamma(1 + 1/k)
    gamma_2 = sp.gamma(1 + 2/k)
    return (gamma_2 / gamma_1**2)

def find_best_k(data_frac):
    k_values = np.arange(1, 5, 0.01)  # Extend the search range if needed
    differences = []
    
    for k in k_values:
        theor_frac = weibull_frac(k)
        difference = np.abs(data_frac - theor_frac)
        differences.append(difference)
    
    best_k_index = np.argmin(differences)
    best_k = k_values[best_k_index]
    min_diff = differences[best_k_index]
    
    return best_k, min_diff

k, min_diff = find_best_k(data_frac)
A = mean_WS/sp.gamma(1 + 1/k)

NU = len(windspeed)
cie = 0.438 * NU

beta_w = A * (np.log(cie))**(1/k)
alpha_w = A *(np.log(cie))**(1/k - 1)/k
U50_w = alpha_w * np.log(T/T_0) + beta_w

print(f"Weibull parameter method :")
print(f"α: {alpha_w}")
print(f"β: {beta_w}")
print(f"Estimated U50: {U50_w:.2f} m/s")
print('\n')

print('simplest method')
########### SIMPLEST METHOD #############
U_mean = windspeed.mean()  
U50_naive = 5 * U_mean

print(f"Simple Method:")
print(f"Mean wind speed (U): {U_mean:.2f} m/s")
print(f"Estimated U50: {U50_naive:.2f} m/s")
print("\n")




##################### Calculate at Nyborg and Korsor ##########################
z_measurement = 70
u_mean = np.mean(sprog['wind_speed'])
vonKarman = 0.4
z_0_water = 0.0002
z_0_land = 0.02
z_hub = 120

sprog['year'] = pd.to_datetime(sprog['timestamp']).dt.year

# Get the top 5 wind speeds per year
top_speeds_by_year = sprog.groupby('year')['wind_speed'].apply(lambda x: x.nlargest(5)).reset_index()

direction_sectors = np.arange(0, 390, 30)
A_weibull = np.zeros((12, 1))
k_weibull = np.zeros((12, 1))


def weibull_parameters(direction_sector_min, direction_sector_max):
    k_wei, loc, A_wei = s.weibull_min.fit(
        sprog['wind_speed'][(sprog['wind_direction'] < direction_sector_max) & (sprog['wind_direction'] >= direction_sector_min)],
        floc=0)
    return A_wei, k_wei


for i in np.arange(1, 13, 1):
    A_weibull[i - 1], k_weibull[i - 1] = weibull_parameters(direction_sectors[i - 1], direction_sectors[i])

U = np.linspace(0, 30, 100)

rot_earth = 7.2921e-5
latitude = 55.3
coriolis_parameter = 2 * rot_earth * np.sin(latitude * np.pi / 180)
A = 1.8
B = 4.5

# GDL function
mean_u_sector_sprogo = np.zeros((12, 1))

# for i in range(12):
#    mean_u_sector_sprogo[i] = np.mean(
#        top_speeds_by_year[(sprog['wind_direction'] < direction_sectors[i + 1]) & (sprog['wind_direction'] >= direction_sectors[i])])

for i in range(12):
    # Filter data for the current sector
    filtered_data = sprog[(sprog['wind_direction'] < direction_sectors[i + 1]) &
                          (sprog['wind_direction'] >= direction_sectors[i])]

    if filtered_data.empty:
        mean_u_sector_sprogo[i] = np.nan
    else:
        # Group by year and find the max wind_speed of each year
        yearly_max = filtered_data.groupby('year')['wind_speed'].max()

        # Sort the yearly max values and take the top 5 highest values
        top_5_max = yearly_max.nlargest(5)

        # Compute the mean of the top 5 max values
        mean_u_sector_sprogo[i] = top_5_max.mean()

print('stop')

def gdl(mean_u, z_0):

    u_star = mean_u * vonKarman / (np.log(z_measurement / z_0_water))
    u_g = u_star / vonKarman * (np.log((u_star / coriolis_parameter) / z_0_water) - A)
    v_g = - B * u_star / vonKarman
    G = np.sqrt(u_g ** 2 + v_g ** 2)

    u_star_guess = 0.0001
    error = 1
    while error > 0.001:
        error = G - u_star_guess / vonKarman * np.sqrt(
            (np.log(u_star_guess / (coriolis_parameter * z_0)) - A) ** 2 + B ** 2)

        u_star_guess = u_star_guess + 0.0001

    mean_u_new = u_star_guess / vonKarman * np.log(z_hub / z_0)
    return mean_u_new, u_star_guess


roughness_nyborg = np.zeros((12, 1))
roughness_korsor = np.zeros((12, 1))
roughness_nyborg[direction_sectors[0:12] <= 180] = z_0_water
roughness_nyborg[direction_sectors[0:12] > 180] = z_0_land
roughness_korsor[direction_sectors[0:12] > 180] = z_0_water
roughness_korsor[direction_sectors[0:12] <= 180] = z_0_land

u_nyborg = np.zeros((12, 1))
u_korsor = np.zeros((12, 1))
u_star_nyborg = np.zeros((12, 1))
u_star_korsor = np.zeros((12, 1))

for i in range(12):
    u_nyborg[i], u_star_nyborg[i] = gdl(mean_u_sector_sprogo[i], roughness_nyborg[i])
    u_korsor[i], u_star_korsor[i] = gdl(mean_u_sector_sprogo[i], roughness_korsor[i])

A_weibull_nyborg = np.zeros((12, 1))
A_weibull_korsor = np.zeros((12, 1))

for i in range(12):
    A_weibull_nyborg[i] = u_nyborg[i] / (gamma(1 + 1 / k_weibull[i]))
    A_weibull_korsor[i] = u_korsor[i] / (gamma(1 + 1 / k_weibull[i]))


# Adjust N to 5, because working with top 5 speeds per year
N = 5
# Sort the adjusted wind speeds for Nyborg and Korsor
#sorted_speeds_nyborg = np.sort(u_nyborg.flatten()) OLD
#sorted_speeds_korsor = np.sort(u_korsor.flatten()) OLD
sorted_speeds_nyborg = np.sort(u_nyborg.flatten())[-N:][::-1]  # Select top N speeds
sorted_speeds_korsor = np.sort(u_korsor.flatten())[-N:][::-1]


mean_WS_nyborg = np.mean(sorted_speeds_nyborg)
b0_nyborg = mean_WS_nyborg
b1_nyborg = np.sum((np.arange(1, N+1) - 1) * sorted_speeds_nyborg) / (N * (N-1))

alpha_nyborg = 2 * b1_nyborg - b0_nyborg * np.log(2)
beta_nyborg = b0_nyborg - gam * alpha_nyborg
U50_nyborg = alpha_nyborg * np.log(T / T_0) + beta_nyborg

# Print results for Nyborg
print("PWM Method for Nyborg:")
print(f"α with PWM: {alpha_nyborg}")
print(f"β with PWM: {beta_nyborg}")
print(f"Estimated U50 for Nyborg: {U50_nyborg:.2f} m/s")
print("\n")

mean_WS_korsor = np.mean(sorted_speeds_korsor)
b0_korsor = mean_WS_korsor
b1_korsor = np.sum((np.arange(1, N+1) - 1) * sorted_speeds_korsor) / (N * (N-1))

alpha_korsor = 2 * b1_korsor - b0_korsor * np.log(2)
beta_korsor = b0_korsor - gam * alpha_korsor
U50_korsor = alpha_korsor * np.log(T / T_0) + beta_korsor

# Print results for Korsor
print("PWM Method for Korsor:")
print(f"α with PWM: {alpha_korsor}")
print(f"β with PWM: {beta_korsor}")
print(f"Estimated U50 for Korsor: {U50_korsor:.2f} m/s")
print("\n")

print('End')
