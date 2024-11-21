### IMPORTS
import pandas as pd
import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns

from IPython import get_ipython
get_ipython().magic('reset -f') # clear all variables
plt.close("all") # close all figures

### READ DATA

headers = ['timestamp', 'wind_speed', 'wind_direction_67.5m', 'wind_direction_70m']
sprog = pd.read_csv("sprog.tsv", sep='\t', header=None, names=headers)

### DATA CLEANING
sprog = sprog.apply(pd.to_numeric, errors='coerce')

#wind direction
sprog.replace(999, np.nan, inplace=True)
sprog['wind_direction'] = sprog['wind_direction_67.5m'].combine_first(sprog['wind_direction_70m'])
sprog[sprog['wind_direction_67.5m'].isna() & sprog['wind_direction_70m'].notna()]
sprog[sprog['wind_direction_67.5m'].isna()]

# Wind Speed
sprog.replace(99.99, np.nan, inplace=True)

#Remove NaN rows
sprog = sprog.dropna(subset=['wind_speed'])
sprog = sprog.dropna(subset=['wind_direction'])

### PART 1

## a

data = {
    "sector": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    "A": [7.750969, 7.525115, 7.761153, 9.236267, 9.603213, 8.461672, 9.407033, 9.931700, 9.951515, 10.194087, 9.630792, 7.896237],
    "k": [1.900057, 1.894834, 2.046636, 2.311559, 2.316891, 2.064537, 2.157974, 2.468474, 2.553114, 2.563006, 2.200351, 1.895504]
}

# Create DataFrame
Weib_parm = pd.DataFrame(data)
Weib_parm_dic = {row['sector']: {'A': row['A'], 'k': row['k']} for _, row in Weib_parm.iterrows()}

# Direction segments
bins = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
labels = np.arange(0, 360, 30)

sprog['sector'] = pd.cut(sprog['wind_direction'], bins=bins, labels=labels, right=False)
sprog.loc[(sprog['wind_direction'] >= 345) | (sprog['wind_direction'] < 15), 'sector'] = 0

sector_dfs = {sector: sprog[sprog['sector'] == sector] for sector in sprog['sector'].unique()}
sector_dfs.pop(np.nan, None)

Urp = 12
Prated = 13
T = 365*24


def power_curve(U):
    if U<=Urp:
        return Prated*(U/Urp)**3
    elif Urp<U<25:
        return Prated
    else :
        return 0
    return False


def weibull_pdf(U,k,A):
    U = np.maximum(U, 1e-6)
    pdf = (k/U)*((U/A)**k) * np.exp(-(U/A)**k)
    return pdf


def ff_i(sector):
    fi = len(sector_dfs[sector])/len(sprog)
    return fi



# E_sector = {}
# AEP = 0  # Annual Energy Production
# for key, df in sector_dfs.items():
#     k = Weib_parm_dic[key]['k']  # Weibull shape parameter
#     A = Weib_parm_dic[key]['A']  # Weibull scale parameter
#     #U = df['wind_speed'].values  # Get wind speeds as a numpy array
#     U = np.arange(0, 40.1, 0.1)

#     integral_f = []
#     delta_Us = []
#     pdfs = []
#     powers = []

#     for i in range(len(U) - 1):  # Iterate through wind speed points
#         delta_U = abs(U[i + 1] - U[i])  # Difference in wind speed
#         delta_Us.append(delta_U)
#         midpoint_U = (U[i] + U[i + 1]) / 2  # Midpoint for better accuracy
#         pdf_value = weibull_pdf(midpoint_U, k, A)
#         pdfs.append(pdf_value)
#         power_value = power_curve(midpoint_U)
#         powers.append(power_value)
#         integral = pdf_value * power_value * delta_U  # Numerical integration
#         integral_f.append(integral)

#     # Save sector energy
#     integral_T = sum(integral_f)
#     E_sector[key] = T * ff_i(key) * integral_T  # Scale by total hours in a year
#     AEP += E_sector[key]  # Add to total energy production

# print(f"Total AEP at Sprogo in MWh: {AEP:.3f}")

####################### AEP at the two other sites ###################################
# Calculate u_star, then geostrophic wind, then new u_star and finally new wind speed for each previous data
kappa = 0.4
z_0_water = 0.02e-2
z_0_land = 3e-2
z = 70
z_hub = 112
A = 1.8
B = 4.5

# Obtaining coriolis parameter from latitude
rot_earth = 7.2921e-5
latitude = 55.3
fc = 2*rot_earth*np.sin(latitude*np.pi/180)

from scipy.optimize import fsolve

def gdl(U, z0, sector):
    u_star = U *kappa/(np.log(z/z_0_water))
    
    U_g = u_star * (np.log(u_star/(fc*z_0_water)) - A)/kappa
    
    V_g = - B*u_star/kappa
    G = np.sqrt(U_g ** 2 + V_g ** 2)

    def equation(u_star_guess):
        G_guess = u_star_guess * np.sqrt((np.log(u_star_guess / (fc * z0)) - A) ** 2 + B ** 2)
        return G_guess - G  # we want 0
    
    # Use fsolve to find the root of the equation
    u_star_guess_initial = 0.0001  
    u_star_solution = fsolve(equation, u_star_guess_initial)
    
    u_star_final = u_star_solution[0]
    
    # Ensure that u_star doesn't exceed reasonable limits
    max_u_star = 0.9
    if u_star_final > max_u_star:
        u_star_final = max_u_star
    
    # Compute new wind speed U
    new_U = u_star_final*np.log(z / z0)/kappa
    return new_U


# Now we have to calculate AEP with new wind speeds 
# Get the roughness according to sectors 
def get_sector_roughness_dict(site):
    if site == "Nyborg":
        return {sector: z_0_water if sector <= 180 else z_0_land for sector in Weib_parm_dic.keys()}
    elif site == "Korsor":
        return {sector: z_0_land if sector <= 180 else z_0_water for sector in Weib_parm_dic.keys()}
    else:
        raise ValueError("Invalid site name. Choose 'Nyborg' or 'Korsor'.")

# Generate the dictionaries
sector_roughness_ny = get_sector_roughness_dict("Nyborg")
sector_roughness_ko = get_sector_roughness_dict("Korsor")


def calculate_site_AEP(sprog, sector_roughness):   
    AEP = 0
    E_sector = {}
    
    for sector, df in sector_dfs.items():
        k = Weib_parm_dic[sector]['k']  # Weibull shape parameter
        A = Weib_parm_dic[sector]['A']  # Weibull scale parameter
        z0 = sector_roughness[sector]

        # Integrate adjusted wind speeds into energy production calculation
        U = np.arange(0, 41, 0.5)
        integral_f = []
        
        for i in range(len(U) - 1): 
            delta_U = abs(U[i + 1] - U[i])  
            midpoint_U = (U[i] + U[i + 1]) / 2
            
            U_new = gdl(midpoint_U, z0, sector) 
            #print(f"Midpoint wind speed: {midpoint_U}, Adjusted wind speed (U_new): {U_new}")
            
            
            pdf_value_new = weibull_pdf(U_new, k, A)
            power_value_new = power_curve(U_new)
            
            #print(f"PDF value: {pdf_value_new}, Power value: {power_value_new}")
            
            integral_new = pdf_value_new * power_value_new * delta_U
            
            integral_f.append(integral_new)

        integral_T = sum(integral_f)
        fi = ff_i(sector)
        E_sector[sector] = T * fi * integral_T  
        AEP += E_sector[sector]  

    return AEP, E_sector


# Calculate AEP for Nyborg and Korsor
AEP_Nyborg, E_sector_Nyborg = calculate_site_AEP(sprog, sector_roughness_ny)
AEP_Korsor, E_sector_Korsor = calculate_site_AEP(sprog, sector_roughness_ko)

print(f"Total AEP at Nyborg in MWh: {AEP_Nyborg:.3f}")
print(f"Total AEP at Korsor in MWh: {AEP_Korsor:.3f}")


print('The End')

