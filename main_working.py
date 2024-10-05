import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from scipy.optimize import minimize
import numpy as np

# Initialize the steam table
steam = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

# Given
t9 = 300  # Temperature at state 9 in °C
# t12 = 20  # Temperature at state 12 in °C
turbine_eff = 0.98
pump_eff = 0.98
solar_collection_area = 50000
m6 = 1
tulsa_Q_in_average = (5.5 * solar_collection_area * 0.18) / 24  # kW, .18=solar collection efficiency

# Function to calculate efficiency
def calculate_efficiency(params):
    p12, p9, m10, m11 = params



    # Pressures at various states
    # p12 = steam.psat_t(t12)
    t12 = steam.tsat_p(p12)
    p1 = p12
    p10 = p12 + 2 * ((p9 - p12) / 3)
    p11 = p12 + ((p9 - p12) / 3)
    p4 = p11
    p5 = p1
    p7 = p10
    p8 = p4

    # State 1
    s1 = steam.sL_t(t12)
    h1 = steam.hL_t(t12)

    # State 2 (after pump)
    h2 = steam.h_ps(p9, s1) * pump_eff
    s2 = steam.s_ph(p9, h2)

    # State 9
    s9 = steam.s_pt(p9, t9)
    h9 = steam.h_pt(p9, t9)

    # States 10, 11, 12 (after turbine)
    rawh10 = steam.h_ps(p10, s9)
    rawh11 = steam.h_ps(p11, s9)
    rawh12 = steam.h_ps(p12, s9)
    h10 = steam.h_ps(p10, s9) * turbine_eff
    h11 = steam.h_ps(p11, s9) * turbine_eff
    h12 = steam.h_ps(p12, s9) * turbine_eff
    s10 = steam.s_ph(p10, h10)
    s11 = steam.s_ph(p11, h11)
    s12 = steam.s_ph(p12, h12)

    # States 7, 8
    h7 = steam.hL_p(p7)
    h8 = h7

    # State 4
    h4 = steam.hL_p(p11)
    h5 = h4

    # State 3
    h3 = (m11 * rawh11 + m10 * h8 - (m11 + m10) * h4 + m6 * h2) / m6

    # State 6
    h6 = (m10 * (rawh10 - h7) + m6 * h3) / m6

    work_turbine = m6 * (h9 - rawh10) + (m6 - m10) * (rawh10 - rawh11) + (m6 - m10 - m11) * (rawh11 - rawh12)
    work_pump = m6 * (h2 - h1)
    work_net = work_turbine - work_pump  # kW
    Q_in = m6 * (h9 - h6)
    efficiency = (work_net / Q_in) * 100

    return -efficiency, Q_in  # Negative because we want to maximize efficiency

def constraint(params):
    _, Q_in = calculate_efficiency(params)
    return tulsa_Q_in_average - Q_in

def constraint2(params):
    _, Q_in = calculate_efficiency(params)
    return Q_in - 0.95 * tulsa_Q_in_average

def pressure_constraint(params):
    p12, p9, _, _ = params
    return p9 - p12


# Initial guess for the parameters
initial_guess = [0.5, 50, 0.33, 0.33]

# Bounds for the parameters
bounds = [(0.1, 1), (30, 80), (0.1, 0.49), (0.1, 0.49)]

# Optimization
result = minimize(lambda params: calculate_efficiency(params)[0],
                  initial_guess, bounds=bounds, method='SLSQP',
                  constraints=[{'type': 'ineq', 'fun': constraint},
                               {'type': 'ineq', 'fun': constraint2},
                               {'type': 'ineq', 'fun': pressure_constraint}])

# Optimal parameters
optimal_params = result.x
optimal_efficiency = -result.fun

print(f"Optimal Parameters: {optimal_params}")
print(f"Optimal Cycle Efficiency: {optimal_efficiency:.2f}%")
print(f"tulsa Q-value: {tulsa_Q_in_average}, Q-value: {calculate_efficiency(result.x)[1]}")


entropies = np.linspace(0, 9.1, 500)
sat_liquid_temps = [steam.tsat_s(s) for s in entropies]
sat_vapor_temps = [steam.tsat_s(s) for s in entropies]

f = plt.figure(figsize=(10,6))
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
tick_spacing = 1


ax.plot(entropies, sat_liquid_temps, label='Saturated Liquid Line')
ax.plot(entropies, sat_vapor_temps, label='Saturated Vapor Line')
ax.set_xticks(np.arange(min(entropies),max(entropies),0.5))
# Add labels and title
plt.xlabel('s')

plt.ylabel('Temperature (°C)')
plt.title('Saturation Line')
# plt.legend()
plt.grid(True)

# Show the plot
plt.show()