import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from scipy.optimize import minimize
import numpy as np

# Initialize the steam table
steam = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

# Given
t9 = 300  # Temperature at state 9 in °C
# t12 = 20  # Temperature at state 12 in °C
turbine_eff = 0.90
pump_eff = 0.90
solar_collection_area = 50000
# m6 = 1
tulsa_Q_in_average = (5.5 * solar_collection_area * 0.18) / 24  # kW, .18=solar collection efficiency

# Function to calculate efficiency
def calculate_efficiency(params):
    p12, p9, m10per, m11per, m6 = params
    m10 = m6 * (m10per / 100)
    m11 = m6 * (m11per / 100)

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


def get_points(params):
    p12, p9, m10per, m11per, m6 = params
    m10 = m6 * (m10per / 100)
    m11 = m6 * (m11per / 100)

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
    h2 = steam.h_ps(p9, s1) / pump_eff
    s2 = steam.s_ph(p9, h2)

    # State 9
    s9 = steam.s_pt(p9, t9)
    h9 = steam.h_pt(p9, t9)

    # States 10, 11, 12 (after turbine)
    rawh10 = steam.h_ps(p10, s9)
    rawh11 = steam.h_ps(p11, s9)
    rawh12 = steam.h_ps(p12, s9)
    h10 = steam.h_ps(p10, s9) / turbine_eff
    h11 = steam.h_ps(p11, s9) / turbine_eff
    h12 = steam.h_ps(p12, s9) / turbine_eff
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
    s3 = steam.s_ph(p9,h3)
    s4 = steam.s_ph(p11,h4)
    s5 = steam.s_ph(p12,h5)
    s6 = steam.s_ph(p9,h6)
    s7 = steam.s_ph(p10,h7)
    s8 = steam.s_ph(p11,h8)
    t1=steam.tsat_p(p12)
    t2=steam.t_hs(h2,s2)
    t3=steam.t_hs(h3,s3)
    t4=steam.t_hs(h4,s4)
    t5=steam.t_hs(h5,s5)
    t6=steam.t_hs(h6,s6)
    t7=steam.t_hs(h7,s7)
    t8=steam.t_hs(h8,s8)
    t10=steam.t_hs(h10,s10)
    t11=steam.t_hs(h11,s11)
    s61 = steam.sL_p(p9)
    s62 = steam.sV_p(p9)
    t61 = steam.t_ps(p9, s61)
    t62 = steam.t_ps(p9, s62)

    return [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s61,s62], [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t61,t62]

def get_cycle(params):
    p12, p9, m10per, m11per, m6 = params
    m10 = m6 * (m10per / 100)
    m11 = m6 * (m11per / 100)

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
    s3 = steam.s_ph(p9,h3)
    s4 = steam.s_ph(p11,h4)
    s5 = steam.s_ph(p12,h5)
    s6 = steam.s_ph(p9,h6)
    s7 = steam.s_ph(p10,h7)
    s8 = steam.s_ph(p11,h8)
    t1=steam.tsat_p(p12)
    t2=steam.t_hs(h2,s2)
    t3=steam.t_hs(h3,s3)
    t4=steam.t_hs(h4,s4)
    t5=steam.t_hs(h5,s5)
    t6=steam.t_hs(h6,s6)
    t7=steam.t_hs(h7,s7)
    t8=steam.t_hs(h8,s8)
    t10=steam.t_hs(h10,s10)
    t11=steam.t_hs(h11,s11)
    s61 = steam.sL_p(p9)
    s62 = steam.sV_p(p9)
    t61 = steam.t_ps(p9,s61)
    t62 = steam.t_ps(p9,s62)
    p2 = steam.p_hs(h2,s2)
    p3 = steam.p_hs(h3,s3)
    p6 = steam.p_hs(h6,s6)

    return ([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12] , [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12],
            [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12], [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])

def constraint(params):
    _, Q_in = calculate_efficiency(params)
    return tulsa_Q_in_average - Q_in

def constraint2(params):
    _, Q_in = calculate_efficiency(params)
    return Q_in - 0.95 * tulsa_Q_in_average

def constraint3(params):
    _, Q_in = calculate_efficiency(params)
    return Q_in - tulsa_Q_in_average

def pressure_constraint(params):
    p12, p9, _, _, _ = params
    return p9 - p12

def m10_constraint(params):
    _, _, m10per, _, _ = params
    return m10per - 0.1

def m11_constraint(params):
    _, _, _, m11per, _ = params
    return m11per - 0.1

def m10_upper_constraint(params):
    _, _, m10per, _, _ = params
    return 0.49 - m10per

def m11_upper_constraint(params):
    _, _, _, m11per, _ = params
    return 0.49 - m11per


# Initial guess for the parameters
initial_guess = [0.1, 50, 15, 7, 1]

# Bounds for the parameters
bounds = [(0.05, 0.5), (40, 60), (15, 25), (5, 10), (.8, 1.2)]

# Optimization
result = minimize(lambda params: calculate_efficiency(params)[0],
                  initial_guess, bounds=bounds, method='SLSQP',
                  constraints=[{'type': 'ineq', 'fun': constraint},
                               # {'type': 'ineq', 'fun': m10_constraint},
                               # {'type': 'ineq', 'fun': m11_constraint},
                               # {'type': 'ineq', 'fun': m10_upper_constraint},
                               # {'type': 'ineq', 'fun': m11_upper_constraint}],
                               ], options={'rhobeg': 0.1, 'maxiter': 10000})

# Optimal parameters
optimal_params = result.x
optimal_efficiency = -result.fun

# print(f"Optimal Parameters: {optimal_params}")
# print(f"Optimal Cycle Efficiency: {optimal_efficiency:.2f}%")
# print(f"tulsa Q-value: {tulsa_Q_in_average}, Q-value: {calculate_efficiency(result.x)[1]}")


entropies = np.linspace(0, 9.1, 500)
sat_liquid_temps = [steam.tsat_s(s) for s in entropies]
sat_vapor_temps = [steam.tsat_s(s) for s in entropies]

f = plt.figure(figsize=(15,15))
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
tick_spacing = 1


ax.plot(entropies, sat_liquid_temps, label='Saturated Liquid Line')
ax.plot(entropies, sat_vapor_temps, label='Saturated Vapor Line')
ax.set_xticks(np.arange(min(entropies),max(entropies),0.5))
points = get_points(result.x)
plt.scatter(points[0], points[1], label='Rankine Cycle')
# Add labels and title
plt.xlabel('s')

plt.ylabel('Temperature (°C)')
plt.title('Saturation Line')
# plt.legend()
plt.grid(True)

# Show the plot
plt.show()
cycle = get_cycle(result.x)
print('Optimized Parameters')
print(f'Turbine Entrance Pressure: {optimal_params[1]:.2f}bar')
print(f'Condensor Pressure: {optimal_params[0]:.2f}bar')
print(f'FWH(ṁ10): {optimal_params[2]:.2f}%')
print(f'FWH(ṁ11): {optimal_params[3]:.2f}%')
print(f'Mass Flow Rate(ṁ6): {optimal_params[4]:.2f}kg/s')
for pnt in range(len(cycle[0])):
    print(f'Point {pnt+1}: h={cycle[0][pnt]:.2f}kJ/kg, T={cycle[1][pnt]:.2f}°C, s={cycle[2][pnt]:.2f}kJ/kg°C, Pressure={cycle[3][pnt]:.2f}bar')

print(f"Optimal Cycle Efficiency: {optimal_efficiency:.2f}%")
print(f"Tulsa Q-in average for year with {solar_collection_area}m² collection area: {tulsa_Q_in_average}")
print(f"Q-in for optimized rankine cycle: {calculate_efficiency(result.x)[1]}")


