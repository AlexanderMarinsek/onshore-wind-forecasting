import numpy as np

class Turbine:
    def __init__ (self, label, nominal_power_kw, calc_power_function):
        self.label = label
        self.nominal_power_kw = nominal_power_kw
        self.calc_power_point = calc_power_function

    def get_label(self):
        return self.label;

    def get_nominal_power_kw(self):
        return self.nominal_power_kw;

    def calc_power(self, wind_speed):
        power = []
        for speed in wind_speed:
            power.append(self.calc_power_point(speed))
        return np.array(power)

    def calc_power_point(self, wind_speed):
        return 0;


# Get point on linear curve
def get_linear_point (p1, p2, x3):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    n = k * p1[0] - p1[1]
    y3 = k * x3 - n
    return y3

"""
Calculate power based on piecewise linear power curve for:
Aircon 10 kW
http://www.urbanwind.net/pdf/CATALOGUE_V2.pdf
"""
def calc_power_aircon_10kw (wind_speed):

    power_kw = 0

    if wind_speed > 25:
        power_kw = 0
    elif wind_speed > 11:
        power_kw = 9.8
    elif wind_speed > 10:
        power_kw = get_linear_point((10,9.3), (11,9.8), wind_speed)
    elif wind_speed > 9:
        power_kw = get_linear_point((9,8.1), (10,9.3), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,6.7), (9,8.1), wind_speed)
    elif wind_speed > 7:
        power_kw = get_linear_point((7,4.9), (8,6.7), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,3.3), (7,4.9), wind_speed)
    elif wind_speed > 5:
        power_kw = get_linear_point((5,2.3), (6,3.3), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,0.9), (5,2.3), wind_speed)
    elif wind_speed > 2.5:
        power_kw = get_linear_point((2.5,0.4), (4,0.9), wind_speed)
    elif wind_speed > 2:
        power_kw = get_linear_point((2,0.0), (2.5,0.4), wind_speed)
    else:
        power_kw = 0

    return float(power_kw)

"""
Calculate power based on piecewise linear power curve for:
Enercon E-82 2050 kW
https://wind-data.ch/tools/powercalc.php
"""
def calc_power_enercon_e82_2050kw (wind_speed):

    power_kw = 0

    if wind_speed > 25:
        power_kw = 0
    elif wind_speed > 13:
        power_kw = 2050
    elif wind_speed > 12:
        power_kw = get_linear_point((12,2000), (13,2050), wind_speed)
    elif wind_speed > 11:
        power_kw = get_linear_point((11,1890), (12,2000), wind_speed)
    elif wind_speed > 10:
        power_kw = get_linear_point((10,1612), (11,1890), wind_speed)
    elif wind_speed > 9:
        power_kw = get_linear_point((9,1180), (10,1612), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,815), (9,1180), wind_speed)
    elif wind_speed > 7:
        power_kw = get_linear_point((7,532), (8,815), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,321), (7,532), wind_speed)
    elif wind_speed > 5:
        power_kw = get_linear_point((5,174), (6,321), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,82), (5,174), wind_speed)
    elif wind_speed > 3:
        power_kw = get_linear_point((3,25), (4,82), wind_speed)
    elif wind_speed > 2:
        power_kw = get_linear_point((2,3), (3,25), wind_speed)
    elif wind_speed > 1:
        power_kw = get_linear_point((1,0.0), (2,3), wind_speed)
    else:
        power_kw = 0

    return power_kw


"""
"""
def calc_power_aria_libellula_20kw (wind_speed):

    power_kw = 0

    if wind_speed > 25:
        power_kw = 0
    elif wind_speed > 20:
        power_kw = 16.4
    elif wind_speed > 19:
        power_kw = get_linear_point((19,16.9), (20,16.4), wind_speed)
    elif wind_speed > 18:
        power_kw = get_linear_point((18,17.2), (19,16.9), wind_speed)
    elif wind_speed > 17:
        power_kw = get_linear_point((17,17.8), (18,17.2), wind_speed)
    elif wind_speed > 16:
        power_kw = get_linear_point((16,18.3), (17,17.8), wind_speed)
    elif wind_speed > 15:
        power_kw = get_linear_point((15,18.9), (16,18.3), wind_speed)
    elif wind_speed > 14:
        power_kw = get_linear_point((14,19.4), (15,18.9), wind_speed)
    elif wind_speed > 13:
        power_kw = get_linear_point((13,19.8), (14,19.4), wind_speed)
    elif wind_speed > 12:
        power_kw = get_linear_point((12,19.9), (13,19.8), wind_speed)
    elif wind_speed > 11:
        power_kw = get_linear_point((11,19.9), (12,19.9), wind_speed)
    elif wind_speed > 10:
        power_kw = get_linear_point((10,19.3), (11,19.9), wind_speed)
    elif wind_speed > 9:
        power_kw = get_linear_point((9,18.3), (10,19.3), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,16.5), (9,18.3), wind_speed)
    elif wind_speed > 7:
        power_kw = get_linear_point((7,14.1), (8,16.5), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,10.7), (7,14.1), wind_speed)
    elif wind_speed > 5:
        power_kw = get_linear_point((5,7.2), (6,10.7), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,3.7), (5,7.2), wind_speed)
    elif wind_speed > 3:
        power_kw = get_linear_point((3,1.0), (4,3.7), wind_speed)
    elif wind_speed > 2:
        power_kw = get_linear_point((2,0), (3,1.0), wind_speed)
    else:
        power_kw = 0

    return power_kw


"""
"""
def calc_power_vestas_v90_3mw (wind_speed):

    power_kw = 0

    if wind_speed > 20:
        power_kw = 0
    elif wind_speed > 16:
        power_kw = 3000
    elif wind_speed > 15:
        power_kw = get_linear_point((15,2995), (16,3000), wind_speed)
    elif wind_speed > 14:
        power_kw = get_linear_point((14,2965), (15,2995), wind_speed)
    elif wind_speed > 13:
        power_kw = get_linear_point((13,2837), (14,2965), wind_speed)
    elif wind_speed > 12:
        power_kw = get_linear_point((12,2544), (13,2837), wind_speed)
    elif wind_speed > 11:
        power_kw = get_linear_point((11,2145), (12,2544), wind_speed)
    elif wind_speed > 10:
        power_kw = get_linear_point((10,1710), (11,2145), wind_speed)
    elif wind_speed > 9:
        power_kw = get_linear_point((9,1273), (10,1710), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,886), (9,1273), wind_speed)
    elif wind_speed > 7:
        power_kw = get_linear_point((7,581), (8,886), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,353), (7,581), wind_speed)
    elif wind_speed > 5:
        power_kw = get_linear_point((5,190), (6,353), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,78), (5,190), wind_speed)
    elif wind_speed > 3:
        power_kw = get_linear_point((3,0), (4,78), wind_speed)
    else:
        power_kw = 0

    return power_kw


"""
"""
def calc_power_yeloblade_1_2 (wind_speed):

    power_kw = 0

    if wind_speed > 44:
        power_kw = 0
    elif wind_speed > 26:
        power_kw = 0.550
    elif wind_speed > 24:
        power_kw = get_linear_point((24,0.530), (26,0.550), wind_speed)
    elif wind_speed > 22:
        power_kw = get_linear_point((22,0.440), (24,0.530), wind_speed)
    elif wind_speed > 20:
        power_kw = get_linear_point((20,0.355), (22,0.440), wind_speed)
    elif wind_speed > 18:
        power_kw = get_linear_point((18,0.278), (20,0.355), wind_speed)
    elif wind_speed > 16:
        power_kw = get_linear_point((16,0.215), (18,0.278), wind_speed)
    elif wind_speed > 14:
        power_kw = get_linear_point((14,0.155), (16,0.215), wind_speed)
    elif wind_speed > 12:
        power_kw = get_linear_point((12,0.1), (14,0.155), wind_speed)
    elif wind_speed > 10:
        power_kw = get_linear_point((10,0.05), (12,0.1), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,0.02), (10,0.05), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,0.008), (8,0.02), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,0.0048), (6,0.008), wind_speed)
    elif wind_speed > 3:
        power_kw = get_linear_point((3,0.0036), (4,0.0048), wind_speed)
    elif wind_speed > 2.7:
        power_kw = get_linear_point((2.7,0.003), (3,0.0036), wind_speed)
    else:
        power_kw = 0

    return power_kw


"""
"""
def calc_power_yeloblade_2_5 (wind_speed):

    power_kw = 0

    if wind_speed > 44:
        power_kw = 0
    elif wind_speed > 28:
        power_kw = 3.0
    elif wind_speed > 26:
        power_kw = get_linear_point((26,2.690), (28,3.0), wind_speed)
    elif wind_speed > 24:
        power_kw = get_linear_point((24,2.279), (26,2.690), wind_speed)
    elif wind_speed > 22:
        power_kw = get_linear_point((22,1.892), (24,2.279), wind_speed)
    elif wind_speed > 20:
        power_kw = get_linear_point((20,1.5265), (22,1.892), wind_speed)
    elif wind_speed > 18:
        power_kw = get_linear_point((18,1.1954), (20,1.5265), wind_speed)
    elif wind_speed > 16:
        power_kw = get_linear_point((16,0.9245), (18,1.1954), wind_speed)
    elif wind_speed > 14:
        power_kw = get_linear_point((14,0.6665), (16,0.9245), wind_speed)
    elif wind_speed > 12:
        power_kw = get_linear_point((12,0.430), (14,0.6665), wind_speed)
    elif wind_speed > 10:
        power_kw = get_linear_point((10,0.255), (12,0.430), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,0.108), (10,0.255), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,0.044), (8,0.108), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,0.026), (6,0.044), wind_speed)
    elif wind_speed > 3:
        power_kw = get_linear_point((3,0.02), (4,0.026), wind_speed)
    elif wind_speed > 2.7:
        power_kw = get_linear_point((2.7,0.015), (3,0.02), wind_speed)
    else:
        power_kw = 0

    return power_kw


"""
"""
def calc_power_yeloblade_1_2_330 (wind_speed):

    power_kw = 0

    if wind_speed > 44:
        power_kw = 0
    elif wind_speed > 20:
        power_kw = 0.440
    elif wind_speed > 18:
        power_kw = get_linear_point((18,0.278), (20,0.330), wind_speed)
    elif wind_speed > 16:
        power_kw = get_linear_point((16,0.215), (18,0.278), wind_speed)
    elif wind_speed > 14:
        power_kw = get_linear_point((14,0.155), (16,0.215), wind_speed)
    elif wind_speed > 12:
        power_kw = get_linear_point((12,0.110), (14,0.155), wind_speed)
    elif wind_speed > 10:
        power_kw = get_linear_point((10,0.075), (12,0.110), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,0.048), (10,0.075), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,0.028), (8,0.048), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,0.012), (6,0.028), wind_speed)
    elif wind_speed > 3:
        power_kw = get_linear_point((3,0.006), (4,0.012), wind_speed)
    elif wind_speed > 2.7:
        power_kw = get_linear_point((2.7,0.003), (3,0.006), wind_speed)
    else:
        power_kw = 0

    return power_kw


"""
"""
def calc_power_yeloblade_2_5_2000 (wind_speed):

    power_kw = 0

    if wind_speed > 44:
        power_kw = 0
    elif wind_speed > 24:
        power_kw = 2.000
    elif wind_speed > 22:
        power_kw = get_linear_point((22,1.892), (24,2.000), wind_speed)
    elif wind_speed > 20:
        power_kw = get_linear_point((20,1.527), (22,1.892), wind_speed)
    elif wind_speed > 18:
        power_kw = get_linear_point((18,1.195), (20,1.527), wind_speed)
    elif wind_speed > 16:
        power_kw = get_linear_point((16,0.925), (18,1.195), wind_speed)
    elif wind_speed > 14:
        power_kw = get_linear_point((14,0.667), (16,0.925), wind_speed)
    elif wind_speed > 12:
        power_kw = get_linear_point((12,0.430), (14,0.667), wind_speed)
    elif wind_speed > 10:
        power_kw = get_linear_point((10,0.280), (12,0.430), wind_speed)
    elif wind_speed > 8:
        power_kw = get_linear_point((8,0.190), (10,0.280), wind_speed)
    elif wind_speed > 6:
        power_kw = get_linear_point((6,0.120), (8,0.190), wind_speed)
    elif wind_speed > 4:
        power_kw = get_linear_point((4,0.060), (6,0.120), wind_speed)
    elif wind_speed > 3:
        power_kw = get_linear_point((3,0.038), (4,0.060), wind_speed)
    elif wind_speed > 2.7:
        power_kw = get_linear_point((2.7,0.015), (3,0.038), wind_speed)
    else:
        power_kw = 0

    return power_kw
