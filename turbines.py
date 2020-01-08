import numpy as np

class Turbine:
    def __init__ (self, name, nominal_power_kw, p_curve):
        self.name = name
        self.nominal_power_kw = nominal_power_kw
        self.p_curve = p_curve

    def get_nominal_power_kw(self):
        return self.nominal_power_kw;

    def calc_power(self, wind_speed):
        power = []
        for speed in wind_speed:
            power.append(self.calc_power_point(speed))
        return np.array(power)

    def calc_power_point(self, speed):

        for i in range(0, self.p_curve.shape[1]):
            if speed < self.p_curve[0, i]:

                if self.p_curve[1, i] == 0:
                    return 0

                return get_linear_point(self.p_curve[:, i-1], self.p_curve[:, i], speed)

        return 0


# Get point on linear curve
def get_linear_point (p1, p2, x3):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    n = k * p1[0] - p1[1]
    y3 = k * x3 - n
    return y3


def main():
    p_curve = np.array([
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25],
        [0, 75, 190, 353, 581, 886, 1273, 1710, 2145, 2544, 2837, 2965, 2295, 3000, 3000]])

    turbine = Turbine("Vestas V90", 10, p_curve)

    for i in range(0,30):
        print (turbine.calc_power_point(i+0.5))
        # print (turbine.c_p(i+0.3))



if __name__ == "__main__":
    main()