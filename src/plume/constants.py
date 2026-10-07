"""Physical constants (SI)."""

G0 = 9.80665  # standard gravity, m/s^2
R_EARTH = 6_371_000.0  # mean Earth radius, m
MU_EARTH = G0 * R_EARTH**2  # chosen so surface gravity is exactly G0 on the sphere
P0 = 101_325.0  # sea-level pressure, Pa
RHO0 = 1.225  # sea-level density, kg/m^3
R_AIR = 287.052_87  # specific gas constant of air, J/(kg K)
GAMMA_AIR = 1.4
