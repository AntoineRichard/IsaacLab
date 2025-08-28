REFERENCE_AIR_DENSITY = {
    0: 1.2250,
    1000: 1.1116,
    2000: 1.0065,
    3000: 0.9091,
    4000: 0.8191,
    5000: 0.7361,
    6000: 0.6557,
    7000: 0.5895,
    8000: 0.5252,
    9000: 0.4664,
    10000: 0.4127,
    11000: 0.3639,
    13000: 0.2655,
    15000: 0.1937,
    17000: 0.1423,
    20000: 0.0880,
    25000: 0.0395,
    30000: 0.0180,
    32000: 0.0132,
    35000: 0.0082,
    40000: 0.0039,
    45000: 0.0019,
    47000: 0.0014,
    50000: 0.0010,
    60000: 0.000288,
    70000: 0.000086,
    80000: 0.000015,
    90000: 0.000003,
    100000: 0.0000005,
    110000: 0.0000001,
}

REFERENCE_WATER_DENSITY = {
    "salt_water": 1025,
    "fresh_water": 1000,
}

class BaseFluidCfg:
    def __init__(self):
        self.current = None

class IncompressibleFluidCfg(BaseFluidCfg):
    def __init__(self):
        self.density = REFERENCE_WATER_DENSITY["fresh_water"]
        self.viscosity = 0.001
        self.current = None

class AtmosphericFluidCfg(BaseFluidCfg):
    def __init__(self):
        super().__init__()
        self.reference_density_values = REFERENCE_AIR_DENSITY.values()
        self.reference_density_heights = REFERENCE_AIR_DENSITY.keys()
        self.initial_altitude = 0
        self.viscosity = 0.0000181
        self.current = None
