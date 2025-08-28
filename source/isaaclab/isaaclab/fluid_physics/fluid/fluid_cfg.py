from .references import REFERENCE_AIR_DENSITY, REFERENCE_WATER_DENSITY, REFERENCE_WATER_VISCOSITY, REFERENCE_AIR_VISCOSITY

class BaseFluidCfg:
    def __init__(self):
        self.current = None

class IncompressibleFluidCfg(BaseFluidCfg):
    def __init__(self):
        super().__init__()
        self.density = REFERENCE_WATER_DENSITY["fresh_water"]
        self.viscosity = REFERENCE_WATER_VISCOSITY["fresh_water"]

class AtmosphericFluidCfg(BaseFluidCfg):
    def __init__(self):
        super().__init__()
        self.reference_density_values = REFERENCE_AIR_DENSITY.values()
        self.reference_density_heights = REFERENCE_AIR_DENSITY.keys()
        self.initial_altitude = 0
        self.viscosity = REFERENCE_AIR_VISCOSITY["20C"]