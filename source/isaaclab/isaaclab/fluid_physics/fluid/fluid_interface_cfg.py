class BaseFluidInterfaceCfg:
    # Always normal to the z-axis.
    def __init__(self):
        self.fluid_surface_type = None

class FluidPlaneInterfaceCfg(BaseFluidInterfaceCfg):
    def __init__(self):
        super().__init__()
        self.fluid_surface_type = "plane"

class FluidWaveInterfaceCfg(BaseFluidInterfaceCfg):
    def __init__(self):
        super().__init__()
        self.fluid_surface_type = "wave"
        self.wave_amplitude = 0.1
        self.wave_frequency = 1.0
        self.wave_direction = (1,0)
        self.wave_speed = None
        self.wave_phase = None