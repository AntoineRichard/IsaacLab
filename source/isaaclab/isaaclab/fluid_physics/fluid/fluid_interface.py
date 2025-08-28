import warp as wp

from .fluid_interface_cfg import BaseFluidSurfaceCfg, FluidPlaneInterfaceCfg, FluidWaveInterfaceCfg

class BaseFluidInterface:
    def __init__(self, cfg: BaseFluidSurfaceCfg, num_envs: int, device: str = "cuda"):
        self.cfg = cfg
        self.surface_type = cfg.fluid_surface_type
        self.offset = wp.zeros((num_envs), dtype=wp.float32, device=device)
        self.is_immerged = wp.zeros((num_envs), dtype=wp.bool, device=device)
        self.immerged_depth = wp.zeros((num_envs), dtype=wp.float32, device=device)
        self.surface_normal = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)

    def get_surface_normal_at_point(self, point: wp.array) -> wp.array:
        raise NotImplementedError

    def is_immerged(self, point: wp.array) -> wp.array:
        raise NotImplementedError

    def get_immerged_depth(self, point: wp.array) -> wp.array:
        raise NotImplementedError


class FluidPlaneInterface(BaseFluidInterface):
    def __init__(self, cfg: FluidPlaneInterfaceCfg, num_envs: int, device: str = "cuda"):
        super().__init__(cfg, num_envs, device)
    
    def get_surface_normal_at_point(self, point: wp.array) -> wp.array:
        return self.surface_normal


class FluidWaveInterface(BaseFluidInterface):
    def __init__(self, cfg: FluidWaveInterfaceCfg, num_envs: int, device: str = "cuda"):
        super().__init__(cfg, num_envs, device)
        self.cfg = cfg
        self.surface_type = cfg.fluid_surface_type

    