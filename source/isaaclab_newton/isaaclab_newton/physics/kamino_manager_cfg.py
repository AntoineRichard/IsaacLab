# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from newton.solvers import SolverKamino

    from isaaclab_newton.physics import NewtonManager


@configclass
class KaminoSolverCfg(NewtonSolverCfg):
    """Configuration for Kamino solver-related parameters.

    Kamino is a Proximal Alternating Direction Method of Multipliers (P-ADMM) based solver for
    constrained multi-body dynamics. It operates in maximal coordinates and supports rigid bodies
    and articulations with hard frictional contacts.

    .. note::

        This solver is currently in **Beta**. Its API and behavior may change in future releases.

    For more information, see the `Newton Kamino documentation`_.

    .. _Newton Kamino documentation: https://newton.readthedocs.io/en/latest/
    """

    class_type: type[NewtonManager] | str = "{DIR}.kamino_manager:NewtonKaminoManager"
    """Manager class for the Kamino solver."""

    solver_type: str = "kamino"
    """Solver type. Can be "kamino"."""

    integrator: str = "euler"
    """Integrator type. Can be "euler" or "moreau"."""

    dynamics_solver: str | None = None
    """Constrained dynamics solver variant.

    When ``None``, Newton's default dynamics solver is used. Set to ``"dvi"``
    to select the experimental DVI solver in compatible Newton versions.
    """

    use_collision_detector: bool = False
    """Whether to use Kamino's internal collision detector instead of Newton's pipeline."""

    use_fk_solver: bool | None = None
    """Whether to enable the forward kinematics solver for state resets.

    When ``None``, Kamino will automatically determine whether to use the FK solver based on the model's
    articulation structure. If the model has loop-closing joints, the FK solver will be used.

    When ``True``, :meth:`NewtonKaminoManager._eval_fk_impl` reconciles body state via
    :meth:`SolverKamino.reset` with :class:`SolverKamino.ResetConfig.from_joints`. Kamino's FK
    solver computes consistent body poses/velocities from the joint coordinates (including the
    base joint for floating bases), resolves passive / loop-closure joints, and writes back a
    consistent full joint state. Environment resets only need to write actuated DOFs in
    ``joint_q``; passive values are filled in by FK. This is required for closed-loop systems.

    When ``False``, Newton's articulated ``eval_fk`` is used instead over the full
    ``joint_q`` / ``joint_qd``. It is then up to the user to specify constraint-consistent
    values. This is the faster option for purely articulated (tree-structured) systems.
    """

    fk_use_regularization: bool = True
    """Whether to regularize the FK reset solve (Tikhonov term on body poses)."""

    fk_regularization_weight: float = 1e-5
    """Weight of the FK reset regularizer, used when :attr:`fk_use_regularization` is ``True``."""

    fk_tolerance: float = 1e-5
    """Convergence tolerance of the FK reset solve."""

    sparse_jacobian: bool = False
    """Whether to use sparse Jacobian computation."""

    sparse_dynamics: bool = False
    """Whether to use sparse dynamics computation."""

    rotation_correction: str = "twopi"
    """Rotation correction mode. Can be "twopi", "continuous", or "none"."""

    angular_velocity_damping: float = 0.0
    """Angular velocity damping factor. Valid range is [0.0, 1.0]."""

    constraints_alpha: float = 0.01
    """Baumgarte stabilization parameter for bilateral joint constraints. Valid range is [0, 1]."""

    constraints_beta: float = 0.01
    """Baumgarte stabilization parameter for unilateral joint-limit constraints. Valid range is [0, 1]."""

    constraints_gamma: float = 0.01
    """Baumgarte stabilization parameter for unilateral contact constraints. Valid range is [0, 1]."""

    constraints_delta: float = 1.0e-6
    """Contact penetration margin [m]."""

    padmm_max_iterations: int = 200
    """Maximum number of P-ADMM solver iterations."""

    padmm_primal_tolerance: float = 1e-6
    """Primal residual convergence tolerance for P-ADMM."""

    padmm_dual_tolerance: float = 1e-6
    """Dual residual convergence tolerance for P-ADMM."""

    padmm_compl_tolerance: float = 1e-6
    """Complementarity residual convergence tolerance for P-ADMM."""

    padmm_rho_0: float = 1.0
    """Initial penalty parameter for P-ADMM."""

    padmm_use_acceleration: bool = True
    """Whether to use acceleration in the P-ADMM solver."""

    padmm_warmstart_mode: str = "containers"
    """Warmstart mode for P-ADMM. Can be "none", "internal", or "containers"."""

    padmm_eta: float = 1e-5
    """Proximal regularization parameter for P-ADMM. Must be greater than zero."""

    padmm_contact_warmstart_method: str = "key_and_position"
    """Contact warm-start method for P-ADMM.

    Can be "key_and_position", "geom_pair_net_force", "geom_pair_net_wrench",
    "key_and_position_with_net_force_backup", or "key_and_position_with_net_wrench_backup".
    """

    padmm_use_graph_conditionals: bool = True
    """Whether to use CUDA graph conditional nodes in the P-ADMM iterative solver.

    When ``False``, replaces ``wp.capture_while`` with unrolled for-loops over max iterations.
    """

    collect_solver_info: bool = False
    """Whether to collect solver convergence and performance info at each step.

    .. warning::

        Enabling this significantly increases solver runtime and should only be used for debugging.
    """

    compute_solution_metrics: bool = False
    """Whether to compute solution metrics at each step.

    .. warning::

        Enabling this significantly increases solver runtime and should only be used for debugging.
    """

    collision_detector_pipeline: str | None = None
    """Collision detection pipeline type. Can be "primitive" or "unified".

    Only used when :attr:`use_collision_detector` is ``True``. If ``None``, Newton's default
    (``"unified"``) is used.
    """

    collision_detector_max_contacts_per_pair: int | None = None
    """Maximum number of contacts to generate per candidate geometry pair.

    Only used when :attr:`use_collision_detector` is ``True``. If ``None``, Newton's default is used.
    """

    max_contacts_per_world: int | None = None
    """Cap the per-world contact pre-allocation handed to Kamino.

    When ``None``, Kamino falls back to ``geoms.world_minimum_contacts`` derived from the
    collision pipeline, which over-allocates dramatically for contact-rich assets. Set this
    to bound GPU memory for multi-env training of contact-heavy tasks (e.g. legged
    locomotion or manipulation). The total ``model.rigid_contact_max`` is computed as
    ``max_contacts_per_world * model.world_count`` before solver construction.
    """

    dynamics_preconditioning: bool = True
    """Whether to use preconditioning in the constrained dynamics solver.

    Preconditioning improves convergence of the PADMM solver by rescaling the
    problem. Disabling may be useful for debugging or profiling solver behavior.
    """

    dynamics_linear_solver_type: str | None = None
    """Linear solver used for constrained dynamics.

    When ``None``, Newton selects its default. DVI sparse dynamics commonly uses
    ``"CR"``.
    """

    dynamics_linear_solver_max_iterations: int | None = None
    """Maximum constrained-dynamics linear-solver iterations.

    When ``None``, Newton selects the iteration budget.
    """

    dvi_max_iterations: int = 100
    """Maximum DVI projected iterations for the all-constraint fallback path."""

    dvi_tolerance: float = 1e-5
    """DVI convergence tolerance on the projected update size."""

    dvi_regularization: float = 1e-6
    """Diagonal regularization applied to DVI projected update denominators."""

    dvi_omega: float = 1.0
    """Relaxation factor applied to DVI projected updates."""

    dvi_block_iterations: int = 32
    """Number of outer DVI block iterations."""

    dvi_contact_iterations: int = 4
    """Number of projected contact sweeps per DVI block iteration."""

    dvi_bilateral_solve_period: int = 1
    """Number of DVI block iterations between bilateral solves."""

    dvi_contact_jacobi_omega: float = 0.3
    """Step size for DVI contact Jacobi updates."""

    dvi_contact_jacobi_relaxation: float = 0.9
    """Solution mixing factor for DVI contact Jacobi updates."""

    dvi_contact_block_preconditioner: bool = False
    """Whether to use a full 3-by-3 contact diagonal block preconditioner."""

    dvi_warmstart_mode: str | None = "containers"
    """DVI warm-start mode. Can be ``"none"``, ``"internal"``, or ``"containers"``.

    When ``None``, the DVI solver receives the ``"none"`` warm-start mode.
    """

    dvi_contact_warmstart_method: str = "key_and_position_with_net_force_backup"
    """Contact matching method used for DVI container warm-starting."""

    def to_solver_config(self) -> SolverKamino.Config:
        """Build a :class:`SolverKamino.Config` from this configuration.

        Converts the flat field layout of :class:`KaminoSolverCfg` into the
        nested dataclass hierarchy expected by :class:`SolverKamino`.

        Returns:
            A ``SolverKamino.Config`` instance ready for solver construction.
        """
        from newton._src.solvers.kamino.config import (
            CollisionDetectorConfig,
            ConstrainedDynamicsConfig,
            ConstraintStabilizationConfig,
            ForwardKinematicsSolverConfig,
            PADMMSolverConfig,
        )
        from newton.solvers import SolverKamino

        # Kamino Manager will set the automatic value before. This is a fallback to true if that mechanism was bypassed.
        if self.use_fk_solver is None:
            self.use_fk_solver = True

        dynamics_kwargs: dict[str, object] = {"preconditioning": self.dynamics_preconditioning}
        if self.dynamics_linear_solver_type is not None:
            dynamics_kwargs["linear_solver_type"] = self.dynamics_linear_solver_type
        if self.dynamics_linear_solver_max_iterations is not None:
            dynamics_kwargs["linear_solver_kwargs"] = {
                "maxiter": self.dynamics_linear_solver_max_iterations,
            }

        solver_kwargs: dict[str, object] = {}
        if self.dynamics_solver is not None:
            solver_kwargs["dynamics_solver"] = self.dynamics_solver
        if self.dynamics_solver == "dvi":
            from newton._src.solvers.kamino.config import DVISolverConfig

            solver_kwargs["dvi"] = DVISolverConfig(
                max_iterations=self.dvi_max_iterations,
                tolerance=self.dvi_tolerance,
                regularization=self.dvi_regularization,
                omega=self.dvi_omega,
                block_iterations=self.dvi_block_iterations,
                contact_iterations=self.dvi_contact_iterations,
                bilateral_solve_period=self.dvi_bilateral_solve_period,
                contact_jacobi_omega=self.dvi_contact_jacobi_omega,
                contact_jacobi_relaxation=self.dvi_contact_jacobi_relaxation,
                contact_block_preconditioner=self.dvi_contact_block_preconditioner,
                warmstart_mode="none" if self.dvi_warmstart_mode is None else self.dvi_warmstart_mode,
                contact_warmstart_method=self.dvi_contact_warmstart_method,
            )

        # Build collision detector config if using Kamino's internal detector
        collision_detector = None
        if self.use_collision_detector:
            cd_kwargs: dict = {}
            if self.collision_detector_pipeline is not None:
                cd_kwargs["pipeline"] = self.collision_detector_pipeline
            if self.collision_detector_max_contacts_per_pair is not None:
                cd_kwargs["max_contacts_per_pair"] = self.collision_detector_max_contacts_per_pair
            collision_detector = CollisionDetectorConfig(**cd_kwargs)

        return SolverKamino.Config(
            integrator=self.integrator,
            use_collision_detector=self.use_collision_detector,
            use_fk_solver=self.use_fk_solver,
            sparse_jacobian=self.sparse_jacobian,
            sparse_dynamics=self.sparse_dynamics,
            rotation_correction=self.rotation_correction,
            angular_velocity_damping=self.angular_velocity_damping,
            collect_solver_info=self.collect_solver_info,
            compute_solution_metrics=self.compute_solution_metrics,
            collision_detector=collision_detector,
            fk=ForwardKinematicsSolverConfig(
                use_regularization=self.fk_use_regularization,
                regularization_weight=self.fk_regularization_weight,
                tolerance=self.fk_tolerance,
            ),
            constraints=ConstraintStabilizationConfig(
                alpha=self.constraints_alpha,
                beta=self.constraints_beta,
                gamma=self.constraints_gamma,
                delta=self.constraints_delta,
            ),
            dynamics=ConstrainedDynamicsConfig(**dynamics_kwargs),
            padmm=PADMMSolverConfig(
                max_iterations=self.padmm_max_iterations,
                primal_tolerance=self.padmm_primal_tolerance,
                dual_tolerance=self.padmm_dual_tolerance,
                compl_tolerance=self.padmm_compl_tolerance,
                rho_0=self.padmm_rho_0,
                eta=self.padmm_eta,
                use_acceleration=self.padmm_use_acceleration,
                use_graph_conditionals=self.padmm_use_graph_conditionals,
                warmstart_mode=self.padmm_warmstart_mode,
                contact_warmstart_method=self.padmm_contact_warmstart_method,
            ),
            **solver_kwargs,
        )
