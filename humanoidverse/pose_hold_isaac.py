"""Minimal Isaac Lab scene for policy evaluation, without motion-task resets."""

import numpy as np
import torch

from humanoidverse.pose_hold import ROOT, control_parameters, transform_actions
from humanoidverse.utils.torch_utils import quat_rotate_inverse


class UsdCollisionGeometry:
    """Actual USD collision meshes/primitives, cached in each rigid link frame."""

    def __init__(self, robot, device):
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        cache = UsdGeom.XformCache()
        names = robot.body_names
        self.parts = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            parent = prim
            while parent.IsValid() and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            if not parent.IsValid() or parent.GetName() not in names:
                raise ValueError(f"Cannot identify collision link for {prim.GetPath()}")
            body = names.index(parent.GetName())
            matrix = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(parent).GetInverse()
            center = np.array(matrix.Transform(Gf.Vec3d(0, 0, 0)))
            axes = np.column_stack([np.array(matrix.TransformDir(Gf.Vec3d(*v))) for v in np.eye(3)])
            kind, points, size = "mesh", None, None
            if prim.IsA(UsdGeom.Mesh):
                raw = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                points = np.array([matrix.Transform(Gf.Vec3d(*p)) for p in raw], dtype=float)
            elif prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                points = np.array(
                    [matrix.Transform(Gf.Vec3d(x, y, z)) for x in (-half, half) for y in (-half, half) for z in (-half, half)]
                )
            elif prim.IsA(UsdGeom.Sphere):
                kind, size = "sphere", float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
                shape = UsdGeom.Capsule(prim) if prim.IsA(UsdGeom.Capsule) else UsdGeom.Cylinder(prim)
                kind = "capsule" if prim.IsA(UsdGeom.Capsule) else "cylinder"
                axis = {"X": 0, "Y": 1, "Z": 2}[str(shape.GetAxisAttr().Get())]
                size = (float(shape.GetRadiusAttr().Get()), float(shape.GetHeightAttr().Get()) / 2, axis)
            else:
                raise ValueError(f"Unsupported USD collision type: {prim.GetTypeName()}")
            if points is not None and len(points) == 0:
                raise ValueError(f"Empty collider at {prim.GetPath()}")
            self.parts.append(
                (
                    body,
                    kind,
                    None if points is None else torch.tensor(points, dtype=torch.float32, device=device),
                    torch.tensor(center, dtype=torch.float32, device=device),
                    torch.tensor(axes, dtype=torch.float32, device=device),
                    size,
                )
            )
        if not self.parts:
            raise ValueError("No USD robot colliders found")

    def minimum(self, transforms):
        # PhysX link transforms use XYZW quaternions.
        low = torch.full((len(transforms),), float("inf"), device=transforms.device)
        vertical = torch.zeros((len(transforms), 3), device=transforms.device)
        vertical[:, 2] = 1
        for body, kind, points, center, axes, size in self.parts:
            t = transforms[:, body]
            row = quat_rotate_inverse(t[:, 3:7], vertical, w_last=True)
            if kind == "mesh":
                z = (row @ points.T).amin(-1)
            else:
                direction = row @ axes
                if kind == "sphere":
                    extent = size * torch.linalg.vector_norm(direction, dim=-1)
                else:
                    radius, half, axis = size
                    length = direction[:, axis].abs()
                    if kind == "capsule":
                        extent = half * length + radius * torch.linalg.vector_norm(direction, dim=-1)
                    else:
                        radial = direction.square().sum(-1) - direction[:, axis].square()
                        extent = half * length + radius * radial.clamp_min(0).sqrt()
                z = row @ center - extent
            low = torch.minimum(low, t[:, 2] + z)
        return low


class IsaacBackend:
    def __init__(self, config, batch_size=3072, clearance=0.01, device="cuda:0"):
        import importlib.metadata
        import os

        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=False, device=device).app
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

        self.config, self.device, self.batch_size, self.clearance = config, device, batch_size, clearance
        r, sim = config["robot"], config["sim"]
        self.params = p = control_parameters(config)
        self.decimation, self.physics_dt = sim["control_decimation"], 1 / sim["fps"]
        self.dt = self.physics_dt * self.decimation
        physx = sim.get("physx", {})
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                device=device,
                dt=self.physics_dt,
                render_interval=self.decimation,
                physx=sim_utils.PhysxCfg(
                    solver_type=physx.get("solver_type", 1),
                    max_position_iteration_count=physx.get("num_position_iterations", 4),
                    max_velocity_iteration_count=physx.get("num_velocity_iterations", 0),
                    bounce_threshold_velocity=physx.get("bounce_threshold_velocity", 0.5),
                ),
            )
        )
        self.scene = InteractiveScene(InteractiveSceneCfg(num_envs=batch_size, env_spacing=5.0, replicate_physics=True))
        floor = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            )
        )
        floor.func("/World/ground", floor)
        asset_root = ROOT / r["asset"]["asset_root"]
        asset = asset_root / r["asset"]["usd_file"]
        if not asset.is_file():
            raise FileNotFoundError(asset)
        names = r["dof_names"]
        actuator = ImplicitActuatorCfg(
            joint_names_expr=names,
            effort_limit_sim=dict(zip(names, p["effort"].tolist())),
            velocity_limit_sim=dict(zip(names, r["dof_vel_limit_list"])),
            stiffness=dict(zip(names, p["kp"].tolist())),
            damping=dict(zip(names, p["kd"].tolist())),
            armature=dict(zip(names, p["armature"].tolist())),
            friction=dict(zip(names, r["dof_joint_friction_list"])),
        )
        cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset),
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=not bool(r["asset"]["self_collisions"]),
                    solver_position_iteration_count=physx.get("num_position_iterations", 4),
                    solver_velocity_iteration_count=physx.get("num_velocity_iterations", 0),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0, 0, 0.8), joint_pos=r["init_state"]["default_joint_angles"], joint_vel={".*": 0.0}
            ),
            actuators={"all": actuator},
        )
        self.robot = Articulation(cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.sim.reset()
        self.joint_ids, _ = self.robot.find_joints(names, preserve_order=True)
        self.body_ids, _ = self.robot.find_bodies(r["body_names"], preserve_order=True)
        self.origins = self.scene.env_origins.clone()
        self.geometry = UsdCollisionGeometry(self.robot, device)
        self.valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.metadata = {
            "backend": "isaacsim",
            "batch_size": batch_size,
            "usd": str(asset),
            "colliders": len(self.geometry.parts),
            "dt": self.dt,
            "controller": "training implicit PD",
            "isaacsim_version": importlib.metadata.version("isaacsim"),
            "isaaclab_version": importlib.metadata.version("isaaclab"),
            "command_writes_per_control": 1,
        }

    def _forward(self):
        self.scene.write_data_to_sim()
        self.sim.physics_sim_view.update_articulations_kinematic()

    def _write(self, q, ids=None):
        if ids is None:
            ids = torch.arange(len(q), device=self.device)
        root = q[:, :7].clone()
        root[:, :3] += self.origins[ids]
        self.robot.write_root_pose_to_sim(root, env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=self.device), env_ids=ids)
        self.robot.write_joint_state_to_sim(
            q[:, 7:], torch.zeros((len(ids), 29), device=self.device), joint_ids=self.joint_ids, env_ids=ids
        )
        self.robot.set_joint_position_target(q[:, 7:], joint_ids=self.joint_ids, env_ids=ids)
        self.robot.set_joint_velocity_target(torch.zeros((len(ids), 29), device=self.device), joint_ids=self.joint_ids, env_ids=ids)

    def reset(self, qpos):
        self.n = len(qpos)
        # Pad the final batch with copies; those environments never enter scores.
        q = torch.as_tensor(qpos, dtype=torch.float32, device=self.device).clone()
        q = torch.cat((q, q[:1].repeat(self.batch_size - self.n, 1)))
        q[:, :2] = 0
        self.robot.reset()
        self.valid[:] = True
        self._write(q)
        self._forward()
        transforms = self.robot.root_physx_view.get_link_transforms().clone()
        minimum = self.geometry.minimum(transforms) - self.origins[:, 2]
        self.shift = self.clearance - minimum
        q[:, 2] += self.shift
        self.initial = q.clone()
        self._write(q)
        self._forward()
        minimum = self.geometry.minimum(self.robot.root_physx_view.get_link_transforms()) - self.origins[:, 2]
        if not torch.allclose(minimum, torch.full_like(minimum, self.clearance), atol=2e-4, rtol=0):
            raise RuntimeError(f"USD clearance alignment failed: {minimum.min().item()} .. {minimum.max().item()}")
        state = self.read()
        if state["qvel"].abs().max().item() > 1e-5:
            raise RuntimeError("Reset left nonzero velocity")
        # With zero root and joint velocities every link is stationary. PhysX
        # may retain the previous link-velocity cache until a physics step;
        # construct the physically correct initial observation without a step.
        state["body_vel"] = torch.zeros_like(state["body_vel"])
        state["body_ang_vel"] = torch.zeros_like(state["body_ang_vel"])
        return state

    def step(self, action):
        applied, target = transform_actions(torch.as_tensor(action, device=self.device), self.config, self.params)
        bad = ~torch.isfinite(target).all(-1)
        self.valid[: self.n] &= ~bad
        target[bad] = self.initial[: self.n, 7:][bad]
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids, env_ids=torch.arange(self.n, device=self.device))
        # Targets persist in PhysX. Its implicit PD runs at every physics step;
        # there are no external wrenches or explicit actuators to refresh here.
        # Refreshing Isaac Lab's diagnostic caches at every substep is redundant.
        self.scene.write_data_to_sim()
        for _ in range(self.decimation):
            self.sim.step(render=False)
        self.scene.update(self.dt)
        state = self.read()
        bad = ~torch.isfinite(state["qpos"]).all(-1) | ~torch.isfinite(state["qvel"]).all(-1)
        self.valid[: self.n] &= ~bad
        if bad.any():
            ids = bad.nonzero().flatten()
            self._write(self.initial[ids], ids)
            self._forward()
            state = self.read()
        state["valid"] = self.valid[: self.n].clone()
        return state, applied

    def read(self):
        view = self.robot.root_physx_view
        self.sim.physics_sim_view.update_articulations_kinematic()
        poses = view.get_link_transforms()[: self.n, self.body_ids].clone()
        velocities = view.get_link_velocities()[: self.n, self.body_ids].clone()
        root = view.get_root_transforms()[: self.n].clone()
        root[:, :3] -= self.origins[: self.n]
        root_vel = view.get_root_velocities()[: self.n].clone()
        root_ang_local = quat_rotate_inverse(root[:, 3:7], root_vel[:, 3:6], w_last=True)
        q = view.get_dof_positions()[: self.n, self.joint_ids]
        v = view.get_dof_velocities()[: self.n, self.joint_ids]
        poses[:, :, :3] -= self.origins[: self.n, None]
        return {
            "qpos": torch.cat((root[:, :3], root[:, [6, 3, 4, 5]], q), dim=-1),
            "qvel": torch.cat((root_vel[:, :3], root_ang_local, v), dim=-1),
            "body_pos": poses[:, :, :3],
            "body_rot": poses[:, :, 3:7],
            "body_vel": velocities[:, :, :3],
            "body_ang_vel": velocities[:, :, 3:6],
            "valid": self.valid[: self.n].clone(),
        }

    def close(self):
        self.app.close()
