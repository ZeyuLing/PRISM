# -*- coding: utf-8 -*-
"""
SMPLPoseProcessor
-----------------
Preprocess (build + normalize) and postprocess (denormalize + recover SMPL/SMPL-X
axis-angle dict for Blender) with batch support.

Key features
------------
- Accepts motion arrays with shape (T, D) or (B, T, D).
- Translation: "abs" (T,3) or "rel" (T,3) with first-frame rel=0, or "abs_rel" (T,6).
- Rotation representation: axis_angle, quaternion, rot6d, rotmat.
- SMPL family: smpl_22 (body), smplh (body+hands), smplx_55 (body+hands+jaw+eyes).
- Robust numerics: std epsilon clamp.
"""

import json
import os
import torch
import numpy as np
from typing import List, Union, Tuple, Dict, Optional

from torch import nn
from prism.models.body_models.smplx_lite import SmplxLite
from prism.models.post_processing.smooth.smoothnet import SmoothNetFilter
from prism.registry import MODELS
from diffusers.models.modeling_outputs import BaseOutput
from einops import rearrange
from mmengine.device import get_device
from prism.utils.geometry.ccd_ik import ik
from prism.utils.geometry.matrix import get_rotation

# Use stable rotation conversion utilities only
from prism.utils.geometry.rotation_convert import (
    ROT_DIM,
    ROTATION_TYPE,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rot_convert,
)

from prism.utils.tensor_utils import tensor_to_array


# -------------------------- Lightweight output container --------------------- #
class ProcessedSMPLOutput(BaseOutput):
    smpl_transl_pose: torch.Tensor
    betas: Optional[torch.Tensor] = None
    expressions: Optional[torch.Tensor] = None
    gender: Optional[torch.Tensor] = None


def _ensure_tensor_1d(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    assert t.dim() == 1, f"Expected 1D vector, got shape {tuple(t.shape)}"
    return t


@MODELS.register_module()
class SMPLPoseProcessor(nn.Module):
    SUPPORTED_ROT = {
        str(ROTATION_TYPE.AXIS_ANGLE),
        str(ROTATION_TYPE.QUATERNION),
        str(ROTATION_TYPE.ROTATION_6D),
        str(ROTATION_TYPE.MATRIX),
    }
    SUPPORTED_TRANSL = {"abs", "rel", "abs_rel"}
    SUPPORTED_SMPL = {"smpl_22", "smplh", "smplx_55"}

    def __init__(
        self,
        smpl_model=dict(
            type="SmplxLiteV437Coco17",
        ),
        do_normalize: bool = True,
        stats_file: str = "stats.json",
        rot_type: str = ROTATION_TYPE.ROTATION_6D,
        transl_type: str = "abs",
        smpl_type: str = "smpl_22",
        eps: float = 1e-6,
        smooth_model=dict(
            type="smoothnet",
            window_size=64,
            output_size=64,
            checkpoint=None,
        ),
    ):
        super().__init__()
        assert rot_type in self.SUPPORTED_ROT
        assert transl_type in self.SUPPORTED_TRANSL
        assert smpl_type in self.SUPPORTED_SMPL

        self.do_normalize = do_normalize
        self.rot_type = rot_type
        self.transl_type = transl_type
        self.smpl_type = smpl_type
        self.eps = float(eps)

        stats = self._load_stats_json(stats_file)
        mean, std = self._build_stats_vectors(stats, rot_type, transl_type, smpl_type)
        std = torch.clamp(std, min=self.eps)

        self.smpl_model: SmplxLite = MODELS.build(smpl_model)
        self.static_joint_ids = [7, 10, 8, 11, 20, 21]

        # smoothnet
        if smooth_model is not None:
            self.smooth_model: SmoothNetFilter = MODELS.build(smooth_model)
        else:
            self.smooth_model = None

        self.register_buffer("mean", mean, persistent=False)  # (D,)
        self.register_buffer("std", std, persistent=False)  # (D,)

        self.freeze()

    # -------------------------- core API -------------------------- #
    def freeze(self):
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False)

    # ---------------------- stats helpers ---------------------- #
    def _load_stats_json(self, stats_file: str) -> dict:
        with open(stats_file, "r") as f:
            stats = json.load(f)
        return stats

    def _get_key_mean_std_from_stats(
        self, stats: dict, key: str, subkey: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if subkey is None:
            mean_list = stats[key]["mean"]
            std_list = stats[key]["std"]
        else:
            mean_list = stats[key][subkey]["mean"]
            std_list = stats[key][subkey]["std"]
        mean = _ensure_tensor_1d(mean_list)
        std = _ensure_tensor_1d(std_list)
        return mean, std

    def _build_stats_vectors(
        self, stats: dict, rot_type: str, transl_type: str, smpl_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if transl_type == "abs":
            t_mean, t_std = self._get_key_mean_std_from_stats(stats, "transl")
        elif transl_type == "rel":
            t_mean, t_std = self._get_key_mean_std_from_stats(stats, "transl_vel")
        else:
            abs_t_mean, abs_t_std = self._get_key_mean_std_from_stats(stats, "transl")
            rel_t_mean, rel_t_std = self._get_key_mean_std_from_stats(
                stats, "transl_vel"
            )
            t_mean = torch.cat([abs_t_mean, rel_t_mean], dim=-1)
            t_std = torch.cat([abs_t_std, rel_t_std], dim=-1)

        go_mean, go_std = self._get_key_mean_std_from_stats(
            stats, "global_orient", subkey=rot_type
        )
        bp_mean, bp_std = self._get_key_mean_std_from_stats(
            stats, "body_pose", subkey=rot_type
        )

        means, stds = [t_mean, go_mean, bp_mean], [t_std, go_std, bp_std]

        if smpl_type == "smplx_55":
            jaw_m, jaw_s = self._get_key_mean_std_from_stats(
                stats, "jaw_pose", subkey=rot_type
            )
            le_m, le_s = self._get_key_mean_std_from_stats(
                stats, "leye_pose", subkey=rot_type
            )
            re_m, re_s = self._get_key_mean_std_from_stats(
                stats, "reye_pose", subkey=rot_type
            )
            means += [jaw_m, le_m, re_m]
            stds += [jaw_s, le_s, re_s]

        if smpl_type in {"smplh", "smplx_55"}:
            lh_m, lh_s = self._get_key_mean_std_from_stats(
                stats, "left_hand_pose", subkey=rot_type
            )
            rh_m, rh_s = self._get_key_mean_std_from_stats(
                stats, "right_hand_pose", subkey=rot_type
            )
            means += [lh_m, rh_m]
            stds += [lh_s, rh_s]

        mean = torch.cat(means, dim=-1).to(torch.float32)
        std = torch.cat(stds, dim=-1).to(torch.float32)
        return mean, std

    # ---------------------- normalize to nearly gaussian distribution ------------------------------ #
    def normalize(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Normalize motion array with precomputed mean/std.

        Args:
            motion: Input motion array with shape (T, D) or (B, T, D).

        Returns:
            Normalized motion array with same shape as input.
        """
        assert motion.dim() in (2, 3)
        if not self.do_normalize:
            return motion
        mean = self.mean.to(motion.dtype).to(motion.device)
        std = self.std.to(motion.dtype).to(motion.device)
        return (motion - mean) / std

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Denormalize motion array with precomputed mean/std.

        Args:
            motion: Input motion array with shape (T, D) or (B, T, D).

        Returns:
            Denormalized motion array with same shape as input.
        """
        assert motion.dim() in (2, 3)
        if not self.do_normalize:
            return motion
        mean = self.mean.to(motion.dtype).to(motion.device)
        std = self.std.to(motion.dtype).to(motion.device)
        return motion * std + mean

    def load_smplx_dict_from_npz(self, npz_path: str) -> Dict[str, np.ndarray]:
        """Load SMPLX dict from npz file."""
        return dict(np.load(npz_path))

    @staticmethod
    def save_smplx_npz(out_path: str, smplx_dict: Dict):
        """Save a Blender-friendly SMPL-X dict to .npz (compressed)."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # ensure arrays are np.float32 where appropriate
        pack = {}
        for k, v in smplx_dict.items():
            if isinstance(v, np.ndarray):
                pack[k] = v.astype(np.float32, copy=False)
            else:
                pack[k] = v
        np.savez_compressed(out_path, **pack)

    # ---------------------- convert transl ------------------------------ #
    def convert_transl(
        self, transl: np.ndarray, transl_type: Optional[str] = None
    ) -> np.ndarray:
        """Convert raw SMPL global tranlsation vector to target format, which suits for network traning.
        transl:  np.ndarray. [T, 3]. Global translation vector.
        transl_type: str. Any in self.SUPPORTED_TRANSL. Default: self.transl_type.
        """
        transl_type = transl_type or self.transl_type
        assert transl_type in self.SUPPORTED_TRANSL
        """Convert translation to absolute or relative."""
        if transl_type == "abs":
            return transl
        elif transl_type == "rel":
            return np.cumsum(transl, axis=0)
        elif transl_type == "abs_rel":
            rel = np.zeros_like(transl)
            rel[1:] = transl[1:] - transl[:-1]
            return np.concatenate([transl, rel], axis=-1)  # (T,6)
        else:
            raise ValueError(f"Unsupported transl_type: {transl_type}")

    def inv_convert_transl(
        self,
        transl: np.ndarray,
        transl_type: Optional[str] = None,
        use_rollout: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert translation back to raw SMPL global translation vector.
        transl: np.ndarray. [T, 3] or [T, 6] or [B, T, 3] or [B, T, 6]. Global translation vector.
        transl_type: str. Any in self.SUPPORTED_TRANSL. Default: self.transl_type.
        use_rollout: bool. Only used when transl_type is "abs_rel". Whether to use rollout to convert translation. Default: True.
        """
        transl_type = transl_type or self.transl_type
        assert transl_type in self.SUPPORTED_TRANSL
        """Convert translation back to absolute or relative."""
        if transl_type == "abs":
            return transl
        elif transl_type == "rel":
            if isinstance(transl, torch.Tensor):
                return transl.cumsum(dim=-2)
            return np.cumsum(transl, axis=-2)
        elif transl_type == "abs_rel":
            if use_rollout:
                pos0 = transl[..., :1, :3]
                rel_t = transl[..., 1:, 3:]
                if isinstance(transl, torch.Tensor):
                    abs_t = torch.cumsum(torch.cat([pos0, rel_t], dim=-2), dim=-2)
                else:
                    abs_t = np.cumsum(np.concatenate([pos0, rel_t], axis=-2), axis=-2)
                return abs_t
            else:
                # directly return the absolute translation
                return transl[..., :3]
        else:
            raise ValueError(f"Unsupported transl_type: {transl_type}")

    # ---------------------- dict -> vector (single sequence) ------------------ #
    def smplx_dict_to_motion_vector(
        self,
        smplx_dict: Union[Dict[str, Union[np.ndarray, torch.Tensor]], str],
        transl_type: Optional[str] = None,
        rot_type: Optional[str] = None,
        smpl_type: Optional[str] = None,
        return_tensor: bool = True,
    ) -> torch.Tensor:
        smpl_type = smpl_type or self.smpl_type
        transl_type = transl_type or self.transl_type
        rot_type = rot_type or self.rot_type

        transl = smplx_dict.get("transl", None)

        go = smplx_dict.get("global_orient", None)
        bp = smplx_dict.get("body_pose", None)

        jaw = smplx_dict.get("jaw_pose", None)
        le = smplx_dict.get("leye_pose", None)
        re = smplx_dict.get("reye_pose", None)

        lhp = smplx_dict.get("left_hand_pose", None)
        rhp = smplx_dict.get("right_hand_pose", None)

        T_candidates = [
            t.shape[0] for t in [transl, go, bp, jaw, le, re, lhp, rhp] if t is not None
        ]
        if not T_candidates:
            raise ValueError("No valid keys to infer T.")
        assert (
            len(set(T_candidates)) == 1
        ), f"Inconsistent T lengths. Got {T_candidates}"
        T = max(T_candidates)

        def ensure_or_zero(t: Optional[np.ndarray], C: int) -> np.ndarray:
            if t is None:
                return np.zeros((T, C))
            return t

        transl = ensure_or_zero(transl, 3)
        go = ensure_or_zero(go, 3)
        bp = ensure_or_zero(bp, 63)
        lhp = ensure_or_zero(lhp, 45)
        rhp = ensure_or_zero(rhp, 45)
        jaw = ensure_or_zero(jaw, 3)
        le = ensure_or_zero(le, 3)
        re = ensure_or_zero(re, 3)

        # translation block
        transl: np.ndarray = self.convert_transl(transl, transl_type)

        poses: np.ndarray = [go, bp]
        if smpl_type == "smplx_55":
            poses += [jaw, le, re]

        if smpl_type == "smplh":
            poses += [lhp, rhp]

        poses = np.concatenate(poses, axis=-1)  # (T, J*3).
        J = poses.shape[-1] // 3
        poses = rearrange(poses, "T (J C) -> (T J) C", C=3)
        poses = rot_convert(poses, "axis_angle", rot_type)
        poses = rearrange(poses, "(T J) C -> T (J C)", J=J)

        motion_vec = np.concatenate([transl, poses], axis=-1)
        if return_tensor:
            motion_vec = torch.from_numpy(motion_vec)
        return motion_vec

    def smplx_dict_to_motion_vector_norm_add_static(
        self,
        smplx_dict,
        transl_type: Optional[str] = None,
        rot_type: Optional[str] = None,
        smpl_type: Optional[str] = None,
    ):
        motion_vec = self.smplx_dict_to_motion_vector(
            smplx_dict,
            transl_type,
            rot_type,
            smpl_type,
        ).unsqueeze(0)
        static_joints = self.get_static_joint_mask_from_motion(
            motion_vec, transl_type, rot_type
        )
        motion_vec = self.normalize(motion_vec)
        motion_vec = torch.cat([motion_vec, static_joints], dim=-1)
        return motion_vec

    def motion_vector_to_static_joints(self, motion_vector: torch.Tensor):
        transl = (
            motion_vector[..., :6]
            if self.transl_type == "abs_rel"
            else motion_vector[..., :3]
        )
        transl = self.inv_convert_transl(transl, self.transl_type)
        poses = motion_vector[..., 6:]
        joints = self.fk(transl, poses, return_intermediate=False)
        static_joints = self.get_static_joint_mask(
            joints[..., self.static_joint_ids, :]
        )
        return static_joints

    def motion_vector_to_smplx_dict(
        self,
        motion_vec: Union[np.ndarray, torch.Tensor],
        jaw_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
        leye_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reye_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
        left_hand_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
        right_hand_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
        betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
        expression: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mocap_framerate: Optional[float] = None,
        gender: Optional[Union[str, List[str]]] = None,
        rot_type: Optional[str] = None,
        transl_type: Optional[str] = None,
        smpl_type: Optional[str] = None,
    ):
        smpl_type = smpl_type or self.smpl_type
        transl_type = transl_type or self.transl_type
        rot_type = rot_type or self.rot_type

        assert len(motion_vec.shape) in [2, 3]
        is_batch = len(motion_vec.shape) == 3
        if isinstance(motion_vec, torch.Tensor):
            motion_vec = tensor_to_array(motion_vec)

        transl_dim = 6 if transl_type == "abs_rel" else 3
        rot_dim = ROT_DIM[rot_type]

        transl = motion_vec[..., :transl_dim]
        poses = motion_vec[..., transl_dim:]
        J = poses.shape[-1] // rot_dim

        # 1) "... (J C) -> ... J C"
        poses = rearrange(poses, "... (J C) -> ... J C", J=J, C=rot_dim)

        # 2) Record prefix shape before flattening (incl. J), then flatten except last dim
        prefix_shape = poses.shape[:-1]  # equivalent to (..., J)
        poses_flat = rearrange(poses, "... C -> (...) C")  # [N, C]

        # 3) Convert
        poses_flat = rot_convert(poses_flat, rot_type, "axis_angle")  # [N, C_out]

        # 4) Restore with recorded prefix shape, then merge J and C_out
        C_out = poses_flat.shape[-1]
        poses = poses_flat.reshape(*prefix_shape, C_out)  # -> (..., J, C_out)
        poses = rearrange(poses, "... J C -> ... (J C)", J=J)
        if smpl_type == "smpl_22":
            poses = np.concatenate(
                [
                    poses,
                    (
                        tensor_to_array(jaw_pose)
                        if jaw_pose is not None
                        else np.zeros((*poses.shape[:-1], 3))
                    ),
                ],
                axis=-1,
            )
            poses = np.concatenate(
                [
                    poses,
                    (
                        tensor_to_array(leye_pose)
                        if leye_pose is not None
                        else np.zeros((*poses.shape[:-1], 3))
                    ),
                ],
                axis=-1,
            )
            poses = np.concatenate(
                [
                    poses,
                    (
                        tensor_to_array(reye_pose)
                        if reye_pose is not None
                        else np.zeros((*poses.shape[:-1], 3))
                    ),
                ],
                axis=-1,
            )
            poses = np.concatenate(
                [
                    poses,
                    (
                        tensor_to_array(left_hand_pose)
                        if left_hand_pose is not None
                        else np.zeros((*poses.shape[:-1], 45))
                    ),
                ],
                axis=-1,
            )
            poses = np.concatenate(
                [
                    poses,
                    (
                        tensor_to_array(right_hand_pose)
                        if right_hand_pose is not None
                        else np.zeros((*poses.shape[:-1], 45))
                    ),
                ],
                axis=-1,
            )
        elif smpl_type == "smplh":
            poses = np.concatenate(
                [
                    poses[..., :66],
                    (
                        tensor_to_array(jaw_pose)
                        if jaw_pose is not None
                        else np.zeros((*poses.shape[:-1], 3))
                    ),
                    (
                        tensor_to_array(leye_pose)
                        if leye_pose is not None
                        else np.zeros((*poses.shape[:-1], 3))
                    ),
                    (
                        tensor_to_array(reye_pose)
                        if reye_pose is not None
                        else np.zeros((*poses.shape[:-1], 3))
                    ),
                    poses[..., 66:],
                ],
                axis=-1,
            )

        if not is_batch:
            smplx_dict = {
                "trans": transl,
                "transl": transl,
                "poses": poses,
                "global_orient": poses[..., :3],
                "body_pose": poses[..., 3:66],
                "jaw_pose": poses[..., 66:69],
                "leye_pose": poses[..., 69:72],
                "reye_pose": poses[..., 72:75],
                "left_hand_pose": poses[..., 75:120],
                "right_hand_pose": poses[..., 120:],
                "gender": gender if gender is not None else "neutral",
                "betas": betas if betas is not None else np.zeros((10,)),
                "expression": (
                    expression
                    if expression is not None
                    else np.zeros((*poses.shape[:-1], 10))
                ),
                "mocap_framerate": (
                    mocap_framerate if mocap_framerate is not None else 30.0
                ),
            }
            return smplx_dict
        else:
            smplx_dict_list = []
            for i in range(poses.shape[0]):
                smplx_dict_list.append(
                    {
                        "trans": transl[i],
                        "transl": transl[i],
                        "poses": poses[i],
                        "global_orient": poses[i, :3],
                        "body_pose": poses[i, 3:66],
                        "jaw_pose": poses[i, 66:69],
                        "leye_pose": poses[i, 69:72],
                        "reye_pose": poses[i, 72:75],
                        "left_hand_pose": poses[i, 75:120],
                        "right_hand_pose": poses[i, 120:],
                        "gender": gender[i] if gender is not None else "neutral",
                        "betas": betas[i] if betas is not None else np.zeros((10,)),
                        "expression": (
                            expression[i]
                            if expression is not None
                            else np.zeros(
                                (
                                    poses.shape[-1],
                                    10,
                                )
                            )
                        ),
                    }
                )
            smplx_dict_list

    def transl_pose_to_smplx_dict(
        self,
        transl,
        poses,
        mocap_framerate=30,
        betas=None,
        expression=None,
        gender=None,
        rot_type=None,
        to_numpy=True,
    ):
        if to_numpy and isinstance(transl, torch.Tensor):
            transl = tensor_to_array(transl)
        if to_numpy and isinstance(poses, torch.Tensor):
            poses = tensor_to_array(poses)
        if betas is not None and to_numpy and isinstance(betas, torch.Tensor):
            betas = tensor_to_array(betas)

        assert len(transl.shape) == 2, f"Expect (T,3), got {transl.shape}"
        assert len(poses.shape) == 2, f"Expect (T,j*d), got {poses.shape}"
        rot_type = rot_type or self.rot_type
        rot_dim = ROT_DIM[rot_type]
        num_joints = poses.shape[-1] // rot_dim

        poses = rot_convert(
            rearrange(poses, "t (j d) -> (t j) d", d=rot_dim),
            from_type=rot_type,
            to_type="axis_angle",
        )

        poses = rearrange(poses, "(t j) d -> t (j d)", j=num_joints)
        T = poses.shape[0]
        if num_joints == 22:
            poses = np.concatenate([poses, np.zeros((T, 99))], axis=-1)
        elif num_joints == 52:
            poses = np.concatenate(
                [poses[..., :66], np.zeros((T, 9)), poses[..., -90:]], axis=-1
            )
        else:
            assert num_joints == 55, num_joints

        smplx_dict = {
            "trans": transl,
            "transl": transl,
            "poses": poses,
            "global_orient": poses[..., :3],
            "body_pose": poses[..., 3:66],
            "jaw_pose": poses[..., 66:69],
            "leye_pose": poses[..., 69:72],
            "reye_pose": poses[..., 72:75],
            "left_hand_pose": poses[..., 75:120],
            "right_hand_pose": poses[..., 120:],
            "gender": gender if gender is not None else "neutral",
            "betas": betas if betas is not None else np.zeros((10,)),
            "expression": (
                expression
                if expression is not None
                else np.zeros((*poses.shape[:-1], 10))
            ),
            "mocap_framerate": (
                mocap_framerate if mocap_framerate is not None else 30.0
            ),
        }
        return smplx_dict

    # ---------------------- preprocess (public) ------------------------------- #
    def preprocess(
        self,
        smplx_dict_or_motion_vec: Union[str, dict, np.ndarray, torch.Tensor],
        transl_type: Optional[str] = None,
        rot_type: Optional[str] = None,
        smpl_type: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> ProcessedSMPLOutput:
        transl_type = transl_type or self.transl_type
        rot_type = rot_type or self.rot_type
        smpl_type = smpl_type or self.smpl_type
        if device is None:
            device = get_device()

        if isinstance(smplx_dict_or_motion_vec, str):
            smplx_dict_or_motion_vec = self.load_smplx_dict_from_npz(
                smplx_dict_or_motion_vec
            )

        if isinstance(smplx_dict_or_motion_vec, dict):
            motion_vec = self.smplx_dict_to_motion_vector(
                smplx_dict_or_motion_vec,
                transl_type=transl_type,
                rot_type=rot_type,
                smpl_type=smpl_type,
                return_tensor=True,
            )  # (T,D)
            betas = smplx_dict_or_motion_vec.get("betas", None)
            expressions = smplx_dict_or_motion_vec.get("expression", None)
            gender = smplx_dict_or_motion_vec.get("gender", None)
        else:
            betas = None
            expressions = None
            gender = None

            # ndarray / tensor path
            if isinstance(smplx_dict_or_motion_vec, np.ndarray):
                motion_vec = torch.from_numpy(smplx_dict_or_motion_vec)
            elif isinstance(smplx_dict_or_motion_vec, torch.Tensor):
                motion_vec = smplx_dict_or_motion_vec
            else:
                raise ValueError(f"Unknown type {type(smplx_dict_or_motion_vec)}")

        assert motion_vec.dim() in (
            2,
            3,
        ), f"Expect (T,D) or (B,T,D), got {motion_vec.shape}"
        motion_vec = motion_vec.to(device, dtype=dtype)
        if self.do_normalize:
            motion_vec = self.normalize(motion_vec)

        out = ProcessedSMPLOutput(
            smpl_transl_pose=motion_vec,
            betas=betas,
            expressions=expressions,
            gender=gender,
        )
        return out

    # ---------------------- postprocess (public) ------------------------------ #
    def postprocess(
        self,
        motion: Union[torch.Tensor, np.ndarray, ProcessedSMPLOutput],
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
    ) -> Union[Dict, List[Dict]]:
        if isinstance(motion, ProcessedSMPLOutput):
            motion = motion.smpl_transl_pose
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)
        assert isinstance(motion, torch.Tensor) and motion.dim() in (
            2,
            3,
        ), "motion must be (T,D) or (B,T,D)"

        device = self.mean.device
        motion = motion.to(device=device, dtype=torch.float32)
        if self.do_normalize:
            motion = self.denormalize(motion)

        # unify to (B,T,D)
        if motion.dim() == 2:
            motion = motion.unsqueeze(0)
        B, T, D = motion.shape

        # dims per block
        d = self.per_joint_dim
        dims = {
            "transl": 6 if self.transl_type == "abs_rel" else 3,
            "go": d * 1,
            "body": d * 21,
            "lh": d * 15,
            "rh": d * 15,
            "jaw": d * 1,
            "le": d * 1,
            "re": d * 1,
        }

        parts = ["transl", "go", "body"]

        if self.smpl_type == "smplx_55":
            parts += ["jaw", "le", "re"]

        if self.smpl_type in {"smplh", "smplx_55"}:
            parts += ["lh", "rh"]

        idx = 0
        blocks = {}
        for p in parts:
            w = dims[p]
            blocks[p] = motion[..., idx : idx + w]
            idx += w
        assert idx == D, f"Dim mismatch: consumed {idx}, total {D}"

        # reconstruct translation
        transl_block = blocks["transl"]  # (B,T,3) or (B,T,6)
        transl = self.inv_convert_transl(transl_block)

        # inverse rotations -> axis-angle per joint
        def inv_block(name: str, K: int) -> torch.Tensor:
            if name not in blocks:
                return torch.zeros((B, T, 3 * K), device=device, dtype=torch.float32)
            blk = blocks[name]  # (B,T,K*d)
            blk = rearrange(blk, "b t (k d) -> (b t k) d", k=K)
            aa = rot_convert(blk, self.rot_type, "axis_angle")
            aa = rearrange(aa, "(b t k) d -> b t (k d)", b=B, t=T, k=K)
            return aa

        go_aa = inv_block("go", 1)
        body_aa = inv_block("body", 21)
        lh_aa = inv_block("lh", 15)
        rh_aa = inv_block("rh", 15)
        jaw_aa = inv_block("jaw", 1)
        le_aa = inv_block("le", 1)
        re_aa = inv_block("re", 1)

        out_list: List[Dict] = []
        for b in range(B):
            T_b = T
            transl_b = tensor_to_array(transl[b])
            go_b = tensor_to_array(go_aa[b])
            body_b = tensor_to_array(body_aa[b])
            jaw_b = tensor_to_array(jaw_aa[b])
            le_b = tensor_to_array(le_aa[b])
            re_b = tensor_to_array(re_aa[b])
            lh_b = tensor_to_array(lh_aa[b])
            rh_b = tensor_to_array(rh_aa[b])
            poses_full = np.concatenate(
                [go_b, body_b, jaw_b, le_b, re_b, lh_b, rh_b], axis=-1
            )

            betas = np.zeros((10,), dtype=np.float32)
            expressions = np.zeros((T_b, 10), dtype=np.float32)

            sample = {
                "transl": transl_b,
                "trans": transl_b,
                "global_orient": go_b,
                "body_pose": body_b,
                "jaw_pose": jaw_b,
                "leye_pose": le_b,
                "reye_pose": re_b,
                "left_hand_pose": lh_b,
                "right_hand_pose": rh_b,
                "poses": poses_full,
                "betas": betas,
                "expressions": expressions,
                "mocap_framerate": float(mocap_framerate),
                "gender": gender,
            }
            out_list.append(sample)

        return out_list[0] if motion.shape[0] == 1 else out_list

    def fk(
        self,
        transl: torch.Tensor,  # [B, T, 3]
        poses: torch.Tensor,  # [B, T, D*J] (D depends on self.rot_type; J>=2 includes root+body)
        rot_type: Optional[str] = None,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert per-joint rotations (given by self.rot_type) to SMPL-X axis-angle
        and run SMPL-X forward in chunks to get joints and vertices.

        Returns:
            joints: [B, T, J_out, 3]
            verts : [B, T, V, 3]
        """
        B, T = transl.shape[:2]
        rot_type = (rot_type or self.rot_type).lower()

        d_per = ROT_DIM[rot_type]

        assert (
            poses.shape[-1] % d_per == 0
        ), f"pose last dim {poses.shape[-1]} must be divisible by {d_per} for type {rot_type}"
        J_all = poses.shape[-1] // d_per
        assert J_all >= 2, "pose must contain root + at least one body joint"

        # Split root and body
        global_orient = poses[..., :d_per]  # [B,T,d_per]
        body_pose = poses[..., d_per:]  # [B,T,(J_all-1)*d_per]
        J_body = J_all - 1

        N = B * T
        global_orient = global_orient.reshape(N, d_per)  # [N,d_per]
        body_pose = body_pose.reshape(N * J_body, d_per)  # [N*Jb,d_per]

        # Use rot_convert -> axis_angle (pass euler config)
        # Note: if rotation_convert uses SciPy bridge for Torch internally, gradients may be cut.
        aa_root = rot_convert(
            global_orient,
            rot_type,
            "axis_angle",
        )  # [N,3]
        aa_body = rot_convert(
            body_pose,
            rot_type,
            "axis_angle",
        ).reshape(
            N, J_body * 3
        )  # [N,Jb*3]

        transl_flat = transl.reshape(N, 3)

        joints, mat, fk_mat = self.smpl_model.fk(
            transl=rearrange(transl_flat, "(b t) c -> b t c", b=B),
            global_orient=rearrange(aa_root, "(b t) c -> b t c", b=B),
            body_pose=rearrange(aa_body, "(b t) c -> b t c", b=B),
        )
        if return_intermediate:
            return joints, mat, fk_mat
        return joints

    def refine_with_static_label(
        self,
        transl: torch.Tensor,
        poses: torch.Tensor,
        static_logits: torch.Tensor,
        rot_type: str = "axis_angle",
    ):
        """
        Refine SMPL-X dict with static label.
        """
        rot_type = rot_type or self.rot_type
        # check static_label is logits or probs or binary label
        static_conf = static_logits.sigmoid()
        joints, local_mat, post_w_mat = self.fk(
            transl=transl, poses=poses, rot_type=rot_type, return_intermediate=True
        )
        post_refine_joints = joints.clone()
        T = joints.shape[1]
        for i in range(1, T):
            prev = post_refine_joints[:, i - 1, self.static_joint_ids]
            this = joints[:, i, self.static_joint_ids]
            c_prev = static_conf[:, i - 1, :, None]
            post_refine_joints[:, i, self.static_joint_ids] = prev * c_prev + this * (
                1 - c_prev
            )

        # ik
        global_rot = get_rotation(post_w_mat)
        body_parents = self.smpl_model.parents_list[:22]
        left_leg_chain = [0, 1, 4, 7, 10]
        right_leg_chain = [0, 2, 5, 8, 11]
        left_hand_chain = [9, 13, 16, 18, 20]
        right_hand_chain = [9, 14, 17, 19, 21]

        local_mat = ik(
            local_mat,
            post_refine_joints[:, :, [7, 10]],
            global_rot[:, :, [7, 10]],
            [3, 4],
            left_leg_chain,
            body_parents,
        )
        local_mat = ik(
            local_mat,
            post_refine_joints[:, :, [8, 11]],
            global_rot[:, :, [8, 11]],
            [3, 4],
            right_leg_chain,
            body_parents,
        )
        local_mat = ik(
            local_mat,
            post_refine_joints[:, :, [20]],
            global_rot[:, :, [20]],
            [4],
            left_hand_chain,
            body_parents,
        )
        local_mat = ik(
            local_mat,
            post_refine_joints[:, :, [21]],
            global_rot[:, :, [21]],
            [4],
            right_hand_chain,
            body_parents,
        )

        body_pose = matrix_to_axis_angle(
            get_rotation(local_mat[:, :, 1:])
        )  # (B, L, J-1, 3, 3)
        body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

        # refined_pose
        refined_pose = torch.cat([poses[..., :3], body_pose, poses[..., 66:]], dim=-1)

        return refined_pose

    # -------------------- Speeds & static mask (GVHMR semantics) --------------------
    def joint_speed(self, w_j3d: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute per-frame joint speed magnitude (m/s) from world joints.
        w_j3d: [B, T, J, 3]
        returns speed: [B, T-1, J]  (no repeat here; caller decides)
        """
        vel = w_j3d[:, 1:, :, :] - w_j3d[:, :-1, :, :]  # [B, T-1, J, 3]
        speed = torch.linalg.norm(vel, dim=-1) / max(dt, 1e-8)  # [B, T-1, J]
        return speed

    def get_static_joint_mask(
        self,
        w_j3d: torch.Tensor,  # [B, T, J, 3]
        vel_thr: float = 0.15,
        fps: float = 30.0,
        repeat_last: bool = False,
    ) -> torch.Tensor:
        """
        static = ( ||Δp||/dt < vel_thr ).
        Returns shape: [B, T(or T-1), J]; if repeat_last=True then [B, T, J]
        """
        B, T, J, _ = w_j3d.shape

        # --- Special case: single frame ---
        if T == 1:
            # Return all False for [B, T, J] to avoid empty tensor
            return torch.zeros(B, T, J, dtype=torch.bool, device=w_j3d.device)

        dt = 1 / fps
        joint_v = self.joint_speed(w_j3d, dt=dt)  # [B, T-1, J]

        static_joint_mask = joint_v < vel_thr  # [B, T-1, J], bool
        if repeat_last:
            last = static_joint_mask[:, -1:, :]  # [B, 1, J]
            static_joint_mask = torch.cat([static_joint_mask, last], dim=1)  # [B, T, J]
        return static_joint_mask

    def get_static_joint_mask_from_motion(
        self,
        motion: torch.Tensor,  # [B, T, 6 + 6*J] or [B, T, 3 + 6*J]
        transl_type: str = None,
        rot_type: str = None,
    ):
        transl_type = transl_type or self.transl_type
        rot_type = rot_type or self.rot_type
        transl = motion[..., :6] if transl_type == "abs_rel" else motion[..., :3]
        transl = self.inv_convert_transl(transl)
        pose = motion[..., 6:] if transl_type == "abs_rel" else motion[..., 3:]
        joints = self.fk(transl, pose, rot_type=rot_type)

        # b t 6
        sj = self.get_static_joint_mask(
            joints[..., self.static_joint_ids, :],
            vel_thr=0.15,
            repeat_last=True,
        )
        return sj

    def post_hoc_static_refine(
        self,
        transl: torch.Tensor,
        poses: torch.Tensor,
        rot_type: str = "axis_angle",
        vel_thr: float = 0.15,
        fps: float = 30.0,
        sharpness: float = 50.0,
    ) -> torch.Tensor:
        """Post-hoc static joint refinement from reconstructed motion.

        Instead of relying on a VQ-VAE-predicted binary channel, compute
        static confidence directly from the decoded joint velocities and
        feed it into the existing ``refine_with_static_label`` IK pipeline.

        The confidence is a **soft** sigmoid:
            conf = sigmoid( sharpness * (vel_thr - speed) )
        This avoids a hard threshold: joints moving much slower than vel_thr
        get conf ≈ 1 (lock in place), joints moving much faster get conf ≈ 0
        (keep original), and joints near the threshold get intermediate blending.
        ``sharpness`` controls the transition steepness (higher = sharper).

        Args:
            transl:  [B, T, 3] world translation.
            poses:   [B, T, J*3] axis-angle poses (or matching rot_type).
            rot_type: rotation representation of ``poses``.
            vel_thr: velocity threshold in m/s (same semantics as
                     ``get_static_joint_mask``).
            fps:     frame rate for speed computation.
            sharpness: steepness of the soft threshold sigmoid.

        Returns:
            Refined poses with the same shape as input ``poses``.
        """
        # FK to get world joint positions
        joints = self.fk(transl, poses, rot_type=rot_type)  # [B, T, J, 3]
        static_joints = joints[:, :, self.static_joint_ids, :]  # [B, T, 6, 3]

        B, T, J_s, _ = static_joints.shape
        if T <= 1:
            return poses  # nothing to refine for single frame

        dt = 1.0 / max(fps, 1e-8)
        speed = self.joint_speed(static_joints, dt=dt)  # [B, T-1, 6]

        # Soft confidence: high when speed << vel_thr, low when speed >> vel_thr
        static_conf = torch.sigmoid(sharpness * (vel_thr - speed))  # [B, T-1, 6]

        # Repeat last frame to get [B, T, 6]
        static_conf = torch.cat([static_conf, static_conf[:, -1:, :]], dim=1)

        # Convert to logits for refine_with_static_label (which applies .sigmoid())
        # conf = sigmoid(logits) => logits = log(conf / (1 - conf))
        eps = 1e-6
        static_conf_clamped = static_conf.clamp(eps, 1.0 - eps)
        static_logits = torch.log(static_conf_clamped / (1.0 - static_conf_clamped))

        return self.refine_with_static_label(
            transl, poses, static_logits, rot_type=rot_type
        )

    @staticmethod
    def smooth_transl(transl: np.ndarray) -> np.ndarray:
        """
        Smooth root translation over time.
        Args:
            transl: np.ndarray of shape [T, 3], dtype float/float32/float64.
        Returns:
            np.ndarray of shape [T, 3], dtype float32.
        """
        if not isinstance(transl, np.ndarray):
            transl = np.asarray(transl)
        assert (
            transl.ndim == 2 and transl.shape[1] == 3
        ), f"Expected [T,3], got {transl.shape}"

        x = transl.astype(np.float32, copy=False)
        T = x.shape[0]
        if T <= 2:
            return x.copy()

        # Choose odd window proportional to sequence length (~15% of T, at least 5)
        win = max(5, int(T * 0.15) // 2 * 2 + 1)
        if win > T:
            win = T if (T % 2 == 1) else T - 1
        if win < 3:
            return x.copy()

        try:
            # 1) Prefer Savitzky-Golay (shape-preserving smooth)
            from scipy.signal import savgol_filter  # type: ignore

            poly = min(3, win - 1)
            y = savgol_filter(
                x, window_length=win, polyorder=poly, axis=0, mode="interp"
            )
            return y.astype(np.float32, copy=False)
        except Exception:
            # 2) Fallback: reflect-padded moving average (same odd window)
            pad = win // 2
            xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
            kernel = np.ones((win,), dtype=np.float32) / float(win)
            # 1D conv per channel (valid output length T)
            y = np.stack(
                [np.convolve(xp[:, c], kernel, mode="valid") for c in range(3)],
                axis=1,
            )
            return y.astype(np.float32, copy=False)

    def smooth_smplx_dict(
        self, smplx_dict: dict, smooth_transl: bool = False, dtype=torch.bfloat16
    ):
        device = get_device()

        poses = smplx_dict["poses"]
        poses = rearrange(poses, "t (j c) -> t j c", c=3)

        # if isinstance(poses, np.ndarray):
        #     poses = torch.from_numpy(poses)
        # poses = poses.to(device, dtype)

        poses = self.smooth_model(poses)

        poses = tensor_to_array(poses)

        poses = rearrange(poses, "t j c -> t (j c)")

        smplx_dict.update(
            {
                "poses": poses,
                "global_orient": poses[..., :3],
                "body_pose": poses[..., 3:66],
                "jaw_pose": poses[..., 66:69],
                "leye_pose": poses[..., 69:72],
                "reye_pose": poses[..., 72:75],
                "left_hand_pose": poses[..., 75:120],
                "right_hand_pose": poses[..., 120:],
            }
        )
        if smooth_transl:
            transl = smplx_dict["transl"]
            transl = self.smooth_transl(transl)

            smplx_dict.update({"transl": transl, "trans": transl})

        return smplx_dict

    def normalize_smplx_dict(self, smplx_dict: dict, smplx_model=None) -> dict:
        """Normalize an SMPL-X motion dict so that frame-0 faces +Z at xz=(0,0)
        and the lowest point across the motion sits at y=0.

        Operates in-place on numpy arrays:
            global_orient (T,3), transl (T,3), poses (T,165).

        Args:
            smplx_dict: dict with at least 'global_orient', 'transl', 'poses'.
            smplx_model: optional SmplxLite for FK-based ground normalization.
                         Falls back to pelvis height if None.
        Returns:
            The same dict, updated in-place.
        """
        global_orient = smplx_dict["global_orient"]  # (T, 3)
        transl = smplx_dict["transl"].copy()  # (T, 3)
        poses = smplx_dict["poses"].copy()  # (T, 165)

        # --- Step 1: determine yaw from first-frame global_orient ---
        R0 = axis_angle_to_matrix(global_orient[0])  # (3, 3)
        forward = R0 @ np.array([0.0, 0.0, 1.0])  # local +Z in world
        yaw = np.arctan2(forward[0], forward[2])  # angle from +Z toward +X

        # Build Y-axis rotation matrix R_yaw(-yaw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        R_yaw = np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float64,
        )

        # --- Step 2: apply yaw correction to entire sequence ---
        T_len = global_orient.shape[0]
        corrected_go = np.zeros_like(global_orient)
        for t in range(T_len):
            R_t = axis_angle_to_matrix(global_orient[t])  # (3, 3)
            R_corrected = R_yaw @ R_t
            corrected_go[t] = matrix_to_axis_angle(R_corrected)

        transl = (R_yaw @ transl.T).T  # (T, 3)

        # --- Step 3: XZ centering (first frame at origin) ---
        transl[:, 0] -= transl[0, 0]
        transl[:, 2] -= transl[0, 2]

        # --- Step 4: ground normalization ---
        _smplx_model = smplx_model if smplx_model is not None else self.smpl_model
        if _smplx_model is not None:
            transl_t = torch.from_numpy(transl).unsqueeze(0).float()
            go_t = torch.from_numpy(corrected_go).unsqueeze(0).float()
            bp = poses[:, 3:66]
            bp_t = torch.from_numpy(bp).unsqueeze(0).float()
            # Ensure tensors are on the same device as the body model (e.g. cuda:rank)
            _dev = next(_smplx_model.buffers()).device
            transl_t = transl_t.to(_dev)
            go_t = go_t.to(_dev)
            bp_t = bp_t.to(_dev)
            joints, _, _ = _smplx_model.fk(
                transl=transl_t,
                global_orient=go_t,
                body_pose=bp_t,
            )
            min_y = joints[..., 1].min().item()
        else:
            min_y = transl[:, 1].min()

        transl[:, 1] -= min_y

        # --- Step 5: update dict in-place ---
        global_orient = corrected_go.astype(np.float32)
        transl = transl.astype(np.float32)
        poses[:, :3] = global_orient
        poses = poses.astype(np.float32)

        smplx_dict["global_orient"] = global_orient
        smplx_dict["transl"] = transl
        smplx_dict["trans"] = transl
        smplx_dict["poses"] = poses

        return smplx_dict

    @property
    def per_joint_dim(self) -> int:
        return ROT_DIM[self.rot_type]


# --------------------------- Tests & examples -------------------------------- #
if __name__ == "__main__":
    import unittest
    import tempfile as _tmp
    import os as _os

    def _zero_stats(rot_type: str, transl_type: str, smpl_type: str, path: str):
        def rot_dim(rt: str) -> int:
            return {
                "axis_angle": 3,
                "quaternion": 4,
                ROTATION_TYPE.ROTATION_6D: 6,
                "rotmat": 9,
            }[rt]

        def block(k: int, d: int):
            return {"mean": [0.0] * (k * d), "std": [1.0] * (k * d)}

        d = rot_dim(rot_type)
        stats = {
            "transl": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            "transl_vel": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            "global_orient": {rot_type: block(1, d)},
            "body_pose": {rot_type: block(21, d)},
            "left_hand_pose": {rot_type: block(15, d)},
            "right_hand_pose": {rot_type: block(15, d)},
            "jaw_pose": {rot_type: block(1, d)},
            "leye_pose": {rot_type: block(1, d)},
            "reye_pose": {rot_type: block(1, d)},
        }
        with open(path, "w") as f:
            json.dump(stats, f)

    class TestSMPLPoseProcessor(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.T = 8
            self.sample = {
                "transl": torch.randn(self.T, 3) * 0.1,
                "global_orient": torch.randn(self.T, 3) * 0.2,
                "body_pose": torch.randn(self.T, 63) * 0.2,
                "left_hand_pose": torch.randn(self.T, 45) * 0.2,
                "right_hand_pose": torch.randn(self.T, 45) * 0.2,
                "jaw_pose": torch.randn(self.T, 3) * 0.1,
                "leye_pose": torch.randn(self.T, 3) * 0.1,
                "reye_pose": torch.randn(self.T, 3) * 0.1,
            }

        def _mk_proc(
            self, rot_type: str, transl_type: str, smpl_type: str = "smplx_55"
        ) -> SMPLPoseProcessor:
            fd, path = _tmp.mkstemp(suffix=".json")
            _os.close(fd)
            _zero_stats(rot_type, transl_type, smpl_type, path)
            proc = SMPLPoseProcessor(
                do_normalize=True,
                stats_file=path,
                rot_type=rot_type,
                transl_type=transl_type,
                smpl_type=smpl_type,
            )
            self._stats_path = path
            return proc

        def tearDown(self):
            if hasattr(self, "_stats_path") and _os.path.exists(self._stats_path):
                try:
                    _os.remove(self._stats_path)
                except Exception:
                    pass

        def _run_for(self, rot_type: str, transl_type: str):
            proc = self._mk_proc(rot_type, transl_type, "smplx_55")
            out = proc.preprocess(self.sample)
            x = out.smpl_transl_pose
            self.assertTrue(isinstance(x, torch.Tensor))
            self.assertEqual(x.dim(), 2)
            rec = proc.postprocess(out, gender="neutral")
            self.assertIn("poses", rec)
            self.assertEqual(rec["poses"].shape[0], self.T)

        def test_abs_axis(self):
            self._run_for("axis_angle", "abs")

        def test_abs_quat(self):
            self._run_for("quaternion", "abs")

        def test_abs_rot6d(self):
            self._run_for(ROTATION_TYPE.ROTATION_6D, "abs")

        def test_abs_rotmat(self):
            self._run_for("rotmat", "abs")

        def test_rel_rot6d(self):
            self._run_for(ROTATION_TYPE.ROTATION_6D, "rel")

        def test_absrel_rot6d(self):
            self._run_for(ROTATION_TYPE.ROTATION_6D, "abs_rel")

        def test_batch_roundtrip(self):
            proc = self._mk_proc(ROTATION_TYPE.ROTATION_6D, "abs", "smplx_55")
            out = proc.preprocess(self.sample)
            x = out.smpl_transl_pose.detach().cpu().numpy()
            T = x.shape[0]
            xb = np.stack([x[:T], x[:T]], axis=0)
            xb_t = torch.from_numpy(xb).to(proc.mean.device)
            nxb = proc.normalize(xb_t)
            rxb = proc.denormalize(nxb)
            self.assertLess((rxb - xb_t).abs().max().item(), 1e-5)

    unittest.main(verbosity=2)
