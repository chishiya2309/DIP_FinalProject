from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PoseBiGRUConfig:
    num_joints: int = 17
    raw_features_per_joint: int = 3
    normalized_features_per_joint: int = 7
    physics_features: int = 18
    fps: float = 24.0
    eps: float = 1e-6
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    num_classes: int = 2
    bidirectional: bool = True

    @property
    def joint_input_size(self) -> int:
        return self.num_joints * self.normalized_features_per_joint

    @property
    def input_size(self) -> int:
        return self.joint_input_size + self.physics_features


class PoseBiGRUAttention(nn.Module):
    def __init__(self, config: PoseBiGRUConfig | None = None) -> None:
        super().__init__()
        self.config = config or PoseBiGRUConfig()
        if self.config.raw_features_per_joint != 3:
            raise ValueError("raw_features_per_joint must be 3.")
        if self.config.normalized_features_per_joint != 7:
            raise ValueError("normalized_features_per_joint must be 7.")
        if self.config.physics_features != 18:
            raise ValueError("physics_features must be 18.")

        gru_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.pose_normalization = LearnablePoseNormalization(
            num_joints=self.config.num_joints,
            fps=self.config.fps,
            eps=self.config.eps,
        )
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        self.gru = nn.GRU(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=self.config.bidirectional,
        )

        recurrent_size = self.config.hidden_size * (2 if self.config.bidirectional else 1)
        self.attention = TemporalAttention(recurrent_size)
        pooled_size = recurrent_size * 3

        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Dropout(self.config.dropout),
            nn.Linear(pooled_size, recurrent_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(recurrent_size, self.config.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        joint_features = None

        if x.ndim == 4:
            batch_size, time_steps, num_joints, features_per_joint = x.shape
            if num_joints != self.config.num_joints:
                raise ValueError(f"Expected {self.config.num_joints} joints, got {num_joints}.")
            if features_per_joint == self.config.raw_features_per_joint:
                joint_features = self.pose_normalization(x, timestamps=timestamps)
            elif features_per_joint != self.config.normalized_features_per_joint:
                expected = (
                    self.config.raw_features_per_joint,
                    self.config.normalized_features_per_joint,
                )
                raise ValueError(f"Expected {expected} features per joint, got {features_per_joint}.")
            else:
                joint_features = x
            x = self.make_sequence_features(joint_features, timestamps=timestamps)
        elif x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, time, joints, features) "
                "or (batch, time, flattened_features)."
            )

        if x.ndim == 3 and x.size(-1) == self.config.num_joints * self.config.raw_features_per_joint:
            batch_size, time_steps, _ = x.shape
            joint_features = x.reshape(
                batch_size,
                time_steps,
                self.config.num_joints,
                self.config.raw_features_per_joint,
            )
            joint_features = self.pose_normalization(joint_features, timestamps=timestamps)
            x = self.make_sequence_features(joint_features, timestamps=timestamps)
        elif x.ndim == 3 and x.size(-1) == self.config.joint_input_size:
            batch_size, time_steps, _ = x.shape
            joint_features = x.reshape(
                batch_size,
                time_steps,
                self.config.num_joints,
                self.config.normalized_features_per_joint,
            )
            x = self.make_sequence_features(joint_features, timestamps=timestamps)

        if x.size(-1) != self.config.input_size:
            raise ValueError(f"Expected input size {self.config.input_size}, got {x.size(-1)}.")

        x = self.input_projection(x)
        sequence, _ = self.gru(x)
        context, attention_weights = self.attention(sequence, mask=mask)
        mean_pool, max_pool = self.pool_sequence(sequence, mask=mask)
        pooled = torch.cat((context, mean_pool, max_pool), dim=-1)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attention_weights
        return logits

    def pool_sequence(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            return sequence.mean(dim=1), sequence.max(dim=1).values

        mask = mask.bool()
        mask_float = mask.unsqueeze(-1).to(sequence.dtype)
        lengths = mask_float.sum(dim=1).clamp_min(1.0)

        mean_pool = (sequence * mask_float).sum(dim=1) / lengths
        max_input = sequence.masked_fill(~mask.unsqueeze(-1), torch.finfo(sequence.dtype).min)
        max_pool = max_input.max(dim=1).values
        valid_rows = mask.any(dim=1).unsqueeze(-1)
        max_pool = torch.where(valid_rows, max_pool, torch.zeros_like(max_pool))

        return mean_pool, max_pool

    def make_sequence_features(
        self,
        joint_features: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, time_steps, _, _ = joint_features.shape
        physics_features = self.pose_normalization.physics_features(joint_features, timestamps=timestamps)
        joint_features = joint_features.reshape(batch_size, time_steps, self.config.joint_input_size)
        return torch.cat((joint_features, physics_features), dim=-1)


class LearnablePoseNormalization(nn.Module):
    def __init__(self, num_joints: int = 17, fps: float = 24.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.fps = fps
        self.eps = eps
        self.left_hip = 11
        self.right_hip = 12
        self.left_shoulder = 5
        self.right_shoulder = 6
        self.nose = 0
        self.left_wrist = 9
        self.right_wrist = 10
        self.left_ankle = 15
        self.right_ankle = 16
        self.bones = (
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
        )

        self.center_logits = nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))
        self.scale_logits = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        gain_init = self.gain_logit(1.0)
        self.scale_gain_logit = nn.Parameter(torch.tensor([gain_init]))
        self.velocity_gain_logit = nn.Parameter(torch.tensor([gain_init]))
        self.acceleration_gain_logit = nn.Parameter(torch.tensor([gain_init]))
        self.joint_gain = nn.Parameter(torch.ones(num_joints, 2))
        self.joint_bias = nn.Parameter(torch.zeros(num_joints, 2))

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        xy = x[..., :2]
        confidence = x[..., 2].clamp(0.0, 1.0)

        hip_center = self.pair_center(xy, confidence, self.left_hip, self.right_hip)
        shoulder_center = self.pair_center(xy, confidence, self.left_shoulder, self.right_shoulder)
        body_center = self.weighted_center(xy, confidence)

        center_weights = torch.softmax(self.center_logits, dim=0)
        center = (
            center_weights[0] * hip_center
            + center_weights[1] * shoulder_center
            + center_weights[2] * body_center
        )

        torso_scale = (shoulder_center - hip_center).norm(dim=-1).clamp_min(self.eps) * 2.0
        shoulder_scale = self.pair_distance(xy, self.left_shoulder, self.right_shoulder).clamp_min(self.eps) * 3.0
        spread_scale = self.weighted_spread(xy, confidence, body_center).clamp_min(self.eps) * 2.5

        scale_weights = torch.softmax(self.scale_logits, dim=0)
        scale = (
            scale_weights[0] * torso_scale
            + scale_weights[1] * shoulder_scale
            + scale_weights[2] * spread_scale
        )
        scale = scale.unsqueeze(-1).unsqueeze(-1) * self.bounded_gain(self.scale_gain_logit)

        normalized_xy = (xy - center.unsqueeze(2)) / scale.clamp_min(self.eps)
        normalized_xy = normalized_xy * self.joint_gain.unsqueeze(0).unsqueeze(0)
        normalized_xy = normalized_xy * confidence.unsqueeze(-1)
        normalized_xy = normalized_xy + self.joint_bias.unsqueeze(0).unsqueeze(0)

        velocity = torch.zeros_like(normalized_xy)
        acceleration = torch.zeros_like(normalized_xy)
        pair_confidence = torch.minimum(confidence[:, 1:], confidence[:, :-1]).unsqueeze(-1)
        inverse_dt = self.inverse_time_delta(normalized_xy, timestamps).unsqueeze(-1)
        interval_valid = (inverse_dt > 0).to(normalized_xy.dtype)
        acceleration_inverse_dt = self.inverse_acceleration_delta(normalized_xy, timestamps).unsqueeze(-1)

        velocity[:, 1:] = (normalized_xy[:, 1:] - normalized_xy[:, :-1]) * inverse_dt * pair_confidence
        velocity = velocity * self.bounded_gain(self.velocity_gain_logit)

        if normalized_xy.size(1) > 2:
            triple_confidence = torch.minimum(pair_confidence[:, 1:], pair_confidence[:, :-1])
            acceleration_weight = acceleration_inverse_dt * interval_valid[:, 1:] * interval_valid[:, :-1]
            acceleration[:, 2:] = (velocity[:, 2:] - velocity[:, 1:-1]) * acceleration_weight * triple_confidence
        acceleration = acceleration * self.bounded_gain(self.acceleration_gain_logit)

        return torch.cat((normalized_xy, confidence.unsqueeze(-1), velocity, acceleration), dim=-1)

    def inverse_time_delta(self, reference: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = reference.size(0)
        time_steps = reference.size(1)

        if time_steps <= 1:
            return reference.new_zeros((batch_size, 0, 1))

        if timestamps is None:
            return reference.new_full((batch_size, time_steps - 1, 1), self.fps)

        timestamps = timestamps.to(device=reference.device, dtype=reference.dtype)

        if timestamps.ndim == 1:
            timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
        elif timestamps.ndim == 3 and timestamps.size(-1) == 1:
            timestamps = timestamps.squeeze(-1)

        if timestamps.ndim != 2:
            raise ValueError("timestamps must have shape (time,) or (batch, time).")

        if timestamps.size(0) == 1 and batch_size > 1:
            timestamps = timestamps.expand(batch_size, -1)

        if timestamps.size(0) != batch_size or timestamps.size(1) != time_steps:
            raise ValueError(f"Expected timestamps shape ({batch_size}, {time_steps}), got {tuple(timestamps.shape)}.")

        delta = timestamps[:, 1:] - timestamps[:, :-1]
        inverse_delta = torch.where(delta > self.eps, delta.reciprocal(), torch.zeros_like(delta))
        return inverse_delta.unsqueeze(-1)

    def inverse_acceleration_delta(self, reference: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = reference.size(0)
        time_steps = reference.size(1)

        if time_steps <= 2:
            return reference.new_zeros((batch_size, 0, 1))

        if timestamps is None:
            return reference.new_full((batch_size, time_steps - 2, 1), self.fps)

        timestamps = timestamps.to(device=reference.device, dtype=reference.dtype)

        if timestamps.ndim == 1:
            timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
        elif timestamps.ndim == 3 and timestamps.size(-1) == 1:
            timestamps = timestamps.squeeze(-1)

        if timestamps.ndim != 2:
            raise ValueError("timestamps must have shape (time,) or (batch, time).")

        if timestamps.size(0) == 1 and batch_size > 1:
            timestamps = timestamps.expand(batch_size, -1)

        if timestamps.size(0) != batch_size or timestamps.size(1) != time_steps:
            raise ValueError(f"Expected timestamps shape ({batch_size}, {time_steps}), got {tuple(timestamps.shape)}.")

        delta = timestamps[:, 1:] - timestamps[:, :-1]
        valid = (delta[:, 1:] > self.eps) & (delta[:, :-1] > self.eps)
        midpoint_delta = 0.5 * (delta[:, 1:] + delta[:, :-1])
        inverse_delta = torch.where(valid, midpoint_delta.reciprocal(), torch.zeros_like(midpoint_delta))
        return inverse_delta.unsqueeze(-1)

    def bounded_gain(self, value: torch.Tensor, low: float = 0.25, high: float = 4.0) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(value)

    def gain_logit(self, gain: float, low: float = 0.25, high: float = 4.0) -> float:
        ratio = torch.tensor((gain - low) / (high - low))
        return float(torch.logit(ratio))

    def weighted_center(self, xy: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        weights = confidence.unsqueeze(-1)
        numerator = (xy * weights).sum(dim=2)
        denominator = weights.sum(dim=2).clamp_min(self.eps)
        return numerator / denominator

    def pair_center(
        self,
        xy: torch.Tensor,
        confidence: torch.Tensor,
        left_index: int,
        right_index: int,
    ) -> torch.Tensor:
        points = torch.stack((xy[:, :, left_index], xy[:, :, right_index]), dim=2)
        weights = torch.stack((confidence[:, :, left_index], confidence[:, :, right_index]), dim=2).unsqueeze(-1)
        fallback = self.weighted_center(xy, confidence)
        denominator = weights.sum(dim=2).clamp_min(self.eps)
        center = (points * weights).sum(dim=2) / denominator
        valid = weights.sum(dim=2) > self.eps
        return torch.where(valid, center, fallback)

    def pair_distance(self, xy: torch.Tensor, left_index: int, right_index: int) -> torch.Tensor:
        return (xy[:, :, left_index] - xy[:, :, right_index]).norm(dim=-1)

    def weighted_spread(
        self,
        xy: torch.Tensor,
        confidence: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        weights = confidence.unsqueeze(-1)
        squared_distance = ((xy - center.unsqueeze(2)) ** 2).sum(dim=-1, keepdim=True)
        numerator = (squared_distance * weights).sum(dim=2)
        denominator = weights.sum(dim=2).clamp_min(self.eps)
        return torch.sqrt(numerator / denominator + self.eps).squeeze(-1)

    def physics_features(
        self,
        joint_features: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xy = joint_features[..., :2]
        confidence = joint_features[..., 2].clamp(0.0, 1.0)
        inverse_dt = self.inverse_time_delta(joint_features, timestamps)
        interval_valid = (inverse_dt > 0).to(joint_features.dtype)
        acceleration_inverse_dt = self.inverse_acceleration_delta(joint_features, timestamps)

        hip_center = self.pair_center(xy, confidence, self.left_hip, self.right_hip)
        shoulder_center = self.pair_center(xy, confidence, self.left_shoulder, self.right_shoulder)
        hip_confidence = torch.maximum(confidence[:, :, self.left_hip], confidence[:, :, self.right_hip])
        shoulder_confidence = torch.maximum(confidence[:, :, self.left_shoulder], confidence[:, :, self.right_shoulder])
        spine_confidence = torch.minimum(hip_confidence, shoulder_confidence).unsqueeze(-1)
        spine = shoulder_center - hip_center
        spine_length = spine.norm(dim=-1, keepdim=True) * spine_confidence
        spine_angle = torch.atan2(spine[..., 0], -spine[..., 1].clamp(min=-1e12, max=1e12))
        sin_angle = torch.sin(spine_angle).unsqueeze(-1) * spine_confidence
        cos_angle = torch.cos(spine_angle).unsqueeze(-1) * spine_confidence

        angular_velocity = torch.zeros_like(spine_angle)
        angular_acceleration = torch.zeros_like(spine_angle)

        if spine_angle.size(1) > 1:
            angle_delta = spine_angle[:, 1:] - spine_angle[:, :-1]
            angular_pair_confidence = torch.minimum(spine_confidence[:, 1:], spine_confidence[:, :-1]).squeeze(-1)
            angular_velocity[:, 1:] = (
                torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
                * inverse_dt.squeeze(-1)
                * angular_pair_confidence
            )

        if spine_angle.size(1) > 2:
            angular_pair_confidence = torch.minimum(spine_confidence[:, 1:], spine_confidence[:, :-1]).squeeze(-1)
            angular_triple_confidence = torch.minimum(
                angular_pair_confidence[:, 1:],
                angular_pair_confidence[:, :-1],
            )
            angular_acceleration[:, 2:] = (
                (angular_velocity[:, 2:] - angular_velocity[:, 1:-1])
                * (acceleration_inverse_dt * interval_valid[:, 1:] * interval_valid[:, :-1]).squeeze(-1)
                * angular_triple_confidence
            )

        width, height = self.body_extent(xy, confidence)
        log_aspect = torch.log((height + self.eps) / (width + self.eps)).unsqueeze(-1)
        aspect_velocity = torch.zeros_like(log_aspect)

        if log_aspect.size(1) > 1:
            aspect_velocity[:, 1:] = (log_aspect[:, 1:] - log_aspect[:, :-1]) * inverse_dt

        left_wrist_confidence = torch.minimum(confidence[:, :, self.left_wrist], confidence[:, :, self.left_hip]).unsqueeze(-1)
        right_wrist_confidence = torch.minimum(confidence[:, :, self.right_wrist], confidence[:, :, self.right_hip]).unsqueeze(-1)
        left_wrist_to_hip = (xy[:, :, self.left_wrist] - xy[:, :, self.left_hip]).norm(dim=-1, keepdim=True)
        right_wrist_to_hip = (xy[:, :, self.right_wrist] - xy[:, :, self.right_hip]).norm(dim=-1, keepdim=True)
        left_wrist_to_hip = left_wrist_to_hip * left_wrist_confidence
        right_wrist_to_hip = right_wrist_to_hip * right_wrist_confidence
        ankle_confidence = torch.maximum(confidence[:, :, self.left_ankle], confidence[:, :, self.right_ankle]).unsqueeze(-1)
        head_floor_confidence = torch.minimum(confidence[:, :, self.nose].unsqueeze(-1), ankle_confidence)
        feet_y = torch.maximum(xy[:, :, self.left_ankle, 1], xy[:, :, self.right_ankle, 1]).unsqueeze(-1)
        head_to_floor = (feet_y - xy[:, :, self.nose, 1].unsqueeze(-1)) * head_floor_confidence
        bone_mean, bone_std, bone_velocity, bone_confidence = self.bone_features(xy, confidence, inverse_dt)
        mean_confidence = confidence.mean(dim=-1, keepdim=True)
        valid_joint_ratio = (confidence > 0.2).to(confidence.dtype).mean(dim=-1, keepdim=True)

        return torch.cat(
            (
                sin_angle,
                cos_angle,
                angular_velocity.unsqueeze(-1),
                angular_acceleration.unsqueeze(-1),
                width.unsqueeze(-1),
                height.unsqueeze(-1),
                log_aspect,
                aspect_velocity,
                head_to_floor,
                left_wrist_to_hip,
                right_wrist_to_hip,
                bone_mean,
                bone_std,
                bone_velocity,
                bone_confidence,
                mean_confidence,
                valid_joint_ratio,
                spine_length,
            ),
            dim=-1,
        )

    def body_extent(self, xy: torch.Tensor, confidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = confidence > self.eps
        valid_expanded = valid.unsqueeze(-1)
        xy_min = xy.masked_fill(~valid_expanded, torch.finfo(xy.dtype).max).amin(dim=2)
        xy_max = xy.masked_fill(~valid_expanded, torch.finfo(xy.dtype).min).amax(dim=2)
        fallback_min = xy.amin(dim=2)
        fallback_max = xy.amax(dim=2)
        has_valid = valid.any(dim=2, keepdim=True)
        xy_min = torch.where(has_valid, xy_min, fallback_min)
        xy_max = torch.where(has_valid, xy_max, fallback_max)
        extent = (xy_max - xy_min).clamp_min(self.eps)
        return extent[..., 0], extent[..., 1]

    def bone_features(
        self,
        xy: torch.Tensor,
        confidence: torch.Tensor,
        inverse_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = []
        bone_confidences = []

        for left_index, right_index in self.bones:
            lengths.append((xy[:, :, left_index] - xy[:, :, right_index]).norm(dim=-1))
            bone_confidences.append(torch.minimum(confidence[:, :, left_index], confidence[:, :, right_index]))

        bone_lengths = torch.stack(lengths, dim=-1)
        bone_confidence = torch.stack(bone_confidences, dim=-1)
        weights = bone_confidence / bone_confidence.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        bone_mean = (bone_lengths * weights).sum(dim=-1, keepdim=True)
        bone_variance = (((bone_lengths - bone_mean) ** 2) * weights).sum(dim=-1, keepdim=True)
        bone_std = torch.sqrt(bone_variance + self.eps)
        bone_velocity = torch.zeros_like(bone_mean)

        if bone_lengths.size(1) > 1:
            delta = (bone_lengths[:, 1:] - bone_lengths[:, :-1]).abs()
            pair_confidence = torch.minimum(bone_confidence[:, 1:], bone_confidence[:, :-1])
            delta_weights = pair_confidence / pair_confidence.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            bone_velocity[:, 1:] = (delta * delta_weights).sum(dim=-1, keepdim=True) * inverse_dt

        return bone_mean, bone_std, bone_velocity, bone_confidence.mean(dim=-1, keepdim=True)


class TemporalAttention(nn.Module):
    def __init__(self, input_size: int, attention_size: int | None = None) -> None:
        super().__init__()
        attention_size = attention_size or input_size

        self.score = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.GELU(),
            nn.Linear(attention_size, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(sequence).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)

        return context, weights


def build_model(
    num_joints: int = 17,
    raw_features_per_joint: int = 3,
    normalized_features_per_joint: int = 7,
    physics_features: int = 18,
    fps: float = 24.0,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_classes: int = 2,
) -> PoseBiGRUAttention:
    config = PoseBiGRUConfig(
        num_joints=num_joints,
        raw_features_per_joint=raw_features_per_joint,
        normalized_features_per_joint=normalized_features_per_joint,
        physics_features=physics_features,
        fps=fps,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    )
    return PoseBiGRUAttention(config)


if __name__ == "__main__":
    model = build_model()
    sample = torch.randn(4, 30, 17, 3)
    sample[..., 2] = sample[..., 2].sigmoid()
    logits, attention = model(sample, return_attention=True)

    print(f"logits shape: {tuple(logits.shape)}")
    print(f"attention shape: {tuple(attention.shape)}")
