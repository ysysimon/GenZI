from __future__ import annotations

import math
import unittest

import torch

from genzi.rotation import axis_angle_to_matrix, matrix_to_axis_angle


class TestRotationHelpers(unittest.TestCase):
    def test_zero_axis_angle(self):
        axis_angle = torch.zeros(2, 3)
        matrix = axis_angle_to_matrix(axis_angle)
        expected = torch.eye(3).expand(2, 3, 3)

        torch.testing.assert_close(matrix, expected)
        torch.testing.assert_close(matrix_to_axis_angle(expected), axis_angle)

    def test_known_quarter_turns(self):
        axis_angles = torch.tensor(
            [
                [math.pi / 2, 0.0, 0.0],
                [0.0, math.pi / 2, 0.0],
                [0.0, 0.0, math.pi / 2],
            ],
            dtype=torch.float64,
        )
        matrices = axis_angle_to_matrix(axis_angles)
        expected = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float64,
        )

        torch.testing.assert_close(matrices, expected, atol=1e-7, rtol=1e-7)

    def test_batch_round_trip_via_matrix(self):
        axis_angles = torch.tensor(
            [
                [0.2, -0.3, 0.4],
                [-0.7, 0.1, 0.5],
                [0.0, 0.0, 1.2],
            ],
            dtype=torch.float64,
        )

        matrices = axis_angle_to_matrix(axis_angles)
        recovered = matrix_to_axis_angle(matrices)

        torch.testing.assert_close(
            axis_angle_to_matrix(recovered), matrices, atol=1e-7, rtol=1e-7
        )

    def test_near_pi_round_trip(self):
        axis_angle = torch.tensor([[math.pi - 1e-5, 0.0, 0.0]], dtype=torch.float64)
        matrix = axis_angle_to_matrix(axis_angle)
        recovered = matrix_to_axis_angle(matrix)

        torch.testing.assert_close(
            axis_angle_to_matrix(recovered), matrix, atol=1e-7, rtol=1e-7
        )

    def test_backward_has_finite_gradients(self):
        axis_angle = torch.tensor(
            [[0.2, -0.4, 0.6], [0.1, 0.3, -0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )

        loss = matrix_to_axis_angle(axis_angle_to_matrix(axis_angle)).pow(2).sum()
        loss.backward()

        self.assertIsNotNone(axis_angle.grad)
        self.assertTrue(torch.isfinite(axis_angle.grad).all())


if __name__ == "__main__":
    unittest.main()
