"""
Minimal optimisation example for ndscan.

Run ``QuadraticOptimisationScan`` from the dashboard, then set:

- Execution mode: ``Optimise``
- Parameter(s): ``x`` and optionally ``y``
- Bounds for ``x``: ``min=-5``, ``initial=4``, ``max=5``
- Bounds for ``y``: ``min=-5``, ``initial=-4``, ``max=5``
- Objective channel: ``objective``
- Direction: ``Minimise``

The optimum is at ``x = 1.25``, ``y = -0.5``.

If you only optimise ``x``, the example still works and converges to the optimum along
that axis for the current fixed value of ``y``.
"""

from ndscan.experiment import *


class QuadraticOptimisationSim(ExpFragment):
    def build_fragment(self):
        self.setattr_param(
            "x",
            FloatParam,
            "x",
            default=4.0,
            min=-5.0,
            max=5.0,
            step=0.1,
        )
        self.setattr_param(
            "y",
            FloatParam,
            "y",
            default=-4.0,
            min=-5.0,
            max=5.0,
            step=0.1,
        )
        self.setattr_result("objective", FloatChannel)

    def run_once(self):
        self.objective.push((self.x.get() - 1.25) ** 2 + (self.y.get() + 0.5) ** 2)


QuadraticOptimisationScan = make_fragment_scan_exp(QuadraticOptimisationSim)
