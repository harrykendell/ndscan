from __future__ import annotations

from dataclasses import dataclass
import traceback

import numpy as np
import numpy.typing as npt

# import BoTorch ML libraries
from random import randint
import torch

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats.qmc import LatinHypercube, scale

# for SAASBO implementation 
from botorch import fit_fully_bayesian_model_nuts
from botorch.optim import optimize_acqf
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize

from .base import (
    AlgorithmParameter,
    OptimizeAlgorithmSpec,
    Optimizer,
    register_algorithm,
)


@dataclass
class SAASBayesianOptimizerOptimizeAlgorithmSpec(OptimizeAlgorithmSpec):
    xatol: float = 1e-3
    fatol: float = 1e-3
    n_init: int = 50
    warmup_steps: int = 128
    thinning: int = 16
    user_seed: int = -1


class SAASBayesianOptimizer():
    """
    Sequential ask/tell SAAS Bayesian Optimization implementation.

    Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO)is used for
    high-dimensional Bayesian optimization. SAASBO uses a hierarchical sparsity 
    prior consisting of a global shrinkage parameter and inverse lengthscales, 
    employing a half-Cauchy distribution for these parameters. This allows for 
    effective optimization in high-dimensional spaces.Further perform inference 
    in the SAAS model using Hamiltonian Monte Carlo (HMC).
    """

    def __init__(
        self,
        initial: tuple[float, ...],         # initial samples from GUI
        lower_bounds: tuple[float, ...],    # lower bounds of active params
        upper_bounds: tuple[float, ...],    # upper bounds of active params
        xatol: float,
        fatol: float,
        n_init: int,                        # no.of initial samples to generate
        warmup_steps: int,                  # no.of warmup steps for HMC
        thinning: int,                      # thinning factor for HMC samples
        user_seed: int,                     # user defined seed
    ):

        # simulation parameters
        self.n_init = n_init

        # select the acquisition function type
        self.acq_func_type = "logEI"

        self.user_seed = user_seed                      # for reproducibility
        self.warmup_steps = warmup_steps
        self.thinning = thinning
        self.iter_idx = 0                               # initial iteration index
        self.sample_idx = 0                             # sample index
        self._termination_reason: str | None = None     # termination reason
        self.num_mcmc_samples = 64

        # experimental info
        self.n_params = len(initial)                    # no.of active parameters
        self.active_bounds_lower = np.array(lower_bounds, dtype=float)
        self.active_bounds_upper = np.array(upper_bounds, dtype=float)
        self._span = self.active_bounds_upper - self.active_bounds_lower

        # create physical and unit bounds
        self.phys_bounds = self.make_physical_bounds()
        self.unit_bounds = self.make_unit_bounds()

        self._xatol = xatol
        self._fatol = fatol

        # generate initial parameter input using LHS sampling
        self.x = self.LHS_sampler()

        # initialize training dataset
        self.init_x = torch.from_numpy(self.normalize(self.x)).double()
        self.init_y = torch.empty((0, 1), dtype=torch.double)
        self.init_y_var = torch.empty((0, 1), dtype=torch.double)
        self.best_init_y = float("-inf")

    def ask(self) -> tuple[float, ...]:
        """
        Suggests the next candidate point to sample and
        updates the training data with the candidate
        points found until the termination condition /
        max iterations are reached.
        Returns ``None`` if no more are needed.

        Outputs:
            - x_point : ((1,d)) active parameters and their values
        """
        if self._termination_reason is not None:
            # don't return points to evaluate
            return None

        if self.iter_idx == 0:
            # initial sampling
            if self.sample_idx < self.n_init:
                # make parameter dictionary
                x_point = self.x[self.sample_idx, :]
                # increment sample index
                self.sample_idx += 1
                return x_point

            
            # stop sampling, begin optimization
            print("Initial sampling complete.")
            self.iter_idx = 1

        # BO loop
        if self.iter_idx > 0:  # 1 < i < N
            # get the next points that maximize the AF
            fit_success, new_candidates = self.get_next_points()
            if not fit_success:
                print("Model fitting failed. Terminating optimization.")
                self._termination_reason = "model_fit_failure"
                return None
            
            # convert the new candidates for experimental use
            x_point = self.denormalize(new_candidates)
            # append the data set with the new candidate
            self.init_x = torch.cat((self.init_x, new_candidates))
            # update the iteration
            self.iter_idx += 1
            return x_point

        return None

    def tell(
        self, point: tuple[float, ...], value: float, std_dev: float = 0.0
    ) -> None:
        """
        Runs the experiment for the given parameters by :meth:`ask`
        and updates the measured objective value and its standard deviation.

        Inputs:
            - x_point : ((1,d) array) input parameters
            - value : ((1,2) array) objective function mean and std

        Outputs:
            - None. Updates the observations
        """

        del point
        if self._termination_reason is not None:
            return None
        # convert to torch tensors and append the data set with the new point(s)
        self.init_y = torch.cat((self.init_y,
                                 torch.tensor(value, dtype=torch.double).reshape(1, 1)
                                 ))  # update y-values
        
        # add small noise floor for numerical stability in GP fitting
        noise_floor = 1e-6
        obs_var = max(std_dev**2, noise_floor)
        self.init_y_var = torch.cat((self.init_y_var, 
                                     torch.tensor(obs_var, dtype=torch.double).reshape(1, 1)
                                     )) # update y-variances
         # obtain the best point so far
        self.best_init_y = self.init_y.max().item()
        
        if self.init_y.numel() > self.n_params:  # only check after enough data
            # check for convergence 
            self._maybe_terminate()

        return None

    def is_done(self) -> bool:
        """Return whether the optimizer has terminated."""
        return self._termination_reason is not None

    def best(self) -> tuple[tuple[float, ...], float] | None:
        """Returns the best point/value pair seen so far,
        if any evaluations completed.

        Outputs:
            - best_x : best parameters/point
            - best_init_y : best objective function value
        """
        # calculate the total no.of elements in a tensor
        if self.init_y.numel() == 0: # PyTorch method
            return None

        # select the single best observation
        best_idx = int(torch.argmax(self.init_y).item())

        # get model parameters at this index
        best_x_norm = self.init_x[best_idx]

        # convert parameters back to physical units
        best_x = self.denormalize(best_x_norm)

        return best_x, self.best_init_y

    def best_std(self) -> float | None:
        """Return the measured standard deviation for the current best point."""
        if self.init_y.numel() == 0:
            return None

        best_idx = int(torch.argmax(self.init_y).item())
        best_var = self.init_y_var[best_idx].item()
        return float(np.sqrt(max(best_var, 0.0)))

    def termination_reason(self) -> str | None:
        """Return the termination reason, or ``None`` while the optimiser is active."""
        return self._termination_reason

    def _maybe_terminate(self) -> None:
        """
        Terminates the optimization if either
        convergence or max evaluations reached.
        """

        # find where the best values occurred
        best_x, best_y = self.best() # in physical units

        # denormalize init_x and flatten init_y 
        all_x_phys = np.array([self.denormalize(self.init_x[i]) for i in range(len(self.init_x))])
        all_y = self.init_y.numpy().flatten()

        # get recent 5 points and values for convergence check
        recent_x = all_x_phys[-5:]
        recent_y = all_y[-5:]

        # find max change in x
        max_x_delta = max(
            np.max(np.abs((point - best_x) / self._span)) for point in recent_x
        )
        
        # find max change in y
        max_f_delta = max(abs(value - best_y) for value in recent_y)
        
        # check if both changes are within the specified tolerances
        if max_x_delta <= self._xatol and max_f_delta <= self._fatol:
            self._termination_reason = "converged"


    def build_surrogate_model(self):
        """
        Builds the surrogate model (Heteroscedastic GP regressor)
        for the BO loop. Fits a second GP to model how noise varies with x.

        Outputs :
            - model : the surrogate model (GP) fitted to the initial data
        """

        # create GP surrogate
        model = SaasFullyBayesianSingleTaskGP(
            train_X=self.init_x,
            train_Y=self.init_y,
            train_Yvar=self.init_y_var,
            outcome_transform=Standardize(m=1)
        )

        return model

    def get_acquisition_function(self, model):
        """
        Creates the acquisition function (AF) for the BO loop.
        Allows user to select from different AF's (EI, logEI, UCB, PI).

        Inputs:
            - model : the surrogate model (GP) fitted to the initial data

        Outputs:
            - acq_func : the acquisition function
        """
        # initialize class for the different AF's
        if self.acq_func_type == "EI":
            acq_func = qExpectedImprovement(model=model, best_f=self.best_init_y)
        elif self.acq_func_type == "logEI":
            acq_func = qLogExpectedImprovement(model=model, best_f=self.best_init_y)
        elif self.acq_func_type == "UCB":
            acq_func = qUpperConfidenceBound(model=model, beta=0.05)
        elif self.acq_func_type == "PI":
            acq_func = qProbabilityOfImprovement(model=model, best_f=self.best_init_y)
        else:
            raise ValueError("Invalid acquisition function type")
        return acq_func
    
  
    def get_next_points(self):
        """
        Obtains the next point(s) to sample in the BO loop.
        It does the following:
            - Builds the surrogate model (GP)
            - Uses fully Bayesian inference for fitting
            - Creates the Acquistion Function (AF)
            - Safely fit a model 
            - Find candidates for the next point to sample by optimizing the AF.

        Outputs :
            - fit_success: boolean indicating whether the model was fitted successfully
            - candidates: candidate(s) found while using a given AF
        """
        # create the GP models
        model = self.build_surrogate_model()

        try: # Attempt to fit the model for hyperparameter optimization
            # sample posterior hyperparameters directly via HMC/NUTS.
            fit_fully_bayesian_model_nuts( 
                    model,
                    warmup_steps=self.warmup_steps,
                    num_samples=self.num_mcmc_samples,
                    thinning=self.thinning,
                    disable_progbar=True,
                ) 
            # create the acquisition function
            acq_func = self.get_acquisition_function(model)

            # find candidates
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self.unit_bounds,
                q=1,
                num_restarts=10,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200},
            )
            return True, candidates

        except ValueError as ve:
            # handle common data-related errors
            print(f"ValueError during fitting: {ve}")
            return False, None

        except Exception as e:
            # catch any other unexpected errors
            print(f"Unexpected error during model fitting: {e}")
            traceback.print_exc()
            return False, None


    def make_physical_bounds(self):
        """Function returns the physical bounds for
        active parameters as torch tensors.

        Outputs :
            - phys_bounds : ((2, d') tensor) bounds where d' =  no.of active parameters
        """
        # convert to torch tensors for GP fitting
        lower = torch.tensor(self.active_bounds_lower, dtype=torch.double)
        upper = torch.tensor(self.active_bounds_upper, dtype=torch.double)

        # stack into shape (2, num_active_params)
        phys_bounds = torch.stack((lower, upper))

        return phys_bounds

    def make_unit_bounds(self):
        """Function returns the unit bounds that will be used
        in the unit hypercube optimization.

        Outputs:
            - unit_bounds : ((2, d') tensor) unit bounds
        """
        # create bounds
        unit_bounds = torch.stack([
            torch.zeros(self.n_params, dtype=torch.double),
            torch.ones(self.n_params, dtype=torch.double),
        ])
        return unit_bounds

    def LHS_sampler(self):
        """Function to generate initial parameters for BO
            using Latin Hypercube Sampling (LHS).

        Outputs:
            - init_params : (n, d) array of initial parameter vectors
        """

        #  generate LHS samples
        seed = self.user_seed if self.user_seed >= 0 else randint(0, 42)
        sampler = LatinHypercube(
            d=self.n_params, seed=seed
        )  # LHS sampler with fixed/None seed
        unit_samples = sampler.random(
            n=self.n_init
        )  # shape (n_init, d), values in [0, 1]

        #  scale from [0,1] to the actual parameter ranges
        init_params = scale(
            unit_samples,
            l_bounds=self.active_bounds_lower,
            u_bounds=self.active_bounds_upper,
        )

        return init_params

    def normalize(self, x: npt.NDArray[np.float64]):
        """Normalizes the input x to be between [0,1]
                x_norm = (x - lower) / (upper - lower)

        Inputs:
            - x : ((n,d) array) input parameters in physical units
            - active_params : list of active parameter names

        Outputs:
            - x_norm: ((n,d) array) : normalized input parameters
        """

        x_norm = (x - self.active_bounds_lower) / (
            self.active_bounds_upper - self.active_bounds_lower
        )

        return x_norm

    def denormalize(self, x_norm: torch.Tensor) -> np.ndarray:
        """Converts the normalized parameters to physical values.
                x = x_norm * (upper - lower) + lower

        Inputs:
            - x_norm: ((n,d) tensor) input parameters between [0,1]
            - phys_bounds : ((2,d) tensor) physical lower and upper bounds

        Outputs:
            - x : ((n,d) array) input parameters in physical units
        """
        # denormalize [0,1] to physical units
        x_phy = (
            x_norm * (self.phys_bounds[1] - self.phys_bounds[0]) + self.phys_bounds[0]
        )

        # convert to numpy arrays
        x = x_phy.detach().cpu().numpy().flatten()

        # if hardware require float32, convert to float32
        # x = x_phy.detach().cpu().numpy().astype(np.float32)

        return x



register_algorithm(
    "saas_bayesian",
    display_name="SAAS Bayesian optimization",
    description="SAAS Bayesian optimization optimizer",
    parameters=[
        AlgorithmParameter(
            name="fatol",
            label="fatol",
            minimum=0.0,
            maximum=10**9,
            default=1e-3,
            tooltip="Terminate when the objective value changes by at most this amount.",
        ),
        AlgorithmParameter(
            name="xatol",
            label="xatol",
            minimum=0.0,
            maximum=1.0,
            default=1e-3,
            step=1e-4,
            tooltip="Terminate when each axis moves by at most this fraction of its configured bounds span.",
        ),
        AlgorithmParameter(
            name="n_init",
            label="n_init",
            minimum=20,
            maximum=100,
            default=50,
            step=1,
            tooltip="Number of initial samples to train the GP.",
        ),
        
         AlgorithmParameter(
            name="warmup_steps",
            label="warmup_steps",
            minimum=64,
            maximum=512,
            default=100,
            step=10,
            tooltip="Number of warmup steps for HMC.",
        ),
         AlgorithmParameter(
            name="thinning",
            label="thinning",
            minimum=4,
            maximum=20,
            default=16,
            step=1,
            tooltip="Thinning factor for HMC samples.",
        ),
        AlgorithmParameter(
            name="user_seed",
            label="user_seed",
            minimum=-1,
            maximum=42,
            default=-1,
            step=1,
            tooltip="User defined seed for initial sampling. If -1, do random selection.",
        ),
    ],
    spec_cls=SAASBayesianOptimizerOptimizeAlgorithmSpec,
    optimizer_cls=SAASBayesianOptimizer,
)

__all__ = [
    "SAASBayesianOptimizerOptimizeAlgorithmSpec",
    "SAASBayesianOptimizer",
]
