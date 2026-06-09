from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import traceback

# standard libraries 
import numpy as np
import numpy.typing as npt
from gpytorch.constraints import Interval
from scipy.stats.qmc import LatinHypercube, scale

# import ML libraries
import torch
from random import randint
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.optim import optimize_acqf
from botorch.exceptions.errors import ModelFittingError
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.transforms import Standardize

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qUpperConfidenceBound,
    qProbabilityOfImprovement
    )
from botorch.acquisition.logei import (
    qLogExpectedImprovement, 
    qLogNoisyExpectedImprovement
    )
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

from gpytorch.priors import LogNormalPrior
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import ScalarizedPosteriorTransform 
from botorch.acquisition.utils import prune_inferior_points



from .base import (
    AlgorithmParameter,
    OptimizeAlgorithmSpec,
    Optimizer,
    register_algorithm,
)



@dataclass
class BayesianOptimizerOptimizeAlgorithmSpec(OptimizeAlgorithmSpec):
    xatol: float = 1e-3
    fatol: float = 1e-3
    n_init: int = 10
    user_seed: int = -1

class BayesianOptimizer(Optimizer):
    """
    Sequential ask/tell Bayesian Optimization implementation.

    Bayesian Optimization is a global optimization technique designed
    for functions that are expensive, noisy, or lack gradient information.
    Builds a probabilistic surrogate model (Gaussian Process) of the objective
    function, and uses an acquisition function maximization to decide the next
    sampling point. This balances exploration and exploitation, and is suitable
    for black-box optimization procedures.
    """

    def __init__(
        self,
        initial: tuple[float, ...],         # initial samples from GUI
        lower_bounds: tuple[float, ...],    # lower bounds of active params
        upper_bounds: tuple[float, ...],    # upper bounds of active params
        xatol: float,                       # tolerance for change in x
        fatol: float,                       # tolerance for change in f(x)
        n_init: int,                        # no.of initial samples to generate
        user_seed: int,                     # user defined seed
        # robust_mode: bool,                # use simple/robust optimizer
    ):

        # select the acquisition function type
        self.acq_func_type = "logEI"
        
        # simulation parameters
        self.user_seed = user_seed                  # initial seed for reproducibility
        self.iter_idx = 0                           # initial iteration index
        self.sample_idx = 0                         # sample index
        self._termination_reason: str | None = None # termination reason
        self._num_asked = 0                         # no.of points asked for
        self.mc_samples = 500                       # no.of Monte Carlo samples (256 - 1024)

        # experimental info
        self.n_params = len(initial)                # no.of active parameters
        self.active_bounds_lower = np.array(lower_bounds, dtype=float)
        self.active_bounds_upper = np.array(upper_bounds, dtype=float)
        self._span = self.active_bounds_upper - self.active_bounds_lower

        # create physical and unit bounds
        self.phys_bounds = self.make_physical_bounds()
        self.unit_bounds = self.make_unit_bounds()

        self._xatol = xatol
        self._fatol = fatol

        # generate initial parameter input using LHS sampling
        if  self.n_params >= 3 : 
            self.n_init = int(n_init * np.sqrt(self.n_params))
        else: 
            self.n_init = self.n_params * n_init

        self.x = self.LHS_sampler()

        # initialize training dataset
        self.init_x = torch.from_numpy(self.normalize(self.x)).double()
        assert self.init_x.ndim == 2, f"init_x must be 2D, got {self.init_x.shape}"

        self.init_y = torch.empty((0, 1), dtype=torch.double)
        self.init_y_var = torch.empty((0, 1), dtype=torch.double)
        self.best_init_y = float("inf")

        # for tracking performance 
        self.n_test = 20
        self.fixed_values = None
        self.output_idx = 0
        self.z_score = 1.96         # 95% confidence interval
        self.time_accumulated = 0.0 # run time 
        
        # self.noise_robust_mode = robust_mode   # change to True for real experiment
        self.noise_robust_mode = False           # default 
    
    def ask(self) -> tuple[float, ...]:
        """
        Suggests the next candidate point to sample and
        updates the training data with the candidate
        points found until the termination condition /
        max iterations are reached.
        Returns ``None`` if no more are needed.

        Outputs:
            - x_point : ((1,d) array) active parameters and their values
        """
        if self._termination_reason is not None:
            # don't return points to evaluate
            return None

        if self.iter_idx == 0:
            # initial sampling
            if self.sample_idx < self.n_init:
                # get point from initial LHS samples
                x_point = self.x[self.sample_idx, :]
                # increment sample index and calls to ask()
                self.sample_idx += 1
                self._num_asked += 1
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
            new_candidates = new_candidates.reshape(1, self.n_params)
            x_point = self.denormalize(new_candidates)
            # append the data set with the new candidate
            self.init_x = torch.cat((self.init_x, new_candidates))
            # update the iteration
            self.iter_idx += 1
            self._num_asked += 1
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
                                 torch.tensor(-value, dtype=torch.double).reshape(1, 1)
                                 ))     # update y-values
        
        # add small noise floor for numerical stability in GP fitting
        if  self.noise_robust_mode is True: 
            noise_floor = 1e-3
        else: 
            noise_floor = 0.0
        obs_var = max(std_dev**2, noise_floor)
        self.init_y_var = torch.cat((self.init_y_var, 
                                     torch.tensor(obs_var, dtype=torch.double).reshape(1, 1)
                                     )) # update y-variances
         # obtain the best point so far
        self.best_init_y = self.init_y.max().item()
        
        if self.init_y.numel() > self.n_init:  # only check after enough data
            # check for convergence 
            self._maybe_terminate()

        return None

    def is_done(self) -> bool:
        """
        Return whether the optimizer has terminated."""
        return self._termination_reason is not None

    def best(self) -> tuple[tuple[float, ...], float] | None:
        """
        Returns the best point/value pair seen so far,
        if any evaluations completed.

        Outputs:
            - best_x : best parameters/point
            - best_y : best objective function value
        """
        # calculate the total no.of elements in a tensor
        if self.init_y.numel() == 0: # PyTorch method
            # no evaluations have been made yet
            return None

        # select the single best observation
        best_idx = int(torch.argmax(self.init_y).item())

        # get model parameters at this index
        best_x_norm = self.init_x[best_idx]

        # convert parameters back to physical units
        best_x = self.denormalize(best_x_norm)
        best_y = -self.best_init_y

        return best_x, best_y

    def best_std(self) -> float | None:
        """
        Return the measured standard deviation for the current best point.
        """
        if self.init_y.numel() == 0:
            return None

        best_idx = int(torch.argmax(self.init_y).item())
        best_var = self.init_y_var[best_idx].item()
        return float(np.sqrt(np.maximum(best_var, 0.0)))

    def termination_reason(self) -> str | None:
        return self._termination_reason

    def _maybe_terminate(self) -> None:
        """
        Terminates the optimization if either
        convergence or max evaluations reached.
        """
        
        if self.iter_idx == 0:
            return  # don't check for termination during initial sampling
        
        # find where the best values occurred
        best_x, best_y = self.best() # in physical units

        # denormalize init_x and flatten init_y
        all_x_phys = np.array([self.denormalize(self.init_x[i]) for i in range(len(self.init_x))])
        all_y = - self.init_y.numpy().flatten() # negate back to original values
        
        # get recent 5 points and values for convergence check
        recent_x = all_x_phys[-5:]
        recent_y = all_y[-5:]

        # find max change in x
        max_x_delta = max(
            np.max(np.abs((point - best_x) / self._span)) for point in recent_x
        )
        
        # find max change in y
        max_f_delta = max(abs(value - best_y) for value in recent_y)
    
        # get best standard deviation
        best_y_std = self.best_std()
        self.dynamic_fatol = 2 * best_y_std  # dynamic fatol based on current noise level

        if self.noise_robust_mode is True: 
            fatol = self.dynamic_fatol
        else:
            fatol = self._fatol
      
        # check if both changes are within the specified tolerances
        if max_x_delta <= self._xatol and max_f_delta <= fatol:
        # if max_x_delta <= self._xatol and max_f_delta <= self._fatol:
            self._termination_reason = "converged"
    

    def get_kernel(self):
        """
        Defines a Matern kernel within a ScaleKernel wrapper.
        - Uses an ARD to give each dimension gets its own lengthscale
        - Constrains the lengthscales for numerical stability
        
        Outputs:
            - covar_module : learned output variance
        """

        if self.noise_robust_mode is True:
            matern_kernel = MaternKernel(nu=2.5,
                                    ard_num_dims=self.n_params,                 # individual lengthscales 
                                    lengthscale_constraint=GreaterThan(1e-3),   # constrained lengthscales.
                                    lengthscale_prior=LogNormalPrior(-2.0, 1.0) # prior
                                    )
        else: 
            matern_kernel = MaternKernel(nu=2.5, 
                                    ard_num_dims=self.n_params,                
                                    )
        # obtain output variance
        covar_module = ScaleKernel(matern_kernel)

        return covar_module

    def build_surrogate_model(self):
        """
        Builds the surrogate model (Heteroscedastic GP regressor)
        for the BO loop. Fits a second GP to model how noise varies with x.

        Outputs : None. Updates the  model and mll
            - model : the surrogate model (GP) fitted to the initial data
            - mll : the marginal log likelihood of the fitted model
        """
        # obtain Matern Kernel
        covar_module = self.get_kernel()

        # deduplicate all three tensors together before fitting:
        self.train_x, self.train_y, self.train_y_var = self.deduplicate_training_data()
        
        # compute custom likelihood 
        likelihood = FixedNoiseGaussianLikelihood(
                    noise=self.train_y_var.squeeze(),
                    learn_additional_noise=False  # disable additional noise
        )

        # build GP model
        if self.noise_robust_mode is True: 
            # create robust model
            self.model = SingleTaskGP(
                train_X=self.train_x,
                train_Y=self.train_y,
                train_Yvar=self.train_y_var,
                covar_module=covar_module,
                outcome_transform=Standardize(m=1),  # m = 1 for single outputs
                likelihood=likelihood,
            )
        else: 
            # create simple model
            self.model = SingleTaskGP(
                train_X=self.train_x,
                train_Y=self.train_y,
                covar_module=covar_module,
                outcome_transform=Standardize(m=1),  # m = 1 for single outputs
            )

        # define the marginal log likelihood
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def obj_callable(self, Z: torch.Tensor, X: Optional[torch.Tensor] = None):
        return Z[..., 0]

    def constraint_callable(self, Z: torch.Tensor):
        return Z[..., 1]



    def get_acquisition_function(self):
        """
        Creates the acquisition function (AF) for the BO loop.
        Allows user to select from different AF's (EI, logEI, UCB, PI).

        Outputs:
            - acq_func : the acquisition function
        """

        # standardize the best observed value 
        best_f_standardized = (self.best_init_y - self.model.outcome_transform.means.item()) \
                       / self.model.outcome_transform.stdvs.item()
        # obtain the objective and constraint callables 
        objective = GenericMCObjective(objective=self.obj_callable)

        # define quasi-MC N(0,1) sampler that uses Sobol sequences
        qmc_sampler = SobolQMCNormalSampler(
                        sample_shape=torch.Size([self.mc_samples]))
        
        # transform a model's posterior
        weights = torch.tensor([1.0])  # add more weights for multi-output objective 
        posterior_transform = ScalarizedPosteriorTransform(weights, offset=0.0)
        
        # add manual pruning of baseline for logNEI
        X_baseline_pruned = prune_inferior_points(  
                model=self.model,  
                X=self.train_x,  
                objective=objective,  
                posterior_transform=posterior_transform,  
        )  

        # initialize class for the different AF's
        if self.acq_func_type == "EI":
                acq_func = qExpectedImprovement(
                model=self.model, 
                best_f=best_f_standardized
        )
        elif self.acq_func_type == "logEI" and self.noise_robust_mode is True:
                acq_func = qLogExpectedImprovement(
                model=self.model, 
                best_f=best_f_standardized,
                sampler=qmc_sampler,
                # objective=objective,
                # constraints=[self.constraint_callable],
        )
        elif self.acq_func_type == "logEI" and self.noise_robust_mode is False:
                acq_func = qLogExpectedImprovement(
                model=self.model, 
                best_f=best_f_standardized,
        )
            
        elif self.acq_func_type == "logNEI":
                acq_func = qLogNoisyExpectedImprovement(
                model=self.model, 
                X_baseline=X_baseline_pruned,
                prune_baseline=True,  # remove redundant baseline points
                sampler=qmc_sampler,
                cache_root=True,      # Caches Cholesky decomposition 
                # objective=objective,
                # constraints=[self.constraint_callable],
        )
        elif self.acq_func_type == "KG":
                acq_func = qKnowledgeGradient(
                model=self.model,
                num_fantasies=self.mc_samples)
        
        elif self.acq_func_type == "UCB":
                acq_func = qUpperConfidenceBound(
                model=self.model, 
                beta=0.05
        )
        elif self.acq_func_type == "PI":
                acq_func = qProbabilityOfImprovement(
                model=self.model, 
                best_f=best_f_standardized
        )
        else:
            raise ValueError("Invalid acquisition function type")
        
        
        return acq_func

    def get_next_points(self):
        """
        Obtains the next point(s) to sample in the BO loop.
        It does the following:
            - Builds and trains the surrogate model (GP)
            - Creates the Acquistion Function (AF)
            - Find candidates for the next point to sample by optimizing the AF.

        Outputs :
            - candidates: candidate(s) found while using a given AF
        """
        # create the GP models
        self.build_surrogate_model()

        try: # Attempt to fit the model for hyperparameter optimization

            fit_gpytorch_mll(self.mll)  # uses L-BFGS-B optimizer
        
        except ModelFittingError:
            print("L-BFGS-B failed, falling back to Adam...")
            
            try: 
                fit_gpytorch_mll_torch(self.mll, step_limit=300)             
            
            except Exception as e : 
                print(f"Adam optimizer also failed. {e}")
                return False, None
            
            
        except ValueError as ve:
            # handle common data-related errors
            print(f"ValueError during fitting: {ve}")
            return False, None

        except Exception as e:
            # catch any other unexpected errors
            print(f"Unexpected error during model fitting: {e}")
            traceback.print_exc()
            return False, None
        
        # create the acquisition function
        acq_func = self.get_acquisition_function()

        if self.noise_robust_mode is True:
            raw_samples = 500
        else: 
            raw_samples = 1024
        
        # find candidates ( assuming one optimizer is successful)
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.unit_bounds,
            q=1,                # no.of candidates to generate in the batch
            num_restarts=10,    # no.of starting points for multi-start optimization.
            raw_samples=raw_samples,    # no.of samples for initial condition generation (256 - 1024)
            options={"batch_limit": 5, "maxiter": 200},
        )

        return True, candidates

    
    def make_physical_bounds(self):
        """
        Function returns the physical bounds for
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
        """
        Function returns the unit bounds that will be used
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
        """
        Function to generate initial parameters for BO
            using Latin Hypercube Sampling (LHS).

        Outputs:
            - init_params : (n, d) array of initial parameter vectors
        """

        #  generate LHS samples
        seed = self.user_seed if self.user_seed >= 0 else randint(0, 42)
        sampler = LatinHypercube(
            d=self.n_params, seed=seed
        )  # LHS sampler with fixed/None seed
        
        unit_samples = sampler.random(n=self.n_init
        )  # shape (n_init, d), values in [0, 1]

        #  scale from [0,1] to the actual parameter ranges
        init_params = scale(
            unit_samples,
            l_bounds=self.active_bounds_lower,
            u_bounds=self.active_bounds_upper,
        )

        return init_params

    def normalize(self, x: npt.NDArray[np.float64]):
        """
        Normalizes the input x to be between [0,1]
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
        """
        Converts the normalized parameters to physical values.
                x = x_norm * (upper - lower) + lower

        Inputs:
            - x_norm: ((1,d) tensor) input parameters between [0,1]
            - phys_bounds : ((2,d) tensor) physical lower and upper bounds

        Outputs:
            - x : ((1,d) array) input parameters in physical units
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
    
    def deduplicate_training_data(self):
        """Remove duplicate x points by averaging their y values and variances.
        Outputs: 
            - unique_x : ((n_unique, d) tensor) unique x points
            - unique_y : ((n_unique, 1) tensor) averaged y values 
            - unique_y_var : ((n_unique, 1) tensor) averaged y variances
        """
        unique_x, inverse_idx = torch.unique(self.init_x, dim=0, 
                                             return_inverse=True)
        
        n_unique = unique_x.shape[0]
        unique_y     = torch.zeros(n_unique, self.init_y.shape[1],
                                   dtype=torch.float64)
        unique_y_var = torch.zeros(n_unique, self.init_y_var.shape[1], 
                                   dtype=torch.float64)
        counts       = torch.zeros(n_unique, dtype=torch.float64)

        for i, idx in enumerate(inverse_idx):
            unique_y[idx]     += self.init_y[i]
            unique_y_var[idx] += self.init_y_var[i]
            counts[idx]       += 1

        # average y across duplicates
        unique_y     /= counts.unsqueeze(1)
        
        # variance of the mean = sum(var_i) / n²
        unique_y_var /= (counts ** 2).unsqueeze(1)

        return unique_x, unique_y, unique_y_var
    
    def get_sim_time(self): 
        """ Returns the total time taken for the optimization 
        process so far. 
        
        Outputs: 
            - total_time : total time taken for optimization so far
        """
        return self.time_accumulated

register_algorithm(
    "bayesian",
    display_name="Bayesian optimization",
    description="Bayesian optimization optimizer",
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
            minimum=10,
            maximum=100,
            default=50,
            step=1,
            tooltip="Number of initial samples to train the GP.",
        ),
        AlgorithmParameter(
            name="user_seed",
            label="user_seed",
            minimum=-1,
            maximum=42,
            default=-1,
            step=1,
            tooltip="User defined seed for initial sampling. If -1, do random selection.",
        )
    ],
    spec_cls=BayesianOptimizerOptimizeAlgorithmSpec,
    optimizer_cls=BayesianOptimizer,
)

__all__ = [
    "BayesianOptimizerOptimizeAlgorithmSpec",
    "BayesianOptimizer",
]