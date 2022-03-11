import bayesian_benchmarks as bb
from bayesian_benchmarks.data import regression_datasets, get_regression_data
import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import seaborn as sns
import argparse

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.models import SingleTaskVariationalGP

def get_mll(gp, x, y):
    N = len(x)
    x = gp.feature_extractor(x)
    x = gp.scale_to_bounds(x)
    covar_matrix = gp.covar_module(x,x).evaluate()
    covar_matrix += gp.likelihood.noise * torch.eye(N).to(x.device)
    log_mll = - 0.5 * (y.T @ torch.inverse(covar_matrix)) @ y 
    log_mll += - 0.5 * torch.logdet(covar_matrix)
    log_mll += - 0.5 * N * np.log(2 * np.pi)

    return log_mll

def CondtionalMLL(gp, x, y, xm, ym):
    return get_mll(gp, x, y) - get_mll(gp, xm, ym)

def RMSE(preds, targets):
    return (preds.squeeze().cpu() - targets.squeeze().cpu()).pow(2).mean().pow(0.5)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear3', torch.nn.Linear(data_dim, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, feature_extractor):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        
def main(args):
    
    data = get_regression_data(args.dataset)
    train_x = torch.FloatTensor(data.X_train)[:args.ntrain]
    train_y = torch.FloatTensor(data.Y_train).squeeze()[:args.ntrain]
    test_x = torch.FloatTensor(data.X_test)
    test_y = torch.FloatTensor(data.Y_test).squeeze()
    data_dim = train_x.size(-1)
    m = int(args.m * train_x.shape[0])
    
    
    rmse = torch.zeros(args.ntrial)
    for trl in range(args.ntrial):

        feature_extractor = LargeFeatureExtractor(data_dim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood,
                                  feature_extractor)

        if torch.cuda.is_available():
            use_cuda=True
            model = model.cuda()
            likelihood = likelihood.cuda()
            train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

        training_iterations = 100

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        def train(losstype='cmll'):
            iterator = tqdm.tqdm(range(training_iterations))
            for i in iterator:
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                output = model(train_x)
                # Calc loss and backprop derivatives
                if losstype=='mll':
                    loss = -mll(output, train_y)
                if losstype=='cmll':
                    order = torch.randperm(train_x.shape[0])        
                    xm = train_x[order[:m]]
                    xstar = train_x[order[m:]]

                    ym = train_y[order[:m]]
                    ystar = train_y[order[m:]]
                    loss = -CondtionalMLL(model, train_x, train_y, xm, ym)

                loss.backward()
                iterator.set_postfix(loss=loss.item())
                optimizer.step()

        train(losstype=args.losstype)
        model.eval();
        test_preds = model(test_x).mean
        rmse[trl] = RMSE(test_preds, test_y)
        
    fpath = "./saved-outputs/"
    fname = "exactdkl" + args.dataset + "_ntrain" + str(args.ntrain) + "_" + args.losstype
    if args.losstype == "cmll":
        fname += "_" + str(args.m) + "m"
    fname += ".pt"
    
    torch.save(rmse, fpath + fname)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        '--losstype',
        type=str,
        default='mll',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mll',
    )
    parser.add_argument(
        '--ntrain',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--ntrial',
        type=int,
        default=10,
    )

    args = parser.parse_args()

    main(args)