import gpytorch
import torch
import tqdm
import numpy as np

def MLL(gp, x, y):
    N = len(x)
    projected_x = gp.feature_extractor(x)
    projected_x = gp.scale_to_bounds(projected_x)
    covar_matrix = gp.covar_module(projected_x,projected_x).evaluate()
    covar_matrix = covar_matrix + gp.likelihood.noise * torch.eye(N).to(x.device)
    log_mll = - 0.5 * (y.T @ torch.inverse(covar_matrix)) @ y 
    log_mll += - 0.5 * torch.logdet(covar_matrix)
    log_mll += - 0.5 * N * np.log(2 * np.pi)

    return log_mll

def CondtionalMLL(gp, x, y, xm, ym):
    return MLL(gp, x, y) - MLL(gp, xm, ym)

def VariationalConditionalMLL(gp, var_mll, x, y, xm, ym):
    return var_mll(gp(x), y) - var_mll(gp(xm), ym)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, embed_dim, widths=[1000, 500]):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear0', torch.nn.Linear(data_dim, widths[0]))
        self.add_module('relu0', torch.nn.ReLU())
        for lyr in range(1, len(widths)):
            self.add_module('linear' + str(lyr), 
                            torch.nn.Linear(widths[lyr-1], widths[lyr]))
            self.add_module('relu' + str(lyr), torch.nn.ReLU())

        self.add_module('linear' + str(len(widths)+1), 
                        torch.nn.Linear(widths[-1], embed_dim))
        
class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, feature_extractor,
                     embed_dim, grid=False, scale=True):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            
            covar = gpytorch.kernels.RBFKernel(ard_num_dims=embed_dim)
            if grid:
                grid_size = gpytorch.utils.grid.choose_grid_size(train_x, 0.9)
                covar = gpytorch.kernels.GridInterpolationKernel(
                        covar, num_dims=embed_dim, grid_size=grid_size)
            if scale:
                covar = gpytorch.kernels.ScaleKernel(covar)

            self.covar_module = covar
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
         
        
def TrainModel(model, likelihood, train_x, train_y, 
               losstype="cmll", m=5, train_iters=400, printing=False,
              lr=0.01):

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
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    iterator = tqdm.tqdm(range(train_iters))
    losses = torch.zeros(train_iters)
    for i in iterator:
        optimizer.zero_grad()

        if losstype=='cmll':
            order = torch.randperm(train_x.shape[0])        
            xm = train_x[order[:m]]
            xstar = train_x[order[m:]]

            ym = train_y[order[:m]]
            ystar = train_y[order[m:]]
            loss = -CondtionalMLL(model, train_x, train_y, xm, ym)
        
        else:
            output = model(train_x)
            loss = -mll(output, train_y)
        loss.backward()
        losses[i] = loss.item()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        
    return losses

