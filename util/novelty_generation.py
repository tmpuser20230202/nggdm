import torch

class MirrorDescentExponentialGradientOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for param in group['params']:
                grad = param.grad
                #param -= group['lr'] * grad
                param *= torch.exp(-group['lr'] * grad) / torch.sum(param * torch.exp(-group['lr'] * grad))
        return loss

class NewClusterAllocatorMean(torch.nn.Module):
    def __init__(self, model_best, i):
        super().__init__()
        self.model_best_ = model_best
        self.K = len(self.model_best_.weights_)
        self.D = self.model_best_.means_.shape[1]

        self.existing_dists_ = [torch.distributions.MultivariateNormal(torch.tensor(means), scale_tril=torch.tril(torch.tensor(cov)))
                                for means, cov in zip(self.model_best_.means_, self.model_best_.covariances_)]

        self.mu_ = torch.nn.Parameter(torch.tensor(self.model_best_.means_[i]))

    def forward(self):
        return self.mu_

class NewClusterAllocatorCovariance(torch.nn.Module):
    def __init__(self, model_best, i):
        super().__init__()
        self.model_best_ = model_best
        self.K = len(self.model_best_.weights_)
        self.D = self.model_best_.means_.shape[1]

        self.existing_dists_ = [torch.distributions.MultivariateNormal(torch.tensor(means), scale_tril=torch.tril(torch.tensor(cov)))
                                for means, cov in zip(self.model_best_.means_, self.model_best_.covariances_)]

        self.covariances_ = torch.nn.Parameter(torch.tensor(self.model_best_.covariances_[i]))

    def forward(self):
        return self.covariances_

class VonNeumannOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.lr = {}
        #self.d2 = {}

        for group in self.param_groups:
            for param in group['params']:
                self.lr[param] = torch.ones_like(param.data) * lr
                #self.d2[param] = torch.ones_like(param.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                d1 = param.grad.data
                #d1 = torch.sign(param.grad.data)
                #t = torch.where(torch.sign(self.d2[param]) == d1, 1.0, 2.0)

                lr = self.lr[param]
                param.data += -d1 * lr
                param.data = ((param.data + param.data.T)/2)
                _, s, V = torch.linalg.svd(param.data)
                s[s < 0] = 0
                param.data = V.T @ torch.diag(s) @ V
                
        return loss

def kl_divergence_gmm(weights, weights_new, eps=1e-2):
    phi = torch.rand(K, K+1)
    phi /= torch.sum(phi, axis=1, keepdims=True)
    psi = torch.rand(K, K+1)
    phi /= torch.sum(psi, axis=1, keepdims=True)

    wkl = weights.reshape(-1, 1) * torch.exp(-kl_pairs)
    
    while True:
        phi_new = wkl * psi/ torch.sum(psi * torch.exp(-kl_pairs), axis=1, keepdims=True)
        psi_new = phi_new * weights_new / torch.sum(phi_new, axis=1, keepdims=True)
        
        if (torch.linalg.norm(phi_new - phi) <= eps) | (torch.linalg.norm(psi_new - psi) <= eps):
            break

        phi = phi_new
        psi = psi_new

    weights = phi.sum(axis=0)
    weights_new = psi.sum(axis=0)

    kl = torch.sum(phi * torch.log(psi/phi))
    return kl


def determine_new_cluster_by_novlety_condition(gmm_model_selection):
    i_cluster = np.random.randint(0, gmm_model_selection.model_best_.means_.shape[0])
    
    alloc_mean = NewClusterAllocatorMean(gmm_model_selection.model_best_, i_cluster)
    alloc_cov = NewClusterAllocatorCovariance(gmm_model_selection.model_best_, i_cluster)
    optimizer_mean = torch.optim.Adam(alloc_mean.parameters(), lr=0.01)
    optimizer_cov = VonNeumannOptimizer(alloc_cov.parameters(), lr=0.01)
    
    kl_list = []
    for p in alloc_mean.existing_dists_:
        for q in alloc_mean.existing_dists_:
            kl_list.append(torch.distributions.kl_divergence(p, q).mean().item())
    
    n_epoch = 500
    thr_kl = max(kl_list)
    loss_list = []
    
    keep_update = True
    while keep_update:
        mean = alloc_mean()
        cov = alloc_cov()
        p = torch.distributions.MultivariateNormal(mean, cov)
        
        for q in alloc_mean.existing_dists_:
            optimizer_mean.zero_grad()
            optimizer_cov.zero_grad()
            loss = -torch.distributions.kl_divergence(p, q).mean()
            loss_list.append(-loss.item())
            loss.backward(retain_graph=True)
            optimizer_mean.step()
            optimizer_cov.step()
        print(min(loss_list))
        keep_update = (min(loss_list) <= thr_kl)
        loss_list.clear()

   return alloc_mean, alloc_cov

def determine_weights_by_reliablity_condition(gmm_model_selection, alloc_mean, alloc_cov):
    kl_pairs = torch.tensor([[torch.distributions.kl_divergence(d1, d2) for d2 in alloc_mean.existing_dists_ + 
                             [torch.distributions.MultivariateNormal(alloc_mean.mu_.detach(), alloc_cov.covariances_.detach())]] for d1 in alloc_mean.existing_dists_])

    K = len(gmm_model_selection.model_best_.means_)

    weights = torch.tensor(gmm_model_selection.model_best_.weights_)
    weights_new = nn.Parameter(torch.ones(K+1) / (K+1))
    optimizer_md = MirrorDescentExponentialGradientOptimizer([weights_new], lr=0.01)

    n_epoch = 100
    for epoch in range(n_epoch):
        optimizer_md.zero_grad()
        loss = kl_divergence_gmm(weights, weights_new)
        loss_list.append(loss)
        loss.backward()
        optimizer_md.step()

    return weigths_new