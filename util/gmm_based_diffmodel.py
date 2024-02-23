sys.path.append('../repos/gmcd')
sys.path.append('../repos/gmcd/src')
sys.path.append('../repos/gmcd/src/datasets')
sys.path.append('../repos/gmcd/src/experiment')
sys.path.append('../repos/gmcd/src/optimizer')
sys.path.append('../repos/gmcd/src/mutils')

sys.path.append('../repos/gmcd/src/model')
from artransformer_diff import ArTransformerDiffusion  # src/model
from gaussian_diff import GaussianDiffusion # src/model
from linear_transformer import DenoisingTransformer # src/model

class NGGDM(nn.Module):
    def __init__(self, run_config, dataset_class, K, encoding_dim, name='NGGDM', figure_path=''):
        super().__init__()
        self.figure_path = figure_path
        self.name = name
        self.run_config = run_config
        self.dataset_class = dataset_class
        #self.S = run_config.batch_size
        self.S = run_config.S
        self.K = K
        self.encoding_dim = encoding_dim
        T = int(self.run_config.T)

        #self.gmm = GMMModelSelection(K_min=2, K_max=10, random_state=0, mode='GMM_DNML')

        self.z_0_layer = MyArTransformerDiffusion(self.run_config, self.S, self.K, 
                                                  self.encoding_dim, T, figure_path=figure_path)
    
    def forward(self, z, ldj=None, reverse=False, **kwargs):
        #self.gmm.fit(z.cpu().detach().numpy())
        #categ = gmm_model_selection.predict(z.cpu().detach().numpy())
        #n_cluster = len(gmm_model_selection.model_best_.means_)
        print(z.shape)
        #z = z.reshape(z.shape[0], 1, z.shape[1])
        if not reverse:
            z, fcdm_ldj = self.z_0_layer(
                z, reverse=reverse, x_cat=categ, **kwargs)
            ldj = fcdm_ldj
        if reverse:
            z, fcdm_ldj = self.z_0_layer(
                z, reverse=True, x_cat=None, **kwargs)

        ldj = fcdm_ldj
        return ldj

    def need_data_init(self):
        return False

from run_config import RunConfig

class PlanarRunConfig(RunConfig):
    def __init__(self,
                 dataset,
                 S, 
                 K) -> None:
        super().__init__()
        self.S = 16  # Number of elements in the sets.
        self.K = K
       
        self.eval_freq = 500
        self.dataset = dataset

        self.T = 10
        self.diffusion_steps = self.T
        self.batch_size = 16
        self.encoding_dim = 2
        self.max_iterations = 1000
        self.transformer_dim = 64
        self.input_dp_rate = 0.2
        self.transformer_heads = 8
        self.transformer_depth = 2
        self.transformer_blocks = 1
        self.transformer_local_heads = 4
        self.transformer_local_size = 64

import torch.distributions as D
import torch.nn.functional as F

from linear_transformer import DenoisingTransformer
from diff_utils import LossType, ModelMeanType, ModelVarType, extract_into_tensor, get_named_beta_schedule, normal_kl

class MyArTransformerDiffusion(GaussianDiffusion):
    def __init__(
            self, diffusion_params,
            S, 
            K, 
            latent_dim,
            T,
            #extActFixed,
            #posterior_sample_fun=None,
            figure_path=""):
        super().__init__(
            sequence_length=S,
            latent_dim=latent_dim,
            T=T,
            denoise_fn=None,
            betas=get_named_beta_schedule('cosine', T),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_LARGE,
            rescale_timesteps=False,  figure_path=figure_path)

        #self.extActFixed = extActFixed
        #K = extActFixed.fixed_embedding.num_embeddings
        self.loss_type = LossType.NOISY_GUIDED_SHARP
        self.K = K
        self.S = S
        self.T = T
        self.d = latent_dim
        self.alpha = diffusion_params.alpha
        if self.alpha is None:
            self.loss_type = LossType.NLL
        #self.posterior_sample = posterior_sample_fun  # p(x|z, t)
        #self.nll_loss = th.nn.BCEWithLogitsLoss()
        self.nll_loss = torch.nn.BCEWithLogitsLoss()
        try:
            self.corrected_var = diffusion_params.corrected_var
        except Exception as e:
            self.corrected_var = False

        self.mixture_weights = DenoisingTransformer(
            K=K, S=S, latent_dim=latent_dim, diffusion_params=diffusion_params)

        # other coeff needed for the posterior N(z_t-1|z_t,x)
        self.sigma_tilde = (self.betas *
                            (1.0 - self.alphas_cumprod_prev)
                            / (1.0 - self.alphas_cumprod)
                            )

    def need_data_init(self):
        return []

    def forward(self, z, ldj=None, reverse=False, x_cat=None, **kwargs):
        batch_size, set_size, hidden_dim = z.size(0), z.size(1), z.size(2)
        if not reverse:

            ldj = z.new_zeros(batch_size, )
            print('z.shape:', z.shape)
            print('self.d_in:', self.d_in)
            z = z.reshape((batch_size, self.d_in))
            device = z.device
            if self.training:
                #t = th.randint(0, self.T, (batch_size, ),
                #               device=device).long()
                t = torch.randint(0, self.T, (batch_size, ),
                                  device=device).long()
                ldj = -self.training_losses(z, t, x_cat, **kwargs)['loss']
                z = z.reshape((batch_size, set_size, hidden_dim))
                return z, ldj
            else:
                ldj = self.log_likelihood(z)
                return z, ldj

        else:
            ldj = self.nll(z)
            return z, ldj

    def log_likelihood(self, z_0):
        b = z_0.size(0)
        device = z_0.device
        log_likelihood = 0

        for t in range(0, self.num_timesteps):
            t_array = (th.ones(b, device=device) * t).long()
            sampled_z_t = self.q_sample(z_0, t_array)
            kl_approx = self.compute_Lt(
                z_0=z_0,
                z_t=sampled_z_t,
                t=t_array)

            log_likelihood += kl_approx

        qt_mean, _, qt_log_variance = self.q_mean_variance(z_0, t_array)
        # THIS SHOULD BE SUPER SMALL
        kl_prior = -normal_kl(qt_mean, qt_log_variance, mean2=0.0, logvar2=0.0)
        #kl_prior = th.sum(kl_prior, dim=1)
        kl_prior = torch.sum(kl_prior, dim=1)

        log_likelihood += kl_prior

        return log_likelihood

    def compute_Lt(self, z_0, z_t, t):

        z_t = z_t.reshape(-1, self.S, self.d)
        logits_output = self.mixture_weights(t, z_t)  # get p_theta
        transformer_probs = logits_to_probs(logits_output)

        dist = self.get_zt_given(z_t, t)

        w = self.get_w(dist, z_0, z_t, t)

        terms = transformer_probs * w  # all p_theta * w
        #approx_kl = th.log(th.sum(terms, dim=2))  # take log of sum accross k
        approx_kl = torch.log(th.sum(terms, dim=2))  # take log of sum accross k
        #approx_kl = th.sum(approx_kl, dim=1)
        approx_kl = torch.sum(approx_kl, dim=1)

        z_0 = z_0.reshape(-1, self.S, self.d)
        stacked_z_0 = z_0[:, :, None, :].repeat(
            1,  1, self.K, 1)  # repeat along K
        log_pdf_z0 = dist.log_prob(stacked_z_0)
        terms = transformer_probs * th.exp(log_pdf_z0)
        #decoder_nll = -th.log(th.sum(terms, dim=2))
        decoder_nll = -torch.log(th.sum(terms, dim=2))
        #decoder_nll = th.sum(decoder_nll, dim=1)
        decoder_nll = torch.sum(decoder_nll, dim=1)

        #mask = (t == th.zeros_like(t)).float()
        mask = (t == torch.zeros_like(t)).float()
        # replace nan to zero as they should not b added in the first place
        #loss = mask * th.nan_to_num(decoder_nll) + (1. - mask) * approx_kl
        loss = mask * torch.nan_to_num(decoder_nll) + (1. - mask) * approx_kl

        return loss

    def get_w(self, dist, z_0, z_t, t):
        z_t = z_t.reshape(-1, self.S * self.d)
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=z_0, x_t=z_t, t=t
        )
        true_mean = true_mean.reshape(-1, self.S, self.d)
        true_log_variance_clipped = true_log_variance_clipped.reshape(
            -1, self.S, self.d)
        stacked_true_mean = true_mean[:, :, None, :].repeat(
            1,  1, self.K, 1)  # repeat along K
        stacked_true_log_variance_clipped = true_log_variance_clipped[:, :, None, :].repeat(
            1,  1, self.K, 1)  # repeat along S

        #kl = normal_kl(stacked_true_mean, stacked_true_log_variance_clipped,
        #               dist.mean, th.log(dist.variance))
        kl = normal_kl(stacked_true_mean, stacked_true_log_variance_clipped,
                       dist.mean, torch.log(dist.variance))
        #kl = th.sum(kl, dim=3)
        kl = torch.sum(kl, dim=3)
        #w = th.exp(-kl)
        w = torch.exp(-kl)
        return w

    def training_losses(self, z_0, t, x_cat, **kwargs):

        #noise = th.randn_like(z_0)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise=noise)
        z_t = z_t.reshape(-1, self.S, self.d)
        # to do, z_t from z_t-1
        check_fraction_same_x = False
        if self.loss_type in [LossType.NOISY_GUIDED_SHARP]:

            dist = self.get_zt_given(z_t, t)
            w = self.get_w(dist, z_0, z_t, t).view(-1, self.S)
            w = w.permute(1, 0)
            #norm_w = th.sum(w, dim=0).view(-1)  # sum over k
            norm_w = torch.sum(w, dim=0).view(-1)  # sum over k
            #p_w = th.div(w, norm_w).permute(1, 0).view(-1, self.S, self.K)
            p_w = torch.div(w, norm_w).permute(1, 0).view(-1, self.S, self.K)
            if self.loss_type == LossType.NOISY_GUIDED_SHARP:
                p_w_power = p_w**self.alpha
                p_w_power = p_w_power.reshape(-1, self.K)
                p_w_power = p_w_power.permute(1, 0)
                #norm_w = th.sum(p_w_power, dim=0).view(-1)
                norm_w = torch.sum(p_w_power, dim=0).view(-1)
                #p_w = th.div(p_w_power, norm_w).permute(
                #    1, 0).view(-1, self.S, self.K)
                p_w = torch.div(p_w_power, norm_w).permute(
                    1, 0).view(-1, self.S, self.K)

            c = D.categorical.Categorical(p_w)
            w = c.sample()
            #mask = (t == th.zeros_like(t)).int().view(-1, 1)
            mask = (t == torch.zeros_like(t)).int().view(-1, 1)
            # at 0 we take x as is
            w = mask * x_cat + (1 - mask) * w
        else:
            w = x_cat
        if check_fraction_same_x:
            self.check_fraction_same_x(w, x_cat, t)

       
       
        logits_output = self.mixture_weights(t, z_t)
        terms = {}
        transformer_probs = logits_to_probs(logits_output)
       

        logits_output_flat = logits_output.reshape(-1, self.K)
        w = x_cat
        w_flat = w.view(-1)
        #w_flat = F.one_hot(w_flat, num_classes=self.K).type(th.float32)
        w_flat = F.one_hot(w_flat, num_classes=self.K).type(torch.float32)
        neg_log_likelihood = F.binary_cross_entropy_with_logits(
            logits_output_flat, w_flat, reduction='none')
        #neg_log_likelihood = th.sum(
        #    neg_log_likelihood.view(-1, self.S, self.K), dim=2)
        neg_log_likelihood = torch.sum(
            neg_log_likelihood.view(-1, self.S, self.K), dim=2)
        #terms['loss'] = th.sum(neg_log_likelihood, dim=1)
        terms['loss'] = torch.sum(neg_log_likelihood, dim=1)

        return terms

    def inspect_mix_pis(self, mixing_logits, t, x_cat=None):
        name = "entropy_transformer.pdf"
        probs = logits_to_probs(mixing_logits)
        self.check_entropy(probs, t, name)

    #@th.no_grad()
    @torch.no_grad()
    def sample(self, num_samples,  watch_z_t=False):

        device = next(self.mixture_weights.parameters()).device
        shape = (num_samples, self.d_in)
        #z = th.randn(*shape, device=device)
        z = torch.randn(*shape, device=device)
        z_T = z
        indices = list(range(self.num_timesteps))[::-1]
        t_to_check = [int(self.num_timesteps/2)]
        return_dict = {}

        for i in indices:

            #t = th.tensor([i] * shape[0], device=device)
            t = torch.tensor([i] * shape[0], device=device)
            #with th.no_grad():
            with torch.no_grad():
                out = self.p_sample(z, t)
                z = out["sample"]
                pi = out["logits"]
                x_w = out["sampled_w"]

        return_dict['z_T'] = z_T
        return_dict.update(out)
        return return_dict

    def p_sample(self, z_t, t):  # denoising step
        # x \sim p(X|z^t)
        # z^t-1 \sim p(Z^t-1|X=x)
        z_t = z_t.reshape(-1, self.S, self.d)
        b = z_t.shape[0]

        logits = self.mixture_weights(t, z_t)  # B, S, K
        p_w_given_past = D.categorical.Categorical(logits=logits)
        w = p_w_given_past.sample()

        norm_of_w = self.get_zt_given(z_t, t, w)
        sample = norm_of_w.sample().type(th.float32)
        return {"sample": sample, "sampled_w": w, 'logits': logits}

    def info(self):
        return 'Task-cognizant Diffusion Model'

    # norm [b, s, k, d] or # norm [b, s, d]
    def get_zt_given(self,  z_t, t, x=None):
        b = z_t.shape[0]
        device = z_t.device
        mu_k, var = self.extActFixed.get_mean_var(device)
        var = var[0, 0]
        shape = (b, self.S, self.K, self.d)
        posterior_mean_coef1 = extract_into_tensor(self.posterior_mean_coef1,
                                                   t, (b, self.K, self.d))
        stacked_means = mu_k * posterior_mean_coef1  # coef * mu_k, [b,K,d]
        stacked_means = stacked_means[:, None, :, :].repeat(
            1,  self.S, 1, 1)  # repeat along S

        if x is not None:
            shape = (b, self.S, self.d)
            x = x.reshape(b*self.S)
            stacked_means = stacked_means.reshape(b*self.S, self.K, self.d)
            index = th.arange(start=0, end=b*self.S, dtype=th.long)
            stacked_means = stacked_means[index, x, :]
            stacked_means = stacked_means.reshape(
                b, self.S, self.d)  # only one normal per S

        posterior_mean_coef2 = extract_into_tensor(self.posterior_mean_coef2,
                                                   t, z_t.shape)

        z_term = posterior_mean_coef2 * z_t  # coef * z_t [b, S, d]
        if x is None:
            z_term = z_term[:, :, None, :].repeat(
                1,  1, self.K, 1)  # repeat along K

        mean_sk_given_z = stacked_means + z_term

        posterior_mean_coef1 = extract_into_tensor(self.posterior_mean_coef1,
                                                   t, shape)
        sigma_tilde = extract_into_tensor(self.sigma_tilde,
                                          t, shape)

        if self.corrected_var:
            stacked_std = th.sqrt(posterior_mean_coef1 **
                                  2 * var**2 + sigma_tilde)
        else:
            stacked_std = posterior_mean_coef1 * var + sigma_tilde

        # norm [b, s, k, d] or # norm [b, s, d]
        return D.Independent(D.Normal(mean_sk_given_z, stacked_std), 1)

import os
import numpy as np
import torch.nn as nn
import torch
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.train_helper import check_if_best_than_saved, export_result_txt, prepare_checkpoint, print_detailed_scores_and_sampling, save_train_model_fun, store_model_dict, check_params
from src.mutils import Tracker, get_device, create_optimizer_from_args, load_model, write_dict_to_tensorboard

class MyTrainTemplate:
    """
    Template class to handle the training loop.
    Each experiment contains a experiment-specific training class inherting from this template class.
    """

    def __init__(self,
                 runconfig,
                 batch_size,
                 checkpoint_path,
                 z_train, 
                 z_val, 
                 z_test, 
                 categ_train, 
                 categ_val, 
                 categ_test, 
                 dataset_name, 
                 name_prefix=""):
        self.NUM_SAMPLES = 1000
        model_name = 'NGGDM'
        path_model_prefix = os.path.join(self.path_model_prefix, model_name)
        name_prefix = os.path.join(name_prefix, path_model_prefix)
        self.batch_size = batch_size
        # Remove possible spaces. Name is used for creating default checkpoint path
        self.name_prefix = name_prefix.strip()
        self.runconfig = runconfig
        
        self.checkpoint_path, self.figure_path = prepare_checkpoint(
            checkpoint_path, self.name_prefix)
        # store model info
        store_model_dict(self.figure_path, runconfig)
        runconfig.checkpoint_path = self.checkpoint_path
        # Load model
        self.model = self._create_model(runconfig, self.figure_path)
        self.model = self.model.to(get_device())

        # Load task
        self.z_train = z_train
        self.z_val = z_val
        self.z_test = z_test

        self.categ_train = categ_train
        self.categ_val = categ_val
        self.categ_test = categ_test

        self.dataset_name = dataset_name
        
        self.task = self._create_task(runconfig)
        # Load optimizer and checkpoints
        self._create_optimizer(runconfig)

    def _create_optimizer(self, optimizer_params):
        parameters_to_optimize = self.model.parameters()
        self.optimizer = create_optimizer_from_args(parameters_to_optimize,
                                                    optimizer_params)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            optimizer_params.lr_decay_step,
            gamma=optimizer_params.lr_decay_factor)
        self.lr_minimum = optimizer_params.lr_minimum

    
    def train_model(self,
                    max_iterations=1e6,
                    loss_freq=50,
                    eval_freq=2000,
                    save_freq=1e5,
                    max_gradient_norm=0.25,
                    no_model_checkpoints=False):

        check_params(self.model)
        start_iter = 0
        best_save_dict = {
            "file": None,
            "metric": 1e6,
            "detailed_metrics": None,
            "test": None
        }
        best_save_iter = best_save_dict["file"]
        evaluation_dict = {}
        last_save = None

        test_NLL = None  # Possible test performance determined in the end of the training

        def save_train_model(index_iter):
            return save_train_model_fun(no_model_checkpoints,
                                        best_save_dict,
                                        evaluation_dict,
                                        index_iter,
                                        self.save_model,
                                        only_weights=True)

        # Initialize tensorboard writer
        writer = SummaryWriter(self.checkpoint_path)

        # "Trackers" are moving averages. We use them to log the loss and time needed per training iteration
        time_per_step = Tracker()
        time_per_step_list = []
        train_losses = Tracker()
        self.model.eval()
        self.task.initialize()

        print("=" * 50 + "\nStarting training...\n" + "=" * 50)

        print("Performing initial evaluation...")

        detailed_scores = self.task.eval(initial_eval=True)
        start = time.time()
        sample_metrics = self.task.evaluate_sample(num_samples=self.NUM_SAMPLES)
        end = time.time()
        time_for_sampling = (end - start)
        print_detailed_scores_and_sampling(detailed_scores,
                                           sample_metrics)
        print('time for sampling ', self.NUM_SAMPLES, ' samples : ',
              "{:.2f}".format((time_for_sampling)), ' sec')

        self.model.train()
        detailed_scores_to_tensorboard = {}
        detailed_scores_to_tensorboard.update(
            sample_metrics.get_printable_metrics_dict())
        detailed_scores_to_tensorboard.update(detailed_scores)
        write_dict_to_tensorboard(writer,
                                  detailed_scores_to_tensorboard,
                                  base_name="eval",
                                  iteration=start_iter)

        index_iter = start_iter
        keep_going = True
        self.loss_prev = None
        while keep_going:

            # Training step
            start_time = time.time()
            loss = self.task.train_step(iteration=index_iter)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()),
                                           max_gradient_norm)
            self.optimizer.step()
            if self.optimizer.param_groups[0]['lr'] > self.lr_minimum:
                self.lr_scheduler.step()
            end_time = time.time()

            time_per_step.add(end_time - start_time)
            time_per_step_list.append(end_time - start_time)
            train_losses.add(loss.item())

            if (index_iter + 1) % loss_freq == 0:

                loss_avg = train_losses.get_mean(reset=True)
                self.loss_prev = loss_avg
                train_time_avg = time_per_step.get_mean(reset=True)
                print(
                    "Training iteration %i|%i (%4.2fs). Loss: %6.5f." %
                    (index_iter + 1, max_iterations, train_time_avg,
                        loss_avg))
                writer.add_scalar("train/loss", loss_avg, index_iter + 1)
                writer.add_scalar("train/learning_rate",
                                  self.optimizer.param_groups[0]['lr'],
                                  index_iter + 1)
                writer.add_scalar("train/training_time", train_time_avg,
                                  index_iter + 1)

                self.task.add_summary(writer,
                                      index_iter + 1,
                                      checkpoint_path=self.checkpoint_path)

            # Performing evaluation every "eval_freq" steps
            if (index_iter + 1) % eval_freq == 0:
                self.model.eval()

                detailed_scores = self.task.eval()
                start = time.time()
                sample_metrics = self.task.evaluate_sample(
                    num_samples=self.NUM_SAMPLES)
                end = time.time()
                time_for_sampling = (end - start)
                print_detailed_scores_and_sampling(
                    detailed_scores, sample_metrics)

                print('time for sampling ', self.NUM_SAMPLES, ' samples : ',
                      "{:.2f}".format((time_for_sampling)), ' sec')
                if 'overfit_detected' in sample_metrics.metrics:
                    if sample_metrics.metrics['overfit_detected']:
                        print('The model is overfitting to the training samples...')
                        keep_going = False
                self.model.train()
                detailed_scores_to_tensorboard = sample_metrics.get_printable_metrics_dict()
                detailed_scores_to_tensorboard.update(detailed_scores)
                write_dict_to_tensorboard(writer,
                                          detailed_scores_to_tensorboard,
                                          base_name="eval",
                                          iteration=index_iter + 1)

                # If model performed better on validation than any other iteration so far => save it and eventually replace old model
                check_if_best_than_saved(last_save, detailed_scores['loss'],
                                         detailed_scores, best_save_dict,
                                         index_iter,
                                         self.get_checkpoint_filename,
                                         self.checkpoint_path,
                                         evaluation_dict, save_train_model)

            if (index_iter + 1) % save_freq == 0:
                save_train_model(index_iter + 1)
                if last_save is not None and os.path.isfile(
                        last_save) and last_save != best_save_iter:
                    print("Removing checkpoint %s..." % last_save)
                    os.remove(last_save)
                last_save = self.get_checkpoint_filename(index_iter + 1)
            index_iter += 1
            keep_going = not index_iter == int(max_iterations)
        # End training loop
        print('time to train ', np.sum(time_per_step_list))

        # Testing the trained model
        detailed_scores = self.task.test()
        print("=" * 50 + "\nTest performance: %lf" % (detailed_scores['loss']))
        detailed_scores["original_NLL"] = test_NLL
        best_save_dict["test"] = detailed_scores

        sample_metrics = self.task.evaluate_sample(
            num_samples=10*self.NUM_SAMPLES)

        export_result_txt(best_save_iter, best_save_dict, self.checkpoint_path)
        writer.close()
        return detailed_scores, sample_metrics

    def get_checkpoint_filename(self, iteration):
        checkpoint_file = os.path.join(
            self.checkpoint_path,
            'checkpoint_' + str(iteration).zfill(7) + ".tar")
        return checkpoint_file

    def save_model(self,
                   iteration,
                   add_param_dict,
                   save_embeddings=False,
                   save_optimizer=True):
        checkpoint_file = self.get_checkpoint_filename(iteration)
        if isinstance(self.model, nn.DataParallel):
            model_dict = self.model.module.state_dict()
        else:
            model_dict = self.model.state_dict()

        checkpoint_dict = {'model_state_dict': model_dict}
        if save_optimizer:
            checkpoint_dict[
                'optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint_dict[
                'scheduler_state_dict'] = self.lr_scheduler.state_dict()
        checkpoint_dict.update(add_param_dict)
        torch.save(checkpoint_dict, checkpoint_file)


from metrics import compute_corr_higher_order, compute_higher_order_stats, get_frac_overlap
from sample_metrics import SamplesMetrics
from mutils import get_device

import torch.utils.data as data

import pickle as pk

class MyTaskTemplate:

    #def __init__(self, model, run_config, name, load_data=True, debug=False, batch_size=64, drop_last=False, num_workers=None):
    def __init__(self, model, run_config, name, 
                 z_train, z_val, z_test, 
                 categ_train, categ_val, categ_test, 
                 dataset_name, 
                 load_data=True, debug=False, batch_size=64, drop_last=False, num_workers=None):
        # Saving parameters
        self.name = name
        self.model = model
        self.run_config = run_config
        self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.debug = debug
        self.sampling_batch = 1000
        # Initializing dataset parameters
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_epoch = 0
        # Load data if specified, and create data loaders
        if load_data:
            #self._load_datasets()
            #self._load_datasets(z_train, z_val, z_test)
            self._load_datasets(z_train, z_val, z_test, categ_train, categ_val, categ_test, dataset_name)
            self._initialize_data_loaders(
                drop_last=drop_last, num_workers=num_workers)
        else:
            self.train_data_loader = None
            self.train_data_loader_iter = None
            self.val_data_loader = None
            self.test_data_loader = None

        # Create a dictionary to store summary metrics in
        self.summary_dict = {}

        # Placeholders for visualization
        self.gen_batch = None
        self.class_colors = None

        # Put model on correct device
        self.model.to(get_device())
        self._precompute_stats_for_metrics()

    def _precompute_stats_for_metrics(self):
        # if the metrics already has been precomputed, load it.
        # Pairwise Covariance
        data_path = self.test_dataset.data_path
        self.higher_order_dict_n_abs = {}
        for n in range(2, min(10, self.model.S)):
            file = str(n)+'_list_highcov_test_abs.pk'
            filepath = os.path.join(data_path, file)

            if not os.path.exists(filepath):
                #print('self.test_dataset.np_data:', self.test_dataset.np_data)
                print('self.test_dataset.np_data.shape:', self.test_dataset.np_data.shape)
                print('n:', n)
                higher_order_dict_abs = compute_higher_order_stats(
                    self.test_dataset.np_data, num_patterns=1000, n=n, absolute_version=True)

                with open(filepath, 'wb') as f:
                    pk.dump(higher_order_dict_abs, f)
            else:
                with open(filepath, 'rb') as f:
                    higher_order_dict_abs = pk.load(f)
            self.higher_order_dict_n_abs[n] = higher_order_dict_abs
        filepath = os.path.join(data_path, 'training_dict.pk')
        if not os.path.exists(filepath):
            training_dict = self.data_to_dict(
                self.train_dataset.np_data)  # todo load and store
            with open(filepath, 'wb') as f:
                pk.dump(training_dict, f)
        else:
            with open(filepath, 'rb') as f:
                training_dict = pk.load(f)
        filepath = os.path.join(data_path, 'test_dict.pk')
        if not os.path.exists(filepath):
            test_dict = self.data_to_dict(
                self.test_dataset.np_data)  # todo load and store
            with open(filepath, 'wb') as f:
                pk.dump(test_dict, f)
        else:
            with open(filepath, 'rb') as f:
                test_dict = pk.load(f)
        self.training_dict = training_dict
        self.test_dict = test_dict

        self.frac_val_seen = get_frac_overlap(
            self.training_dict, self.test_dict)

    def data_to_dict(self, np_data):
        dict_data = {}
        for i in tqdm(range(np_data.shape[0])):
            x = list(np_data[i, :])
            str_x = '-'.join(str(s) for s in x)
            if str_x in dict_data:
                dict_data[str_x] += 1
            else:
                dict_data[str_x] = 1

        return dict_data

    def _initialize_data_loaders(self, drop_last, num_workers):
        if num_workers is None:
            if isinstance(self.model, nn.DataParallel) and torch.cuda.device_count() > 1:
                num_workers = torch.cuda.device_count()
            else:
                num_workers = 1

        def _init_fn(worker_id):
            np.random.seed(42)
        # num_workers = 1
        # Initializes all data loaders with the loaded datasets
        if hasattr(self.train_dataset, "get_sampler"):
            self.train_data_loader = data.DataLoader(self.train_dataset, batch_sampler=self.train_dataset.get_sampler(self.train_batch_size, drop_last=drop_last), pin_memory=True,
                                                     num_workers=num_workers, worker_init_fn=_init_fn)
            self.val_data_loader = data.DataLoader(self.val_dataset, batch_sampler=self.val_dataset.get_sampler(
                self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)
            self.test_data_loader = data.DataLoader(self.test_dataset, batch_sampler=self.test_dataset.get_sampler(
                self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)
        else:
            self.train_data_loader = data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                                     worker_init_fn=_init_fn)
            self.val_data_loader = data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                                   shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)
            self.test_data_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                    shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)

        self.train_data_loader_iter = iter(self.train_data_loader)

    def train_step(self, iteration=0):
        # Check if training data was correctly loaded
        if self.train_data_loader_iter is None:
            print("[!] ERROR: Iterator of the training data loader was None. Additional parameters: " +
                  "train_data_loader was %sloaded, " % ("not " if self.train_data_loader is None else "") +
                  "train_dataset was %sloaded." % ("not " if self.train_dataset is None else ""))

        # Get batch and put it on correct device
        batch = self._get_next_batch()
        #batch = TaskTemplate.batch_to_device(batch)
        batch = MyTaskTemplate.batch_to_device(batch)
        
        # Perform task-specific training step
        #return self._train_batch(batch, iteration=iteration)
        return self._train_batch(batch, iteration=iteration)

    def sample(self, num_samples):
        sampling_batch = self.sampling_batch
        num_samples_total = 0
        samples_list = []
        for _ in range(math.ceil(num_samples/sampling_batch)):
            if num_samples_total+sampling_batch > num_samples:
                batch_num_samples = num_samples - num_samples_total
            else:

                batch_num_samples = sampling_batch
            num_samples_total += batch_num_samples
            samples = self.model.sample(num_samples=batch_num_samples)
            samples_list.append(samples['x'].cpu().detach().numpy())
        samples = np.array(samples_list)
        samples = samples.reshape(-1, samples.shape[2])

        return samples

    def __get_sample_stats(self, samples):

        r20_corr_abs = compute_corr_higher_order(
            samples, self.higher_order_dict_n_abs)
        samples_dict = self.data_to_dict(samples)
        frac_seen_samples = get_frac_overlap(
            self.training_dict, samples_dict)
        return frac_seen_samples, r20_corr_abs

    # computes samples metrics on the samples_np and wrap up the results in SamplesResults
    def get_sample_metrics(self, samples_np):
        frac_seen_samples, r20_corr_abs = self.__get_sample_stats(
            samples_np)
        overfit_detected = frac_seen_samples > self.frac_val_seen*1.5
        metrics = {'frac_seen_samples': frac_seen_samples,
                   'overfit_detected': overfit_detected}
        for i, r20_abs in enumerate(r20_corr_abs):
            if r20_abs[1] < 0.05:
                metrics['r_20_abs'+str(i+2)] = r20_abs[0]
            else:
                metrics['r_20_abs'+str(i+2)] = -2

        samples_results = SamplesMetrics(
            samples_np, metrics)

        return samples_results

    def eval(self, data_loader=None, **kwargs):
        # Default: if no dataset is specified, we use validation dataset
        if data_loader is None:

            data_loader = self.val_data_loader
        is_test = (data_loader == self.test_data_loader)

        start_time = time.time()
        torch.cuda.empty_cache()
        self.model.eval()

        # Prepare metrics
        nll_counter = 0
        result_batch_dict = {}
        # Evaluation loop
        with torch.no_grad():
            for batch_ind, batch in enumerate(data_loader):

                print("Evaluation process: %4.2f%%" %
                      (100.0 * batch_ind / len(data_loader)), end="\r")
                # Put batch on correct device
                #batch = TaskTemplate.batch_to_device(batch)
                batch = MyTaskTemplate.batch_to_device(batch)
                # Evaluate single batch
                batch_size = batch[0].size(0) if isinstance(
                    batch, tuple) else batch.size(0)
                batch_dict = self._eval_batch(
                    batch, is_test=is_test)
                for key, batch_val in batch_dict.items():
                    if key in result_batch_dict:
                        result_batch_dict[key] += batch_val.item() * batch_size
                    else:
                        result_batch_dict[key] = batch_val.item() * batch_size

                nll_counter += batch_size

                if self.debug and batch_ind > 10:
                    break
        detailed_metrics = {}
        for key, batch_val in result_batch_dict.items():
            detailed_metrics[key] = batch_val / max(1e-5, nll_counter)

        self.model.train()
        eval_time = int(time.time() - start_time)
        print("Finished %s with loss of %4.3f, (%imin %is)" % ("testing" if data_loader ==
                                                               self.test_data_loader else "evaluation", detailed_metrics["loss"], eval_time/60, eval_time % 60))
        torch.cuda.empty_cache()

        return detailed_metrics

    def _train_batch(self, batch, iteration):
        print(batch)
        #x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
        x_in, x_length, x_channel_mask = self._preprocess_batch(batch[0])
        neg_boundnl = -self.model(
            x_in,
            categ=batch[1], 
            reverse=False,
            get_ldj_per_layer=True,
            beta=self.beta_scheduler.get(iteration),
            length=x_length)

        loss = (neg_boundnl / x_length.float()).mean()
        self.summary_dict["ldj"].append(loss.item())
        self.summary_dict["beta"] = self.beta_scheduler.get(iteration)

        return loss



    def test(self, **kwargs):
        return self.eval(data_loader=self.test_data_loader, **kwargs)

    def add_summary(self, writer, iteration, checkpoint_path=None):
        # Adding metrics collected during training to the tensorboard
        # Function can/should be extended if needed
        for key, val in self.summary_dict.items():
            summary_key = "train_%s/%s" % (self.name, key)
            # If it is not a list, it is assumably a single scalar
            if not isinstance(val, list):
                writer.add_scalar(summary_key, val, iteration)
                self.summary_dict[key] = 0.0
            elif len(val) == 0:  # Skip an empty list
                continue
            # For a list of scalars, report the mean
            elif not isinstance(val[0], list):
                writer.add_scalar(summary_key, mean(val), iteration)
                self.summary_dict[key] = list()
            else:  # List of lists indicates a histogram
                val = [v for sublist in val for v in sublist]
                writer.add_histogram(summary_key, np.array(val), iteration)
                self.summary_dict[key] = list()

    def _get_next_batch(self):
        # Try to get next batch. If one epoch is over, the iterator throws an error, and we start a new iterator
        try:
            batch = next(self.train_data_loader_iter)
        except StopIteration:
            self.train_data_loader_iter = iter(self.train_data_loader)
            batch = next(self.train_data_loader_iter)
            self.train_epoch += 1
        return batch

    def _eval_batch(self, batch, is_test=False):
        #print(batch)
        #x_in, x_length, x_channel_mask = self._preprocess_batch(batch)
        x_in, x_length, x_channel_mask = self._preprocess_batch(batch[0])
        #print(x_in.shape, x_length.shape, x_channel_mask.shape)
        #print(self.model)
        ldj = self.model(x_in,
                         categ=batch[1], 
                         reverse=False,
                         get_ldj_per_layer=False,
                         beta=1,
                         length=x_length)

        loss = -(ldj / x_length.float()).mean()
        std_loss = (ldj / x_length.float()).std()

        return {'loss': loss, 'std_loss': std_loss}

    @staticmethod
    def batch_to_device(batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = tuple([b.to(get_device()) for b in batch])
        else:
            batch = batch.to(get_device())
        return batch

import pickle as pk
import os
from mutils import PARAM_CONFIG_FILE
from train_template import TrainTemplate
from experiment.synthetic_task import TaskSyntheticModeling
#from model.GMCD import GMCD
from src.datasets.synthetic import SyntheticDataset

# Training class for the synthetic experiment.


class MyTrainSyntheticModeling(MyTrainTemplate):
    def __init__(self,
                 runconfig,
                 batch_size,
                 checkpoint_path,
                 z_train, 
                 z_val, 
                 z_test, 
                 categ_train, 
                 categ_val, 
                 categ_test, 
                 dataset_name, 
                 path_experiment="",
                 **kwargs):
        self.path_model_prefix = os.path.join(
            runconfig.dataset, "S_" + str(runconfig.S) + "_K_" + str(runconfig.K))
        super().__init__(runconfig,
                         batch_size,
                         checkpoint_path,
                         z_train, 
                         z_val, 
                         z_test, 
                         categ_train, 
                         categ_val, 
                         categ_test, 
                         dataset_name, 
                         name_prefix=path_experiment,
                         **kwargs)

        self.z_train = z_train
        self.z_val = z_val
        self.z_test = z_test

        self.categ_train = categ_train
        self.categ_val = categ_val
        self.categ_test = categ_test

        self.dataset_name = dataset_name

    def _create_model(self, runconfig, figure_path):
        #model = GMCD(run_config=runconfig,
        #             dataset_class=SyntheticDataset, figure_path=figure_path)
        #model = MyArTransformerDiffusion(runconfig, 
        #                                 runconfig.S, runconfig.K, runconfig.encoding_dim, runconfig.T, figure_path=figure_path)
        model = NGGDM(runconfig, MySyntheticDataset, runconfig.K, runconfig.encoding_dim)
        
        return model

    def _create_task(self, runconfig):
        #task = TaskSyntheticModeling(self.model,
        #                             runconfig,
        #                             batch_size=self.batch_size)
        task = MyTaskSyntheticModeling(self.model,
                                       runconfig,
                                       z_train=self.z_train, 
                                       z_val=self.z_val, 
                                       z_test=self.z_test, 
                                       categ_train=self.categ_train, 
                                       categ_val=self.categ_val, 
                                       categ_test=self.categ_test, 
                                       dataset_name=self.dataset_name, 
                                       batch_size=self.batch_size)
        return task


def start_training(runconfig,
                   z_train, 
                   z_val, 
                   z_test, 
                   categ_train, 
                   categ_val, 
                   categ_test, 
                   dataset_name, 
                   return_result=False):

    # Setup training
    #trainModule = TrainSyntheticModeling(runconfig,
    #                                     batch_size=runconfig.batch_size,
    #                                     checkpoint_path=runconfig.checkpoint_path,
    #                                     path_experiment='')
    trainModule = MyTrainSyntheticModeling(runconfig,
                                           batch_size=runconfig.batch_size,
                                           checkpoint_path=runconfig.checkpoint_path,
                                           z_train=z_train, 
                                           z_val=z_val, 
                                           z_test=z_test, 
                                           categ_train=categ_train, 
                                           categ_val=categ_val, 
                                           categ_test=categ_test, 
                                           dataset_name=dataset_name, 
                                           path_experiment='')
    # store the config of the run
    args_filename = os.path.join(trainModule.checkpoint_path,
                                 PARAM_CONFIG_FILE)

    with open(args_filename, "wb") as f:
        pk.dump(runconfig, f)

    # Start training

    result = trainModule.train_model(
        runconfig.max_iterations,
        loss_freq=50,
        eval_freq=runconfig.eval_freq,
        save_freq=runconfig.save_freq)

    # Cleaning up the checkpoint directory afterwards if selected

    if return_result:
        return result

from sample_metrics import SamplesMetrics
from optimizer.scheduler import ExponentialScheduler
#from src.datasets.synthetic import SyntheticDataset
from metrics import get_diff_metric
#from src.task_template import TaskTemplate
import torch


class MyTaskSyntheticModeling(MyTaskTemplate):
    def __init__(self,
                 model,
                 run_config, 
                 z_train, 
                 z_val, 
                 z_test, 
                 categ_train, 
                 categ_val, 
                 categ_test, 
                 dataset_name, 
                 load_data=True,
                 debug=False,
                 batch_size=64):
        super().__init__(model,
                         run_config,
                         z_train=z_train, 
                         z_val=z_val, 
                         z_test=z_test, 
                         categ_train=categ_train, 
                         categ_val=categ_val, 
                         categ_test=categ_test, 
                         dataset_name=dataset_name, 
                         load_data=load_data,
                         debug=debug,
                         batch_size=batch_size,
                         name="MyTaskSyntheticModeling")
        self.beta_scheduler = self.create_scheduler(self.run_config)

        self.summary_dict = {
            "log_prob": list(),
            "ldj": list(),
            "z": list(),
            "beta": 0
        }

    def create_scheduler(self, scheduler_params, param_name=None):
        end_val = scheduler_params.beta_scheduler_end_val
        start_val = scheduler_params.beta_scheduler_start_val
        stepsize = scheduler_params.beta_scheduler_step_size
        logit = scheduler_params.beta_scheduler_logit
        delay = scheduler_params.beta_scheduler_delay

        return ExponentialScheduler(start_val=start_val, end_val=end_val, logit_factor=logit, stepsize=stepsize, delay=delay, param_name=param_name)

    def _load_datasets(self, z_train, z_val, z_test, categ_train, categ_val, categ_test, dataset_name):
        self.S = self.run_config.S
        self.K = self.run_config.K
        #print("Loading synthetic dataset K=%s..." % self.S)
        print("Loading synthetic dataset K=%s..." % self.K)

        #self.train_dataset = SyntheticDataset(S=self.S, K=self.K,
        #                                      train=True)
        #self.train_dataset = MySyntheticDataset(z_train, train=True)
        self.train_dataset = MySyntheticDataset(z_train, categ_train, dataset_name=dataset_name, train=True)
        #self.val_dataset = SyntheticDataset(S=self.S, K=self.K,
        #                                    val=True)
        #self.val_dataset = MySyntheticDataset(z_val, val=True)
        self.val_dataset = MySyntheticDataset(z_val, categ_val, dataset_name=dataset_name, val=True)
        #self.test_dataset = SyntheticDataset(S=self.S, K=self.K,
        #                                     test=True)
        #self.test_dataset = MySyntheticDataset(z_test, test=True)
        self.test_dataset = MySyntheticDataset(z_test, categ_test, dataset_name=dataset_name, test=True)

    def evaluate_sample(self, num_samples):
        samples_np = self.sample(num_samples)  # obtain the samples
        # compute the metrics on the samples
        return self.get_sample_metrics(samples_np)

    def get_sample_metrics(self, samples_np):
        m = samples_np.shape[0]  # num samples
        base_samples_results = super().get_sample_metrics(samples_np)
        histogram_samples_per_p = self.train_dataset.samples_to_dict(
            samples_np)
        size_support_dict = self.train_dataset.get_size_support_dict()
        empirical_prob = {}
        for _, dict_p in histogram_samples_per_p.items():
            for key, val in dict_p.items():
                empirical_prob[key] = val / m

        all_ps = list(histogram_samples_per_p.keys())
        if 0 in all_ps:
            all_ps.remove(0)
        if len(all_ps) == 0:
            p_likely = p_rare = 0
        else:
            p_likely = sum(histogram_samples_per_p[max(
                all_ps)].values()) / m
            p_rare = sum(histogram_samples_per_p[min(
                all_ps)].values()) / m
        p_total = p_likely+p_rare

        d_tv, H, tv_ood = get_diff_metric(
            histogram_samples_per_p, size_support_dict, M=m)

        metrics = {'p_rare': p_rare,
                   'p_likely': p_likely, 'p_total': p_total, 'd_tv': d_tv, 'H': H, 'tv_ood': tv_ood, 'histogram_samples_per_p': histogram_samples_per_p}

        sample_results = SamplesMetrics(
            samples_np, metrics, histogram_samples_per_p)
        sample_results.add_new_metrics(base_samples_results.metrics)
        return sample_results

    def _train_batch_discrete(self, x_in, x_length):
        #print(x_in.shape)
        #print(x_in)
        _, ldj = self.model(x_in, reverse=False, length=x_length)
        loss = (-ldj / x_length.float()).mean()
        return loss

    def _calc_loss(self, neg_ldj, neglog_prob, x_length, take_mean=True):
        neg_ldj = (neg_ldj / x_length.float())
        neglog_prob = (neglog_prob / x_length.float())
        loss = neg_ldj + neglog_prob
        if take_mean:
            loss = loss.mean()
            neg_ldj = neg_ldj.mean()
            neglog_prob = neglog_prob.mean()
        return loss, neg_ldj, neglog_prob

    def _preprocess_batch(self, batch):
        x_in = batch
        x_length = x_in.new_zeros(x_in.size(0),
                                  dtype=torch.long) + x_in.size(1)
        x_channel_mask = x_in.new_ones(x_in.size(0),
                                       x_in.size(1),
                                       1,
                                       dtype=torch.float32)
        return x_in, x_length, x_channel_mask

    def initialize(self, num_batches=16):

        if self.model.need_data_init():
            # print("Preparing data dependent initialization...")
            batch_list = []
            for _ in range(num_batches):
                batch = self._get_next_batch()
                #batch = TaskTemplate.batch_to_device(batch)
                batch = MyTaskTemplate.batch_to_device(batch)
                x_in, x_length, _ = self._preprocess_batch(batch)
                batch_tuple = (x_in, {"length": x_length})
                batch_list.append(batch_tuple)
            self.model.initialize_data_dependent(batch_list)

from mutils import PARAM_CONFIG_FILE

def my_start_training(runconfig,
                      z_train, 
                      z_val, 
                      z_test, 
                      categ_train, 
                      categ_val, 
                      categ_test, 
                      dataset_name, 
                      return_result=False):
    # Setup training
    trainModule = MyTrainSyntheticModeling(runconfig,
                                           batch_size=runconfig.batch_size,
                                           checkpoint_path=runconfig.checkpoint_path, 
                                           z_train=z_train, 
                                           z_val=z_val, 
                                           z_test=z_test, 
                                           categ_train=categ_train, 
                                           categ_val=categ_val, 
                                           categ_test=categ_test, 
                                           dataset_name=dataset_name, 
                                           path_experiment='')
    # store the config of the run
    args_filename = os.path.join(trainModule.checkpoint_path,
                                 PARAM_CONFIG_FILE)
    if not os.path.exists(trainModule.checkpoint_path):
        os.makedirs(trainModule.checkpoint_path)
    
    with open(args_filename, "wb") as f:
        pk.dump(runconfig, f)

    # Start training

    result = trainModule.train_model(
        runconfig.max_iterations,
        loss_freq=50, 
        eval_freq=runconfig.eval_freq, 
        save_freq=runconfig.save_freq)

    # Cleaning up the checkpoint directory afterwards if selected

    if return_result:
        return result

