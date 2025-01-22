#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from scipy import sparse
from nltk import download as nltk_download

# If you need these corpora/dictionaries in local environments:
nltk_download('punkt', quiet=True)
nltk_download('wordnet', quiet=True)

##########################################################
# 1. Helper: Load preprocessed data from .npz files
##########################################################
##########################################################
# 1. Helper: Load preprocessed data from .npz files
##########################################################
def load_preprocessed_data(folder_path, prefix="ideology", device='cpu'):
    """
    Loads preprocessed data from the specified folder and prefix. 
    Returns a dictionary containing:
        - docs_word_matrix_tensor: (num_docs x vocab_size) float tensor
        - env_index_tensor:        (num_docs x num_envs) float tensor
        - vocab_size:              int
        - num_envs:                int
    """
    train_data_path = os.path.join(folder_path, f"{prefix}_train.npz")
    env_data_path   = os.path.join(folder_path, f"{prefix}_env.npz")
    # file_check = "path/to/preprocessed_data/ideology/ideology_train.npz"
    with np.load(train_data_path) as data:
        print("Keys:", data.files)  # Should see ['data', 'indices', 'indptr', 'shape']
        print("data[\"data\"][:10]:", data['data'][:10])  
    # 1) Load the training sparse matrix
    # ----> ADD allow_pickle=True here
    train_data_npz = np.load(train_data_path, allow_pickle=True)
    train_data_csr = sparse.csr_matrix(
        (train_data_npz['data'], train_data_npz['indices'], train_data_npz['indptr']),
        shape=train_data_npz['shape']
    )
    # Convert to dense if feasible
    docs_word_matrix_tensor = torch.from_numpy(train_data_csr.toarray()).float().to(device)
    
    # 2) Load the environment index file
    # ----> ADD allow_pickle=True here
    env_data_npz = np.load(env_data_path, allow_pickle=True)
    env_np = env_data_npz['data']  # shape should be (num_docs, num_envs)
    env_index_tensor = torch.from_numpy(env_np).float().to(device)
    
    # 3) Extract metadata
    vocab_size = train_data_csr.shape[1]        # number of words
    num_envs   = env_index_tensor.shape[1]      # number of distinct environments
    
    return {
        'docs_word_matrix_tensor': docs_word_matrix_tensor,
        'env_index_tensor': env_index_tensor,
        'vocab_size': vocab_size,
        'num_envs': num_envs
    }


##########################################################
# 2. The MTM model class
##########################################################
class MTM(nn.Module):
    def __init__(self, num_topics, num_words, num_envs,
                 device='cpu',
                 empirical_bayes=True,
                 fixed_alpha_a=None, fixed_alpha_b=None):
        super(MTM, self).__init__()

        def init_param(shape):
            return nn.Parameter(torch.randn(shape, device=device))

        def init_param_zeros(shape):
            return nn.Parameter(torch.zeros(shape, device=device))

        self.num_topics = num_topics
        self.num_words  = num_words
        self.num_envs   = num_envs
        self.empirical_bayes = empirical_bayes

        # ------------------------
        # Global Beta, β₀,k
        # ------------------------
        self.beta = init_param([num_topics, num_words])
        self.beta_logvar = init_param_zeros([num_topics, num_words])
        self.beta_prior = Normal(
            torch.zeros([num_topics, num_words], device=device),
            torch.ones([num_topics, num_words],  device=device)
        )

        # ------------------------
        # α hyperparameters (for gamma prior), possibly empirical Bayes
        # ------------------------
        if empirical_bayes:
            self.log_alpha_a = nn.Parameter(torch.tensor(1.0, device=device))
            self.log_alpha_b = nn.Parameter(torch.tensor(1.0, device=device))
        else:
            # Use provided or defaults
            alpha_a_fixed = fixed_alpha_a if fixed_alpha_a is not None else 4.7
            alpha_b_fixed = fixed_alpha_b if fixed_alpha_b is not None else 0.05
            self.log_alpha_a = torch.tensor(alpha_a_fixed, device=device)
            self.log_alpha_b = torch.tensor(alpha_b_fixed, device=device)

        # Initialize sigma
        if empirical_bayes:
            alpha_a = torch.exp(self.log_alpha_a)
            alpha_b = torch.exp(self.log_alpha_b)
        else:
            alpha_a = self.log_alpha_a
            alpha_b = self.log_alpha_b

        self.sigma = torch.distributions.Gamma(alpha_a, alpha_b).rsample(
            [num_envs, num_topics, num_words]
        ).to(device)

        # ------------------------
        # gamma and gamma prior
        # ------------------------
        self.gamma = init_param_zeros([num_envs, num_topics, num_words])
        self.gamma_logvar = -torch.log(self.sigma).add(1e-8)
        self.gamma_prior = Normal(
            torch.zeros_like(self.gamma),
            torch.sqrt(1.0 / self.sigma).add(1e-8)
        )

        # ------------------------
        # θ global prior and network
        # ------------------------
        self.theta_global_prior = Normal(
            torch.zeros(num_topics, device=device),
            torch.ones(num_topics,  device=device)
        )

        self.theta_global_net = nn.Sequential(
            nn.Linear(num_words, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, num_topics * 2)
        )

    def get_learned_parameters(self):
        """Returns the learned alpha parameters when using empirical Bayes"""
        if self.empirical_bayes:
            return (torch.exp(self.log_alpha_a).item(),
                    torch.exp(self.log_alpha_b).item())
        return None

    def forward(self, bow, x_d):
        # bow: (batch_size, vocab_size)
        # x_d: (batch_size, num_envs)
        # 1) Document-specific global θ params
        self.theta_global_params = self.theta_global_net(bow)  # shape: [batch_size, num_topics*2]
        theta_global_mu, theta_global_logvar = self.theta_global_params.split(self.num_topics, dim=-1)
        theta_global_logvar = theta_global_logvar.add(1e-8)

        # sample θ from Normal
        theta_sample = Normal(
            theta_global_mu, torch.exp(0.5 * theta_global_logvar).add(1e-8)
        ).rsample()

        # convert to probabilities
        theta_softmax = F.softmax(theta_sample, dim=-1)  # [batch_size, num_topics]

        # 2) sample global beta
        beta_dist = Normal(self.beta, torch.exp(0.5 * self.beta_logvar).add(1e-8))
        beta_sample = beta_dist.rsample()  # [num_topics, num_words]

        # 3) sample gamma
        gamma_dist = Normal(self.gamma, torch.exp(0.5 * self.gamma_logvar).add(1e-8))
        gamma_sample = gamma_dist.rsample()  # [num_envs, num_topics, num_words]

        # 4) effect of doc-specific covariates on gamma
        # x_d: (batch_size, num_envs), gamma_sample: (num_envs, num_topics, num_words)
        gamma_effect = torch.einsum('be,etv->btv', x_d, gamma_sample)

        # 5) adjusted beta for each document
        #    original beta plus gamma effect
        adjusted_beta = self.beta.unsqueeze(0) + gamma_effect
        adjusted_beta_softmax = F.softmax(adjusted_beta, dim=-1)

        # 6) combine doc-topic and topic-word
        eta_d = torch.einsum('bt,btv->bv', theta_softmax, adjusted_beta_softmax)
        return eta_d

##########################################################
# 3. KL and ELBO utilities
##########################################################
def calculate_kl_divergences(MTM, env, empirical_bayes=True):
    theta_global_mu, theta_global_logvar = MTM.theta_global_params.split(MTM.num_topics, dim=-1)
    theta_global_logvar = theta_global_logvar.add(1e-8)

    theta_global = Normal(
        theta_global_mu, torch.exp(0.5 * theta_global_logvar).add(1e-8)
    )
    theta_global_kl = torch.distributions.kl.kl_divergence(
        theta_global, MTM.theta_global_prior
    ).sum()

    beta = Normal(MTM.beta, torch.exp(0.5 * MTM.beta_logvar))
    beta_kl = torch.distributions.kl.kl_divergence(beta, MTM.beta_prior).sum()

    if not empirical_bayes:
        gamma = Normal(MTM.gamma, torch.exp(0.5 * MTM.gamma_logvar))
        gamma_kl = torch.distributions.kl.kl_divergence(gamma, MTM.gamma_prior).sum()
    else:
        gamma_kl = 0  # No KL divergence for gamma if EB is True

    return theta_global_kl, beta_kl, gamma_kl

def bbvi_update(minibatch, env_index, MTM, optimizer, n_samples):
    optimizer.zero_grad()

    # forward pass
    z = MTM(minibatch, env_index)

    # KL divergences
    kl_theta, kl_beta, kl_gamma = calculate_kl_divergences(
        MTM, env_index, empirical_bayes=False
    )

    # log-likelihood and ELBO
    log_likelihood = (minibatch * torch.log(z + 1e-10)).sum(-1)
    elbo = log_likelihood.mul(n_samples) - (kl_theta + kl_beta + kl_gamma)
    elbo_total = -elbo.sum()  # negative for gradient descent

    # backward
    elbo_total.backward(retain_graph=True)
    optimizer.step()

    return elbo_total.item()

def empirical_bayes_update(MTM, optimizer_hyper, empirical_bayes=True,
                           num_epochs_hyper=2, kl_threshold=1e-5):
    """Empirical Bayes update for the hyperparameters of the Gamma distribution."""
    if not empirical_bayes:
        return

    previous_gamma_kl = float('inf')

    for _ in range(num_epochs_hyper):
        optimizer_hyper.zero_grad()
        with torch.set_grad_enabled(True):
            alpha_a = F.softplus(MTM.log_alpha_a)
            alpha_b = F.softplus(MTM.log_alpha_b)
            sigma_sample = torch.distributions.Gamma(alpha_a, alpha_b).rsample(
                [MTM.num_envs, MTM.num_topics, MTM.num_words]
            )

            gamma_prior = Normal(
                torch.zeros_like(MTM.gamma),
                torch.sqrt(1.0 / sigma_sample + 1e-10)
            )

            gamma = Normal(
                MTM.gamma, torch.exp(0.5 * MTM.gamma_logvar + 1e-10)
            )
            gamma_kl = torch.distributions.kl.kl_divergence(gamma, gamma_prior).sum()

            delta_gamma_kl = abs(gamma_kl.item() - previous_gamma_kl)
            if delta_gamma_kl < kl_threshold:
                break

            gamma_kl.backward(retain_graph=True)

        optimizer_hyper.step()
        previous_gamma_kl = gamma_kl.item()

##########################################################
# 4. Main training functions
##########################################################
def train_model(MTM, docs_word_matrix_tensor, env_index_tensor,
                num_epochs=80, minibatch_size=16, lr=0.01, empirical_bayes=True):
    device = next(MTM.parameters()).device  # get device from model
    MTM.train()

    optimizer = torch.optim.Adam(MTM.parameters(), lr=lr, betas=(0.9, 0.999))
    # For hyper-parameters
    optimizer_hyper = torch.optim.Adam(
        [MTM.log_alpha_a, MTM.log_alpha_b], lr=lr, betas=(0.9, 0.999)
    )

    # ensure data is on same device
    docs_word_matrix_tensor = docs_word_matrix_tensor.to(device)
    env_index_tensor        = env_index_tensor.to(device)

    for epoch in range(num_epochs):
        elbo_accumulator = 0.0

        # random permutation
        perm = torch.randperm(docs_word_matrix_tensor.size(0))

        for i in range(0, docs_word_matrix_tensor.size(0), minibatch_size):
            indices = perm[i : i + minibatch_size]
            minibatch = docs_word_matrix_tensor[indices]
            minibatch_env_index = env_index_tensor[indices]

            elbo_value = bbvi_update(
                minibatch, minibatch_env_index, MTM, optimizer,
                docs_word_matrix_tensor.size(0)
            )
            elbo_accumulator += elbo_value

        # empirical bayes step
        if empirical_bayes:
            empirical_bayes_update(MTM, optimizer_hyper)

        # average ELBO
        avg_elbo = elbo_accumulator / (docs_word_matrix_tensor.size(0) / minibatch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Average ELBO: {avg_elbo:.4f}")

def train_model_two_phase(mtm_model, docs_word_matrix_tensor, env_index_tensor,
                          num_epochs_phase1=10, num_epochs_phase2=150,
                          minibatch_size=1024, lr=0.01):
    device = next(mtm_model.parameters()).device

    if mtm_model.empirical_bayes:
        print("Phase 1: Training with Empirical Bayes...")
        train_model(
            mtm_model, docs_word_matrix_tensor, env_index_tensor,
            num_epochs=num_epochs_phase1,
            minibatch_size=minibatch_size,
            lr=lr,
            empirical_bayes=True
        )

        learned_alpha_a, learned_alpha_b = mtm_model.get_learned_parameters()
        print(f"[PHASE 1] Learned alpha_a={learned_alpha_a:.4f}, alpha_b={learned_alpha_b:.4f}")
        
        alpha_a_softplus = F.softplus(mtm_model.log_alpha_a).item()
        alpha_b_softplus = F.softplus(mtm_model.log_alpha_b).item()
        print(f"[PHASE 1] Softplus: alpha_a={alpha_a_softplus:.4f}, alpha_b={alpha_b_softplus:.4f}")

        print("\nPhase 2: Training with learned fixed parameters...")
        new_model = MTM(
            num_topics=mtm_model.num_topics,
            num_words=mtm_model.num_words,
            num_envs=mtm_model.num_envs,
            device=device,
            empirical_bayes=False,
            fixed_alpha_a=alpha_a_softplus,
            fixed_alpha_b=alpha_b_softplus
        ).to(device)

        train_model(
            new_model, docs_word_matrix_tensor, env_index_tensor,
            num_epochs=num_epochs_phase2,
            minibatch_size=minibatch_size,
            lr=lr,
            empirical_bayes=False
        )
        return new_model

    else:
        print("Training with fixed parameters (no Empirical Bayes).")
        train_model(
            mtm_model, docs_word_matrix_tensor, env_index_tensor,
            num_epochs=num_epochs_phase2,
            minibatch_size=minibatch_size,
            lr=lr,
            empirical_bayes=False
        )
        return mtm_model

##########################################################
# 5. main() 
##########################################################
def main():
    parser = argparse.ArgumentParser(
        description="Train the Multi-Environment Topic Model (MTM)."
    )
    parser.add_argument("--data_prefix", type=str, default="ideology",
                        help="Prefix for the dataset (e.g., 'ideology').")
    parser.add_argument("--data_folder", type=str, default=None,
                        help="Path to the preprocessed data folder. If not provided, "
                             "defaults to 'data/preprocessed_data'.")
    parser.add_argument("--num_topics", type=int, default=20,
                        help="Number of topics for the MTM.")
    parser.add_argument("--phase1_epochs", type=int, default=10,
                        help="Number of epochs in Phase 1 (EB).")
    parser.add_argument("--phase2_epochs", type=int, default=150,
                        help="Number of epochs in Phase 2 (Fixed).")
    parser.add_argument("--minibatch_size", type=int, default=1024,
                        help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate.")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Set data folder and prefix
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = args.data_folder or os.path.join(script_dir, "data", "preprocessed_data")
    data_prefix = args.data_prefix

    # Check if folder exists
    folder_path = os.path.join(data_folder, data_prefix)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"[ERROR] Folder '{folder_path}' does not exist. "
                                f"Ensure the data folder and prefix are correct.")

    print(f"[INFO] Loading data from: {folder_path}")
    processed_data = load_preprocessed_data(
        folder_path=folder_path, prefix=data_prefix, device=device
    )

    # Initialize and train the model as before
    print("[INFO] Initializing MTM model...")
    mtm_model = MTM(
        num_topics=args.num_topics,
        num_words=processed_data['vocab_size'],
        num_envs=processed_data['num_envs'],
        device=device,
        empirical_bayes=True  # You can flip this if you don't want EB
    ).to(device)

    print("[INFO] Starting training in two phases...")
    final_model = train_model_two_phase(
        mtm_model,
        docs_word_matrix_tensor=processed_data['docs_word_matrix_tensor'],
        env_index_tensor=processed_data['env_index_tensor'],
        num_epochs_phase1=args.phase1_epochs,
        num_epochs_phase2=args.phase2_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr
    )
    print("[INFO] Training complete.")

if __name__ == "__main__":
    main()
