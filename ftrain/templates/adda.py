"""Adversarial Discriminative Domain Adaptation (ADDA) PyTorch Lightning template for Domain Adaptation.

This module implements the ADDA framework, which facilitates unsupervised domain adaptation.
The primary goal is to adapt a model trained on a source domain to perform well on a target domain without labeled target data.
This is achieved by learning domain-invariant representations using adversarial training.

References:
    - Adversarial Discriminative Domain Adaptation (ADDA)
      Paper: https://arxiv.org/abs/1702.05464

Classes:
    ADDA(pl.LightningModule): Main class for ADDA implementation.

Attributes:
    __all__ (List[str]): Exports a list of symbols from this module.


Notes:
    - Assumes source_mapper and discriminator are defined and initialized.
    - Using a custom target_mapper is not recommended; initializing it with source_mapper weights is advised.
    - Need the trainloader to be a tuple of (source, target) data.
    

Todo:
    - Explore advanced domain adaptation techniques for integration.
"""

import warnings
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn  # noqa: TCH002
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT  # noqa: TCH002

__all__ = ['ADDA']


class ADDA(pl.LightningModule):
    """Adversarial Discriminative Domain Adaptation (ADDA) PyTorch Lightning template for Domain Adaptation.

    This class implements the ADDA framework, which enables unsupervised domain adaptation. The main objective is to adapt a model trained on a source domain for optimal performance on a target domain, even when target domain labels are unavailable. The adaptation is achieved by learning domain-invariant representations through adversarial training.

    Args:
        source_mapper (nn.Module): A pretrained neural network that maps source domain data to a latent space.
        discriminator (nn.Module): Discriminator neural network for distinguishing between source and target data.
        target_mapper (nn.Module | None): Neural network to map target domain data to the latent space. If None, source_mapper is copied.
        wasserstein_loss (bool): Flag indicating whether to use the Wasserstein loss. Default is False.
        gradient_penalty (float | None): Weight for gradient penalty when using Wasserstein loss. Default is None.
        n_critic (int): Number of critic iterations during training. Must be greater than 0. Default is 5.

    Attributes:
        __all__ (List[str]): Exports a list of symbols from this module.

    Methods:
        forward(x_target: torch.Tensor) -> torch.Tensor: Forward pass for mapping target data to latent space.
        configure_optimizers() -> List[nn.Module]: Configures optimizers for target_mapper and discriminator.
        training_step(batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT: Training step for the ADDA model.
        on_train_start() -> None: Executed before training starts.
        on_train_end() -> None: Executed after training ends.

    Notes:
        - It's assumed that source_mapper and discriminator are defined and initialized externally.
        - Using a custom target_mapper is discouraged due to potential convergence issues. Initializing it with source_mapper weights is recommended.

    Todo:
        - Investigate advanced domain adaptation techniques for potential integration.

    """

    def __init__(
        self,
        source_mapper: nn.Module,
        discriminator: nn.Module,
        target_mapper: nn.Module | None = None,
        wasserstein_loss: bool = False,
        gradient_penalty: float | None = None,
        n_critic: int = 5,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the ADDA model.

        Args:
            source_mapper (nn.Module): A pretrained neural network that maps source domain data to a latent space.
            discriminator (nn.Module): A discriminator neural network that distinguishes source from target data.
            target_mapper (nn.Module | None): A neural network to map target domain data to the latent space. If None, a copy of source_mapper is used.
            wasserstein_loss (bool): Whether to use Wasserstein loss instead of GAN loss. Default is False.
            gradient_penalty (float | None): Weight for the gradient penalty when using Wasserstein loss. Default is None.
            n_critic (int): Number of critic iterations for training. Must be greater than 0. Default is 5.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Notes:
            - If target_mapper is None, it is suggested to initialize it with pretrained source_mapper weights.
            - The module assumes that source_mapper and discriminator are externally defined and properly initialized.

        Attributes:
            source_mapper (nn.Module): The neural network that maps source domain data to a latent space.
            target_mapper (nn.Module): The neural network that maps target domain data to the latent space.
            discriminator (nn.Module): The discriminator neural network for domain differentiation.
            _real_label (int): Label used for real samples during training.
            _fake_label (int): Label used for fake samples during training.
            _use_wasserstein_loss (bool): Flag indicating whether Wasserstein loss is used.
            _gradient_penalty (float | None): Weight for the gradient penalty.
            _n_critic (int): Number of critic iterations per training step.
            automatic_optimization (bool): Whether to enable manual optimization. Default is False.
        """
        super().__init__(*args, **kwargs)

        # This is the source mapper which maps the source data to a latent space
        # This is supposed to be a pretrained model
        self._source_mapper = source_mapper

        if target_mapper is not None:
            warnings.warn(
                'Using custom target mapper is not recommended. \
                The target mapper is suggested to be a copy of the source mapper with pretrained weights. \
                Target mapper will probably converge to degenerate solution.',
                stacklevel=1,
            )
            self.target_mapper = target_mapper
        else:
            # This is the target representation which maps the target data to a latent space
            # Initially, this is a copy of the source representation but it will be updated during training
            self.target_mapper = deepcopy(source_mapper)

        # This is the discriminator which tries to distinguish between the source and target data
        self.discriminator = discriminator

        # Lables
        self._real_label = 1
        self._fake_label = 0

        # Wasserstein loss
        self._use_wasserstein_loss = wasserstein_loss

        # Gradient penalty
        if not self._use_wasserstein_loss:
            assert (
                gradient_penalty is None
            ), "Gradient penalty can only be used with the Wasserstein loss."

        # Gradient penalty
        if gradient_penalty is not None:
            self._gradient_penalty = gradient_penalty

        # Number of critic iterations
        assert n_critic > 0, "The number of critic iterations must be greater than 0"
        self._n_critic = n_critic

        # Enable manual optimization
        self.automatic_optimization = False

    def forward(self, x_target: torch.Tensor) -> torch.Tensor:
        """Forward the target data through the target mapper into latent space.

        Args:
            x_target (torch.Tensor): Samples from the target domain.

        Returns:
            torch.Tensor: The latent representation of the target data.
        """
        return self.target_mapper(x_target)

    def configure_optimizers(self) -> list[nn.Module]:
        """Configure the optimizers for the target mapper and discriminator."""
        # Optimizer for the target mapper
        lr = 1e-4
        # Adam init
        beta1, beta2 = 0.0, 0.9

        return [
            torch.optim.Adam(
                self.target_mapper.parameters(), lr=lr, betas=(beta1, beta2)
            ),
            torch.optim.Adam(
                self.discriminator.parameters(), lr=lr, betas=(beta1, beta2)
            ),
        ]

    def training_step(
        self, batch: torch.Tensor, batch_idx: int  # noqa: ARG002
    ) -> STEP_OUTPUT:
        """Training step for the ADDA model.

        Args:
            batch (torch.Tensor): Batch of source and target data. (source, target)
            batch_idx (int): Batch index.

        Returns:
            STEP_OUTPUT: _description_
        """
        # Get the source and target data
        source_t, target_t = batch

        with torch.no_grad():
            # Get the latent representation of the source data no gradient tracking size source is freeze
            source_latent_t = self._source_mapper(source_t)

        # Get the latent representation of the target data
        target_latent_t = self.target_mapper(target_t)  # Has gradient tracking

        # do the discriminator step for n_critic times
        for _ in range(self._n_critic):
            # Discriminator step
            loss = self._discriminator_step(
                source_latent_t, target_latent_t.detach()
            )  # Detach the target latent representation since we don't want to update the target mapper

            # Log the discriminator loss
            self.log(
                "discriminator_loss", loss, on_step=True, on_epoch=True, prog_bar=True
            )

        # Target step
        loss_t = self._target_step(
            target_latent_t
        )  # Do not detach the target latent representation since we want to update the target mapper

        # Log the target loss
        self.log("target_loss", loss_t, on_step=True, on_epoch=True, prog_bar=True)

        return {
            "loss": loss_t,
        }

    def _discriminator_step(
        self, source_latent_t: torch.Tensor, target_latent_t: torch.Tensor
    ) -> torch.Tensor:
        # Compute discriminator loss conditioned on the GAN loss or Wasserstein loss

        # Get the discriminator optimizer
        _, discriminator_optimizer = self.optimizers()

        # Zero the gradients
        self.discriminator.zero_grad()

        # Compute the discriminator loss
        if self._use_wasserstein_loss:
            loss = self._compute_wasserstein_loss_for_discriminator(
                source_latent_t, target_latent_t
            )

        else:
            loss = self._compute_gan_loss_for_discriminator(
                source_latent_t, target_latent_t
            )

        # Backpropagate the loss
        self.manual_backward(loss)

        # Update the discriminator
        discriminator_optimizer.step()

        return loss

    def _target_step(self, target_latent_t: torch.Tensor) -> torch.Tensor:
        # Compute target loss conditioned on the GAN loss or Wasserstein loss

        # Get the target mapper optimizer
        target_mapper_optimizer, _ = self.optimizers()

        # Zero the gradients
        self.target_mapper.zero_grad()

        # Compute the target loss
        if self._use_wasserstein_loss:
            loss = self._compute_wasserstein_loss_for_target(target_latent_t)
        else:
            loss = self._compute_gan_loss_for_target(target_latent_t)

        # Backpropagate the loss
        self.manual_backward(loss)

        # Update the target mapper
        target_mapper_optimizer.step()

        return loss

    def _compute_gan_loss_for_discriminator(
        self,
        source_latent_t: torch.Tensor,
        target_latent_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the GAN loss for the discriminator."""
        # Forward the source latent representation through the discriminator
        source_discriminator_output_t = self.discriminator(source_latent_t)

        # Forward the target latent representation through the discriminator
        target_discriminator_output_t = self.discriminator(target_latent_t)

        # Compute the cross entropy loss for the source data and target data
        return F.binary_cross_entropy(
            source_discriminator_output_t,
            torch.full(
                (source_discriminator_output_t.shape[0], 1),
                self._real_label,
                dtype=torch.float32,
                device=self.device,
            ),
        ) + F.binary_cross_entropy(
            target_discriminator_output_t,
            torch.full(
                (target_discriminator_output_t.shape[0], 1),
                self._fake_label,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _compute_wasserstein_loss_for_discriminator(
        self,
        source_latent_t: torch.Tensor,
        target_latent_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Wasserstein loss for the discriminator."""
        # Forward the source latent representation through the discriminator
        source_discriminator_output_t = self.discriminator(source_latent_t)

        # Forward the target latent representation through the discriminator
        target_discriminator_output_t = self.discriminator(target_latent_t)

        # Compute the Wasserstein loss for the source data and target data
        wasserstein_l = -torch.mean(source_discriminator_output_t) + torch.mean(
            target_discriminator_output_t
        )

        # If the gradient penalty is 0, return the Wasserstein loss
        if not hasattr(self, "_gradient_penalty"):
            return wasserstein_l

        # Compute the gradient penalty

        # Calculate interpolation
        alpha = torch.rand(
            source_latent_t.shape[0],
            *[1 for i in range(len(source_discriminator_output_t.shape) - 1)],
            device=self.device,
        )
        # Interpolation
        x_hat = torch.lerp(source_latent_t, target_latent_t, alpha)
        # Set requires_grad attribute of tensor. Important for gradient penalty
        x_hat.requires_grad_(True)

        # Forward the interpolated latent representation through the discriminator
        output = self.discriminator(x_hat)

        # Compute the gradient of the output with respect to x_hat
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=x_hat,
            grad_outputs=torch.ones_like(
                output, dtype=torch.float32, device=self.device
            ),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute the gradient penalty
        gradient_penalty = (
            self._gradient_penalty * (torch.norm(gradients, p='fro') - 1) ** 2
        )
        return wasserstein_l + gradient_penalty

    def _compute_gan_loss_for_target(
        self, target_latent_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the GAN loss for the target mapper."""
        # Forward the target latent representation through the discriminator
        target_discriminator_output_t = self.discriminator(target_latent_t)

        # Compute the cross entropy loss for the target data
        return F.binary_cross_entropy(
            target_discriminator_output_t,
            torch.full(
                (target_discriminator_output_t.shape[0], 1),
                self._real_label,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _compute_wasserstein_loss_for_target(
        self, target_latent_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Wasserstein loss for the target mapper."""
        # Forward the target latent representation through the discriminator
        target_discriminator_output_t = self.discriminator(target_latent_t)

        # Compute the Wasserstein loss for the target data
        return -1 * torch.mean(target_discriminator_output_t)
