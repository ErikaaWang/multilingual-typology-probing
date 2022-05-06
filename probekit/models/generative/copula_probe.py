from overrides import overrides
from typing import Dict, Any, List

from copulae import GaussianCopula

import torch
from torch.distributions import MultivariateNormal

from probekit.utils.types import PyTorchDevice
from probekit.models.generative_probe import GenerativeProbe


class CopulaProbe(GenerativeProbe):
    """Defines a generative model where embeddings are drawn from a Gaussian copula."""
    def __init__(
            self, class_probabilities: torch.Tensor,
            embeddings: List[torch.Tensor],
            #copulas: List[torch.Tensor],
            #means: List[torch.Tensor],
            #covariance_matrices: List[torch.Tensor],
            device: PyTorchDevice = "cpu"):

        assert torch.isclose(class_probabilities.sum(), torch.tensor(1.)), \
            "Class probabilities must sum up to 1."

        assert len(embeddings) == class_probabilities.shape[0], \
            f"Incorrect number of means provided. Expected {class_probabilities.shape[0]}, \
                got {len(embeddings)}."

        #assert len(copulas) == class_probabilities.shape[0], \
        #    f"Incorrect number of covariances provided. Expected {class_probabilities.shape[0]}, \
        #        got {len(copulas)}."
        
        self._class_probabilities = class_probabilities.to(device)
        self._embedding_distributions = [
            GaussianCopula(dim=embedding.shape[1]).fit(embedding.numpy())
            for embedding in embeddings 
        ]
        #self._embedding_distributions = [
        #    MultivariateNormal(loc=mean.to(device), covariance_matrix=cov.to(device))
        #    for mean, cov in zip(means, covariance_matrices)]

        super().__init__(device=device, num_classes=len(class_probabilities))

    @classmethod
    def from_specification(
            cls, parameters: Dict[str, Any],
            device: PyTorchDevice = "cpu"):
        return cls(
            class_probabilities=parameters["class_probabilities"],
            embeddings=parameters["embeddings"], #copulas=parameters["copulas"],
            device=device)

    @overrides
    def _log_prob_input_given_class(self, inputs: torch.Tensor):
        # Compute sample probs according to each distribution
        print(self._embedding_distributions)
        log_probs = [x.log_prob(inputs) for x in self._embedding_distributions]
        log_probs_tensor = torch.stack(log_probs, dim=1)
        assert log_probs_tensor.shape == (inputs.shape[0], self._num_classes)

        return log_probs_tensor

    @overrides
    def _log_prob_class(self):
        return self._class_probabilities.log()
