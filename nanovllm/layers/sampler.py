import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        if temperatures is None:
            return logits.argmax(dim=-1)
        if torch.all(temperatures <= 1e-10):
            return logits.argmax(dim=-1)
        return self.sample(logits, temperatures)

    @torch.compile
    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy = temperatures <= 1e-10
        greedy_tokens = logits.argmax(dim=-1)
        safe_temps = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits.div(safe_temps.unsqueeze(dim=1)), dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy, greedy_tokens, sample_tokens)
