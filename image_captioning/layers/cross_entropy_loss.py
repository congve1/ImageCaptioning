import torch
from torch import nn
from torch.nn import functional as F

from image_captioning.modeling.utils import to_contiguous


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, inputs, targets, cap_lens):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        masks = self._build_masks(batch_size, seq_len, cap_lens, inputs.device)
        inputs = F.log_softmax(inputs, dim=2)
        inputs = to_contiguous(inputs).view(-1, inputs.shape[-1])
        targets = to_contiguous(targets).view(-1, 1)
        masks = to_contiguous(masks).view(-1, 1)
        output = - inputs.gather(1, targets) * masks
        output = torch.sum(output) / batch_size
        return output

    @staticmethod
    def _build_masks(batch_size, seq_len, cap_lens, device):
        masks = torch.zeros((batch_size, seq_len)).to(device)
        for i in range(batch_size):
            masks[i, :cap_lens[i]] = torch.ones(cap_lens[i])
        return masks
