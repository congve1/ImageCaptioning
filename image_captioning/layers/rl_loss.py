import torch
from torch import nn


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, log_probs, rewards, sample_seqs, vocab):
        batch_size = log_probs.size(0)
        seq_length = log_probs.size(1)
        end_idx = vocab['<end>']
        pad_idx = vocab['<pad>']
        masks = sample_seqs.new_ones(
            sample_seqs.size(), dtype=torch.float
        ).to(sample_seqs.device)
        bool_indexes = (sample_seqs == end_idx)
        end_indexes = torch.nonzero(bool_indexes)
        previous_index = -1
        for index in end_indexes:
            if previous_index == index[0]:
                continue
            previous_index = index[0]
            bool_indexes[index] = 0
            bool_indexes[index[0], index[1]+1:] = 1
        masks[bool_indexes] = 0.
        masks = masks.reshape(-1)
        log_probs = log_probs.view(-1)
        rewards = rewards.view(-1)
        reward_loss = -log_probs * rewards * masks
        reward_loss = torch.sum(reward_loss) / batch_size

        return reward_loss
