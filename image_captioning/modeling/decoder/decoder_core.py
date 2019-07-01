import torch
import torch.nn as nn
from torch.nn import functional as F
from image_captioning.modeling.utils import cat
from image_captioning.modeling import registry
from image_captioning.modeling.decoder.decoder_attention import build_decoder_attention


def build_decoder_core(cfg, vocab_size, core_name):
    assert core_name in registry.DECODER_CORES,\
        "DECODER.CORE: {} is not registered in registry".format(
            core_name
    )
    return registry.DECODER_CORES[core_name](cfg, vocab_size)


@registry.DECODER_CORES.register("TopDownCore")
class TopDownCore(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(TopDownCore, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB
        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            hidden_size * 2, hidden_size
        )
        self.logit = nn.Linear(hidden_size, vocab_size)
        self.attention = build_decoder_attention(cfg, "TopDownAttention")
        self.filter_h = nn.Sequential(
            nn.Conv1d(1, 1, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        filter_h_att = self.filter_h(next_h_att.unsqueeze(1)).squeeze(1)
        weighted_att_features, att_weights = self.attention(att_feats, filter_h_att)
        input_lang_lstm = cat([weighted_att_features, next_h_att], 1)
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states, att_weights

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param)


@registry.DECODER_CORES.register("TopDownCoreNoConv")
class TopDownCoreNoConv(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(TopDownCoreNoConv, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB

        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            hidden_size * 2, hidden_size
        )
        self.logit = nn.Linear(hidden_size, vocab_size)
        self.attention = build_decoder_attention(cfg, "TopDownAttention")
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        weighted_att_features, att_weights = self.attention(att_feats, next_h_att)
        input_lang_lstm = cat([weighted_att_features, next_h_att], 1)
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states, att_weights

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param)


@registry.DECODER_CORES.register("DualCore")
class ConvHiddenCore(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(ConvHiddenCore, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB

        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        feature_size = cfg.MODEL.ENCODER.FEATURE_SIZE
        locations = cfg.MODEL.ENCODER.ATT_SIZE ** 2
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            locations+hidden_size*2, hidden_size
        )
        self.logit = nn.Linear(hidden_size, vocab_size)
        self.filter_h = nn.Sequential(
            nn.Conv1d(1, 1, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attention = build_decoder_attention(cfg, "DualAttention")
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        filter_h_att = self.filter_h(next_h_att.unsqueeze(1)).squeeze(1)
        weighted_spatial_features,\
            weighted_channel_features,\
            att_weights = self.attention(att_feats, filter_h_att)
        input_lang_lstm = cat([weighted_spatial_features,
                               weighted_channel_features, next_h_att], 1)
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states, att_weights

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param)


@registry.DECODER_CORES.register("DualCoreNoConv")
class DualCoreNoConv(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(DualCoreNoConv, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB

        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        feature_size = cfg.MODEL.ENCODER.FEATURE_SIZE
        locations = cfg.MODEL.ENCODER.ATT_SIZE ** 2
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            locations+hidden_size*2, hidden_size
        )
        self.logit = nn.Linear(hidden_size, vocab_size)
        self.attention = build_decoder_attention(cfg, "DualAttention")
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        weighted_spatial_features,\
            weighted_channel_features,\
            att_weights = self.attention(att_feats, next_h_att)
        input_lang_lstm = cat([weighted_spatial_features,
                               weighted_channel_features, next_h_att], 1)
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states, att_weights

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param)


@registry.DECODER_CORES.register("ChannelCore")
class ChannelCore(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(ChannelCore, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB

        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        locations = cfg.MODEL.ENCODER.ATT_SIZE ** 2
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            hidden_size + locations, hidden_size
        )
        self.logit = nn.Linear(hidden_size, vocab_size)
        self.filter_h = nn.Sequential(
            nn.Conv1d(1, 1, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attention = build_decoder_attention(cfg, "ChannelAttention")
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        filter_h_att = self.filter_h(next_h_att.unsqueeze(1)).squeeze(1)
        weighted_channel_features, att_weights = self.attention(att_feats, filter_h_att)
        input_lang_lstm = cat([weighted_channel_features, next_h_att], 1)
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states, att_weights

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param)


@registry.DECODER_CORES.register("ChannelCoreNoConv")
class ChannelCoreNoConv(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(ChannelCoreNoConv, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB

        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        locations = cfg.MODEL.ENCODER.ATT_SIZE ** 2
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            hidden_size + locations, hidden_size
        )
        self.logit = nn.Linear(hidden_size, vocab_size)
        self.attention = build_decoder_attention(cfg, "ChannelAttention")
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        weighted_channel_features, att_weights = self.attention(att_feats, next_h_att)
        input_lang_lstm = cat([weighted_channel_features, next_h_att], 1)
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states, att_weights

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param)


@registry.DECODER_CORES.register("Baseline")
class Baseline(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(Baseline, self).__init__()
        self.dropout_prob = cfg.MODEL.DECODER.DROPOUT_PROB

        embedding_size = cfg.MODEL.DECODER.EMBEDDING_SIZE
        hidden_size = cfg.MODEL.DECODER.HIDDEN_SIZE
        locations = cfg.MODEL.ENCODER.ATT_SIZE ** 2
        self.att_lstm = nn.LSTMCell(
            embedding_size + hidden_size * 2, hidden_size
        )
        self.lang_lstm = nn.LSTMCell(
            hidden_size, hidden_size
        )
        self.logit = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def forward(self, xt, fc_feats, att_feats, hidden_states):
        att_feats = att_feats.mean(2).mean(1)
        fc_feats = (fc_feats+att_feats-att_feats) # avoid distributed training error
        prev_h_lang = hidden_states[0][1]
        prec_c_lang = hidden_states[1][1]
        prev_h_att = hidden_states[0][0]
        prev_c_att = hidden_states[1][0]
        input_att_lstm = cat([prev_h_lang, fc_feats, xt], 1)
        next_h_att, next_c_att= self.att_lstm(
            input_att_lstm, (prev_h_att, prev_c_att)
        )
        input_lang_lstm = next_h_att
        next_h_lang, next_c_lang = self.lang_lstm(
            input_lang_lstm, (prev_h_lang, prec_c_lang)
        )
        output = F.dropout(next_h_lang, self.dropout_prob, self.training)
        output = self.logit(output)
        hidden_states = (
            torch.stack([next_h_att, next_h_lang]),
            torch.stack([next_c_att, next_c_lang])
        )
        return output, hidden_states

    def _init_weights(self):
        """
        use orthogonal to initialize lstm weights
        Returns:

        """
        for name, param in self.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param)
