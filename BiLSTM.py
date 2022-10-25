import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
import math
from build_word_vocab import load_vocab, load_type

class BiLSTM_softmax(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, num_attention_heads=8):
        super(BiLSTM_softmax, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # Soft-lexicon
        _, self.lexicon_size = load_vocab()
        self.lexicon_embeds = nn.Embedding(self.lexicon_size, embedding_dim)

        _, self.type_size = load_type()
        self.type_embeds = nn.Embedding(self.type_size, embedding_dim)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 5, 1024,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(2048, self.tagset_size)
        self.dropout = nn.Dropout(0.2)
        self.crf = CRF(self.tagset_size, batch_first=True)

        # Self-attention
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_dim / num_attention_heads)
        self.all_head_size = hidden_dim

        self.key_layer = nn.Linear(embedding_dim, hidden_dim)
        self.query_layer = nn.Linear(embedding_dim, hidden_dim)
        self.value_layer = nn.Linear(embedding_dim, hidden_dim)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_data, batch_B, batch_M, batch_E, batch_S, B_type, M_type, E_type, S_type, model_type, labels=None):
        embeds = self.word_embeds(input_data)

        # B_set
        mask_B = batch_B.gt(0).unsqueeze(-1)
        for b in range(mask_B.shape[0]):
            for s in range(mask_B.shape[1]):
                mask_B[b][s][0] = True
        batch_B = self.lexicon_embeds(batch_B)
        batch_B = torch.mul(batch_B, mask_B)
        batch_B = torch.sum(batch_B, dim=-2)

        mask_B_sum = torch.sum(mask_B, dim=-2)
        batch_B = batch_B / mask_B_sum

        # type of B_set
        mask_B_type = B_type.gt(0).unsqueeze(-1)
        for b in range(mask_B_type.shape[0]):
            for s in range(mask_B_type.shape[1]):
                mask_B_type[b][s][0] = True
        B_type = self.lexicon_embeds(B_type)
        B_type = torch.mul(B_type, mask_B_type)
        B_type = torch.sum(B_type, dim=-2)

        mask_B_type_sum = torch.sum(mask_B_type, dim=-2)
        B_type = B_type / mask_B_type_sum

        # print(torch.cat((batch_B, B_type), dim=-1).shape)

        # M_set
        mask_M = batch_M.gt(0).unsqueeze(-1)
        for b in range(mask_M.shape[0]):
            for s in range(mask_M.shape[1]):
                mask_M[b][s][0] = True
        batch_M = self.lexicon_embeds(batch_M)
        batch_M = torch.mul(batch_M, mask_M)
        batch_M = torch.sum(batch_M, dim=-2)
        mask_M_sum = torch.sum(mask_M, dim=-2)
        batch_M = batch_M / mask_M_sum

        # type of M_set
        mask_M_type = M_type.gt(0).unsqueeze(-1)
        for b in range(mask_M_type.shape[0]):
            for s in range(mask_M_type.shape[1]):
                mask_M_type[b][s][0] = True
        M_type = self.lexicon_embeds(M_type)
        M_type = torch.mul(M_type, mask_M_type)
        M_type = torch.sum(M_type, dim=-2)
        mask_M_type_sum = torch.sum(mask_M_type, dim=-2)
        M_type = M_type / mask_M_type_sum

        # E_set
        mask_E = batch_E.gt(0).unsqueeze(-1)
        for b in range(mask_E.shape[0]):
            for s in range(mask_E.shape[1]):
                mask_E[b][s][0] = True
        batch_E = self.lexicon_embeds(batch_E)
        batch_E = torch.mul(batch_E, mask_E)
        batch_E = torch.sum(batch_E, dim=-2)
        mask_E_sum = torch.sum(mask_E, dim=-2)
        batch_E = batch_E / mask_E_sum

        # type of E_set
        mask_E_type = E_type.gt(0).unsqueeze(-1)
        for b in range(mask_E_type.shape[0]):
            for s in range(mask_E_type.shape[1]):
                mask_E_type[b][s][0] = True
        E_type = self.lexicon_embeds(E_type)
        E_type = torch.mul(E_type, mask_E_type)
        E_type = torch.sum(E_type, dim=-2)
        mask_E_type_sum = torch.sum(mask_E_type, dim=-2)
        E_type = E_type / mask_E_type_sum

        # S_set
        batch_S = self.lexicon_embeds(batch_S)
        batch_S = batch_S.squeeze(-2)

        # type of S_set
        S_type = self.lexicon_embeds(S_type)
        S_type = S_type.squeeze(-2)

        embeds = torch.cat((embeds, B_type, M_type, E_type, S_type), dim=-1)

        x, _ = self.lstm(embeds)
        x = self.dropout(x)

        logits = self.hidden2tag(x)

        if model_type == 'Bi-LSTM':
            outputs = (logits,)

            if labels is not None:
                loss_mask = labels.gt(-1)
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if loss_mask is not None:
                    active_loss = loss_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.tagset_size)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs
        else:
            loss_mask = input_data.gt(0)
            output = self.crf.decode(logits, mask=loss_mask)

            outputs = (output,)

            if labels is not None:
                loss = -self.crf(logits, labels, mask=loss_mask)
                outputs = (loss,) + outputs

            return outputs