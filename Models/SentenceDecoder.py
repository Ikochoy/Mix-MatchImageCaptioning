import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wix = nn.Linear(input_size, hidden_size)
        self.Wim = nn.Linear(hidden_size, hidden_size)

        self.Wfx = nn.Linear(input_size, hidden_size)
        self.Wfm = nn.Linear(hidden_size, hidden_size)

        self.Wox = nn.Linear(input_size, hidden_size)
        self.Wom = nn.Linear(hidden_size, hidden_size)

        self.Wcx = nn.Linear(input_size, hidden_size)
        self.Wcm = nn.Linear(hidden_size, hidden_size)

        self.probabilities = []

    def forward(self, x, m_prev, c_prev):
        """Forward pass of the LSTM computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size
            c_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
            c_new: batch_size x hidden_size
        """
        i_t = torch.sigmoid(self.Wix(x) + self.Wim(m_prev))
        f_t = torch.sigmoid(self.Wfx(x) + self.Wfm(m_prev))
        o_t = torch.sigmoid(self.Wox(x) + self.Wom(m_prev))
        c_new = f_t * c_prev + i_t * torch.tanh(self.Wcx(x) + self.Wcm(m_prev))
        m_new = o_t * c_new
        return m_new, c_new


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_outputs, captions):
        """
        Write forward function for the LSTM decoder for the show and tell image captioning task.
        Returns predicted captions and its probability.
        """
        # batch_size x seq_len x hidden_size
        embeddings = self.embedding(captions)
        embeddings = torch.cat((torch.zeros(embeddings.shape[0], 1, embeddings.shape[2]), embeddings), dim=1)
        # batch_size x seq_len x hidden_size
        h_t = torch.zeros(encoder_outputs.shape[0], 1, self.hidden_size)
        c_t = torch.zeros(encoder_outputs.shape[0], 1, self.hidden_size)
        # batch_size x seq_len x vocab_size
        logits = []
        captions = []
        caption_probs = []
        for i in range(embeddings.shape[1]):
            h_t, c_t, p_t = self.rnn(embeddings[:, i, :], h_t, c_t)
            logit = self.out(h_t)
            logits.append(logit)
            # get the max probability caption
            pred_cap_probs, pred_cap_index = torch.max(logit, dim=1)
            captions.append(pred_cap_index)
            caption_probs.append(pred_cap_probs)
        logits = torch.stack(logits, dim=1)
        captions = torch.stack(captions, dim=1)
        caption_probs = torch.stack(caption_probs, dim=1)
        return logits, caption_probs, captions
        


class MyRNNCell(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        """Initialize RNN Cell."""
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        

        # self.relu = nn.ReLU()
        # self.Ww = nn.Linear(obs_dim, obs_dim) # TODO: Check this

        # self.We = nn.Linear(obs_dim, hidden_size)

        # self.Wf = nn.Linear(hidden_size, hidden_size)

        # self.Wb = nn.Linear(hidden_size, hidden_size)

        # self.Wd = nn.Linear(hidden_size, hidden_size)

        self.Whi = 1 # Last layer of CNN
        self.Whx = nn.Linear(obs_dim, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.Woh = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    
    def forward(self, x, h_f, h_b, CNN_last_layer):
        """Compute forward pass for this RNN cell."""
        # x_t = self.Ww() # Some identity?

        # e_t = self.relu(self.We(x_t))

        # h_ft = self.relu(e_t + self.Wf(h_f))

        # h_bt = self.relu(e_t + self.Wb(h_b))

        # s_t = self.relu(self.Wd(h_ft + h_bt))

        b_v = self.Whi(CNN_last_layer)
        h_t = self.relu(self.Whx(x) + self.Whh(h_t_prev)) # TODO
        y_t = self.softmax(self.Woh(h_t))

        return s_t
    
class MyRNN(nn.Module):
    def __init__(self, obs_dim, hidden_size, output_dim):
        """Initialize RNN."""
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size

        self.rnn_cell = MyRNNCell(obs_dim, hidden_size)

    def forward(self, x):
        """Compute forward pass on sequence x.
        
        Input sequence x has shape (B x L x D), where:
        B is batch size, L is sequence length, and D is the number of features.
        """
        batch_size, seq_len, n_feat = x.size()
        
        # Stores outputs of RNN cell
        output_arr = torch.zeros((batch_size, seq_len, self.output_dim))
        hidden_arr = torch.zeros((batch_size, seq_len, self.hidden_size))
        
        # Send to GPU. This is a gotcha, make sure to send Tensors created
        # in a model to the same device as input Tensors.
        output_arr = output_arr.float().to(x.device)
        hidden_arr = hidden_arr.float().to(x.device)

        hidden = self.init_hidden(batch_size, x.device)

        for i in range(seq_len):
            # For each iteration, compute RNN on input for current position
            output, hidden = self.rnn_cell(x[:, i, :], hidden)

            output_arr[:, i, :] = output
            hidden_arr[:, i, :] = hidden

        return output_arr, hidden_arr

    def init_hidden(self, batch_size, device):
        """Initialize RNN hidden state.
        
        Some people advocate for using random noise instead of zeros, or 
        training for the initial state. Personally, I don't know if it matters!
        """
        return torch.zeros(batch_size, self.hidden_size, device=device)


class SentenceDecoder(nn.Module):
    def __init__(self, choice, vocab_size, hidden_size):
        if choice == 'LSTM':
            self.rnn = LSTMDecoder(vocab_size=vocab_size, hidden_size=hidden_size)
        elif choice == 'RNN':
            # not sure what obs_dim should be
            self.rnn = MyRNN(obs_dim=obs_dim, hidden_size=hidden_size, output_dim=vocab_size)
    
    def forward(self, captions, encoder_outputs):
        return self.rnn.forward(captions=captions, encoder_outputs=encoder_outputs)
        
