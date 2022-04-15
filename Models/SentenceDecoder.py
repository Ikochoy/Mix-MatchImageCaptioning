from json import encoder
from unicodedata import bidirectional
import torch.nn as nn
import torch



class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)

        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, encoder_outputs, captions):
        """
        Write forward function for the LSTM decoder for the show and tell image captioning task.
        Returns predicted captions and its probability.
        """
        embeddings = self.embedding(captions)
        max_captions_length = captions.shape[-1]

        # # batch_size x seq_len x hidden_size
        h_t = torch.zeros(encoder_outputs.shape[0], 1, self.hidden_size)
        c_t = torch.zeros(encoder_outputs.shape[0], 1, self.hidden_size)

        # batch_size x seq_len x vocab_size
        logits = []
        captions = []
        caption_probs = []

        for i in range(max_captions_length-1):
            embeddings_i = embeddings[:, i, :]
            inputs = torch.cat((encoder_outputs, embeddings_i), dim=1)

            h_t, c_t = self.rnn(inputs, (h_t, c_t))

            logit = self.out(h_t).squeeze(1)
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
    def __init__(self, vocab_size, hidden_size, cnn_last_layer_shape):
        """Initialize RNN Cell."""
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.Whi = nn.Linear(cnn_last_layer_shape, hidden_size)  # Last layer of CNN dimensions
        self.Whx = nn.Linear(vocab_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.Woh = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x, h_t_prev, CNN_last_layer, t):
        """Compute forward pass for this RNN cell."""

        b_v = self.Whi(CNN_last_layer)

        if t == 1:
            h_t = self.relu(self.Whx(x) + self.Whh(h_t_prev) + b_v)
        else:
            h_t = self.relu(self.Whx(x) + self.Whh(h_t_prev))

        y_t = self.softmax(self.Woh(h_t))

        return y_t, h_t


class MyRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, CNN_last_layer_size, device):
        """Initialize RNN."""
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cnn_last_layer_size = CNN_last_layer_size


        self.rnn_cell = MyRNNCell(vocab_size, hidden_size, CNN_last_layer_size)  # TODO: Double check if the last layer size is correct

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.device = device


    def forward(self, captions, encoder_outputs):
        """
        Write forward function for the RNN decoder for the Karpathy & Li Fei-Fei paper.
        Returns the predicted word sequence and the hidden state sequence.
        """
        # batch_size x seq_len x hidden_size
        # B is batch size, L is sequence length, and D is the number of features.
        embeddings = self.embedding(captions)
        embeddings = torch.cat((torch.zeros(embeddings.shape[0], 1, embeddings.shape[2]), embeddings), dim=1)  # The START vector is just a 0 vector here

        batch_size, seq_len, n_feat = embeddings.size()

        # Stores outputs of RNN cell
        output_arr = torch.zeros((batch_size, seq_len, self.vocab_size))
        hidden_arr = torch.zeros((batch_size, seq_len, self.hidden_size))

        # Send to GPU. Send Tensors created in a model to the same device as input Tensors.
        embeddings = embeddings.to(self.device)
        output_arr = output_arr.float().to(self.device)
        hidden_arr = hidden_arr.float().to(self.device)

        # Initialize RNN hidden state: h_0 is the 0 vector
        hidden = torch.zeros(batch_size, self.hidden_size, device=self.device)

        for i in range(seq_len):

            # For each iteration, compute RNN on input for current position
            output, hidden = self.rnn_cell(x=embeddings[:, i, :], h_t_prev=hidden, CNN_last_layer=encoder_outputs, t=i)

            output_arr[:, i, :] = output
            hidden_arr[:, i, :] = hidden

        return output_arr, hidden_arr


class SentenceDecoder(nn.Module):

    def __init__(self, choice, vocab_size, hidden_size):
        self.choice = choice
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if choice == 'LSTM':
            self.rnn = LSTMDecoder(vocab_size=vocab_size, hidden_size=hidden_size)
        elif choice == 'RNN':

            self.rnn = MyRNN(vocab_size=vocab_size, hidden_size=hidden_size, CNN_last_layer=4096, device=device)
    
    def forward(self, encoder_outputs, captions):
        return self.rnn.forward(captions=captions, encoder_outputs=encoder_outputs)

    def teacher_forcing(self, features, captions):
        embeddings = self.rnn.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        rnn_output = self.rnn(embeddings)[0]
        rnn_output = rnn_output[:,1:,:]
        outputs = self.out(rnn_output)
        return outputs
