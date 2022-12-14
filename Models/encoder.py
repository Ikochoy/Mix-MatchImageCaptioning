import torch

class Encoder(torch.nn.Module):

    def __init__(
                self,
                source_vocab_size: int,
                pad_id: int = -1,
                word_embedding_size: int = 1024,
                num_hidden_layers: int = 2,
                hidden_state_size: int = 512,
                dropout: float = 0.1,
                cell_type: str = 'lstm'):
            '''Initialize the encoder
            '''

            if -source_vocab_size <= pad_id < 0:
                pad_id = source_vocab_size + pad_id

            super().__init__()
            self.source_vocab_size = source_vocab_size
            self.pad_id = pad_id
            self.word_embedding_size = word_embedding_size
            self.num_hidden_layers = num_hidden_layers
            self.hidden_state_size = hidden_state_size
            self.dropout = dropout
            self.cell_type = cell_type
            self.embedding = self.rnn = None
            self.init_submodules()

    def init_submodules(self):

        self.embedding = torch.nn.Embedding(num_embeddings=self.source_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)

        if self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(input_size=self.word_embedding_size,
                                hidden_size=self.hidden_state_size,
                                num_layers=self.num_hidden_layers,
                                dropout =self.dropout,
                                bidirectional=True
                                )
        elif self.cell_type=='lstm':
            self.rnn = torch.nn.LSTM(input_size=self.word_embedding_size,
                                    hidden_size=self.hidden_state_size,
                                    num_layers=self.num_hidden_layers,
                                    dropout=self.dropout,
                                    bidirectional=True
                                    )
        else:
            self.rnn = torch.nn.GRU(input_size=self.word_embedding_size,
                                     hidden_size=self.hidden_state_size,
                                     num_layers=self.num_hidden_layers,
                                     dropout=self.dropout,
                                     bidirectional=True
                                     )


    def forward(
            self,
            F: torch.LongTensor,
            F_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        '''Defines the structure of the encoder'''

        '''input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden'''

        embedding_seq = self.get_all_rnn_inputs(F)
        seq_hidden = self.get_all_hidden_states(embedding_seq, F_lens, h_pad)

        #padding
        for s in range(F.shape[0]):
            for m in range(F.shape[1]):
                if F[s,m] == 0:
                    seq_hidden[s,m,:] = h_pad

        return seq_hidden

    def get_all_rnn_inputs(self, F: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)

        '''Get all input vectors to the RNN at once

        '''

        x = self.embedding.forward(input=F)
        for s in range(F.shape[0]):
            for m in range(F.shape[1]):
                if F[s,m] == self.pad_id:
                    x[s,m,:] = 0
        return x



    def get_all_hidden_states(
            self,
            x: torch.FloatTensor,
            F_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:

        output = torch.nn.utils.rnn.pack_padded_sequence(input=x, lengths=F_lens.cpu(),
                                                         enforce_sorted=False)
        h,_ = self.rnn.forward(output)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence=h, padding_value=h_pad,
                                                      total_length=x.shape[0])
        return h
