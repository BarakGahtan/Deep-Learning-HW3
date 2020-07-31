
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO: Create two maps as described in the docstring above.
    # It's best if you also sort the chars before assigning indices, so that
    # they're in lexical order.
    # ====== YOUR CODE: ======
    unique_chars = sorted(list(set(text)))
    char_to_idx = {char: idx for (idx, char) in enumerate(unique_chars)}
    idx_to_char = {idx: char for (idx, char) in enumerate(unique_chars)}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    regex = re.compile('[%s]' % ''.join(chars_to_remove))
    n_removed = len(regex.findall(text))
    text_clean = regex.sub('', text)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    result = torch.zeros(
        [len(text), max(char_to_idx.values()) + 1], dtype=torch.int8)
    for (idx, char) in enumerate(text):
        result[idx][char_to_idx[char]] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    non_zero_idxs = torch.nonzero(embedded_text)
    result = ''
    for row in non_zero_idxs:
        result += idx_to_char[row[1].item()]
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    embedded = chars_to_onehot(text, char_to_idx)
    embedded_labels = embedded.clone()[1:]
    embedded_labels = torch.argmax(embedded_labels, dim=1)
    embedded = embedded[:-1]
    split_labels = embedded_labels.split(seq_len)
    split_samples = embedded.split(seq_len)
    if split_labels[-1].size(0) < seq_len:
        split_labels = split_labels[:-1]
        split_samples = split_samples[:-1]
    labels = torch.stack(split_labels)
    samples = torch.stack(split_samples)
    samples = samples.to(device)
    labels = labels.to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    result = F.softmax(y.mul(1/temperature), dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        model.train(False)
        embedded = chars_to_onehot(out_text, char_to_idx).unsqueeze(dim=0)
        to_forward = embedded
        h_s = None
        while len(out_text) < n_chars:
            if h_s is not None:
                y, h_s = model(to_forward.to(
                    dtype=torch.float).to(device), h_s.to(device))
            else:
                y, h_s = model(to_forward.to(dtype=torch.float).to(device))
            last_y = y[0, -1, :]
            distribution = hot_softmax(last_y, dim=0, temperature=T)
            char_idx = torch.multinomial(distribution, 1).item()
            char = idx_to_char[char_idx]
            out_text += char
            to_forward = chars_to_onehot(char, char_to_idx).unsqueeze(dim=0)

    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents  one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of indices is takes, samples in the same index of
        #  adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        num_of_batchs = len(self.dataset)//self.batch_size
        new_data_set = self.dataset[:self.batch_size*num_of_batchs]
        idx = [iter_batch * num_of_batchs +
               j for j in range(num_of_batchs) for iter_batch in range(self.batch_size)]
        # raise NotImplementedError()
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model.
        # To implement the affine transforms you can use either nn.Linear
        # modules (recommended) or create W and b tensor pairs directly.
        # Create these modules or tensors and save them per-layer in
        # the layer_params list.
        # Important note: You must register the created parameters so
        # they are returned from our module's parameters() function.
        # Usually this happens automatically when we assign a
        # module/tensor as an attribute in our module, but now we need
        # to do it manually since we're not assigning attributes. So:
        #   - If you use nn.Linear modules, call self.add_module() on them
        #     to register each of their parameters as part of your model.
        #   - If you use tensors directly, wrap them in nn.Parameter() and
        #     then call self.register_parameter() on them. Also make
        #     sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        self.dropout_rate = dropout

        input_dim = in_dim
        for i in range(self.n_layers):
            if i > 0:
                input_dim = h_dim
            fc_xz = nn.Linear(input_dim, h_dim, bias=False)
            fc_hz = nn.Linear(h_dim, h_dim, bias=True)
            sigma_z = nn.Sigmoid()
            fc_xr = nn.Linear(input_dim, h_dim, bias=False)
            fc_hr = nn.Linear(h_dim, h_dim, bias=True)
            sigma_r = nn.Sigmoid()
            fc_xg = nn.Linear(input_dim, h_dim, bias=False)
            fc_hg = nn.Linear(h_dim, h_dim, bias=True)
            tanh = nn.Tanh()
            drop = nn.Dropout(dropout, inplace=False)

            module_dict = dict()

            module_dict["fc_hz"] = fc_hz
            module_dict["fc_xz"] = fc_xz
            module_dict["fc_hr"] = fc_hr
            module_dict["fc_xr"] = fc_xr
            module_dict["fc_hg"] = fc_hg
            module_dict["fc_xg"] = fc_xg
            module_dict["sigma_z"] = sigma_z
            module_dict["sigma_r"] = sigma_r
            module_dict["tanh"] = tanh
            module_dict["drop"] = drop
            self.layer_params.append(module_dict)

        # Register all modules as a part of the model
        for i, layer_modules in enumerate(self.layer_params):
            self.add_module("fc_hz_"+str(i), layer_modules["fc_hz"])
            self.add_module("fc_xz_"+str(i), layer_modules["fc_xz"])
            self.add_module("sigmoid_z_"+str(i), layer_modules["sigma_z"])
            self.add_module("fc_hr_"+str(i), layer_modules["fc_hr"])
            self.add_module("fc_xr_"+str(i), layer_modules["fc_xr"])
            self.add_module("sigmoid_r_"+str(i), layer_modules["sigma_r"])
            self.add_module("fc_hg_"+str(i), layer_modules["fc_hg"])
            self.add_module("fc_xg_"+str(i), layer_modules["fc_xg"])
            self.add_module("tanh_"+str(i), layer_modules["tanh"])
            self.add_module("dropout_"+str(i), layer_modules["drop"])

        fc_hy = nn.Linear(h_dim, out_dim, bias=True)
        self.layer_params.append(fc_hy)
        self.add_module("fc_hy", fc_hy)
        self.hidden_state = None
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(torch.zeros(
                    batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: Implement the model's forward pass.
        # You'll need to go layer-by-layer from bottom to top (see diagram).
        # Tip: You can use torch.stack() to combine multiple tensors into a
        # single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        hidden_state = torch.zeros(
            batch_size, self.n_layers, self.h_dim, device=input.device)
        for i in range(self.n_layers):
            params = self.layer_params[i]
            sequence_states = []
            h = layer_states[i]
            for t in range(seq_len):
                t_input = layer_input[:, t, :].to(input.device)

                z = params["sigma_z"](params["fc_xz"](t_input)
                                      + params["fc_hz"](h))
                r = params["sigma_r"](params["fc_xr"](t_input) +
                                      params["fc_hr"](h))
                g = params["tanh"](params["fc_xg"](t_input) +
                                   params["fc_hg"](r.mul(h)))
                h = z.mul(h) + (1 - z).mul(g)

                sequence_states.append(h)

            layer_input = params["drop"](torch.stack(sequence_states, dim=1))

            hidden_state[:, i, :] = h

        layer_output = self.layer_params[self.n_layers](layer_input)

        # final hidden state unrelated to back propagation
        hidden_state = hidden_state.detach()

        self.hidden_state = hidden_state
        # ========================
        return layer_output, hidden_state
