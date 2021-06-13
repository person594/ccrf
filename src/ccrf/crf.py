import torch
import torch.nn as nn
import torch.nn.functional as F

from ccrf.utils import logmmexp


class CRF(nn.Module):
    def __init__(self, n_labels, start_and_end_transitions=True):
        r"""A(n unconstrained) CRF output layer.
        Takes label-wise emission potentials as input, and keeps track of transition potentials as a
        parameter matrix.

        Args:
            n_labels: The number of labels/tags (int)
            start_and_end_transitinos: Should the CRF add implicit start and end tokens,
                and manage their transition scores 
        """

        super().__init__()

        self.n_labels = n_labels
        self.start_and_end_transitions = start_and_end_transitions

        self.transitions = nn.Parameter(0.1 * torch.randn(self.n_labels, self.n_labels))
        if self.start_and_end_transitions:
            self.start_transitions = nn.Parameter(0.1 * torch.randn(self.n_labels))
            self.end_transitions = nn.Parameter(0.1 * torch.randn(self.n_labels))
                            
        
    def _logZ(self, x):
        # x: float[batch_size, sequence_length, self.n_labels]
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        logits = x[:,0,:]
        # logits: float32[batch_size, self.n_labels]
        if self.start_and_end_transitions:
            logits = logits + self.start_transitions.view(1, -1)
        
        for i in range(1, sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, self.n_labels]
            logits = logmmexp(logits, self.transitions) + xi

        if self.start_and_end_transitions:
            logits = logits + self.end_transitions.view(1, -1)

        return torch.logsumexp(logits, dim=1)
        
    def _score(self, x, labels):
        # x: float[batch_size, sequence_length, self.n_labels]
        # labels: long[batch_size, sequence_length]
        batch_size, sequence_length, _ = x.shape
        s = x.new_zeros(batch_size)
        # s: float[batch_size]

        for i in range(sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, n_labels]
            li = labels[:,i]
            # ti: long[batch_size]

            if i > 0:
                last_label = labels[:,i-1]
                # last_label: int64[batch_size]
                # get the transition scores for each element of the batch
                indices = last_label * self.n_labels + li
                # indices: long[batch_size]
                transition_scores = self.transitions.view(-1)[indices]
                # transition_scores: float32[batch_size]
            else:
                if self.start_and_end_transitions:
                    transition_scores = self.start_transitions[li]
                    # transition_scores: float32[batch_size]
                else:
                    transition_scores = 0

            # get feature scores for each element of the batch
            feature_scores = torch.gather(xi, 1, li.view(batch_size, 1)).view(batch_size)
            # feature_scores: float[batch_size]

            s = s + transition_scores + feature_scores

        if self.start_and_end_transitions:
            s = s + self.end_transitions[labels[:,-1]]
        return s

            
    def map(self, x):
        """Performs MAP inference, returning the most probable tag sequence y given x
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
            Returns:
                output tags; int64[batch, sequence_length]
        """
        # x: float[batch_size, sequence_length, n_labels]
        batch_size, sequence_length, _ = x.shape

        logits = x.clone()
        # logits: float[batch_size, sequence_length, self.n_labels]
        if self.start_and_end_transitions:
            logits[:,0,:] += self.start_transitions.view(1, -1)

        for i in range(1, sequence_length):
            logits[:,i] += logmmexp(logits[:,i-1], transitions)

        if self.start_and_end_transitions:
            logits[:,-1] += self.end_transitions.view(1, -1)

        sample_sequence = x.new_full((batch_size, sequence_length), 0, dtype=torch.int64)
        # sample_sequence: int64[batch, sequence_length]
        
        sample_logits = logits[:,-1,:]
        # sample_logits: float32[batch, self.n_tags
        
        sample_sequence[:,-1] = torch.argmax(sample_logits, dim=1)
        
        for i in reversed(range(sequence_length-1)):
            sample_logits = logits[:,i,:]
            # sample_logits: float32[batch_size, self.n_tags]
            forward_transitions = (self.transitions.T)[sample_sequence[:,i+1],:]
            # forward_transitions: float32[batch_size, self.n_tags]
            sample_logits = sample_logits + forward_transitions
            # sample_logits: float32[batch_size, self.n_tags]
            sample_sequence[:,i] = torch.argmax(sample_logits, dim=1)
        return sample_sequence

    def loss(self, x, y):
        """Returns NLL for y given x, averaged across batches
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
                y: output tags; int64[batch, sequence_length]
            Returns:
                Average NLL loss; float32[]
        """
        nll = self._logZ(x) - self._score(x, y)
        return torch.mean(nll)

    def sample(self, x, k=1, temp=1):
        """Samples ys from the distribution P(y|x)
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
                k: number of samples per input x (independent and with replacement)
                temp: temperature parameter
            Returns:
                output tags; int64[batch, k, sequence_length]
        """
        # x : float[batch_size, sequence_length, self.n_labels]
        batch_size, sequence_length, _ = x.shape

        logits = x.clone() / temp
        # logits: float[batch_size, sequence_length, self.n_labels]
        if self.start_and_end_transitions:
            logits[:,0,:] += self.start_transitions.view(1, -1) / temp

        for i in range(1, sequence_length):
            logits[:,i] += logmmexp(logits[:,i-1], self.transitions / temp)

        if  self.start_and_end_transitions:
            logits[:,-1,:] += self.end_transitions.view(1, -1) / temp
            
        sample_sequence = x.new_full((batch_size, k, sequence_length), 0, dtype=torch.int64)
        # sample_sequence: int64[batch, k, sequence_length]
        
        sample_logits = logits[:,-1,:]
        
        sample_p = F.softmax(sample_logits, dim=-1)
        # sample_p: float32[batch, self.n_tags]
        
        sample_sequence[:,:,-1] = torch.multinomial(sample_p, k, replacement=True)
        
        for i in reversed(range(sequence_length-1)):
            sample_logits = logits[:,i,:].view(batch_size, 1, self.n_tags)
            # sample_logits: float32[batch_size, 1, self.n_tags]
            forward_transitions = (self.transitions.T)[sample_sequence[:,:,i+1],:] / temp
            # forward_transitions: float32[batch_size, k, self.n_tags]
            sample_logits = sample_logits + forward_transitions
            # sample_logits: float32[batch_size, k, self.n_tags]
            sample_p = F.softmax(sample_logits + forward_transitions, dim=-1)
            # sample_p: float32[batch_size, k, self.n_tags]
            sample_p = sample_p.view(batch_size*k, -1)
            samples = torch.multinomial(sample_p, 1)
            sample_sequence[:,:,i] = samples.view(batch_size, k)
        return sample_sequence

    def forward(self, x):
        """Performs MAP inference, returning the most probable tag sequence y given x
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
            Returns:
                output tags; int64[batch, sequence_length]
        """
        return self.map(x)
