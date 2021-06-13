import torch
import torch.nn as nn
import torch.nn.functional as F

from ccrf.utils import logmmexp

import automic

# close enough :P
inf = 1e4

class RegCCRF(nn.Module):
    def __init__(self, automaton):
        r"""A regular-constrained CRF output layer.
        Takes label-wise emission potentials as input, and keeps track of transition potentials as a
        parameter matrix.

        Args:
            automaton: An instance of automic.Automaton representing the regular language to constrain to.
                This automaton must be unambiguous -- this is not checked automatically.
                Smaller automata will generally result in better performance.
        """
        super().__init__()
        self.automaton = automic.epsilon_remove(automaton)
        # TODO: disambiguate automatically
        #assert not self.automaton.is_ambiguous()

        # Also TODO: start and end transitions.
        
        # build our tags from the automaton
        # assume now that all tags are in (token, state) form
        # we could also chose (state, token), which could potentially even give us fewer
        # tags, but that is left as a TODO for now
        
        tags = set()
        start_tags = set()
        end_tags = set()
        for i in range(self.automaton.n_states):
            for token, successors in self.automaton.transitions[i].items():
                for successor in successors:
                    tags.add((token, successor))
                    if i == 0:
                        start_tags.add((token, successor))
                    if successor in self.automaton.accepting:
                        end_tags.add((token, successor))

        self.tag_pairs = list(tags)
        self.tag_pairs_t = {tag: self.tag_pairs.index(tag) for tag in self.tag_pairs}
        self.start_tags = {self.tag_pairs.index(st) for st in start_tags}
        self.end_tags = {self.tag_pairs.index(st) for st in end_tags}

        self.n_tags = len(self.tag_pairs)
        self.labels = list(automaton.alphabet)
        self.labels.sort() # just to introduce some determinism here
        self.n_labels = len(self.labels)

        self.register_buffer('tag2label', torch.tensor([self.labels.index(tp[0]) for tp in self.tag_pairs], dtype=torch.int64))
        
        self.label_transitions = nn.Parameter(0.1 * torch.randn(self.n_labels, self.n_labels))
                            
        self.register_buffer('transition_constraints', torch.zeros(self.n_tags, self.n_tags, dtype=torch.float32))
        for i, (token1, state1) in enumerate(self.tag_pairs):
            for j, (token2, state2) in enumerate(self.tag_pairs):
                if state2 not in self.automaton.transitions[state1][token2]:
                    self.transition_constraints[i, j] = -inf


        indices = torch.zeros(self.n_tags, self.n_tags, dtype=torch.int64)
        for i in range(self.n_tags):
            for j in range(self.n_tags):
                i_label = self.tag2label[i]
                j_label = self.tag2label[j]
                indices[i,j] = i_label * self.n_labels + j_label

        self.register_buffer('indices', indices)


    def _transitions(self, temp=1):
        # make sure the constraints are unaffected by the temperature, i.e. we can't break them at high temperatures
        return (self.label_transitions.view(-1)[self.indices] / temp) + self.transition_constraints

    def _logZ(self, x):
        # x: float[batch_size, sequence_length, self.n_tags]
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        logits = x.new_full((batch_size, self.n_tags), -inf)
        # logits: float32[batch_size, self.n_tags]

        transitions = self._transitions()
        
        for tag in self.start_tags:
            logits[:, tag] = x[:,0,tag]
        # logits: float[batch_size, self.n_tags]
        for i in range(1, sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, self.n_tags]
            logits = logmmexp(logits, transitions) + xi

        for tag in range(self.n_tags):
            if tag not in self.end_tags:
                logits[:,tag] = -inf

        return torch.logsumexp(logits, dim=1)
        
    def _score(self, x, tags):
        # x: float[batch_size, sequence_length, n_tags]
        # tags: long[batch_size, sequence_length]
        # assume tags[:,0] are in self.start_tags, and tags[:,-1] are in end_tags
        batch_size, sequence_length, _ = x.shape
        s = x.new_zeros(batch_size)
        # s: float[batch_size]

        transitions = self._transitions()

        for i in range(sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, n_tags]
            ti = tags[:,i]
            # ti: long[batch_size]

            if i > 0:
                last_tag = tags[:,i-1]
                # get the transition scores for each element of the batch
                indices = last_tag * self.n_tags + ti
                # indices: long[batch_size]
                transition_scores = torch.take(transitions, indices)
            else:
                transition_scores = 0

            # get feature scores for each element of the batch
            feature_scores = torch.gather(xi, 1, ti.view(batch_size, 1)).view(batch_size)
            # feature_scores: float[batch_size]

            s += transition_scores + feature_scores

        return s


    def log_p(self, x, y):
        """Returns the log probability of label sequence ``y`` given ``x``.
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
                y: output tags; int64[batch, sequence_length]

            Returns:
                log probabilities; float32[batch]
        """
        x = x[:,:,self.tag2label]
        return self._score(x, y) - self._logZ(x)

    def p(self, x, y):
        """Returns the probability of label sequence ``y`` given ``x``.
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
                y: output tags; int64[batch, sequence_length]

            Returns:
                probabilities; float32[batch]
        """
        return torch.exp(self.log_p(x, y))

    def encode(self, labels):
        """Encodes a label sequence into a tag sequence, which can be used as a ``y'' variable.
            Args:
                labels: a sequence of labels which are accepted by the RegCCRF's automaton.
            Returns:
                A sequence of tags.
        """
        paths = self.automaton.paths(labels)
        assert len(paths) == 1
        # if zero, the string is not in the language
        # if >1, the automaton is ambiguous
        path = next(iter(paths))
        encoded = []
        for token, state in zip(labels, path[1:]):
            encoded.append(self.tag_pairs_t[token, state])
        return encoded
        #return next(iter(paths))

    
    def decode(self, y):
        """Decodes a tag sequence back into a label sequence
            Args:
                y: a sequence of tags
            Returns:
                The corresponding sequence of labels.
        """
        string = []
        for yi in y:
            string.append(self.tag_pairs[yi][0])
        return string


    def map(self, x):
        """Performs MAP inference, returning the most probable tag sequence y given x
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
            Returns:
                output tags; int64[batch, sequence_length]
        """
        # x : float[batch_size, sequence_length, self.n_labels]
        x = x[:,:,self.tag2label]
        # x: float[batch_size, sequence_length, n_tags]
        batch_size, sequence_length, _ = x.shape

        transitions = self._transitions()

        logits = x.new_full((batch_size, sequence_length, self.n_tags), -inf)
        # logits: float[batch_size, sequence_length, self.n_tags]
        for tag in self.start_tags:
            logits[:,0,tag] = x[:,0,tag]
        for i in range(1, sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, self.n_tags]
            logits[:,i] = logmmexp(logits[:,i-1], transitions) + xi

        for tag in range(self.n_tags):
            if tag not in self.end_tags:
                logits[:,-1,tag] = -inf
            
        sample_sequence = x.new_full((batch_size, sequence_length), 0, dtype=torch.int64)
        # sample_sequence: int64[batch, sequence_length]
        
        sample_logits = logits[:,-1,:]
        # sample_logits: float32[batch, self.n_tags
        
        sample_sequence[:,-1] = torch.argmax(sample_logits, dim=1)
        
        for i in reversed(range(sequence_length-1)):
            sample_logits = logits[:,i,:]
            # sample_logits: float32[batch_size, self.n_tags]
            forward_transitions = (transitions.T)[sample_sequence[:,i+1],:]
            # forward_transitions: float32[batch_size, self.n_tags]
            sample_logits = sample_logits + forward_transitions
            # sample_logits: float32[batch_size, self.n_tags]
            sample_sequence[:,i] = torch.argmax(sample_logits, dim=1)
        return sample_sequence

    def forward(self, x):
        """Performs MAP inference, returning the most probable tag sequence y given x
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
            Returns:
                output tags; int64[batch, sequence_length]
        """
        return self.map(x)

    def loss(self, x, y):
        """Returns NLL for y given x, averaged across batches
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
                y: output tags; int64[batch, sequence_length]
            Returns:
                Average NLL loss; float32[]
        """

        x = x[:,:,self.tag2label]
        nll = self._logZ(x) - self._score(x, y)
        return torch.mean(nll)

    def sample(self, x, k, temp=1):
        """Samples ys from the distribution P(y|x)
            Args:
                x: input potentials; float32[batch, sequence_length, n_labels]
                k: number of samples per input x (independent and with replacement)
                temp: temperature parameter
            Returns:
                output tags; int64[batch, k, sequence_length]
        """
        # x : float[batch_size, sequence_length, n_labels]
        x = x[:,:,self.tag2label] / temp
        # x: float[batch_size, sequence_length, n_tags]
        batch_size, sequence_length, _ = x.shape

        transitions = self._transitions(temp)

        logits = x.new_full((batch_size, sequence_length, self.n_tags), -inf)
        # logits: float[batch_size, sequence_length, self.n_tags]
        for tag in self.start_tags:
            logits[:,0,tag] = x[:,0,tag]
        for i in range(1, sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, self.n_tags]
            logits[:,i] = logmmexp(logits[:,i-1], transitions) + xi

        for tag in range(self.n_tags):
            if tag not in self.end_tags:
                logits[:,-1,tag] = -inf
            
        sample_sequence = x.new_full((batch_size, k, sequence_length), 0, dtype=torch.int64)
        # sample_sequence: int64[batch, k, sequence_length]
        
        sample_logits = logits[:,-1,:]
        
        sample_p = F.softmax(sample_logits, dim=-1)
        # sample_p: float32[batch, self.n_tags]

        sample_sequence[:,:,-1] = torch.multinomial(sample_p, k, replacement=True)
        for i in reversed(range(sequence_length-1)):
            sample_logits = logits[:,i,:].view(batch_size, 1, self.n_tags)
            # sample_logits: float32[batch_size, 1, self.n_tags]
            forward_transitions = (transitions.T)[sample_sequence[:,:,i+1],:]
            # forward_transitions: float32[batch_size, k, self.n_tags]
            sample_logits = sample_logits + forward_transitions
            # sample_logits: float32[batch_size, k, self.n_tags]
            sample_p = F.softmax(sample_logits + forward_transitions, dim=-1)
            # sample_p: float32[batch_size, k, self.n_tags]
            sample_p = sample_p.view(batch_size*k, -1)
            samples = torch.multinomial(sample_p, 1)
            sample_sequence[:,:,i] = samples.view(batch_size, k)
        return sample_sequence
