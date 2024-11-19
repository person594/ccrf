from typing import List, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccrf.utils import logmmexp



import automic as at

# close enough :P
inf = 1e4

class StateLabeledAutomaton(at.Automaton):
    def __init__(self, n_states, accepting):
        super().__init__(n_states, accepting)
        self.labels = [None for _ in range(n_states)]

    def add_state(self):
        self.labels.append(None)
        return super().add_state()

    def string_labels(self, s):
        states = {0}
        label_seq = []
        for sym in s:
            next_states = set()
            emitted_labels = []
            for state in states:
                next_states |= self.transitions[state][sym]
            for state in next_states:
                emitted_labels.append(self.labels[state])
            label_seq.append(emitted_labels)
            states = next_states
        return label_seq


def make_state_labeled_automaton(patterns: List[at.Automaton], alphabet = set()):
    for p in patterns:
        alphabet |= p.alphabet
        
    Any = at.Automaton(1, set())
    for symbol in alphabet:
        Any = Any | at.literal([symbol])

    dotstar = Any[:]

    new_patterns = []
    for i, p in enumerate(patterns):
        dotstarp = at.cat(dotstar, p)
        new_patterns.append(at.cat(dotstarp, at.star(dotstarp)))
        #new_patterns.append((Any[:] + p)[1:])
    patterns = new_patterns
    #patterns = [(Any[:] + p)[1:] for p in patterns]
    patterns = [at.determinize(p, alphabet) for p in patterns]
    patterns = [at.minimize(p) for p in patterns]        
        
    A = StateLabeledAutomaton(1, set())

    tuple2state = {(0,) * len(patterns): 0}
    state2tuple = {0: (0,) * len(patterns)}

    frontier = {0}
    while len(frontier) > 0:
        new_frontier = set()
        for source_state in frontier:
            for symbol in alphabet:
                source_tuple = state2tuple[source_state]
                target_tuple = tuple(next(iter(p.transitions[s][symbol])) for p, s in zip(patterns, source_tuple))
                if target_tuple in tuple2state:
                    target_state = tuple2state[target_tuple]
                else:
                    target_state = A.add_state()
                    tuple2state[target_tuple] = target_state
                    state2tuple[target_state] = target_tuple
                    new_frontier.add(target_state)
                A.transitions[source_state][symbol] = {target_state}
                label = set()
                for i, (p, s) in enumerate(zip(patterns, target_tuple)):
                    if s in p.accepting:
                        label.add(i)
                A.labels[target_state] = label
        frontier = new_frontier
    A.accepting = set(range(A.n_states))
    return A


class PatCRF(nn.Module):
    def __init__(self, patterns: List[at.Automaton], alphabet=set()):
        super().__init__()
        alphabet = set(alphabet)
        self.automaton = make_state_labeled_automaton(patterns, alphabet)

        tags = set()
        start_tags = set()
        
        for i in range(self.automaton.n_states):
            for token, successors in self.automaton.transitions[i].items():
                for successor in successors:
                    tags.add((token, successor))
                    if i == 0:
                        start_tags.add((token, successor))

        self.tags = list(tags)
        self.tags.sort() # making things deterministic
        self.tags_t = {tag: self.tags.index(tag) for tag in self.tags}
        self.start_tags = {self.tags.index(st) for st in start_tags}

        self.n_tags = len(self.tags)
        self.labels = list(self.automaton.alphabet)
        self.labels.sort() # making things deterministic
        self.n_labels = len(self.labels)
        self.n_patterns = len(patterns)

            
        self.register_buffer('tag2label', torch.tensor([self.labels.index(tp[0]) for tp in self.tags], dtype=torch.int64))
        
        self.label_transitions = nn.Parameter(0.1 * torch.randn(self.n_labels, self.n_labels))
                            
        self.register_buffer('transition_constraints', torch.zeros(self.n_tags, self.n_tags, dtype=torch.float32))


        self.register_buffer('tag_patterns', torch.zeros(self.n_tags, len(patterns)))
        
        for i, (token1, state1) in enumerate(self.tags):
            if self.automaton.labels[state1] is not None:
                    for pattern_id in self.automaton.labels[state1]:
                        self.tag_patterns[i, pattern_id] = 1

            for j, (token2, state2) in enumerate(self.tags):
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

    def _logZ(self, x, enforce_boundaries=True):
        # x: float[batch_size, sequence_length, self.n_tags]
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        logits = x.new_full((batch_size, self.n_tags), -inf)
        # logits: float32[batch_size, self.n_tags]

        transitions = self._transitions()

        if enforce_boundaries:
            for tag in self.start_tags:
                logits[:, tag] = x[:,0,tag]
        else:
            for tag in range(self.n_tags):
                logits[:, tag] = x[:,0,tag]
        # logits: float[batch_size, self.n_tags]
        for i in range(1, sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, self.n_tags]
            logits = logmmexp(logits, transitions) + xi

        return torch.logsumexp(logits, dim=1)
        
    def _score(self, x, tags):
        # x: float[batch_size, sequence_length, n_tags]
        # tags: long[batch_size, sequence_length]
        # assume tags[:,0] are in self.start_tags
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

            # get emission scores for each element of the batch
            emission_scores = torch.gather(xi, 1, ti.view(batch_size, 1)).view(batch_size)
            # emission_scores: float[batch_size]

            s += transition_scores + emission_scores

        return s


    def log_p(self, label_emissions, pattern_emissions, y):
        """Returns the log probability of label sequence ``y`` given ``x``.
            Args:
                label_emissions: label-wise emission potentials; float32[batch, sequence_length, n_labels]
                pattern_emissions: pattern-wise emission potentials; float32[batch, sequence_length, n_patterns]
                y: output tags; int64[batch, sequence_length]
                
            Returns:
                log probabilities; float32[batch]
        """
        x = label_emissions[:,:,self.tag2label] + pattern_emissions @ self.tag_patterns.T
        return self._score(x, y) - self._logZ(x)

    def p(self, label_emissions, pattern_emissions, y):
        """Returns the probability of label sequence ``y`` given ``x``.
            Args:
                label_emissions: label-wise emission potentials; float32[batch, sequence_length, n_labels]
                pattern_emissions: pattern-wise emission potentials; float32[batch, sequence_length, n_patterns]
                y: output tags; int64[batch, sequence_length]

            Returns:
                probabilities; float32[batch]
        """
        return torch.exp(self.log_p(label_emissions, pattern_emissions, y))

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
            encoded.append(self.tags_t[token, state])
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
            string.append(self.tags[yi][0])
        return string


    def map(self, label_emissions, pattern_emissions):
        """Performs MAP inference, returning the most probable tag sequence y given x
            Args:
                label_emissions: label-wise emission potentials; float32[batch, sequence_length, n_labels]
                pattern_emissions: pattern-wise emission potentials; float32[batch, sequence_length, n_patterns]
            Returns:
                output tags; int64[batch, sequence_length]
        """
        # Mathematically, this if shouldn't be needed, but there's an autograd bug with empty patterns...
        if torch.numel(pattern_emissions) == 0:
            x = label_emissions[:,:,self.tag2label]
        else:
            x = label_emissions[:,:,self.tag2label] + pattern_emissions @ self.tag_patterns.T

        # x: float[batch_size, sequence_length, n_tags]
        batch_size, sequence_length, _ = x.shape
        
        transitions = self._transitions()

        logits = x.new_full((batch_size, sequence_length, self.n_tags), -inf)
        back_pointers = []
        # logits: float[batch_size, sequence_length, self.n_tags]
        for tag in self.start_tags:
            logits[:,0,tag] = x[:,0,tag]
        for i in range(1, sequence_length):
            xi = x[:,i,:]
            # xi: float[batch_size, self.n_tags]
            #logits[:,i] = logmmexp(logits[:,i-1], transitions) + xi
            incoming_scores = transitions.unsqueeze(0) + logits[:,i-1].unsqueeze(-1) + xi.unsqueeze(1)
            # incoming_scores: float32[batch_size, self.n_tags, self.n_tags]
            # conceptually, element [i, j, k] is the score for tag k if it came from tag j (for batch i)
            logits[:,i], bps = torch.max(incoming_scores, dim=1)
            back_pointers.append(bps)

        sample_sequence = x.new_full((batch_size, sequence_length), 0, dtype=torch.int64)
        # sample_sequence: int64[batch, sequence_length]
        
        sample_logits = logits[:,-1,:]
        # sample_logits: float32[batch, self.n_tags
        
        sample_sequence[:,-1] = torch.argmax(sample_logits, dim=1)
        
        for i, bps in reversed(list(enumerate(back_pointers))):
            # bps: int64[batch_size, self.n_tags]
            sample_sequence[:,i] = bps[torch.arange(batch_size), sample_sequence[:,i+1]]
            """
            sample_logits = logits[:,i,:]
            # sample_logits: float32[batch_size, self.n_tags]
            forward_transitions = (transitions.T)[sample_sequence[:,i+1],:]
            # forward_transitions: float32[batch_size, self.n_tags]
            sample_logits = sample_logits + forward_transitions
            # sample_logits: float32[batch_size, self.n_tags]
            sample_sequence[:,i] = torch.argmax(sample_logits, dim=1)
            """
        return sample_sequence

    def forward(self, label_emissions, pattern_emissions):
        """Performs MAP inference, returning the most probable tag sequence y given x
            Args:
                label_emissions: label-wise emission potentials; float32[batch, sequence_length, n_labels]
                pattern_emissions: pattern-wise emission potentials; float32[batch, sequence_length, n_patterns]
            Returns:
                output tags; int64[batch, sequence_length]
        """
        return self.map(label_emissions, pattern_emissions)

    def loss(self, label_emissions, pattern_emissions, y, enforce_boundaries=True):
        """Returns NLL for y given x, averaged across batches
            Args:
                label_emissions: label-wise emission potentials; float32[batch, sequence_length, n_labels]
                pattern_emissions: pattern-wise emission potentials; float32[batch, sequence_length, n_patterns]
                y: output tags; int64[batch, sequence_length]
            Returns:
                Average NLL loss; float32[]
        """
        if torch.numel(pattern_emissions) == 0:
            x = label_emissions[:,:,self.tag2label]
        else:
            x = label_emissions[:,:,self.tag2label] + pattern_emissions @ self.tag_patterns.T
        nll = self._logZ(x, enforce_boundaries) - self._score(x, y)
        return torch.mean(nll)

    def sample(self, label_emissions, pattern_emissions, k, temp=1.0):
        """Samples ys from the distribution P(y|x)
            Args:
                label_emissions: label-wise emission potentials; float32[batch, sequence_length, n_labels]
                pattern_emissions: pattern-wise emission potentials; float32[batch, sequence_length, n_patterns]
                k: number of samples per input x (independent and with replacement)
                temp: temperature parameter
            Returns:
                output tags; int64[batch, k, sequence_length]
        """
        x = label_emissions[:,:,self.tag2label] + pattern_emissions @ self.tag_patterns.T
        # x: float[batch_size, sequence_length, n_tags]
        batch_size, sequence_length, _ = x.shape

        transitions = self._transitions(temp)
        raise NotImplementedError()

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
