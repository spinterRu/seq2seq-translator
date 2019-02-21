import torch


class OneHotEncoder:

    def __init__(self, texts, device):
        self.device = device
        self.token_index = {}
        self.index_token = {}
        idx = 0
        self.max_length = 0
        for line in texts:
            l = len(line)
            if l > self.max_length:
                self.max_length = l
            for char in line:
                if self.token_index.get(char) is None:
                    self.token_index[char] = idx
                    self.index_token[idx] = char
                    idx = idx + 1
        self.max = len(self.token_index.keys())
        self.tensor_dict = {}
        for key in self.token_index:
            self.tensor_dict[key] = self.get_encoding(key)

    def get_encoding(self, char):
        vec = torch.zeros([1, 1, self.max], device=self.device, requires_grad=False)
        vec[0, 0, self.token_index[char]] = 1
        return vec

    def get_encoding_for_sentence(self, sentence):
        l = []
        for c in sentence:
            l.append(self.tensor_dict[c])

        return l

    def get_sentence(self, indexes):
        s = ""
        for i in indexes:
            s = s + self.index_token[i]
        return s

    def get_sentences(self, sentences):
        l = []
        for tuple in sentences:
            s = "Score = " + str(tuple[2]) + ", sentence: " + self.get_sentence(tuple[1]) + '\n'
            l.append(s)
        return l

    def get_encoding_for_sentence_single_tensor(self, sentence):
        vec = torch.zeros([len(sentence), 1, self.max], device=self.device)
        for i in range(0, len(sentence)):
            vec[i, 0, self.token_index[sentence[i]]] = 1
        return vec


