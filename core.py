import spacy
import random
import gensim.downloader as api
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
import hashlib
import math


class PrivacyPerturbator:
    def __init__(self, total_epsilon=3.0):
        self.total_epsilon = total_epsilon
        self.skip_words = {"year", "years", "old", "decade", "decades", "month", "months", "day", "days"}
        self.nlp = spacy.load("en_core_web_trf")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.word2vec_model = api.load('word2vec-google-news-300')
        self.classifier = self._train_classifier(self._get_default_train_data())

    class SimpleClassifier(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_size, 2)
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x):
            logits = self.linear(x)
            probs = self.softmax(logits)
            return probs

    def _get_default_train_data(self):
        return [
            ("Alice is 3 years old.", 0),
            ("The average temperature is 23.5 °C.", 1),
            ("The year 2024 is a leap year.", 0),
            ("The stock price increased by 1.5% today.", 1),
            ("John has 7 apples.", 0),
            ("The distance to the nearest star is approximately 4.2 light-years.", 1),
        ]

    def _train_classifier(self, train_data, num_epochs=5):
        classifier = self.SimpleClassifier(hidden_size=768)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            classifier.train()
            for context, label in train_data:
                inputs = self.tokenizer(context, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)

                optimizer.zero_grad()
                labels = torch.tensor([label])
                outputs = classifier(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return classifier

    def _compute_privacy_budget(self, text, entities):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
        norms = torch.norm(token_embeddings, dim=1).cpu().numpy()
        token_norm_map = dict(zip(tokens, norms))
        entity_token_map = {}
        entity_norms = {}
        for ent in entities:
            ent_tokens = self.tokenizer.tokenize(ent.text)
            token_norms = [token_norm_map.get(tok, 1e-5) for tok in ent_tokens]
            avg_norm = np.mean(token_norms)
            entity_token_map[ent.text] = ent_tokens
            entity_norms[ent.text] = avg_norm

        total_weight = sum(entity_norms.values())
        entity_epsilons = {
            ent_text: self.total_epsilon * (entity_norms[ent_text] / total_weight)
            for ent_text in entity_token_map
        }

        token_epsilon_map = {}
        for ent_text, tokens in entity_token_map.items():
            per_token_epsilon = entity_epsilons[ent_text] / len(tokens)
            for tok in tokens:
                token_epsilon_map[tok] = per_token_epsilon
        return token_epsilon_map

    def _perturb_word(self, word, epsilon):
        if word in self.word2vec_model:
            similar_words = self.word2vec_model.similar_by_word(word, topn=5)
            similar_words = [w[0] for w in similar_words]
            similar_words.append(word)
            p = np.exp(epsilon) / (np.exp(epsilon) + len(similar_words) - 1)
            q = 1 / (np.exp(epsilon) + len(similar_words) - 1)
            probs = [q] * len(similar_words)
            probs[similar_words.index(word)] = p
            return random.choices(similar_words, probs)[0]
        return word

    class Piecewise:
        def __init__(self, eps, input_range=(0, 100)):
            self.eps = eps
            self.alpha, self.beta = input_range

        def __call__(self, x):
            if self.eps == 0:
                return x
            t = (x - self.alpha) / (self.beta - self.alpha)
            t = 2 * t - 1
            exp_eps = math.exp(self.eps)
            exp_eps_half = math.exp(self.eps / 2)
            C = (exp_eps_half + 1) / (exp_eps_half - 1) if exp_eps_half - 1 != 0 else float('inf')
            P = (exp_eps - exp_eps_half) / (2 * exp_eps_half + 2)
            L = t * (C + 1) / 2 - (C - 1) / 2
            R = L + C - 1
            x = torch.rand(1).item()
            if x < P * (L + C) / exp_eps:
                t = torch.rand(1).item() * (L + C) - C
            elif x < P * (L + C) / exp_eps + P * (R - L):
                t = torch.rand(1).item() * (R - L) + L
            else:
                t = torch.rand(1).item() * (C - R) + R
            return (self.beta - self.alpha) * (t + 1) / 2 + self.alpha

    def _local_hashing_perturb(self, value, epsilon=1, g=100):
        salt = random.randint(0, 10000)
        x = self._hash_function(value, salt) % g
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1 / (math.exp(epsilon) + g - 1)
        probs = [q] * g
        probs[x] = p
        return random.choices(range(g), probs)[0]

    def _hash_function(self, value, salt):
        value_str = str(value) + str(salt)
        hash_obj = hashlib.sha256(value_str.encode())
        return int(hash_obj.hexdigest(), 16)

    def _perturb_number(self, number, epsilon, is_discrete, entity_label=None):
        if entity_label == "AGE":
            number = min(max(number, 0), 120)
        if is_discrete:
            return self._local_hashing_perturb(number, epsilon)
        else:
            pm = self.Piecewise(epsilon)
            perturbed = pm(number)
            if entity_label == "AGE":
                perturbed = min(max(perturbed, 0), 120)
            return round(perturbed, 2 if number < 1 else 0)

    def _is_discrete_context(self, context, entity_label=None):
        if entity_label == "AGE":
            return True
        inputs = self.tokenizer(context, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            probs = self.classifier(embeddings)
        return probs[0][0].item() > 0.5

    def perturb(self, text):
        doc = self.nlp(text)
        privacy_budgets = self._compute_privacy_budget(text, doc.ents)
        perturbed_text = text

        for ent in doc.ents:
            ent_text = ent.text
            ent_label = ent.label_

            if re.fullmatch(r'[-+]?\d*\.\d+|\d+', ent_text):
                epsilon = privacy_budgets.get(ent_text, self.total_epsilon / 10)
                is_discrete = self._is_discrete_context(ent_text, ent_label)
                perturbed = self._perturb_number(float(ent_text), epsilon, is_discrete, entity_label=ent_label)
                perturbed_text = perturbed_text.replace(ent_text, str(perturbed), 1)
                continue

            tokens = ent_text.split()
            perturbed_tokens = []

            for tok in tokens:
                tok_lower = tok.lower()

                if tok_lower in self.skip_words:
                    perturbed_tokens.append(tok)
                    continue

                if re.fullmatch(r'[-+]?\d*\.\d+|\d+', tok):
                    epsilon = privacy_budgets.get(tok, self.total_epsilon / 10)
                    is_discrete = self._is_discrete_context(tok, ent_label)
                    perturbed = self._perturb_number(float(tok), epsilon, is_discrete, entity_label=ent_label)
                    perturbed_tokens.append(str(perturbed))
                else:
                    epsilon = privacy_budgets.get(tok, self.total_epsilon / 10)
                    perturbed = self._perturb_word(tok, epsilon)
                    perturbed_tokens.append(perturbed)

            perturbed_ent = ' '.join(perturbed_tokens)
            perturbed_text = perturbed_text.replace(ent_text, perturbed_ent, 1)

        return perturbed_text


if __name__ == "__main__":
    perturbator = PrivacyPerturbator(total_epsilon=3.0)
    original = "My name is Alex and I am 40 years old. The stock rose by 2.5 today."
    perturbed = perturbator.perturb(original)
    print("Original: ", original)
    print("Perturbed:", perturbed)
