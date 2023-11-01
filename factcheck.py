# factcheck.py
import nltk
import torch
from typing import List
import numpy as np
import spacy
import gc


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """

    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # raise Exception("Not implemented")
        labels = ["entailment", "neutral", "contradiction"]
        probs = torch.nn.functional.softmax(logits)
        choice = int(np.argmax(probs))

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        return labels[choice], probs


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):

    def predict(self, fact: str, passages: List[dict]) -> str:

        stopwords = [".", "``", ",", "(", ")", "''", "/s", "s", ";", ">", "<", "!", "[", "]", "-", "is", "in", "have",
                     "has", "then", "into", "he", "they", "was", "a", "the", "an", "to", "on", "as", "with", "by",
                     "for", "of", "from", "at", "about"]
        original_fact = fact
        fact = fact.replace("-", " ")
        stopwords.extend([word.lower() for word in nltk.word_tokenize(passages[0]['title'])])

        fact_bow = nltk.word_tokenize(fact)
        fact_words = [word.lower() for word in fact_bow if
                      (word.lower() not in stopwords) and (
                          word.lower().isascii())]
        fact_bow = set(nltk.bigrams(fact_words))

        whole_p = ""
        title = passages[0]['title']
        for p in passages:
            temp = p['text']
            temp = temp.replace("-", " ")
            whole_p += temp
        whole_p.replace(passages[0]['title'], "")
        pass_bow = nltk.word_tokenize(whole_p)
        pass_words = [word.lower() for word in pass_bow if
                      (word.lower() not in stopwords) and (
                          word.lower().isascii()) and word not in title]
        pass_bow = set(nltk.bigrams(pass_words))
        pass_bow = set(pass_bow)

        if len(fact_bow) <= 1:
            for word in fact_words:
                if word in pass_words:
                    # print(f"Manually guessed for fact {original_fact}")
                    return "S"

        intersection = len(fact_bow.intersection(pass_bow))
        if len(fact_bow) == 0:
            modified_jac = 0
        else:
            modified_jac = intersection / len(fact_bow)

        # print(f"title: {title}, fact: {original_fact}")
        # print(f"fact_bow: {fact_bow}")
        # print(f"pass_bow: {pass_bow}")
        # print(f"jaccard similarity: {modified_jac}\n")
        if modified_jac > 0.33:
            return "S"
        else:
            return "NS"

    def predict_sentence(self, fact: str, sentence: str, threshold: float) -> str:

        stopwords = [".", "``", ",", "(", ")", "''", "/s", "s", ";", ">", "<", "!", "[", "]", "-", "is", "in", "have",
                     "has", "then", "into", "he", "they", "was", "a", "the", "an", "to", "on", "as", "with", "by",
                     "for", "of", "from", "at", "about"]
        original_fact = fact
        fact = fact.replace("-", " ")
        # stopwords.extend([word.lower() for word in nltk.word_tokenize(passages[0]['title'])])

        fact_bow = nltk.word_tokenize(fact)
        fact_words = [word.lower() for word in fact_bow if
                      (word.lower() not in stopwords) and (
                          word.lower().isascii())]
        fact_bow = set(nltk.bigrams(fact_words))

        pass_bow = nltk.word_tokenize(sentence)
        pass_words = [word.lower() for word in pass_bow if
                      (word.lower() not in stopwords) and (
                          word.lower().isascii())]
        pass_bow = set(nltk.bigrams(pass_words))
        pass_bow = set(pass_bow)

        if len(fact_bow) <= 1:
            for word in fact_words:
                if word in pass_words:
                    # print(f"Manually guessed for fact {original_fact}")
                    return "S"

        intersection = len(fact_bow.intersection(pass_bow))
        if len(fact_bow) == 0:
            modified_jac = 0
        else:
            modified_jac = intersection / len(fact_bow)

        # print(f"title: {title}, fact: {original_fact}")
        # print(f"fact_bow: {fact_bow}")
        # print(f"pass_bow: {pass_bow}")
        # print(f"jaccard similarity: {modified_jac}\n")
        if modified_jac > threshold:
            return "S"
        else:
            return "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.prev_model = WordRecallThresholdFactChecker()

    def predict(self, fact: str, passages: List[dict]) -> str:
        whole_p = ""
        for p in passages:
            whole_p += p['text']
        whole_p = whole_p.replace("<s>", "")
        whole_p = whole_p.replace("</s>", "")

        sentences = nltk.sent_tokenize(whole_p)
        chosen_sentences = []

        for sentence in sentences:
            if self.prev_model.predict_sentence(fact, sentence, 0) == "NS":
                continue
            chosen_sentences.append(sentence)
        probs = []
        for sentence in chosen_sentences:
            label, prob = self.ent_model.check_entailment(sentence, fact)
            if label == "entailment":
                return "S"
            probs.append(prob)

        if len(probs) == 0:
            return "NS"

        choice = np.argsort(np.mean(np.vstack(probs), axis=0))[-1]
        if choice == 1:
            choice = np.argsort(np.mean(np.vstack(probs), axis=0))[-2]

        if choice == 0:
            return "S"
        else:
            return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
