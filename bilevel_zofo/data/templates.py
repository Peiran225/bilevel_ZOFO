import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

class Template:
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer/candidate)
        """
        raise NotImplementedError

    def verbalize(self, sample, candidate):
        """
        Return the prompted version of the example (with the answer/candidate)
        """
        return candidate

    def encode_sfc(self, sample):
        """
        Same as encode, but for SFC (calibration) -- this usually means the input is not included
        """
        return "<mask>"

    def verbalize_sfc(self, sample, candidate):
        """
        Same as verbalize, but for SFC (calibration) -- this usually means the input is not included
        """
        return candidate


class SST2Template(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"


class SST2TemplateEmpty(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} "

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {self.verbalizer[candidate]}"


class CopaTemplate(Template):
    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def get_conjucture(self, sample):
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def capitalize(self, c):
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        conjunction = self.get_conjucture(sample)
        return conjunction.strip()

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class CopaTemplateEmpty(Template):
    capitalization: str = "correct"
    effect_conj: str = " "
    cause_conj: str = " "

    def get_conjucture(self, sample):
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def capitalize(self, c):
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        conjunction = self.get_conjucture(sample)
        return conjunction.strip()

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class BoolQTemplate(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV2(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV3(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class MultiRCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n"

    def verbalize(self, sample, candidate):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class CBTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WICTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n"

    def verbalize(self, sample, candidate):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WSCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ReCoRDTemplate(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer:"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"Answer:"

    def verbalize_sfc(self, sample, candidate):
        return f"Answer: {candidate}"


class ReCoRDTemplateGPT3(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        return f"{passage}\n-"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"{passage}\n- {query}"

        # passage = sample.data['passage']
        # query = sample.data['query']
        # return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"-"

    def verbalize_sfc(self, sample, candidate):
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"- {query}"


class RTETemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class RTETemplateEmpty(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\n\"{hypothesis}\"\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\n\"{hypothesis}\"\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class SQuADv2Template(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class DROPTemplate(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class WinoGrandeTemplate(Template):
    @staticmethod
    def get_prompt(sample):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = sample.data["sentence"]
        context, target = sentence.split("_")
        return context

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + candidate

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class TweetEvalSentimentTemplate(Template):
    verbalizer = {0: "terrible", 1: "neutral", 2: "neutral"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"


class TweetEvalSentimentTemplateEmpty(Template):
    verbalizer = {0: "terrible", 1: "neutral", 2: "neutral"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} "

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {self.verbalizer[candidate]}"


class IMDBTemplate(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"


class IMDBTemplateEmpty(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} "

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {self.verbalizer[candidate]}"


class RottenTomatoesTemplate(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"


class RottenTomatoesTemplateEmpty(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} "

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {self.verbalizer[candidate]}"


class EmotionTemplate(Template):
    verbalizer = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} The emotion was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} The emotion was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" The emotion was"

    def verbalize_sfc(self, sample, candidate):
        return f" The emotion was {self.verbalizer[candidate]}"


class TweetEvalEmotionTemplate(Template):
    verbalizer = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} The emotion was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} The emotion was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" The emotion was"

    def verbalize_sfc(self, sample, candidate):
        return f" The emotion was {self.verbalizer[candidate]}"


class TweetEvalIronyTemplate(Template):
    verbalizer = {0: "non-ironic", 1: "ironic"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} The text was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} The text was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" The text was"

    def verbalize_sfc(self, sample, candidate):
        return f" The text was {self.verbalizer[candidate]}"


class AmazonPolarityTemplate(Template):
    verbalizer = {0: "negative", 1: "positive"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} The sentiment was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} The sentiment was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" The sentiment was"

    def verbalize_sfc(self, sample, candidate):
        return f" The sentiment was {self.verbalizer[candidate]}"


class PoemSentimentTemplate(Template):
    verbalizer = {0: "negative", 1: "positive", 2: "neutral", 3: "mixed"}

    def encode(self, sample):
        text = sample.data["verse_text"]
        return f"{text} The sentiment was"

    def verbalize(self, sample, candidate):
        text = sample.data["verse_text"]
        return f"{text} The sentiment was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" The sentiment was"

    def verbalize_sfc(self, sample, candidate):
        return f" The sentiment was {self.verbalizer[candidate]}"


class YelpReviewPolarityTemplate(Template):
    verbalizer = {0: "negative", 1: "positive"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} The sentiment was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} The sentiment was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" The sentiment was"

    def verbalize_sfc(self, sample, candidate):
        return f" The sentiment was {self.verbalizer[candidate]}"


class FinancialPhrasebankTemplate(Template):

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} The sentiment was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} The sentiment was {candidate}"

    def encode_sfc(self, sample):
        return f" The sentiment was"

    def verbalize_sfc(self, sample, candidate):
        return f" The sentiment was {candidate}"


class EmoTemplate(Template):
    verbalizer = {0: "others", 1: "happy", 2: "sad", 3: "angry"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} The emotion was"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text} The emotion was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" The emotion was"

    def verbalize_sfc(self, sample, candidate):
        return f" The emotion was {self.verbalizer[candidate]}"


class GLUERTETemplate(Template):
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['sentence1']
        hypothesis = sample.data['sentence2']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['sentence1']
        hypothesis = sample.data['sentence2']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class GLUEMRPCTemplate(Template):
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        sentence1 = sample.data["sentence1"]
        sentence2 = sample.data["sentence2"]
        return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences paraphrases of each other? Yes or No?\n"

    def verbalize(self, sample, candidate):
        sentence1 = sample.data["sentence1"]
        sentence2 = sample.data["sentence2"]
        return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences paraphrases of each other? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class GLUEQQPTemplate(Template):
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        question1 = sample.data["question1"]
        question2 = sample.data["question2"]
        return f"Question 1: {question1}\nQuestion 2: {question2}\nAre these questions duplicates of each other? Yes or No?\n"

    def verbalize(self, sample, candidate):
        question1 = sample.data["question1"]
        question2 = sample.data["question2"]
        return f"Question 1: {question1}\nQuestion 2: {question2}\nAre these questions duplicates of each other? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class GLUEMNLITemplate(Template):
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Yes, No, or Maybe?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Yes, No, or Maybe?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class SciTailTemplate(Template):

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Yes, No, or Maybe?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Yes, No, or Maybe?\n {candidate}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{candidate}"


class GLUEWNLITemplate(Template):
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        sentence1 = sample.data["sentence1"]
        sentence2 = sample.data["sentence2"]
        return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences entailed by each other? Yes or No?\n"

    def verbalize(self, sample, candidate):
        sentence1 = sample.data["sentence1"]
        sentence2 = sample.data["sentence2"]
        return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences entailed by each other? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class SICKTemplate(Template):
    verbalizer = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def encode(self, sample):
        sentence1 = sample.data["sentence_A"]
        sentence2 = sample.data["sentence_B"]
        return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences entailment, contradiction, or neutral?\n"

    def verbalize(self, sample, candidate):
        sentence1 = sample.data["sentence_A"]
        sentence2 = sample.data["sentence_B"]
        return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences entailment, contradiction, or neutral?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ANLITemplate(Template):
    verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Yes, No, or Maybe?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Yes, No, or Maybe?\n {candidate}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class MedicalQuestionsPairsTemplate(Template):
    verbalizer = {1: "Yes", 0: "No"}

    def encode(self, sample):
        question1 = sample.data["question_1"]
        question2 = sample.data["question_2"]
        return f"Question 1: {question1}\nQuestion 2: {question2}\nAre these questions similar to each other? Yes or No?\n"

    def verbalize(self, sample, candidate):
        question1 = sample.data["question_1"]
        question2 = sample.data["question_2"]
        return f"Question 1: {question1}\nQuestion 2: {question2}\nAre these questions similar to each other? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class HatexplainTemplate(Template):
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text} Does the text contain hate speech? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text}\n Does the text contain hate speech? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class HateSpeechOffensiveTemplate(Template):
    verbalizer = {0: "hate speech", 1: "offensive", 2: "neither"}

    def encode(self, sample):
        text = sample.data["tweet"]
        return f"{text}\nIs the text hate speech, offensive, or neither?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["tweet"]
        return f"{text}\nIs the text hate speech, offensive, or neither?\n {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class TweetEvalOffensiveTemplate(Template):
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text}\n Is the text offensive? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text}\n Is the text offensive? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class EthosTemplate(Template):
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text}\nIs the text offensive? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text}\nIs the text offensive? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class TweetEvalHateTemplate(Template):
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text}\n Is the text hate speech? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text}\n Is the text hate speech? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class HateSpeech18Template(Template):
    verbalizer = {0: "No", 1: "Yes", 2: "skip", 3: "depends"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text}\nIs the text hate speech? Yes, No, Skip, or Depends?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text}\nIs the text hate speech? Yes, No, Skip, or Depends?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class KiltFeverTemplate(Template):
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data["input"]
        return f"{text}\nIs the claim true? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["input"]
        return f"{text}\nIs the claim true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class LiarTemplate(Template):
    verbalizer = {0: "Pants-on-Fire", 1: "False", 2: "Barely-True", 3: "Half-True", 4: "Mostly-True", 5: "True"}

    def encode(self, sample):
        text = sample.data["statement"]
        return f"{text}\nWhat is the label for this claim? Pants-on-Fire, False, Barely-True, Half-True, Mostly-True, or True?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["statement"]
        return f"{text}\nWhat is the label for this claim? Pants-on-Fire, False, Barely-True, Half-True, Mostly-True, or True?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class LiarTemplateJustified(Template):
    verbalizer = {0: "Pants-on-Fire", 1: "False", 2: "Barely-True", 3: "Half-True", 4: "Mostly-True", 5: "True"}

    def encode(self, sample):
        text = sample.data["statement"]
        justification = sample.data["justification"]
        return f"Claim: {text}\nJustification: {justification}\nWhat is the label for this claim? Pants-on-Fire, False, Barely-True, Half-True, Mostly-True, or True?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["statement"]
        justification = sample.data["justification"]
        return f"Claim: {text}\nJustification: {justification}\nWhat is the label for this claim? Pants-on-Fire, False, Barely-True, Half-True, Mostly-True, or True?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class HealthFactsTemplate(Template):
    verbalizer = {0: "False", 1: "True", 2: "Unverified", 3: "Mixture"}

    def encode(self, sample):
        text = sample.data["claim"]
        return f"{text}\nWhat is the label for this claim? False, True, Unverified, or Mixture?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["claim"]
        return f"{text}\nWhat is the label for this claim? False, True, Unverified, or Mixture?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class HealthFactsEvidenceTemplate(Template):
    verbalizer = {0: "False", 1: "True", 2: "Unverified", 3: "Mixture"}

    def encode(self, sample):
        text = sample.data["claim"]
        explanation = sample.data["explanation"]
        return f"Claim: {text}\nExplanation: {explanation}\nWhat is the label for this claim? False, True, Unverified, or Mixture?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["claim"]
        explanation = sample.data["explanation"]
        return f"Claim: {text}\nExplanation: {explanation}\nWhat is the label for this claim? False, True, Unverified, or Mixture?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ClimateFeverTemplate(Template):
    verbalizer = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def encode(self, sample):
        text = sample.data["claim"]
        return f"{text}\nWhat is the label for this claim? entailment, neutral, or contradiction?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["claim"]
        return f"{text}\nWhat is the label for this claim? entailment, neutral, or contradiction?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ClimateFeverEvidenceTemplate(Template):
    verbalizer = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def encode(self, sample):
        text = sample.data["claim"]
        explanation = sample.data["evidence"]
        return f"Claim: {text}\nEvidence: {explanation}\nWhat is the label for this claim? entailment, neutral, or contradiction?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["claim"]
        explanation = sample.data["evidence"]
        return f"Claim: {text}\nEvidence: {explanation}\nWhat is the label for this claim? entailment, neutral, or contradiction?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class TabFactTemplate(Template):
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        text = sample.data["statement"]
        evidence = sample.data["table_text"]
        return f"Claim: {text}\nEvidence: {evidence}\nIs the claim true? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["statement"]
        evidence = sample.data["table_text"]
        return f"Claim: {text}\nEvidence: {evidence}\nIs the claim true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class TweetEvalStanceFeminismTemplate(Template):
    verbalizer = {0: "neutral", 1: "against", 2: "favor"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text}\nWhat is the stance of the tweet towards feminism? neutral, against, or favor?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text}\nWhat is the stance of the tweet towards feminism? neutral, against, or favor?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class TweetEvalStanceAtheismTemplate(Template):
    verbalizer = {0: "neutral", 1: "against", 2: "favor"}

    def encode(self, sample):
        text = sample.data["text"]
        return f"{text}\nWhat is the stance of the tweet towards atheism? neutral, against, or favor?\n"

    def verbalize(self, sample, candidate):
        text = sample.data["text"]
        return f"{text}\nWhat is the stance of the tweet towards atheism? neutral, against, or favor?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WikiQATemplate(Template):
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"Question: {question}\nAnswer: {answer}\nIs the answer correct? Yes or No?\n"

    def verbalize(self, sample, candidate):
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"Question: {question}\nAnswer: {answer}\nIs the answer correct? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class DirectMetaICLTemplate(Template):
    method = "direct"

    def encode(self, sample):
        text = sample.data["input"]
        if sample.candidates is not None:
            text += "\n" + ", ".join(sample.candidates)
        return f"{text} "

    def verbalize(self, sample, candidate):
        text = sample.data["input"]
        if sample.candidates is not None:
            text += "\n" + ", ".join(sample.candidates)
        return f"{text}\n{candidate}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {candidate}"


class ChannelMetaICLTemplate(Template):

    method = "channel"

    def encode(self, sample):
        text = sample.data["input"]
        return f"{text} "

    def verbalize(self, sample, candidate):
        text = sample.data["input"]
        return f"{text}\n{candidate}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {candidate}"
