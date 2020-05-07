"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset
import spacy
# import time

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

nlp = spacy.load('en_core_web_sm')
spacy.prefer_gpu()
spacy_checkpoint = 0

bigger = 0
smaller = 0

class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _) in samples:
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self, args, path):
        self.args = args
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

    # score the sentence based on a set of heuristics
    def score_sentence(self, sentence, question):
        score = 0
        for token in sentence:
            for word in question:
                if token.text == word.text and token.dep_ == word.dep_:
                    score += 1
        return score

    def spacy_adjustment(self, processed_passage, question, answer_start, answer_end):
        passage = processed_passage
        question = nlp(question)
        passage_tokens = [token.text for token in passage]
        question_tokens = [token.text for token in question]
        SCORE = 1
        THRESHOLD = 2
        CHECK_NER = True
        CHECK_PROPN = True
        CHECK_SENT_LEN = True
        CHECK_SUBJ = True
    
        # for sent in passage.sents:
        #     end_of_sent = index + len(sent)
        #     contains_entity = len(sent.ents) != 0
        #     if contains_entity:
        #         sent_cands.append(sent)
        #         index += len(sent)
        #     elif index < answer_start < end_of_sent:
        #         print("cheating")
        #         sent_cands.append(sent)
        #         index += len(sent)
        #     elif end_of_sent < answer_start:
        #         answer_start -= len(sent)
        #         answer_end -= len(sent)

        # create array to store each sentence and its score
        sentences = [[sentence, 0] for sentence in passage.sents]

        # parse the question for useful attributes
        question_proper_nouns = [noun.text for noun in question if noun.pos_ == 'PROPN']
        question_noun_subjs = [noun.text for noun in question if noun.dep_ == 'nsubj']

        for sent in sentences:
            # check if there's a named entity
            if CHECK_NER:
                # are there detected named entities?
                if len(sent[0].ents) > 0:
                    sent[SCORE] += 1
                # do any of the entities match those in the question?
                for ent in sent[0].ents:
                    if ent in question.ents:
                        sent[SCORE] += 1
            # check if sentence contains proper noun from question
            if CHECK_PROPN:
                for word in sent[0]:
                    if word.pos_ == 'PROPN' and word.text in question_proper_nouns:
                        sent[SCORE] += 1
            # check the sentence length
            if CHECK_SENT_LEN:
                pass
            # check matching subjects
            if CHECK_SUBJ:
                for word in sent[0]:
                    if word.dep_ == 'nsubj' and word.text in question_noun_subjs:
                        sent[SCORE] += 1

        
        avg_sent_len = sum([len(sent[0]) for sent in sentences]) / len(sentences)
        min_sent_len = min([len(sent[0]) for sent in sentences])
        max_sent_len = max([len(sent[0]) for sent in sentences])
        ans_len = 0
        index = 0
        for sent in sentences:
            end_of_sent = index + len(sent[0])
            if index < answer_start < end_of_sent:
                ans_len += len(sent[0])
            index += len(sent[0])
        global bigger, smaller
        if ans_len == min_sent_len:
            bigger += 1
        else:
            smaller += 1

        # print('min', min_sent_len)
        # print('avg', avg_sent_len)
        # print('max', max_sent_len)

        # go through sentence candidates and remove ones with too low of a score
        passage_tokens = []
        index = 0
        for sent in sentences:
            end_of_sent = index + len(sent[0])
            # keep the sentence if the score is high enough
            if sent[SCORE] > THRESHOLD:
                passage_tokens.extend([token.text for token in sent[0]])
                index += len(sent[0])
            elif index < answer_start < end_of_sent:
                # the sentence containing the answer is being removed
                print("Sentence with answer is removed")
            elif end_of_sent < answer_start:
                # sentence not containing the answer is removed, adjust start and end
                answer_start -= len(sent[0])
                answer_end -= len(sent[0])

        if len(passage_tokens) == 0:
            print("PASSAGE TOKENS LENGTH IS ZERO")

        return passage_tokens, question_tokens, answer_start, answer_end
            

    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """

        print('Creating samples')
        samples = []
        spacy_checkpoint = 0
        for elem in self.elems[:1000]:
            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            # tic = time.perf_counter()
            passage_str = elem['context']
            processed_passage = nlp(passage_str)
            spacy_checkpoint += 1
            for qa in elem['qas']:
                # Get the passage and question string
                passage_str = elem['context']
                question_str = qa['question']
                qid = qa['qid']

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive. These will need to be adjusted if sentences 
                # are eliminated
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]


                # Run through Spacy
                passage, question, answer_start, answer_end = self.spacy_adjustment(processed_passage, question_str, answer_start, answer_end)

                # Adjust size for max length and lowercase
                passage = passage[:self.args.max_context_length]
                passage = [token.lower() for token in passage]
                question = question[:self.args.max_question_length]
                question = [token.lower() for token in question]
                
                samples.append(
                    (qid, passage, question, answer_start, answer_end)
                )
                # print('OUR ANSWER   ', passage[answer_start:answer_end + 1])
                # print('ACTUAL ANSWER',answers[0]['text'])

            print('num bigger', bigger)
            print('num smaller', smaller)
            print('frac', (bigger / (smaller + bigger)))
            if spacy_checkpoint % 1000 == 0:
                print('Finished ' + str(spacy_checkpoint) + ' samples')
        print()
        return samples

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        start_positions = []
        end_positions = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, answer_start, answer_end = self.samples[idx]

            # Convert words to tensor.
            passage_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(passage)
            )
            question_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(question)
            )
            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)

        return zip(passages, questions, start_positions, end_positions)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]
                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages = torch.zeros(bsz, max_passage_length)
            padded_questions = torch.zeros(bsz, max_question_length)
            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages, questions)):
                passage, question = passage_question
                padded_passages[iii][:len(passage)] = passage
                padded_questions[iii][:len(question)] = question

            # Create an input dictionary
            batch_dict = {
                'passages': cuda(self.args, padded_passages).long(),
                'questions': cuda(self.args, padded_questions).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long()
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
