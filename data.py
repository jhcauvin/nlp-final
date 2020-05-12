"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch

from torch.utils.data import Dataset
from random import shuffle
import random as rand
from utils import cuda, load_dataset
import spacy
from spacy.tokenizer import Tokenizer
import warnings
warnings.filterwarnings("ignore")

# import time

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

nlp = spacy.load('en_core_web_md')
tokenizer = nlp.tokenizer
#spacy.require_gpu()

spacy_checkpoint = 0
ner_count = 0
propn_count = 0
nsubj_count = 0
simil_count = 0
nchunk_count = 0
num_unprocessed = 0
total = 0
sents_removed = 0

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
    def __init__(self, args, path, do_train=True, is_train=True,):
        self.args = args
        self.is_train = is_train
        self.do_train = do_train
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

    def spacy_adjustment(self, processed_passage, question, answer_start, answer_end):
        passage = processed_passage
        question = nlp(question)
        orig_ans_st = answer_start
        orig_ans_end = answer_end
        question_tokens = [token.text for token in question]
        SCORE = 1
        THRESHOLD = 2
        CHECK_NER = True
        CHECK_PROPN = True
        CHECK_SUBJ = True
        CHECK_SIMILARITY = True
        CHECK_NOUN_CHNKS = True
        global ner_count, propn_count, nsubj_count, simil_count, nchunk_count, num_unprocessed, total, sents_removed
        total += 1
    
        # create array to store each sentence and its score
        sentences = [[sentence, 0] for sentence in passage.sents]

        # parse the question for useful attributes
        question_proper_nouns = [noun.lemma_ for noun in question if noun.pos_ == 'PROPN']
        question_noun_subjs = [noun.lemma_ for noun in question if noun.dep_ == 'nsubj']
        avg_similarity = sum([sent[0].similarity(question) for sent in sentences]) / len(sentences)

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
                        ner_count += 1
            # check if sentence contains proper noun from question
            if CHECK_PROPN:
                for word in sent[0]:
                    if word.pos_ == 'PROPN' and word.lemma_ in question_proper_nouns:
                        sent[SCORE] += 1
                        propn_count += 1
            # check matching subjects
            if CHECK_SUBJ:
                for word in sent[0]:
                    if word.dep_ == 'nsubj' and word.lemma_ in question_noun_subjs:
                        sent[SCORE] += 1
                        nsubj_count += 1
                        
            # use spacy's built-in similarity estimate
            if CHECK_SIMILARITY:
                if sent[0].similarity(question) > avg_similarity:
                    sent[SCORE] += 1
                    simil_count += 1
            # check for matching noun chunks
            if CHECK_NOUN_CHNKS:
                for chunk in sent[0].noun_chunks:
                    for qchunk in question.noun_chunks:
                        if chunk.text == qchunk.text:
                            sent[SCORE] += 1
                            nchunk_count += 1
        if len(question) == 0:
            print('question with nothing')

        # go through sentence candidates and remove ones with too low of a score
        passage_tokens = []
        index = 0
        removed = 0
        for sent in sentences:
            end_of_sent = index + len(sent[0])
            # keep the sentence if the score is high enough
            if sent[SCORE] >= THRESHOLD:
                passage_tokens.extend([token.text for token in sent[0]])
                index += len(sent[0])
            elif self.is_train and (index <= answer_start < end_of_sent or index <= answer_end < end_of_sent or answer_start <= index <= answer_end):
                # a sentence containing the answer is being removed. Don't process this during training
                num_unprocessed += 1
                orig_pass = [token.text for token in passage]
                return orig_pass, question_tokens, orig_ans_st, orig_ans_end
            elif end_of_sent <= answer_start:
                # sentence not containing the answer is removed, adjust start and end
                answer_start -= len(sent[0])
                answer_end -= len(sent[0])
                removed += 1

        # if all sentences are removed, return the original values
        if len(passage_tokens) == 0:
            orig_pass = [token.text for token in passage]
            return orig_pass, question_tokens, orig_ans_st, orig_ans_end
            
        sents_removed += removed
        return passage_tokens, question_tokens, answer_start, answer_end
    
    # Search for the start index of ar2 inside ar1
    def find_subarray(self, ar1, ar2):
        for index1 in range(len(ar1)):
            if index1 > len(ar1) - len(ar2):
                return -1
            sub_ar = ar1[index1:index1 + len(ar2)]
            worked = True
            for el in range(len(sub_ar)):
                if sub_ar[el].text != ar2[el].text:
                    worked = False
                    break
            if worked:
                return index1
        return -1
    
    def _original(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.
        Returns:
            A list of words (string).
        """
        samples = []
        for elem in self.elems:
            # Unpack the context paragraph. Shorten to max sequence length.
            passage = [
                token.lower() for (token, offset) in elem['context_tokens']
            ][:self.args.max_context_length]

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for qa in elem['qas']:
                qid = qa['qid']
                question = [
                    token.lower() for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]
                samples.append(
                    (qid, passage, question, answer_start, answer_end)
                )
                
        return samples

    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        if self.is_train and self.do_train:
            print('Creating samples for train set')
        elif self.is_train:
            print('Only processing test set. Calling original code for train')
            return self._original()
        else:
            print('Creating samples for test set')
        samples = []
        spacy_checkpoint = 0
        # rand.seed(4)
        # rand.shuffle(self.elems)
        for elem in self.elems:
            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            passage_str = elem['context']
            processed_passage = nlp(passage_str)
            spacy_checkpoint += 1
            for qa in elem['qas']:
                # get the passage and question string
                passage_str = elem['context']
                question_str = qa['question']
                qid = qa['qid']

                # select the first answer span, which is formatted as (start_position, end_position), where the end_position
                # is inclusive. These will need to be adjusted if sentences are eliminated
                answers = qa['detected_answers']
                orig_ans = answers[0]['text']

                # spacy tokenizes the passages in a slightly different way
                # so we tokenize the answer with spacy and then find it within the spacy-tokenized passage
                processed_ans = tokenizer(orig_ans)
                answer_start = self.find_subarray(processed_passage, processed_ans)
                answer_end = answer_start + len(processed_ans) - 1

                ans = processed_passage[answer_start:answer_end + 1].text

                # the spacy tokenizer makes a mistake on 4 out of roughly 86,000 test cases. We ignore these during training
                if ans != orig_ans and self.is_train:
                    continue

                # run through Spacy
                passage, question, answer_start, answer_end = self.spacy_adjustment(processed_passage, question_str, answer_start, answer_end)

                ans = passage[answer_start:answer_end + 1]
                real = [token.text for token in processed_ans]

                # training should never have the wrong answer
                if self.is_train and real != ans:
                    [print(sent) for sent in processed_passage.sents]
                    print(real)
                    print(ans)

                # adjust size for max length and lowercase
                passage = passage[:self.args.max_context_length]
                passage = [token.lower() for token in passage]
                question = question[:self.args.max_question_length]
                question = [token.lower() for token in question]

                # preprocessing got something wrong on testing set, return a wrong answer that's in bounds to avoid erroring out
                if (not self.is_train) and (len(passage) <= answer_start or len(passage) <= answer_end or answer_start > answer_end):
                    answer_start = 0
                    answer_end = 0
                    
                samples.append(
                    (qid, passage, question, answer_start, answer_end)
                )

            if spacy_checkpoint % 1000 == 0:
                print('Finished ' + str(spacy_checkpoint) + ' samples')
        global ner_count, propn_count, nsubj_count, simil_count, nchunk_count, num_unprocessed, total, sents_removed
        print()
        if self.is_train:
            # only relevant when training
            print(str(num_unprocessed) + " out of " + str(total) + " examples were not processed")
        print(str(sents_removed / total) + " average sentences removed per passage/question pair")
        print()
        print('NER triggered ' + str(ner_count) + ' times')
        print('Proper Noun triggered ' + str(propn_count) + ' times')
        print('Noun Subject triggered ' + str(nsubj_count) + ' times')
        print('Similarity triggered ' + str(simil_count) + ' times')
        print('Noun chunk triggered ' + str(nchunk_count) + ' times')
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
