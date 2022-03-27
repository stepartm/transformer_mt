import math
import random
import os
from typing import Optional

import torch
from tqdm import tqdm
import youtokentome as yttm


class SentencesIterable:
    '''
    The iterable in this class is used multiple times and since the default
    implementation of __iter__ exhausts the iterable after the first call
    to this method, the method is redefined.
    '''

    def __init__(self, path: str, lowercase=False):
        self.path = path
        self.iterator = None
        self.lowercase = lowercase
        self.__iter__()
    
    def __iter__(self):
        self.iterator = map(str.strip, open(self.path))

        if self.lowercase:
            self.iterator = map(str.lower, self.iterator)

        return self.iterator
    
    def __len__(self):
        sentence_iterable = self.__iter__()
        num_objects = 0

        for _ in sentence_iterable:
            num_objects += 1

        return num_objects
    
    def __next__(self):
        return next(self.iterator)


class MTData(torch.utils.data.IterableDataset):
    
    def __init__(
        self, source_path: str, target_path: str, sort_data=False,
        tokenizer: Optional[yttm.BPE]=None, lowercase=True,
        batch_size: Optional[int]=100
    ):
        '''
        It is expected, that the data for source and target
        language are aligned, meaning that the number of
        sentences for both languages is equal and source_sentence[i]
        corresponds to target_sentence[i] semantically.
        '''
        self.source_path = source_path
        self.target_path = target_path
        self.lowercase = lowercase

        self.n_objects = len(
            SentencesIterable(self.source_path)
        )
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
        if sort_data:
            self.generate_sorted_data()

        self.source_sentences = SentencesIterable(self.source_path)
        self.target_sentences = SentencesIterable(self.target_path)
        
    def generate_sorted_data(self):
        '''
        The data is sorted by source sequence length
        to ensure that distributed processing latency
        for operations on source sequences is reduced.
        '''
        assert self.tokenizer

        results = []

        for batch in tqdm(self.batches, desc='Sorting sentences'):
            source_sentences, target_sentences = zip(*batch)
            source_tokens = self.tokenizer.encode(
                source_sentences, output_type=yttm.OutputType.ID
            )
            source_lengths = [len(_) for _ in source_tokens]
            results.extend(
                list(
                    zip(source_sentences, target_sentences, source_lengths)
                )
            )
        
        results.sort(key=lambda x: x[2])
        '''
        The batches will contain the sentences of approximately equal length,
        but the iterator is biased, meaning that the first batches will contain
        the shortest sequences and the last batches will contain the longest
        ones, therefore batches should be shuffled.
        '''
        batches = [
            results[i: i + self.batch_size]
            for i in range(0, self.n_objects, self.batch_size)
        ]
        random.shuffle(batches)

        sorted_source = 'sorted_source_sentences.txt'
        if os.path.exists(sorted_source):
            os.remove(sorted_source)

        sorted_target = 'sorted_target_sentences.txt'
        if os.path.exists(sorted_source):
            os.remove(sorted_target)

        source_f = open(sorted_source, 'w')
        target_f = open(sorted_target, 'w')

        for batch in batches:
            for sample in batch:
                source_sentence, target_sentence, _ = sample
                source_f.write(source_sentence + '\n')
                target_f.write(target_sentence + '\n')
        
        source_f.close()
        target_f.close()

        self.source_path = 'sorted_source_sentences.txt'
        self.target_path = 'sorted_target_sentences.txt'
    
    def get_batch_iterator(self):
        num_batches = math.ceil(
            self.n_objects / self.batch_size
        )
        
        source_iterator = SentencesIterable(
            path=self.source_path, lowercase=self.lowercase
        )
        target_iterator = SentencesIterable(
            path=self.target_path, lowercase=self.lowercase
        )

        remaining_objects = self.n_objects

        for _ in range(num_batches):
            source_data = []
            target_data = []

            for _ in range(
                min(self.batch_size, remaining_objects)
            ):
                source_data.append(next(source_iterator))
                target_data.append(next(target_iterator))

            batch = list(zip(source_data, target_data))
            yield batch

            remaining_objects -= self.batch_size
    
    @property
    def batches(self):
        return self.get_batch_iterator()

    def __iter__(self):
        return zip(self.source_sentences, self.target_sentences)