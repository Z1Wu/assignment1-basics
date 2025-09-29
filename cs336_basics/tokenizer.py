import pickle
from collections.abc import Iterable, Iterator
from cs336_basics.bpe import PAT_PRE_TOKEN
import regex as re
from memory_profiler import profile


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None): # Construct a tokenizer from a given
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.merges_set = set(self.merges)
        self.vocab_bytes2id = {v : k for (k, v) in self.vocab.items()}
        # print(self.vocab_bytes2id)
        # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
        if special_tokens != None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.PAT_SPECIAL_TOKENS = re.compile(
                f"({"|".join([re.escape(e) for e in self.special_tokens])})"
            )
            for s in special_tokens:
                sb = s.encode('utf-8')
                if sb not in self.vocab_bytes2id:
                    tid = len(self.vocab)
                    self.vocab_bytes2id[sb] = tid
                    self.vocab[tid] = sb
        else:
            self.special_tokens = None
            self.PAT_SPECIAL_TOKENS = None


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # generate vocabulary and merges object from file 
        with open(vocab_filepath, 'wb') as fv, open(merges_filepath, 'wb') as fm:
            return Tokenizer(
                pickle.load(fv),
                pickle.load(fm),
                special_tokens
            )

    def _encode_pretoken(self, text: str) -> list[int]:
        if self.special_tokens != None and text in self.special_tokens:
            return [self.vocab_bytes2id[text.encode('utf-8')]]
        # Encode single token
        encoded_str = text.encode('utf-8')
        bytes_seq = [bytes([e]) for e in encoded_str]

        # 复杂度过高
        # for merge in self.merges:
        #     if merge[0] + merge[1] in encoded_str:
        #         p = 0
        #         l = len(bytes_seq)
        #         new_bytes_seq = []
        #         while p < l:
        #             if p + 1 < l and (bytes_seq[p], bytes_seq[p + 1]) == merge:
        #                 new_bytes_seq.append(bytes_seq[p] + bytes_seq[p + 1])
        #                 p += 2
        #             else:
        #                 new_bytes_seq.append(bytes_seq[p])
        #                 p += 1
        #         bytes_seq = new_bytes_seq

        # 找到最先的合并
        while 1:
            min_token_id = float('inf')
            merge_idx = -1
            # 根据 token id 找到第一个 apply 的 merge
            for i in range(len(bytes_seq) - 1):
                bp = (bytes_seq[i], bytes_seq[i + 1])
                if bp in self.merges_set:
                    new_token = bp[0] + bp[1]
                    if new_token in self.vocab_bytes2id and self.vocab_bytes2id[new_token] < min_token_id:
                        min_token_id = self.vocab_bytes2id[new_token]
                        merge_idx = i
            # end of merge
            if merge_idx == -1: 
                break
            
            bytes_seq = (
                bytes_seq[:merge_idx] + 
                [bytes_seq[merge_idx] + bytes_seq[merge_idx + 1]] + 
                bytes_seq[merge_idx + 2:]
            )

        return [self.vocab_bytes2id[bs] for bs in bytes_seq]


    def _get_pretokens(self, text: str) -> Iterator[str]:
        for s in self.PAT_SPECIAL_TOKENS.split(text) if self.PAT_SPECIAL_TOKENS != None else [text]:
            if len(s) == 0:
                continue
            if self.special_tokens != None and s in self.special_tokens:
                yield s
                continue
            # pretoken should not across boundary
            for e in re.finditer(PAT_PRE_TOKEN, s):
                yield e.group(0)

    def encode(self, text: str) -> list[int]:
        # FIXME: Failed to pass memory test 
        # in unit test, the result of this is a list with 1328129 elements
        # if represent token id with int , it will at least occupy 10M memory which is 10 x memory limit of unit-test
         
        
        # Try1: split into chunks of string and pass into encode_iterable
        # def split_str(ss, chunk_num):
        #     chunk_num = 10
        #     chunk_size = len(ss) // 10
        #     for i in range(chunk_num):
        #         begin = i * chunk_size
        #         end = (i + 1) * chunk_size if i != chunk_num - 1 else len(ss)
        #         chunk = ss[begin: end]
        #         if len(chunk) == 0:
        #             continue
        #         yield chunk


        # def split_generator():
        #     max_spec_len = max([len(tk) for tk in self.special_tokens]) if self.special_tokens != None else 0
        #     for s in self.PAT_SPECIAL_TOKENS.split(text) if self.PAT_SPECIAL_TOKENS != None else [text]:
        #         if len(s) == 0:
        #             continue
                
        #         if len(s) > max(max_spec_len, 10):
        #             yield from split_str(s, chunk_num=10)
        #         else:
        #             yield s
        # return list(self.encode_iterable(split_generator()))
    
        
        return [
            e for tk in self._get_pretokens(text) 
            for e in self._encode_pretoken(tk)
        ]

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        #  This is required for memory-efficient tokenization of large files that we cannot directly load into
        # memory.
        
        prev_last: str = ""
        for text in iterable:
            pre_tokens = list(self._get_pretokens(prev_last + text))
            assert len(pre_tokens) >= 1, f"illeage string [{text}]"
            prev_last = pre_tokens[-1]
            for tk in pre_tokens[:-1]:
                for e in self._encode_pretoken(tk):
                    yield e
        
        if len(prev_last) != 0:
            for e in self._encode_pretoken(prev_last):
                yield e

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        bytes_seq = b''
        for e in ids:
            bytes_seq += self.vocab[e]
        # ignore invalid utf-8 string here
        return bytes_seq.decode('utf-8', 'ignore')
        
if __name__ == "__main__":
    # check simple case
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: '<|endoftext|>'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    test_tokenizer = Tokenizer(vocab, merges, ['<|endoftext|>'])
    test_str = 'the cat ate the cat ate the cat ate <|endoftext|>the cat ate'
    print(
        test_tokenizer.encode(test_str)
    )

    # test iter 
    str_list = ['the ca', 't', ' ate']
    print([e for e in test_tokenizer.encode_iterable(str_list)])