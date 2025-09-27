import os
from typing import BinaryIO, List, Tuple, Set
import regex as re
from collections import Counter, defaultdict
import multiprocessing
from pathlib import Path
import pickle
import datetime
import heapq
from sortedcontainers import SortedList

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenization_worker(start: int, end: int, file: str, out_dir: str, re_split_token, re_pre_token):
    _print = lambda *args, **kwargs: print(f"[Worker {start}:{end}]", *args, **kwargs)
    with open(file, "rb") as f:
        f.seek(start)
        # find_chunk_boundaries ensure that chuk can be decoded without error
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        _print("chunk charcter number", len(chunk))

        chunk = ''.join(re.split(re_split_token, chunk))
        _print("chunk charcter number after remove special tokens", len(chunk))

        t2c = Counter()
        for e in re.finditer(re_pre_token, chunk):
            token = e.group(0)
            t2c[token] += 1
        _print(f"pretoekn number: {len(t2c)} , 10 samples {list(t2c.items())[:10]}")
        outfile_name = f"token2count_{start}_{end}.pkl"
        with open(os.path.join(out_dir, outfile_name), "wb") as out_f:
            pickle.dump(t2c, out_f)
            _print(f"Dumped to {os.path.join(out_dir, outfile_name)}")

def pretokenization(train_file_path: str, num_processes: int, special_tokens: List[str], resume_dir: str | None = None) -> Counter[str]:
    if resume_dir is not None:
        PRETOKEN_RESULT_DIR = resume_dir
    else:
        PRETOKEN_RESULT_DIR = f'./data/token2count_debug/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        # split special token，不保留分隔符
        PAT_SPLIT_SPEC_TOKEN = re.compile(
            "|" .join([re.escape(e) for e in special_tokens])
        )
        PAT_PRE_TOKEN = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        # create result dir if not exist
        Path(PRETOKEN_RESULT_DIR).mkdir(parents=True, exist_ok=True)

        # multiple process pre-tokenization
        with open(train_file_path, "rb") as f:
            # compute bound, processing number to your core number
            num_processes = num_processes
            processes = []
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                p = multiprocessing.Process(
                    target=pretokenization_worker,
                    args=(start, end, train_file_path, PRETOKEN_RESULT_DIR, PAT_SPLIT_SPEC_TOKEN, PAT_PRE_TOKEN)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                if (p.exitcode != 0):
                    raise RuntimeError("One of the processes failed!")
            print("All processes finished!")

        # merge t2c
    total_t2c = Counter()
    for fname in os.listdir(PRETOKEN_RESULT_DIR):
        if not fname.startswith("token2count_"):
            continue
        with open(os.path.join(PRETOKEN_RESULT_DIR, fname), "rb") as f:
            total_t2c.update(pickle.load(f))
    print(f"Total pre-token number: {len(total_t2c)}, 10 samples {list(total_t2c.items())[:10]}")
    return total_t2c

class BPC:
    def __init__(self, bp: Tuple[bytes, ...], count) -> None:
        self.bp = bp
        self.count = count
    
    def __lt__(self, other: 'BPC') -> bool:
        return (-self.count, other.bp) < (-other.count, self.bp)
    
    def __repr__(self) -> str:
        return f"bpc(count={self.count}, bp={self.bp})"
    
    def __eq__(self, other: 'BPC'):
        return isinstance(other, BPC) and self.bp == other.bp and self.count == other.count



def bytes_reduce(bps: Tuple[bytes, ...]) -> bytes:
    from functools import reduce
    return reduce(lambda a, b: a + b, bps)

def update_sorted_list(sorted_list: SortedList, old: BPC, new: BPC):
    sorted_list.remove(old)
    sorted_list.add(new)

# merge token to get reult
def bpe_train(round_num: int, pre_token_counter: Counter[str]) -> List[Tuple[bytes, ...]]:
    bp2c: Counter[Tuple[bytes, ...]] = Counter()
    bp2token: defaultdict[Tuple[bytes, ...], Set[str]] = defaultdict(set)
    token2bytes_list: defaultdict[str, List[Tuple[bytes, ...]]] = defaultdict(list)
    sorted_list: SortedList = SortedList()
    result = []

    for s in pre_token_counter:
        bytes_s = s.encode('utf-8')
        tokens = [bytes([b]) for b in bytes_s]
        for i in range(len(tokens)-1):
            bp = (tokens[i], tokens[i+1])
            bp2c[bp] += pre_token_counter[s]
            token2bytes_list[s].append(bp)
            bp2token[bp].add(s)
    for bp in bp2c:
        sorted_list.add(BPC(bp, bp2c[bp]))
    print("init: ", sorted_list)
    for i in range(round_num):
        if len(sorted_list) == 0:
            break
        # 找到当前最高的频数
        top:BPC = sorted_list[0] #type: ignore
        sorted_list.remove(top)
        cur_bp = top.bp
        merged_bp = bytes_reduce(cur_bp)
        # 最新的 token
        result.append(cur_bp)
        total_new_bp_counter: Counter[Tuple[bytes, ...]] = Counter()
        for tk in bp2token[cur_bp]:
            # 找到所有的 token
            bytes_tuple_list = token2bytes_list[tk]
            tmp_new_bytes_tuple_list = []
            p = 0
            l = len(bytes_tuple_list)
            while p < l:
                ele = bytes_tuple_list[p]
                if p + 1 < l and bytes_tuple_list[p + 1] == cur_bp:
                    # 之前的元素进行合并
                    assert cur_bp[0] == ele[-1], (cur_bp, ele)
                    new_bp = ele[:-1] + (merged_bp,)
                    tmp_new_bytes_tuple_list.append(new_bp)
                    bp2token[new_bp].add(tk)
                    total_new_bp_counter[new_bp] += pre_token_counter[tk]
                    update_sorted_list(sorted_list, BPC(ele, bp2c[ele]), BPC(ele, bp2c[ele] - pre_token_counter[tk]))
                    bp2c[ele] -= pre_token_counter[tk]
                elif p > 0 and bytes_tuple_list[p - 1] == cur_bp:
                    # 和之后的元素进行合并
                    assert cur_bp[-1] == ele[0], (cur_bp, ele)
                    new_bp = (merged_bp,) + ele[1:]
                    tmp_new_bytes_tuple_list.append(new_bp)
                    bp2token[new_bp].add(tk)
                    total_new_bp_counter[new_bp] += pre_token_counter[tk]
                    update_sorted_list(sorted_list, BPC(ele, bp2c[ele]), BPC(ele, bp2c[ele] - pre_token_counter[tk]))
                    bp2c[ele] -= pre_token_counter[tk]
                elif ele != cur_bp:
                    tmp_new_bytes_tuple_list.append(ele)
                p += 1
            token2bytes_list[tk] = tmp_new_bytes_tuple_list
        
        # 被选中的 bytes pair 需要移除掉，后续不可能
        del bp2token[cur_bp]
        del bp2c[cur_bp]
        bp2c.update(total_new_bp_counter)
        for k in total_new_bp_counter:
            assert bp2c[k] != 0, k
            sorted_list.add(BPC(k, total_new_bp_counter[k]))
        print(cur_bp, sorted_list)
    return result


def test_sorted_list():
    eles = [BPC((b'ab',), 3), BPC((b'bc',), 3), BPC((b'c',), 1)]
    sl = SortedList()
    for e in eles:
        sl.add(e)
    print(sl)
    sl.remove(BPC((b'ab',), 3))
    print(sl)
    

def test_train_bpe():
    # 生成 bpe
    # train_file_path = "/home/wuziyi/code/cs336/assignment1-basics/data/mini_debug.txt"
    # special_tokens = [
    #     "<|endoftext|>"
    # ]
    # pretokenization(train_file_path, num_processes=4, special_tokens=special_tokens)
    w2c = Counter({'low': 5, 'lower': 2, 'widest': 3, 'newest': 6})
    res = bpe_train(20, w2c)
    print(res)
    

if __name__ == "__main__":
    # train_file_path = "/home/wuziyi/code/cs336/assignment1-basics/data/debug_tiny_story_set.txt"
    # special_tokens = [
    #     "<|endoftext|>"
    # ]
    # pretokenization(train_file_path, num_processes=4, special_tokens=special_tokens)

    # test_sorted_list()
    test_train_bpe()