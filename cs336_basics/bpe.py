import os
from typing import BinaryIO, List, Tuple, Set
import regex as re
from collections import Counter, defaultdict
import multiprocessing
from pathlib import Path
import pickle
import datetime
from sortedcontainers import SortedList
import logging
from functools import reduce

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DUMP_FILE_PREFIX = "pretoken_dump"


class BPC:
    def __init__(self, bp: Tuple[bytes, ...], count) -> None:
        self.bp = bp
        self.count = count

    def __lt__(self, other: "BPC") -> bool:
        return (-self.count, other.bp) < (-other.count, self.bp)

    def __repr__(self) -> str:
        return f"bpc(count={self.count}, bp={self.bp})"

    def __eq__(self, other: "BPC"):
        return (
            isinstance(other, BPC) and self.bp == other.bp and self.count == other.count
        )

def current_time_str_micro():
    now = datetime.datetime.now()
    return f'{now.strftime("%Y%m%d_%H%M%S")}_{now.microsecond:06d}'

# borrow from cs336_basics/pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

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


def pretokenization_worker(
    start: int, end: int, file: str, outfile_path: str, re_split_token, re_pre_token
):
    _print = lambda s: logging.info(f"[Worker {start}:{end}] {s}")
    with open(file, "rb") as f:
        f.seek(start)
        # find_chunk_boundaries ensure that chuk can be decoded without error
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        _print(f"chunk charcter number: {len(chunk)}")

        t2c = Counter()
        for text in re_split_token.split(chunk):
            # pretoken should not across boundary
            for e in re.finditer(re_pre_token, text):
                token = e.group(0)
                t2c[token] += 1
        _print(f"pretoekn number: {len(t2c)} , 10 samples {list(t2c.items())[:10]}")
        with open(outfile_path, "wb") as out_f:
            pickle.dump(t2c, out_f)
            _print(f"Dumped to {outfile_path}")


def pretokenization(
    train_file_path: str | os.PathLike,
    num_processes: int,
    special_tokens: List[str],
    resume_dir: str | None = None,
) -> Counter[str]:
    if resume_dir is not None:
        PRETOKEN_RESULT_DIR = resume_dir
    else:
        # avoid different run file
        PRETOKEN_RESULT_DIR = f'./data/token2count_debug/{current_time_str_micro()}'
        logging.info(f"Dump pretoken result into dir {PRETOKEN_RESULT_DIR}")
        # split special token，不保留分隔符
        PAT_SPLIT_SPEC_TOKEN = re.compile(
            "|".join([re.escape(e) for e in special_tokens])
        )
        PAT_PRE_TOKEN = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        # create result dir if not exist
        Path(PRETOKEN_RESULT_DIR).mkdir(parents=True, exist_ok=True)
        assert (
            len(os.listdir(PRETOKEN_RESULT_DIR)) == 0
        ), f"dump file directory should be empty ! {PRETOKEN_RESULT_DIR}"

        # multiple process pre-tokenization
        with open(train_file_path, "rb") as f:
            # compute bound, processing number to your core number
            num_processes = num_processes
            processes = []
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                outfile_path = os.path.join(
                    PRETOKEN_RESULT_DIR, f"{DUMP_FILE_PREFIX}_{start}_{end}.pkl"
                )
                p = multiprocessing.Process(
                    target=pretokenization_worker,
                    args=(
                        start,
                        end,
                        train_file_path,
                        outfile_path,
                        PAT_SPLIT_SPEC_TOKEN,
                        PAT_PRE_TOKEN,
                    ),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError("One of the processes failed!")
            logging.info("All processes finished!")

    # merge t2c
    total_t2c = Counter()
    for fname in os.listdir(PRETOKEN_RESULT_DIR):
        if not fname.startswith(DUMP_FILE_PREFIX):
            continue
        with open(os.path.join(PRETOKEN_RESULT_DIR, fname), "rb") as f:
            total_t2c.update(pickle.load(f))
    logging.info(
        f"Total pre-token number: {len(total_t2c)}, 10 samples {list(total_t2c.items())[:10]}"
    )
    return total_t2c


def bytes_reduce(bps: Tuple[bytes, ...]) -> bytes:
    return reduce(lambda a, b: a + b, bps)


def update_sorted_list(sorted_list: SortedList, old: BPC, new: BPC):
    sorted_list.remove(old)
    if new.count > 0:
        sorted_list.add(new)


def _train_bpe(
    round_num: int, pre_token_counter: Counter[str]
) -> List[Tuple[bytes, bytes]]:

    bp2c: Counter[Tuple[bytes, ...]] = Counter()
    bp2token: defaultdict[Tuple[bytes, ...], Set[str]] = defaultdict(set)
    token2bytes_list: defaultdict[str, List[bytes]] = defaultdict(list)
    sorted_list: SortedList = SortedList()
    result = []

    for s in pre_token_counter:
        bytes_s = s.encode("utf-8")
        tokens = [bytes([b]) for b in bytes_s]
        token2bytes_list[s] = [bytes([b]) for b in bytes_s]
        for i in range(len(tokens) - 1):
            bp = (tokens[i], tokens[i + 1])
            bp2c[bp] += pre_token_counter[s]
            bp2token[bp].add(s)

    for bp in bp2c:
        sorted_list.add(BPC(bp, bp2c[bp]))
    for i in range(round_num):
        if len(sorted_list) == 0:
            break
        # get token with highest frequency
        top: BPC = sorted_list[0]  # type: ignore
        sorted_list.remove(top)
        cur_bp = top.bp
        merged_bp = bytes_reduce(cur_bp)
        result.append(cur_bp)
        total_new_bp_counter: Counter[Tuple[bytes, ...]] = Counter()
        logging.info(f"run step {i} with byte pair {cur_bp}")
        for tk in bp2token[cur_bp]:
            tokens_list = token2bytes_list[tk]
            tmp_tokens_list = []
            p = 0
            while p < len(tokens_list):
                if (
                    p + 1 < len(tokens_list)
                    and tokens_list[p] + tokens_list[p + 1] == merged_bp
                ):
                    tmp_tokens_list.append(merged_bp)
                    p += 2
                else:
                    tmp_tokens_list.append(tokens_list[p])
                    p += 1
            old_bp_counter = Counter()
            new_bp_counter = Counter()
            for t in range(len(tokens_list) - 1):
                old_bp_counter[(tokens_list[t], tokens_list[t + 1])] += 1

            for t in range(len(tmp_tokens_list) - 1):
                new_bp_counter[(tmp_tokens_list[t], tmp_tokens_list[t + 1])] += 1

            for e in set(old_bp_counter.keys()) | set(new_bp_counter.keys()):
                # bp needs to decrease frequeny due to merge
                if new_bp_counter[e] < old_bp_counter[e]:
                    if e == cur_bp:
                        continue
                    cnt = old_bp_counter[e] - new_bp_counter[e]
                    new_val = bp2c[e] - pre_token_counter[tk] * cnt
                    update_sorted_list(sorted_list, BPC(e, bp2c[e]), BPC(e, new_val))
                    bp2c[e] = new_val
                    if new_val == 0:
                        del bp2c[e]
                        del bp2token[e]
                # 1. only in new byte pairs and not in old byte paris
                # 2. or count in new byte pairs > count in old byte pair
                elif new_bp_counter[e] > old_bp_counter[e]:
                    bp2token[e].add(tk)
                    total_new_bp_counter[e] += pre_token_counter[tk] * (
                        new_bp_counter[e] - old_bp_counter[e]
                    )

            token2bytes_list[tk] = tmp_tokens_list

        del bp2token[cur_bp]
        del bp2c[cur_bp]
        bp2c.update(total_new_bp_counter)
        for k in total_new_bp_counter:
            assert bp2c[k] != 0, k
            sorted_list.add(BPC(k, bp2c[k]))
        # logging.info(f'current bp {cur_bp}: sorted result {sorted_list}')
    return result


def train_bpe(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str],
    resume_pretoken_dir: str | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pre_token_counter = pretokenization(input_path, 8, special_tokens, resume_pretoken_dir)
    bpe_res = _train_bpe(
        vocab_size - len(special_tokens) - 256, pre_token_counter=pre_token_counter
    )
    vocab = {}
    idx = 0
    for e in special_tokens:
        vocab[idx] = e.encode("utf-8")
        idx += 1
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    for bp in bpe_res:
        vocab[idx] = bytes_reduce(bp)
        idx += 1

    return (vocab, bpe_res)


def test_sorted_list():
    eles = [BPC((b"ab",), 3), BPC((b"bc",), 3), BPC((b"c",), 1)]
    sl = SortedList()
    for e in eles:
        sl.add(e)
    logging.info(sl)
    sl.remove(BPC((b"ab",), 3))
    logging.info(sl)


def test_train_bpe():
    w2c = Counter({"low": 5, "lower": 2, "widest": 3, "newest": 6})
    res = _train_bpe(20, w2c)
    logging.info(res)


def test_train_bpe_full():
    train_file_path = "/home/wuziyi/code/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    special_tokens = ["<|endoftext|>"]
    res = _train_bpe(1000, pretokenization(train_file_path, 8, special_tokens))
    logging.warn(res)


def train_bpe_tinystories():
    # train 
    train_file_path = "/home/wuziyi/code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    out_dir = f"./data/vocab_result/{current_time_str_micro()}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (vocab, merges) = train_bpe(
        train_file_path,
        vocab_size=10000,
        special_tokens=special_tokens,
        resume_pretoken_dir = '/home/wuziyi/code/cs336/assignment1-basics/data/token2count_debug/20250928_181205_158537'
    )
    logging.info("dump result into out_dir")
    with open(os.path.join(out_dir, 'vocab.pkl'), 'wb') as f :
        pickle.dump(vocab, f)
    with open(os.path.join(out_dir, 'merges.pkl'), 'wb') as f :
        pickle.dump(merges, f)

def visualize_report(vocab: dict[int, bytes], merges:  list[tuple[bytes, bytes]]):
    # top 10 longest vocab 
    top10_kvs = sorted(vocab.items(), key = lambda kvs: -len(kvs[1]))[:20]
    for (k, v) in top10_kvs:
        logging.info(f'id {k} : {v} with len {len(v)}')


def show_bpe_result(vocab_dump_path: str, merges_dump_path: str):
    with open(vocab_dump_path, 'rb') as fv, open(merges_dump_path, 'rb') as fm:
        vocab = pickle.load(fv)
        merges = pickle.load(fm)
        visualize_report(vocab, merges)


if __name__ == "__main__":
    # test_sorted_list()
    # test_train_bpe()
    # test_train_bpe_full()
    # train_bpe_tinystories()
    #
    show_bpe_result(
        vocab_dump_path="/home/wuziyi/code/cs336/assignment1-basics/data/vocab_result/20250928_184448_180765/vocab.pkl",
        merges_dump_path="/home/wuziyi/code/cs336/assignment1-basics/data/vocab_result/20250928_184448_180765/vocab.pkl"
    )




