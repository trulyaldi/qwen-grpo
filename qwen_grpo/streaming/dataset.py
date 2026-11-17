"""Disk-backed prompt iterator with stable sample IDs and a persistable, deterministic sampler state (Ticket 09).

PromptStream.build writes the training prompts once to <dir>/prompts.jsonl (one JSON record per line: id, text,
label; ids are the stable sample ids of training.balanced_subset) and <dir>/prompts.meta.json (its sha256 and the
byte offset of every line). Only the offsets stay in memory; records are read from disk by offset when a batch is
drawn. A later build on the same directory must find the same file (dataset identity): a different sha256 is refused.

The sampler reproduces training.shuffled_batches exactly: one random.Random(seed) stream; each epoch shuffles the
record order with it (rng.shuffle of the index list, the same permutation as shuffling the records), and batches of B
are cut from the concatenated epochs, so a batch may straddle two epochs. Its state is small and JSON-serializable:
  dataset_sha256, seed, batch_size, epoch, position (next index into the epoch's order), batches (drawn so far),
  epoch_rng (the random.Random state at the start of the current epoch: the order is recomputed from it),
  order_sha256 (of the current epoch's order: checked when the state is restored).
next_batch() returns (records, occurrences): occurrences = [(epoch, position, sample_id)] of the batch's rows.
"""
import hashlib
import json
import os
import random
from pathlib import Path


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def atomic_write_text(path, text):
    """Write `text` to `path` crash-consistently: temp file in the same directory, fsync, rename, fsync the directory."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class PromptStream:
    def __init__(self, directory, seed, batch_size):
        self.dir = Path(directory)
        meta = json.loads((self.dir / "prompts.meta.json").read_text())
        self.sha256, self.offsets = meta["sha256"], meta["offsets"]
        if _sha((self.dir / "prompts.jsonl").read_bytes()) != self.sha256:
            raise RuntimeError(f"{self.dir / 'prompts.jsonl'} does not match its recorded sha256 (dataset changed)")
        self.seed, self.B = seed, batch_size
        self.rng = random.Random(seed)
        self.epoch, self.position, self.batches = -1, 0, 0
        self.order, self.epoch_rng = [], None
        self._new_epoch()

    @classmethod
    def build(cls, directory, examples, seed, batch_size):
        """Write the prompts (first time) or check that the directory holds exactly these prompts; return the stream."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"id": e["id"], "text": e["text"], "label": e["label"]}, sort_keys=True) + "\n"
                 for e in examples]
        data = "".join(lines).encode()
        if len({e["id"] for e in examples}) != len(examples):
            raise ValueError("sample ids must be unique")
        path = d / "prompts.jsonl"
        if path.exists():
            if _sha(path.read_bytes()) != _sha(data):
                raise RuntimeError(f"{path} holds a different dataset than this run's training prompts: refusing to mix")
        else:
            offsets, pos = [], 0
            for line in lines:
                offsets.append(pos)
                pos += len(line.encode())
            with open(path.with_name(".prompts.jsonl.tmp"), "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            atomic_write_text(d / "prompts.meta.json", json.dumps({"sha256": _sha(data), "offsets": offsets,
                                                                  "n": len(lines)}))
            os.replace(path.with_name(".prompts.jsonl.tmp"), path)
            fsync_dir(d)
        return cls(d, seed, batch_size)

    def __len__(self):
        return len(self.offsets)

    def record(self, i):
        """Record i (the i-th line), read from disk by offset."""
        with open(self.dir / "prompts.jsonl", "rb") as f:
            f.seek(self.offsets[i])
            return json.loads(f.readline())

    def _new_epoch(self):
        self.epoch += 1
        self.position = 0
        self.epoch_rng = self.rng.getstate()
        self.order = list(range(len(self.offsets)))
        self.rng.shuffle(self.order)

    def next_batch(self):
        records, occurrences = [], []
        for _ in range(self.B):
            if self.position == len(self.order):
                self._new_epoch()
            i = self.order[self.position]
            r = self.record(i)
            records.append(r)
            occurrences.append((self.epoch, self.position, r["id"]))
            self.position += 1
        self.batches += 1
        return records, occurrences

    def state(self):
        return {"dataset_sha256": self.sha256, "seed": self.seed, "batch_size": self.B, "epoch": self.epoch,
                "position": self.position, "batches": self.batches, "epoch_rng": _jsonable(self.epoch_rng),
                "order_sha256": _sha(json.dumps(self.order).encode())}

    def restore(self, s):
        if s["dataset_sha256"] != self.sha256 or s["seed"] != self.seed or s["batch_size"] != self.B:
            raise RuntimeError(f"sampler state is for another dataset / seed / batch size: {s['dataset_sha256'][:12]}, "
                               f"{s['seed']}, {s['batch_size']} vs {self.sha256[:12]}, {self.seed}, {self.B}")
        self.rng.setstate(_state(s["epoch_rng"]))
        self.epoch = s["epoch"] - 1
        self._new_epoch()
        if _sha(json.dumps(self.order).encode()) != s["order_sha256"]:
            raise RuntimeError("the restored sampler does not reproduce the saved epoch order")
        self.position, self.batches = s["position"], s["batches"]


def _jsonable(state):
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _state(j):
    version, internal, gauss = j
    return (version, tuple(internal), gauss)
