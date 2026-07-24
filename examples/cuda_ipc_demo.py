"""Minimal CUDA-IPC inter-process tensor demo (same GPU, two processes).

Run:  python cuda_ipc_demo.py
Requires: a CUDA GPU and PyTorch.

What it shows:
  - A CUDA tensor created in the *producer* process is sent over a Pipe.
    torch.multiprocessing serializes it as a CUDA IPC *handle* (a few dozen
    bytes), NOT as its data bytes.
  - The *consumer* reconstructs a tensor from that handle that points at the
    SAME physical GPU memory (zero-copy). Each process maps it into its own
    virtual address space, so data_ptr() differs even though the memory is
    shared.
  - The consumer's in-place write is visible to the producer afterwards,
    proving the memory is shared.
"""
import torch
import torch.multiprocessing as mp


def producer(conn):
    t = torch.arange(10, device="cuda")          # lives in the producer process
    print(f"[producer] created tensor: {t.tolist()}  data_ptr={t.data_ptr()}")
    conn.send(t)                                  # serialized as a CUDA IPC handle, NOT bytes
    conn.recv()                                   # block until consumer is done
    # Same device memory is shared -> producer observes the consumer's writes.
    print(f"[producer] after consumer wrote: {t.tolist()}  data_ptr={t.data_ptr()}")


def consumer(conn):
    t = conn.recv()                               # reconstructed from the IPC handle (zero-copy)
    print(f"[consumer] received:     {t.tolist()}  data_ptr={t.data_ptr()}")
    t.add_(100)                                   # in-place write into the shared memory
    print(f"[consumer] wrote +100:   {t.tolist()}")
    conn.send("done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parent, child = mp.Pipe()
    p = mp.Process(target=producer, args=(parent,))
    c = mp.Process(target=consumer, args=(child,))
    p.start()
    c.start()
    p.join()
    c.join()
