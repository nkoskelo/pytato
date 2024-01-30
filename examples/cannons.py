#!/usr/bin/env python

from mpi4py import MPI  # pylint: disable=import-error
comm = MPI.COMM_WORLD

import pytato as pt
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

from pytato import (find_distributed_partition, generate_code_for_partition,
        number_distributed_tags,
        execute_distributed_partition,
        staple_distributed_send, make_distributed_recv)


def main():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng()

    #x_in = rng.integers(100, size=(3, 3))
    #b_in = rng.integers(100, size=(3, 3))
    x_in = np.arange(9).reshape(3,3) + rank
    b_in = np.arange(9).reshape(3,3) + rank
    c_out= rng.integers(100, size=(3, 3, 3, 3)) * 0
    c_out0 = rng.integers(100,size=(3,3)) * 0
    c_outs = [rng.integers(100,size=(3,3))*0 for i in range(size)]


    x_in_dev = cl_array.to_device(queue, x_in)
    b_in_dev = cl_array.to_device(queue, b_in)
    c_outs_dev = [cl_array.to_device(queue, c_outs[i]) for i in range(size)]
    x = pt.make_data_wrapper(x_in_dev)
    B = pt.make_data_wrapper(b_in_dev)
    data_wrappers = [pt.make_data_wrapper(c_outs_dev[i]) for i in range(size)]

    if size < 2:
        raise RuntimeError("it doesn't make sense to run the "
                           "distributed-memory test single-rank"
                           # and self-sends aren't supported for now
                           )

    # My goal is to compute the matrix matrix multiplication 
    # using Cannon's algorithm
    N = len(x)
    i = rank // 3
    j = rank % 3

    k = (i + j) % N

    a = x # assume the block is passed to us.
    b = B # assume we have the block passed to us.
    #c[i][j] = 0

    for l in range(0, N):
        data_wrappers[rank] = data_wrappers[rank] + a * b
        mytag_a = (main, "a")
        mytag_b = (main, "b")

        staple_distributed_send(a, dest_rank=(i*3 + (j + N -1) % N) % size, comm_tag=mytag_a,
            stapled_to=make_distributed_recv(src_rank=(i*3 + (j + 1) % N) % size, comm_tag=mytag_a, shape=(3,3), dtype=int))

        staple_distributed_send(b, dest_rank=(((i + N - 1) % N) * N + j) % size, comm_tag=mytag_b,
            stapled_to=make_distributed_recv(src_rank=(((i + 1)%N)*N + j), comm_tag=mytag_b, shape=(3,3), dtype=int))

        # These stabled receives should allow a, and b
        # to be updated


    # Find the partition
    outputs = pt.make_dict_of_named_arrays({"out" + str(i): data_wrappers[i] for i in range(size)})
    distributed_parts = find_distributed_partition(comm, outputs)

    distributed_parts, _ = number_distributed_tags(
            comm, distributed_parts, base_tag=42)
    prg_per_partition = generate_code_for_partition(distributed_parts)

    if 0:
        from pytato.visualization import show_dot_graph
        if rank==0:
            show_dot_graph(distributed_parts)

    if 1:
        # Sanity check
        from pytato.visualization import get_dot_graph_from_partition
        if rank==0:
            with open("out_file.out", "w+") as out_file:
                out_file.write(get_dot_graph_from_partition(distributed_parts))

    pt.verify_distributed_partition(comm, distributed_parts)

    context = execute_distributed_partition(distributed_parts, prg_per_partition,
                                             queue, comm)

    #final_res = context["out"].get(queue)



    print("Rank: ", rank,"x:",x, "b: ",b, " Out: ", context["out"+str(rank)].get(queue))

if __name__ == "__main__":
    main()
