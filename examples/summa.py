
from mpi4py import MPI
comm = MPI.COMM_WORLD

import pytato as pt
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

from pytato import (find_distributed_partition, generate_code_for_partition,
                    number_distributed_tags,
                    execute_distributed_partition,
                    staple_distributed_send, make_distributed_recv)

from arraycontext import PytatoPyOpenCLArrayContext
from pyopencl.tools import ImmediateAllocator

class MyArrayContext(PytatoPyOpenCLArrayContext):
    def transform_loopy_program(self, t_unit):
        #print(t_unit)
        return t_unit

# Make a new tag type
# Slap onto each piece of the pytato arrays
# Mapper global to local.
# Sends and receives tag name (Triple) (sender, receiver, "comm_tag")
#   -> For each such triple, there must be exactly one send and one receive

class Mapper:
    def __init__(self):
        # staple these to a result on the right rank at the end
        self.rank_to_sends: Dict[int, List[DistributedSend]] = {}

def main():

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = MyArrayContext(queue, allocator=ImmediateAllocator(queue))
    

    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng()

    # The data
    bs_A = (4, 3)
    bs_B = (3, 2)
    assert bs_A[1] == bs_B[0]
    A_in = rng.integers(100, size=(bs_A[0]*3, bs_A[1]*3))
    B_in = rng.integers(100, size=(bs_B[0]*3, bs_B[0]*4))
    assert A_in.shape[1] == B_in.shape[0]

    def slice_into_blocks(mat, blocksize):
        blocksize_r, blocksize_c = blocksize
        nblocks_r, rem_r = divmod(mat.shape[0], blocksize_r)
        nblocks_c, rem_c = divmod(mat.shape[1], blocksize_c)
        if rem_r or rem_c:
            raise ValueError("matrix size not divisible by block size")
        mat_blocks = np.zeros((nblocks_r, nblocks_c), dtype=object)
        for ib in range(nblocks_r):
            for jb in range(nblocks_c):
                mb = mat[
                    blocksize_r*ib:blocksize_r*(ib+1), 
                    blocksize_c*jb:blocksize_c*(jb+1),
                    ]
                assert mb.shape == blocksize
                mat_blocks[ib, jb] = mb
        return mat_blocks

    A_blocks = slice_into_blocks(actx.from_numpy(A_in), blocksize=bs_A)
    B_blocks = slice_into_blocks(actx.from_numpy(B_in), blocksize=bs_B)
	
    #A_in = np.array([[[rng.integers(100, size=(2,1)) for i in range(2)] for j in range(2)] for k in range(2)])
    #B_in = np.array([[[rng.integers(100, size=(2,1)) for i in range(2)] for j in range(2)] for k in range(2)])
    #c_out = np.array([[rng.integers(100, size=(2,2)) for i in range(2)] for j in range(2)])

    #print(type(c_out[0,0]))
    #from pytools.obj_array import make_obj_array
    #A_in_dev = make_obj_array([[[cl_array.to_device(queue, A_in[i][j][k]) for i in range(2)] for j in range(2)] for k in range(2)])
    #B_in_dev = np.array([[[cl_array.to_device(queue, B_in[i][j][k]) for i in range(2)] for j in range(2)] for k in range(2)])
    #c_out_dev = np.array([[cl_array.to_device(queue, c_out[i][j]) for i in range(2)] for j in range(2)])
    #pu.db

    #A_data_wrappers = np.array([[[pt.make_data_wrapper(A_in_dev[i][j][k]) for i in range(2)] for j in range(2)] for k in range(2)])
    #B_data_wrappers = np.array([[[pt.make_data_wrapper(B_in_dev[i][j][k]) for i in range(2)] for j in range(2)] for k in range(2)])

    #data_wrappers = np.array([[pt.make_data_wrapper(c_out_dev[i][j]) for i in range(2)] for j in range(2)])
    #print(type(data_wrappers[0,0]))

    def outer(a, b):
        return a.reshape(-1, 1) * b.reshape(1, -1)

    C_blocks = np.zeros((A_blocks.shape[0], B_blocks.shape[1]), dtype=object)
    n_outer_product_blocks = B_blocks.shape[0]
    n_outer_products = B_blocks[0,0].shape[0]
    assert n_outer_product_blocks * n_outer_products == B_blocks.shape[0] * bs_B[0]
    for ib in range(C_blocks.shape[0]):
        for jb in range(C_blocks.shape[1]):
            iproc = ...
            for k_op_block in range(n_outer_product_blocks):
                for k_op in range(n_outer_products):
                    op = outer(
                        A_blocks[ib, k_op_block][:, k_op],
                        B_blocks[k_op_block, jb][k_op])
                    C_blocks[ib, jb] = C_blocks[ib, jb] + op

    C_blocks_host = actx.to_numpy(actx.freeze(C_blocks))

    
    #for i in range(4):
    #   for k in range(2):
    #       processor_row = i // (2)
    #       processor_col = i % 2
    #       a = A_data_wrappers[processor_row,processor_col,k] # This is a 2x1 column array
    #       b = B_data_wrappers[k, processor_col, processor_row] # This is a 2x1 row array

    #       data_wrappers[processor_row, processor_col] += pt.dot(a, b.T) # outer product. 


    #"""
    #     for j in range(processor_rows):
    #         if j == p_row:
    #            break


    #          target = p_row*3+ p_col+j
    #          mytag_a = (main, "a", k, target) # loc, type, number, target

    #          target = (p_row+1)*3 + p_col
    #          mytag_b = (main, "b", k, target)




    #          # Send your row data to all the processors in your
    #          # current processors' column
    #          stapled_distributed_send(a, dest_rank=(p_row*3 + p_col + j),
    #                   comm_tag=mytag_a,
    #                   stapled_to=make_distributed_recv(src_rank=(p_row*3 + p_col + j),
    #                        comm_tag=mytag_a, shape=(3,3), dtype=int))


    #           # Send all your column data to the processors in the
    #           # row of the current processor

    #           stapled_distributed_send(b, dest_rank=((p_row+1)*3+ p_col +j),comm_tag=mytag_b,
    #            stapled_to=make_distributed_recv(src_rank=((p_row+1)*3+ p_col +j),
    #                comm_tag=mytag_b, shape=(3,3), dtype=int))

    #            # Compute your update value.
    #            c = c + pt.dot(a, b.T)
    # """
    # outs = pt.make_dict_of_named_array({"out: "+str(i): data_wrappers[i // 2][i % 2] for i in range(4)})
    # distributed_parts = find_distributed_partition(comm, outs)
    # print(distributed_parts)

    # #prg_per_partition = generate_code_for_partition(c)

    # pt.verify_distributed_partition(comm, c)

    # print("Rank: ",rank,"out: ", context["out" + str(rank)].get(queue))

if __name__ == "__main__":
    main()
