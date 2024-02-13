
from mpi4py import MPI
comm = MPI.COMM_WORLD

import pytato as pt
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

from pytato import (find_distributed_partition, generate_code_for_partition,
                    number_distributed_tags,
                    execute_distributed_partition,
                    staple_distributed_send, make_distributed_recv, transform)

from arraycontext import PytatoPyOpenCLArrayContext
from pyopencl.tools import ImmediateAllocator
import arraycontext
import pytools
from typing import Union

class MyArrayContext(PytatoPyOpenCLArrayContext):

    def transform_loopy_program(self, t_unit):
        #print(t_unit)
        return t_unit


# Make a new tag type
# Slap onto each piece of the pytato arrays
# Mapper global to local.
# Sends and receives tag name (Triple) (sender, receiver, "comm_tag")
#   -> For each such triple, there must be exactly one send and one receive

class MyTagType(pytools.tag.Tag):
    def __init__(self, array_location):
        self.array_location = array_location

    def get_location(self):
        return self.array_location

    def __repr__(self):
        return "P" + str(self.array_location)


class Mapper:
    def __init__(self, dag_tagged_arrays):
        # staple these to a result on the right rank at the end
        self.rank_to_sends: Dict[int, List[DistributedSend]] = {}
        self.map = dag_tagged_arrays


    def descendants(self, node):
        return self.map[node]

def main():

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = MyArrayContext(queue, allocator=ImmediateAllocator(queue))
    
    # We choose the grid of processors. 
    # We state that there will be one processor per major C block.
    # However, then we need to divide up the major blocks of A and B.

    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng()

    # The data
    bs_A = (4, 3)
    bs_B = (3, 2)
    assert bs_A[1] == bs_B[0]
    A_in = rng.integers(100, size=(bs_A[0]*3, bs_A[1]*3))
    B_in = rng.integers(100, size=(bs_B[0]*3, bs_B[1]*4))
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
	

    def outer(a, b):
        return a.reshape(-1, 1) * b.reshape(1, -1)

    C_blocks = np.zeros((A_blocks.shape[0], B_blocks.shape[1]), dtype=object)
    n_outer_product_blocks = B_blocks.shape[0]
    n_outer_products = B_blocks[0,0].shape[0]
    assert n_outer_product_blocks * n_outer_products == B_blocks.shape[0] * bs_B[0]

    print("shape: ",C_blocks.shape[0:2])
    n_processors = C_blocks.shape[0] * C_blocks.shape[1]
    n_blocks_A = A_blocks.shape[0] * A_blocks.shape[1]
    print("A shape: ",n_blocks_A)
    for ib in range(C_blocks.shape[0]):
        for jb in range(C_blocks.shape[1]):
            processor_for_C = ib*C_blocks.shape[1] + jb
            print("")
            print("processor: ", processor_for_C)
            print("")
            iproc = ...
            for k_op_block in range(n_outer_product_blocks):
                for k_op in range(n_outer_products):
                    processor_for_A = (k_op_block * n_outer_products + k_op) % n_processors
                    processor_for_B = (k_op_block * n_outer_products + k_op) % n_processors
                    input1 = A_blocks[ib, k_op_block][:, k_op]
                    input2 = B_blocks[k_op_block, jb][k_op]
                    tagged1 = input1.tagged([MyTagType(processor_for_A)])
                    tagged2 = input2.tagged([MyTagType(processor_for_B)])

                    op = outer(
                        A_blocks[ib, k_op_block][:, k_op],
                        B_blocks[k_op_block, jb][k_op])
                    C_blocks[ib, jb] = C_blocks[ib, jb] + op
                    C_blocks[ib, jb] = C_blocks[ib, jb].tagged([MyTagType(processor_for_C)])
                    print("C_blocks[ib,jb[: ",ib,jb,C_blocks[ib, jb])

    myMapper = transform.Mapper()
    for ib in range(C_blocks.shape[0]):
       for jb in range(C_blocks.shape[1]):
           print("")
           print(len(C_blocks[ib,jb][0]))
           print((pt.to_index_lambda(C_blocks[ib,jb][0][0]).bindings))
           print(type(C_blocks[ib,jb]))
           print(C_blocks[ib,jb].tags)
           print("")


    print((C_blocks[0,0]))


    my_scheduler_send_nodes = [[] for i in range(9)]
    build_send_lists(C_blocks[0,0],my_scheduler_send_nodes)


    print(my_scheduler_send_nodes)

    C_blocks_host = actx.to_numpy(actx.freeze(C_blocks))
    #print(C_blocks_host)


def build_send_lists(array, proc_ids_to_comm_ids):


    stack = [array]
    cnt = 0
    while len(stack) > 0:
        next_array = stack.pop()
        loc = None
        for tag in next_array.tags:
            if isinstance(tag, MyTagType):
                loc = tag.get_location()

        if len(array.bindings) > 0:
            print("\n",array.tags,"\n")
            
            print("the first array should be located at: ",loc)
            for key in next_array.bindings.keys():
                your_loc = None
                descend = next_array.bindings[key]
                print(descend, "and tags: ", descend.tags)
                for tag in descend.tags:
                    print(MyTagType)
                    print(type(tag))
                    if type(tag) == MyTagType:
                         your_loc = tag.get_location()
                    else:
                         print("Stated as no match. However, we have: ", MyTagType, " and the instant: ", type(tag))
                    #if isinstance(tag, MyTagType):
                    #     your_loc = tag.get_location()
                #your_loc = next_array.bindings[key].tags.array_location
                print("first depend is located at: ", your_loc)
                if your_loc is None:
                   # is null then we are just going to add it onto the stack.
                   stack.append(descend)
                elif loc is None:
                   # Missing data about the combine step.
                   stack.append(descend)
                elif loc != your_loc:
                    # We are splitting this edge. This means that we need to make a Send/Receive pair.
                    comm_id = tuple([your_loc, loc, len(proc_ids_to_comm_ids[your_loc]) + 1])
                    send = pt.make_distributed_send(next_array.bindings[key], loc, comm_id)
                    receive = pt.make_distributed_recv(your_loc, comm_id, next_array.bindings[key].shape)
                    proc_ids_to_comm_ids[your_loc].append(send)
                else:
                    print(type(next_array.bindings[key]))
                    if isinstance(next_array.bindings[key], pt.IndexLambda):
                        stack.append(next_array.bindings[key])
        cnt += 1
        print("cnt: ", cnt)
            # else we are good to keep going on this path.
    return

def divide(array, my_dag, proc_of_interest: int):

    """ 

    # If array has dependencies we need to check the children.
    if array has descendants:
        left = None
        if array.bindings[0].tags == proc_of_interest:
            left = divide(array.bindings[0], my_dag, proc_of_interest)
        else:
            left = RecieveNodeFromProcessorLEFT
        
        right = None
        if array.bindings[1].tags == proc_of_interest:
            right = divide(array.bindings[1], my_dag, proc_of_interest)
        else:
            right = RecieveNodeFromProcessorRIGHT
        array.bindings[0] = left
        array.bindings[1] = right
        return array
    # else
    else:
        if array.tags:
            pass
        return array


    #if isinstance(array, BasicIndex):
        # This is just an array so we can just add it to the dag
     #   pass

    #elif isinstance(array, IndexLambda):
        # Looks like we are able to make more deep down. 
     #   pass

    assert False
    """

if __name__ == "__main__":
    main()
