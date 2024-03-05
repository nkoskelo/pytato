
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

import pdb # DEBUGGER

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

def get_rank(tags):
    loc = None
    for tag in tags:
        if isinstance(tag, MyTagType):
           loc = tag.get_location()
    return loc

class SliceyMapper(pt.transform.CopyMapperWithExtraArgs):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sends_global = dict()


    def get_comm_id(self, where_you_want_it: int, its_location: int):
        next_id = None # get scope right.
        if where_you_want_it in self.sends_global.keys():
            if its_location in self.sends_global[where_you_want_it].keys():
                next_id = len(self.sends_global[where_you_want_it][its_location])
            else:
                next_id = 0
            self.sends_global[where_you_want_it][its_location].append(tuple([next_id, where_you_want_it, its_location]))

        else:
            # Need to set up this portion of the global.
            self.sends_global[where_you_want_it] = dict()
            self.sends_global[where_you_want_it][its_location] = [tuple([next_id, where_you_want_it, its_location])]
        return tuple([next_id, where_you_want_it, its_location])

    def add_send_node_to_global(self, send_node, comm_id):
        id_number, dest_rank, src_rank = comm_id
        if dest_rank in self.sends_global.keys():
             if src_rank in self.sends_global[dest_rank].keys():
                 self.sends_global[dest_rank][src_rank][id_number] = tuple([id_number, dest_rank, src_rank, send_node])
                 return
        assert False # We should never hit this portion as the global should always already be populated.


    def __call__(self, expr, where_should_be):
        return self.rec(expr, where_should_be)

    def rec(self, expr, where_should_be):
        loc = get_rank(expr.tags)
        if loc != where_should_be and loc != None:
            # Then, we are going to need to split the system.

            comm_id = self.get_comm_id(where_should_be, loc)
            descend = super().rec(expr, where_should_be)
            send = pt.make_distributed_send(descend, where_should_be, comm_id)
            receive = pt.make_distributed_recv(loc, comm_id, descend.shape, descend.dtype)
            self.add_send_node_to_global(send, comm_id)
            return receive
        else:
            return super().rec(expr, where_should_be)

    """
    def map_index_lambda(self, expr: pt.IndexLambda, where_should_be):
        return super().map_index_lambda(expr, where_should_be)

    def map_basic_index(self, expr: pt.BasicIndex, where_should_be):
        return super().map_basic_index(expr, where_should_be)

    def map_reshape(self, expr, where_should_be):
        return super().map_reshape(expr, where_should_be)

    def map_data_wrapper(self, expr, where_should_be):
        return super().map_data_wrapper(expr, where_should_be)
    """

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
	

    def outer(a, b, loc_):
        t1 = a.reshape(-1,1).tagged([MyTagType(loc_)])
        t2 = b.reshape(1,-1).tagged([MyTagType(loc_)])
        return (t1 * t2).tagged([MyTagType(loc_)])

    C_blocks = np.zeros((A_blocks.shape[0], B_blocks.shape[1]), dtype=object)
    n_outer_product_blocks = B_blocks.shape[0]
    n_outer_products = B_blocks[0,0].shape[0]
    print("N prod: ", n_outer_products, "blocks: ", n_outer_product_blocks)
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

                    op = outer(tagged1, tagged2, processor_for_C)
                    C_blocks[ib, jb] = C_blocks[ib, jb] + op
                    C_blocks[ib, jb] = C_blocks[ib, jb].tagged([MyTagType(processor_for_C)])
                    print("C_blocks[ib,jb[: ",ib,jb,C_blocks[ib, jb])
                    print("Processors (A,B,C):", processor_for_A, processor_for_B, processor_for_C)
                    print("tagged1 in bindings: ", C_blocks[ib, jb].bindings)
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
    my_graph = pt.get_dot_graph(C_blocks[1,0])
    with open("out_file.out", "w+") as out_file:
        out_file.write(my_graph)

    print()
    print("GOING INTO THE SLICER")
    print()
    mySlicer = SliceyMapper()
    sliced = mySlicer.rec(C_blocks[0,0],1)
    print("MY SLICER HAS SLICED")
    print(sliced)
    """
    my_scheduler_send_nodes = [[] for i in range(9)]
    for row_block in range(C_blocks.shape[0]):
        for col_block in range(C_blocks.shape[1]):
            build_send_lists(C_blocks[row_block, col_block], my_scheduler_send_nodes)
            for i in range(len(my_scheduler_send_nodes)):
                print("processor: ",i," has ",len(my_scheduler_send_nodes[i])," send nodes")

    C_blocks_host = actx.to_numpy(actx.freeze(C_blocks))
    #print(C_blocks_host)
    """

def build_send_lists(array, proc_ids_to_comm_ids):


    stack = [array]
    cnt = 0
    while len(stack) > 0:
        next_array = stack.pop()
        loc = None
        for tag in next_array.tags:
            if isinstance(tag, MyTagType):
                loc = tag.get_location()
        if isinstance(next_array,pt.Reshape):
            # There is not a bindings attribute.
            descend = next_array.array
            for tag in descend.tags:
                if type(tag) == MyTagType:
                   your_loc = tag.get_location()
                if your_loc is None:
                   # is null then we are just going to add it onto the stack.
                   stack.append(descend)
                elif loc is None:
                   # Missing data about the combine step.
                   stack.append(descend)
                elif loc != your_loc:
                    # We are splitting this edge. This means that we need to make a Send/Receive pair.
                    comm_id = tuple([your_loc, loc, len(proc_ids_to_comm_ids[your_loc]) + 1])
                    send = pt.make_distributed_send(descend, loc, comm_id)
                    receive = pt.make_distributed_recv(your_loc, comm_id, descend.shape, descend.dtype)
                    proc_ids_to_comm_ids[your_loc].append(send)
                    #next_array.array = receive
                else:
                    stack.append(descend)
            

        if  hasattr(next_array,"bindings"):
            for key in next_array.bindings.keys():
                your_loc = None
                descend = next_array.bindings[key]
                if isinstance(descend, pt.IndexLambda) or isinstance(descend, pt.BasicIndex) or isinstance(descend, pt.Reshape):
                    for tag in descend.tags:
                        if type(tag) == MyTagType:
                             your_loc = tag.get_location()
                #breakpoint()
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
                    receive = pt.make_distributed_recv(your_loc, comm_id, next_array.bindings[key].shape, next_array.bindings[key].dtype)
                    proc_ids_to_comm_ids[your_loc].append(send)
                    next_array.bindings[key] = receive
                else:
                    stack.append(next_array.bindings[key])
        cnt += 1
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
