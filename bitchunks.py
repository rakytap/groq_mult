import numpy as np


bitchunk = 7
num_of_chunks = 2
full_bitwidth = num_of_chunks*bitchunk
bound = pow(2, full_bitwidth)
#bound = 4

# function to split higher bitwidth array into int8 arrays bit-slices
def divide_into_bitchunks(inp, num_of_chunks):

    # determine the dimensions of the return
    shape = tuple( [num_of_chunks] + list(inp.shape))

    sign_bits = np.sign(inp)
    sign_bits = (np.abs(sign_bits) - sign_bits)/2
    sign_bits = sign_bits.astype(np.uint8)

    #print(inp)
    inp_loc = inp.copy().astype(np.uint32)
    #print(inp_loc)



    # reserve space for the return
    ret = np.zeros(shape, dtype=np.int8)
    
    
    for chunk in range(num_of_chunks):

        # extract decimal bits
        for bit_idx in range(bitchunk):
            bit = (inp_loc % 2)
            ret[chunk] += pow(2,bit_idx)*bit
            inp_loc = np.right_shift(inp_loc,1)

        
        # adding sign bit
        if( chunk == num_of_chunks-1) :
            ret[chunk] += pow(2,bitchunk)*sign_bits
                

    return ret



