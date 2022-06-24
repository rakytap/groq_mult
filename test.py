import groq.api as g
import groq.api.nn as nn
import numpy as np
import bitchunks as bch
import time
print("Python packages imported successfully")

######################################################################################x
# Create and compile program for Groq chip

dim = 70


matrix1 = g.input_tensor(shape=(bch.num_of_chunks, dim), dtype=g.int8, name="matrix1", layout="H1(W), -1, S2(22-23)")
matrix20 = g.input_tensor(shape=(bch.num_of_chunks*dim, dim), dtype=g.int8, name="matrix20", layout="H1(W), -1, S16")
#matrix21 = g.input_tensor(shape=(bch.num_of_chunks*dim, dim), dtype=g.int8, name="matrix21", layout="H1(E), -1, S16")

class TopLevel(g.Component):  # Create our top level component
    def __init__(self):
        super().__init__()
        self.mm = nn.MatMul(name="MyMatMul", use_vxm_accum=True, planes=[0,1])     #Matmul: using the nn.MatMul() component.
        #self.mm2 = nn.MatMul(name="MyMatMul2", use_vxm_accum=True, planes=[2,3])     #Matmul: using the nn.MatMul() component.


    def build(self, mat1_mt, mat20_mt, time=0):   #Provide input matrices and a default time
    #def build(self, mat1_mt, mat20_mt, mat21_mt, time=0):   #Provide input matrices and a default time

        with g.ResourceScope(name="cpscope", is_buffered=True, time=0) as cpscope :   
            in1_st = mat1_mt.read(streams=g.SG2, time=0)
            in1_copy_mt = in1_st.write(name="write_copy", layout="H1(E), -1, S2")    #Assign a layout preferable to the MXM for the second matrix

        with g.ResourceScope(name="mmscope", is_buffered=True, time=None, predecessors=[cpscope]) as mmscope :   
            #result_mt = self.mm(mat1_mt, mat2_mt, time=0).write(name="mm_result", layout="H1(W), -1, S4")  #recommended layout for the matmul result

            #print( mat20_mt.physical_shape)

            result_st = self.mm(mat1_mt, mat20_mt, time=0)  #int8 multiplication results on SG4_E[4] when plane 0 is used
            #result_st2 = self.mm2(in1_copy_mt, mat21_mt, time=0)  #int8 multiplication results on SG4_E[4] when plane 0 is used


            print(result_st.shape)
            print(result_st.physical_shape)

            #print(result_st2.shape)
            #print(result_st2.physical_shape)

            splitted_st = g.split_inner_splits(result_st)
            print(len(splitted_st))

            '''
            #print(g.split.__doc__)
            separeted_bits_st = g.split(result_st,1, 0)
            lower_bits_st  = separeted_bits_st[0]
            higher_bits_st = separeted_bits_st[1]


            lower_bits_mt  = lower_bits_st.write(name="lower_bits", layout="H1(W), -1, S4(0-3)")
            higher_bits_mt = higher_bits_st.write(name="higher_bits", layout="H1(W), -1, S4(4-7)")
            '''
            result_mt = result_st.write(name="result", layout="H1(E), -1, S4")
            #result_mt2 = result_st2.write(name="result2", layout="H1(W), -1, S4") 

        #with g.ResourceScope(name="mmscope2", is_buffered=True, time=None, predecessors=[mmscope]) as mmscope2 :   

        '''
        with g.ResourceScope(name="bsscope", is_buffered=True, time=None, predecessors=[mmscope]) as bsscope :   

            lower_bits_st  = lower_bits_mt.read(streams=g.SG4_E[0])
            higher_bits_st = higher_bits_mt.read(streams=g.SG4_E[4])
            
            
            shape = (1,dim)
            shape2 = (1,2*dim)
            dtype = g.int8
            bitshift_higher_bits_st = g.constant_tensor(shape2, dtype, name="bitshift_tensor")
            bitshift_higher_bits_st.data = np.concatenate( (np.ones(shape, dtype=dtype.to_nptype() ) * bch.bitchunk, np.ones(shape, dtype=dtype.to_nptype() ) * 2*bch.bitchunk), axis=1 )
            
            bitshift_lower_bits_st = g.constant_tensor(shape2, dtype, name="bitshift_tensor")
            bitshift_lower_bits_st.data = np.concatenate( (np.zeros(shape, dtype=dtype.to_nptype() ), np.ones(shape, dtype=dtype.to_nptype() ) * bch.bitchunk), axis=1 )
            

           
            higher_bits_st = g.left_shift(higher_bits_st, bitshift_higher_bits_st, output_streams=g.SG4_E[2], alus=[4])
            lower_bits_st  = g.left_shift(lower_bits_st, bitshift_lower_bits_st, output_streams=g.SG4_E[0], alus=[0])

            result_st2 = g.add(higher_bits_st, lower_bits_st, output_streams=g.SG4_E[0], alus=[1])

            #lower_bits_mt  = lower_bits_st.write(name="lower_bits", layout="H1(E), -1, S4")
            #higher_bits_mt = higher_bits_st.write(name="higher_bits", layout="H1(E), -1, S4")
            result_mt = result_st2.write(name="result", layout="H1(E), -1, S4")
        ''' 


        return result_mt
        #return result_mt, result_mt2
        #return lower_bits_mt, higher_bits_mt


top = TopLevel()    # instantiate the top level component
result = top(matrix1, matrix20, time=0)    # call into the instance of the top level, providing your inputs and time
#result = top(matrix1, matrix20, matrix21, time=0)    # call into the instance of the top level, providing your inputs and time

iop_file = g.compile(base_name="matmul", result_tensor=result)
g.write_visualizer_data("matmul")
######################################################################################



# calculate the expected result with numpy
print('Calculating expected numpy result')

# create 16 bit input data
t1_data = np.random.randint(-bch.bound, bch.bound, (1, dim), dtype=np.int64)
t2_data = np.random.randint(-bch.bound, bch.bound, (dim, dim), dtype=np.int64)

# create 8 bit chunks
t1_data_8 = bch.divide_into_bitchunks(t1_data, bch.num_of_chunks)
t1_data_64 = 0
for idx in range(len(t1_data_8)):
    t1_data_64 = t1_data_64 + np.left_shift(t1_data_8[idx,:,:].astype(np.int64), bch.bitchunk*idx)

print('test t1_data: '+str(np.allclose(t1_data, t1_data_64, rtol=1e-1, atol=1e-1, equal_nan=True)))



t2_data_8 = bch.divide_into_bitchunks(t2_data, bch.num_of_chunks)
t2_data_64 = 0
for idx in range(len(t2_data_8)):
    t2_data_64 = t2_data_64 + np.left_shift(t2_data_8[idx,:,:].astype(np.int64), bch.bitchunk*idx)

print('test t2_data: '+str(np.allclose(t2_data, t2_data_64, rtol=1e-1, atol=1e-1, equal_nan=True)))

t0 = time.time()
# matmul with 32 bit arithmetics
mult_result = np.matmul(t1_data.astype(np.int64), t2_data.transpose().astype(np.int64))
print("numpy time: " + str( time.time()-t0) )

# multiplication with 8bit arithmetics
mult_result_8 = np.matmul( t1_data_8.reshape((bch.num_of_chunks,dim)).astype(np.int64), t2_data_8.reshape((bch.num_of_chunks*dim,dim)).astype(np.int64).transpose() )

mult_result_64 = 0
for idx in range(bch.num_of_chunks):
    for jdx in range(bch.num_of_chunks):
        mult_result_64 += np.left_shift(mult_result_8[idx,jdx*dim:(jdx+1)*dim].astype(np.int64), bch.bitchunk*(idx+jdx))

print('test matmul with numpy: '+str(np.allclose(mult_result, mult_result_64, rtol=1e-1, atol=1e-1, equal_nan=True)))
#print(mult_result)


##########################################################################################################
# Run the compiled code on the Groq chip
print('Running the code on the Groq chip')


program = g.create_tsp_runner(iop_file)
t0 = time.time()
result = program(matrix1=t1_data_8.reshape((bch.num_of_chunks,dim)), matrix20=t2_data_8.reshape((bch.num_of_chunks*dim,dim)))
groq_result = result['result']


# combining lower and upper 8 bits of the calculated products to get the result of corresponding to the 16bit inputs
print( groq_result.shape )
groq_result_64 = 0
for idx in range(bch.num_of_chunks):
    for jdx in range(bch.num_of_chunks):
        groq_result_64 += np.left_shift(groq_result[idx,jdx*dim:(jdx+1)*dim].astype(np.int64), bch.bitchunk*(idx+jdx))


print("Groq time: " + str( time.time()-t0) )

###########################################################################################################
# Check Result


print("Matrix Multiplication for input tensors of size {} x {}.  Results are: ".format(t1_data.shape, t2_data.shape))
print('Groq chip matmul: '+str(np.allclose(mult_result, groq_result_64, rtol=1e-1, atol=1e-1, equal_nan=True)))
print(t1_data)
#print(t1_data)
