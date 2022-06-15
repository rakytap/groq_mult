import groq.api as g
import groq.api.nn as nn
import numpy as np
print("Python packages imported successfully")

######################################################################################x
# Create and compile program for Groq chip

matrix1 = g.input_tensor(shape=(2, 50), dtype=g.int8, name="matrix1", layout="H1(W), -1, S2")
matrix2 = g.input_tensor(shape=(100, 50), dtype=g.int8, name="matrix2", layout="H1(W), -1, S16(4-38)")

class TopLevel(g.Component):  # Create our top level component
    def __init__(self):
        super().__init__()
        self.mm = nn.MatMul(name="MyMatMul")     #Matmul: using the nn.MatMul() component.

    def build(self, mat1_mt, mat2_mt, time=0):   #Provide input matrices and a default time
        with g.ResourceScope(name="mmscope", is_buffered=True, time=0) as mmscope :   
            #result_mt = self.mm(mat1_mt, mat2_mt, time=0).write(name="mm_result", layout="H1(W), -1, S4")  #recommended layout for the matmul result

            result_st = self.mm(mat1_mt, mat2_mt, time=0)  #recommended layout for the matmul result
            result_st = g.reinterpret(result_st, g.int8)
            result_mt = result_st.write(name="mm_result", layout="H1(W), -1, S4")

        return result_mt


top = TopLevel()    # instantiate the top level component
result = top(matrix1, matrix2, time=0)    # call into the instance of the top level, providing your inputs and time

iop_file = g.compile(base_name="matmul", result_tensor=result)
#g.write_visualizer_data("matmul")
######################################################################################x



# calculate the expected result with numpy


# prepare input data
t1_data_0 = np.random.randint(-2, 2, (1, 50), dtype=np.int8).astype(np.int8) # the lower 8 bits of 16 bits integers
t1_data_1 = np.random.randint(-2, 2, (1, 50), dtype=np.int8).astype(np.int8) # the upper 8 bits of 16 bits integers
t1_data = np.concatenate( (t1_data_0, t1_data_1), axis=0)                    # store the upper and lower bits in a single matrix
t1_data_16 = t1_data_0.astype(np.int16) + np.left_shift(t1_data_1.astype(np.int16), 7) # the combined 16bit integers
print(t1_data.shape)



t2_data_0 = np.random.randint(-2, 2, (50, 50), dtype=np.int8).astype(np.int8) # the lower 8 bits of 16 bits integers
t2_data_1 = np.random.randint(-2, 2, (50, 50), dtype=np.int8).astype(np.int8) # the upper 8 bits of 16 bits integers
t2_data = np.concatenate( (t2_data_0, t2_data_1), axis=0)                     # store the upper and lower bits in a single matrix
t2_data_transpose = t2_data.transpose()
t2_data_16 = t2_data_0.astype(np.int16) + np.left_shift(t2_data_1.astype(np.int16), 7) # the combined 16bit integers
t2_data_transpose_16 = t2_data_16.transpose()
print(t2_data_transpose.shape)

   
oracle = np.matmul(t1_data, t2_data_transpose, dtype=np.int32)                            # the result of the 8bit input matrix multiplication
print(oracle[0,0:50] + np.left_shift(oracle[1,0:50]+oracle[0,50:], 7) + np.left_shift(oracle[1,50:], 14) ) # combining lower and upper 8 bits of the calculated products to get the result of corresponding to the 16bit inputs
print(' ')

oracle_16 = np.matmul(t1_data_16, t2_data_transpose_16, dtype=np.int32)                   # the result of the 16bit input matrix multiplication
print(oracle_16)
print('oracle shape:'+str(oracle.shape))
#print(oracle)

##########################################################################################################
# Run the compiled code on the Groq chip


program = g.create_tsp_runner(iop_file)
result = program(matrix1=t1_data, matrix2=t2_data)
print('Groq result shape:'+str(result['mm_result'].shape))
#print(result['mm_result'].astype(np.int8))

# combing the result of the products of the 8bit input matrices to get the result of the product of 16bit input matrices

result_data = np.zeros( (t1_data.shape[0], t2_data.shape[0]), dtype=np.int32 )
#print( result['mm_result'][0].shape )
# combining 8bit result parts into 32bit results corresponding to the product of 8bit input matrices
for idx in range( result['mm_result'][0].shape[0] ):
    result_tmp = result['mm_result'][:,idx,:]
    result_data = result_data + np.left_shift(result_tmp, 8*idx).astype(np.int32)

print(result_data.shape )


# combining lower and upper 8 bits of the calculated products to get the result of corresponding to the 16bit inputs
result_data = result_data[0,0:50] + np.left_shift(result_data[1,0:50]+result_data[0,50:], 7) + np.left_shift(result_data[1,50:], 14 )


###########################################################################################################
# Check Result


print("Matrix Multiplication for input tensors of size {} x {}.  Results are: ".format(t1_data.shape, t2_data.shape))
print(np.allclose(oracle_16, result_data, rtol=1e-1, atol=1e-1, equal_nan=True))

#print(t1_data)

print(result_data)
