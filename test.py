import groq.api as g
import groq.api.nn as nn
import numpy as np
print("Python packages imported successfully")

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
            result_st = g.reinterpret(result_st, g.uint8)
            result_mt = result_st.write(name="mm_result", layout="H1(W), -1, S4")

        return result_mt


top = TopLevel()    # instantiate the top level component
result = top(matrix1, matrix2, time=0)    # call into the instance of the top level, providing your inputs and time

iop_file = g.compile(base_name="matmul", result_tensor=result)
#g.write_visualizer_data("matmul")




# calculate the result with numpy


# prepare input data
t1_data_0 = np.random.randint(-2, 2, (1, 50), dtype=np.int8).astype(np.int8)
t1_data_1 = np.random.randint(-2, 2, (1, 50), dtype=np.int8).astype(np.int8)
t1_data = np.concatenate( (t1_data_0, t1_data_1), axis=0)
t1_data_16 = t1_data_0.astype(np.int16) + pow(2,7)*t1_data_1.astype(np.int16)
print(t1_data.shape)



t2_data_0 = np.random.randint(-2, 2, (50, 50), dtype=np.int8).astype(np.int8)
t2_data_1 = np.random.randint(-2, 2, (50, 50), dtype=np.int8).astype(np.int8)
t2_data = np.concatenate( (t2_data_0, t2_data_1), axis=0)
t2_data_transpose = t2_data.transpose()
t2_data_16 = t2_data_0.astype(np.int16) + pow(2,7)*t2_data_1.astype(np.int16)
t2_data_transpose_16 = t2_data_16.transpose()
print(t2_data_transpose.shape)


oracle = np.matmul(t1_data.astype(np.int32), t2_data_transpose.astype(np.int32), dtype=np.int32)
oracle_16 = np.matmul(t1_data_16, t2_data_transpose_16, dtype=np.int32)
print(oracle[0,0:50] + pow(2,7)*(oracle[1,0:50]+oracle[0,50:]) + pow(2,14)*oracle[1,50:])
print(' ')
print(oracle_16)
print('oracle shape:'+str(oracle.shape))
#print(oracle)


# Run the compiled code on the Groq chip
program = g.create_tsp_runner(iop_file)
result = program(matrix1=t1_data, matrix2=t2_data)
print('Groq result shape:'+str(result['mm_result'].shape))
#print(result['mm_result'].astype(np.int8))

result_data = np.zeros( (t1_data.shape[0], t2_data.shape[0]), dtype=np.uint32 )
#print( result['mm_result'][0].shape )
for idx in range( result['mm_result'][0].shape[0] ):
    result_tmp = result['mm_result'][:,idx,:]
    #print(result_tmp.shape)
    result_data = result_data + result_tmp*pow(2,8*idx)

print(result_data.shape )

result_data = result_data.astype(dtype=np.int32)
result_data = result_data[0,0:50] + pow(2,7)*(result_data[1,0:50]+result_data[0,50:]) + pow(2,14)*result_data[1,50:]

# Check Result


print("Matrix Multiplication for input tensors of size {} x {}.  Results are: ".format(t1_data.shape, t2_data.shape))
print(np.allclose(oracle_16, result_data, rtol=1e-1, atol=1e-1, equal_nan=True))

#print(t1_data)

print(result_data)