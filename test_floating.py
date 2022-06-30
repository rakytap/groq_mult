import groq.api as g
import groq.api.nn as nn
import numpy as np
import bitchunks as bch
import time
print("Python packages imported successfully")

######################################################################################x
# Create and compile program for Groq chip

dim = 70
num_of_chunks_result = 9

matrix1 = g.input_tensor(shape=(num_of_chunks_result, dim), dtype=g.int8, name="matrix1", layout="H1(W), -1, S2(42-43)")
matrix20 = g.input_tensor(shape=(bch.num_of_chunks*dim, dim), dtype=g.int8, name="matrix20", layout="H1(W), -1, S16(22-41)")
#matrix21 = g.input_tensor(shape=(bch.num_of_chunks*dim, dim), dtype=g.int8, name="matrix21", layout="H1(E), -1, S16")

class TopLevel(g.Component):  # Create our top level component
    def __init__(self):
        super().__init__()
        self.mm = nn.MatMul(name="MyMatMul", use_vxm_accum=True, planes=[0,1])     #Matmul: using the nn.MatMul() component.
        #self.mm2 = nn.MatMul(name="MyMatMul2", use_vxm_accum=True, planes=[2,3])     #Matmul: using the nn.MatMul() component.
        
        # ALU request for bitshift operations (The resource is reused within iterative calls)
        self.bitshift_alu_rq = g.tensor.create_alu_request(alus=[4])
        
        # ALU request for element-wise add operations (The resource is reused within iterative calls)
        self.add_alu_rq = g.tensor.create_alu_request(alus=[1])     
        
        # ALU request for element-wise and operations (The resource is reused within iterative calls)
        self.and_alu_rq = g.tensor.create_alu_request(alus=[0])    
        
        # resource request for element shifting over inner dimension
        self.perm_rq = g.tensor.create_permutor_request(perm=[0], num_perm=1)
           



    def build(self, mat1_mt, mat20_mt, time=0):   #Provide input matrices and a default time
    #def build(self, mat1_mt, mat20_mt, mat21_mt, time=0):   #Provide input matrices and a default time


        with g.ResourceScope(name="mmscope", is_buffered=True, time=0) as mmscope :   
            #result_mt = self.mm(mat1_mt, mat2_mt, time=0).write(name="mm_result", layout="H1(W), -1, S4")  #recommended layout for the matmul result

            #print( mat20_mt.physical_shape)

            result_st = self.mm(mat1_mt, mat20_mt, time=0)  #int8 multiplication results on SG4_E[4] when plane 0 is used
            
            # while calculationg multiplication, creating some constant data
            dtype = g.int32
            bitshift_shape = (1, result_st.shape[1])
            bitshift_mt = g.constant_tensor(bitshift_shape, dtype, name="bitshift_tensor", layout="H1(W), A2, S4(11-14)")
            bitshift_mt.data = np.ones(bitshift_shape, dtype=dtype.to_nptype()) * bch.bitchunk
            
            
            # number full of ones used to extract the least significant 7 bits of the chunks
            bits_extract = int(pow(2, bch.bitchunk))-1            
            
            # array to be used to extract lower 7 bits from a stream tensor
            dtype = g.int8
            array_extract_shape = bitshift_shape
            array_extract_mt = g.constant_tensor(array_extract_shape, dtype, name="bitextract_tensor", layout="H1(W), A2, S1(10)")
            array_extract_mt.data = np.ones( array_extract_shape, dtype=np.int8 ) * bits_extract  
            

            print(result_st.shape)
            print(result_st.physical_shape)

            #print(result_st2.shape)
            #print(result_st2.physical_shape)

            #splitted_st = g.split_inner_splits(result_st)
            #print(len(splitted_st))

            '''
            #print(g.split.__doc__)
            separeted_bits_st = g.split(result_st,1, 0)
            lower_bits_st  = separeted_bits_st[0]
            higher_bits_st = separeted_bits_st[1]


            lower_bits_mt  = lower_bits_st.write(name="lower_bits", layout="H1(W), -1, S4(0-3)")
            higher_bits_mt = higher_bits_st.write(name="higher_bits", layout="H1(W), -1, S4(4-7)")
            '''
            
            #result_storreq = g.tensor.create_storage_request(layout="H1(W), A18(1-18), S4(0-3)")
            #print( result_storreq )
            #result_mt = result_st.write(name="result", storage_req=result_storreq)            
            
            #result_mem = result_mem = g.instruction.malloc( hemis=["W"],slices=range(0,4),banks=[0],count=2,reserve_key="example_key")
            #result_mem2 = result_mem = g.instruction.malloc( hemis=["W"],slices=range(0,4),banks=[0],count=18,reserve_key="example_key2")
            #result_mt = result_st.write(name="result", storage_req=result_mem2)  
          
            result_mt = result_st.write(name="result", layout="H1(W), A18, S4(4-7)")
            
            
            #g.resolve_storage_requests(mmscope)
            #print( result_mt.addrs )
            
        with g.ResourceScope(name="bitreducescope", is_buffered=True, time=None, predecessors=[mmscope]) as brscope :             
            
            
            print('***************** tmp scope **********************')
            
            
            
            
            # storage request for iteratively used memory space storing individual rows
            row_storreq = g.tensor.create_storage_request(layout="H1(W), A2, S4(0-3)")
            
            # storage request for iteratively used bitshift stream tensor
            #bitshift_storreq = g.tensor.create_storage_request(layout="H1(W), A2, S1(8)")
            
            
            # some data used to read out data for stream tensors from memory
            split_sizes = result_mt.physical_shape.inner_dims
            split_num = result_mt.physical_shape.splits
            vectors = result_mt.physical_shape.vectors            
            
            next_row_mt = None
            extracted_bits_mt_list = []    
            
            for row_idx in range(num_of_chunks_result-1):
                print('Iteration: row_dx='+str(row_idx))
                
                if row_idx == 0:
            
                    # first create a stream tensor from the row of the memory tensor
                    row_mt_list = []
                    for split_idx in range(len(split_sizes)):
                        addrs_0 = np.array([g.instruction.parse_address(f"W4-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1)]).reshape(1, 1)
                        addrs_1 = np.array([g.instruction.parse_address(f"W5-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1)]).reshape(1, 1)
                        addrs_2 = np.array([g.instruction.parse_address(f"W6-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1)]).reshape(1, 1)
                        addrs_3 = np.array([g.instruction.parse_address(f"W7-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1)]).reshape(1, 1)
                
                        addrs = np.concatenate( (addrs_0, addrs_1, addrs_2, addrs_3), axis=1 )
                        print(addrs)
                
                                                                
                        #print(addrs)                
                        row_mt_list.append( g.from_addresses(addrs, inner_dim=split_sizes[split_idx], dtype=g.int32, name='row_mt_'+str(split_idx)  ) ) 
                
                
                    row_mt = g.concat_inner_splits( row_mt_list )          
                    row_st = row_mt.read(streams=g.SG4_E[2], time=0)
                    
                else:
                
                    # use the rox created in the previous iteration
                    row_st = next_row_mt.read(streams=g.SG4_E[2], time=18*row_idx) # 15 clicks needed to do one cylce + 3 cycles of memory control latency
            
                # now shift the bits of the row 
                #dtype = g.int8
                #bitshift_st = g.constant_tensor(row_st.shape, dtype, name="bitshift_tensor", storage_req=bitshift_storreq)
                #bitshift_st.data = np.ones(row_st.shape, dtype=dtype.to_nptype()) * bch.bitchunk
                bitshift_st = bitshift_mt.read(streams=g.SG4_E[3], time=None)   

            
                row_st = g.right_shift(row_st, bitshift_st, output_streams=g.SG4_E[2], alus=self.bitshift_alu_rq)
            
                       
            
                #now add the bit-shifted data to the next row
                next_row_mt_list = []
                for split_idx in range(len(split_sizes)):
                    addrs_0 = np.array([g.instruction.parse_address(f"W4-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1,2)]).reshape(1, 1)
                    addrs_1 = np.array([g.instruction.parse_address(f"W5-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1,2)]).reshape(1, 1)
                    addrs_2 = np.array([g.instruction.parse_address(f"W6-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1,2)]).reshape(1, 1)
                    addrs_3 = np.array([g.instruction.parse_address(f"W7-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1,2)]).reshape(1, 1)
                
                    addrs = np.concatenate( (addrs_0, addrs_1, addrs_2, addrs_3), axis=1 )
                    print(addrs)
                
                                                                
                    #print(addrs)                
                    next_row_mt_list.append( g.from_addresses(addrs, inner_dim=split_sizes[split_idx], dtype=g.int32, name='next_row_mt_'+str(split_idx)  ) ) 
                
                
                next_row_mt = g.concat_inner_splits( next_row_mt_list )     
                next_row_st = next_row_mt.read(streams=g.SG4_E[3], time=None)   
                next_row_st = g.add( row_st, next_row_st, output_streams=g.SG4_W[2], alus=self.add_alu_rq, time=None)
            
            
                # save the data of next_row = next_row + 2^(-7)*row into memory (2^(-7) is the right-bitshift operation)
                next_row_mt = next_row_st.write(name="next_row", storage_req=row_storreq)   
            
                #row_mt = row_st.write(name="row", layout="H1(E), A2, S4(0-3)")  
                #next_row_mt = next_row_st.write(name="next_row", layout="H1(W), A2, S4")   
            
            
                #if row_idx == 1:
                #    break
            
            
                #######  extract lower 7 bits of the row and store them into memory  ###########
                print('extracting lower bits')
                if row_idx==0:
                    lower_bits_mt_list = []
                    for split_idx in range(len(split_sizes)):
                        addrs = np.array([g.instruction.parse_address(f"W4-{i+row_idx+split_idx*num_of_chunks_result}") for i in range(1)]).reshape(1, 1)
                        print(addrs)                
                        lower_bits_mt_list.append( g.from_addresses(addrs, inner_dim=split_sizes[split_idx], dtype=g.int8, name='lower_bits_mt_'+str(split_idx))  )                
               
                
                    lower_bits_mt = g.concat_inner_splits( lower_bits_mt_list )          
                    lower_bits_st = lower_bits_mt.read(streams=g.SG1_E[0], time=5)
                    
                    
                else:
                    lower_bits_mt_list = []
                    for split_idx in range(len(split_sizes)):
                        addrs = np.array([g.instruction.parse_address(f"W0-{split_idx}")]).reshape(1, 1)
                        print(addrs)                
                        lower_bits_mt_list.append( g.from_addresses(addrs, inner_dim=split_sizes[split_idx], dtype=g.int8, name='lower_bits_mt_'+str(split_idx)+'_'+str(row_idx))  )                
               
               
                    lower_bits_mt = g.concat_inner_splits( lower_bits_mt_list )          
                    lower_bits_st = lower_bits_mt.read(streams=g.SG1_E[0], time=2+row_idx*18)
                    
                    # add memory tensor exclusion excxeption for the used tensors
                    g.add_mem_constraints(lower_bits_mt_list, [next_row_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                


            
                # now extract the lower 7 bits of the forst row            
            
                # number full of ones used to extract the least significant 7 bits of the chunks
                #bits_extract = int(pow(2, bch.bitchunk))-1
            
                #dtype = g.int8
                #array_extract_st = g.constant_tensor(lower_bits_mt.shape, dtype, name="bitextract_tensor")
                #array_extract_st.data = np.ones( lower_bits_mt.shape, dtype=np.int8 ) * bits_extract  
                array_extract_st = array_extract_mt.read(streams=g.SG4_E[1], time=None) 
            
                lower_bits_st = g.bitwise_and( lower_bits_st, array_extract_st, output_streams=g.SG4_W[0], alus=self.and_alu_rq )
                lower_bits_mt = lower_bits_st.write(name=f"lower_bits_{row_idx}", layout="H1(W), A2, S1(8)")
                
                # add memory tensor exclusion excxeption for the used tensors
                g.add_mem_constraints(extracted_bits_mt_list, [lower_bits_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
            
                extracted_bits_mt_list.append( lower_bits_mt )
            
            
                #lower_bits_mt = row_mt
            

            

            # excract the lower 8 bits from the last row (together with the sign bit)
            print('extracting lower bits --- last row')
            lower_bits_mt_list = []
            for split_idx in range(len(split_sizes)):
                addrs = np.array([g.instruction.parse_address(f"W0-{split_idx}")]).reshape(1, 1)
                print(addrs)                
                lower_bits_mt_list.append( g.from_addresses(addrs, inner_dim=split_sizes[split_idx], dtype=g.int8, name='lower_bits_mt_'+str(split_idx)+'_'+str(num_of_chunks_result-1))  )                
               
               
            lower_bits_mt = g.concat_inner_splits( lower_bits_mt_list )          
            lower_bits_st = lower_bits_mt.read(streams=g.SG1_W[0], time=(num_of_chunks_result-1)*18-1)
            lower_bits_mt = lower_bits_st.write(name=f"lower_bits_{num_of_chunks_result-1}", layout="H1(W), A2, S1(8)")
            
            # add memory tensor exclusion excxeption for the used tensors
            g.add_mem_constraints(extracted_bits_mt_list, [lower_bits_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)       
            
            extracted_bits_mt_list.append( lower_bits_mt )
                    
            # add memory tensor exclusion excxeption for the used tensors
            g.add_mem_constraints(lower_bits_mt_list, [next_row_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)            

            
            # add memory tensor exclusion excxeption for the used tensors
            g.add_mem_constraints([result_mt]+row_mt_list+next_row_mt_list+lower_bits_mt_list, [next_row_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            
            
            #g.resolve_storage_requests(brscope)               
            #print( next_row_mt.addrs )            
            
   
    
                   
        # component to shift element along the inner dimension by amount of dim
        with g.ResourceScope(name="sftscope", is_buffered=True, time=None,  predecessors=[brscope]) as sftscope :  
         
            print('***************** sftscope scope **********************')
            
            row_orig_mt = extracted_bits_mt_list[0]
            row_orig_st  = row_orig_mt.read(streams=g.SG1_W[0], time=0)
            
            
            addrs = np.array([g.instruction.parse_address(f"W8-1")]).reshape(1, 1)
            row_upper_half_orig_mt = g.from_addresses(addrs, inner_dim=split_sizes[1], dtype=g.int8, name='row_upper_half_orig_0')
            row_upper_half_orig_st = row_upper_half_orig_mt.read(streams=g.SG1_W[0], time=2)
            
            # add memory tensor exclusion excxeption for the used tensors
            g.add_mem_constraints(extracted_bits_mt_list, [row_upper_half_orig_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)       
                    
            row_shifted_st            = g.shift(row_orig_st, 70, permutor_id=self.perm_rq, shift_src=[g.instruction.NEW_SRC], output_streams=g.SG1_E[0])
            row_upper_half_shifted_st = g.shift(row_upper_half_orig_st, 70-320, permutor_id=self.perm_rq, shift_src=[g.instruction.NEW_SRC], output_streams=g.SG1_E[0])
            
            # buffer shifted data to feed them to ALU in the same time
            row_shifted_mt            = row_shifted_st.write(name="shifted_data_buffered", layout="H1(W), A2, S1(41)")
            row_upper_half_shifted_mt = row_upper_half_shifted_st.write(name="row_upper_half_shifted_0", layout="H1(W), A2, S1(42)")
            
            # padding result_data2_st with a zero split
            dtype = g.int8
            zeros_shape = (1, split_sizes[1])
            zeros_mt = g.constant_tensor(zeros_shape, dtype, name="zeros_tensor", layout="H1(W), A2, S1(43)")
            zeros_mt.data = np.zeros(zeros_shape, dtype=dtype.to_nptype())
            
            row_upper_half_shifted_mt  = g.concat_inner_splits( [row_upper_half_shifted_mt, zeros_mt])
            row_upper_half_shifted_mt.name = "shifted_data2"
            
            
            row_shifted_st            = row_shifted_mt.read(streams=g.SG4_E[1], time=None)
            row_upper_half_shifted_st = row_upper_half_shifted_mt.read(streams=g.SG4_E[0], time=None)
                      
            
            row_shifted_st = g.add( row_shifted_st, row_upper_half_shifted_st, output_streams=g.SG4_W[2], alus=self.add_alu_rq, time=52)
            row_shifted_mt = row_shifted_st.write(name="shifted_row_0", layout="H1(W), A2, S1(40)")
            
            
            g.resolve_storage_requests(sftscope)  
            print(row_upper_half_shifted_st.shape)
            print(row_orig_mt.addrs)
            print(row_upper_half_orig_mt.addrs)            
        

        return [result_mt] + extracted_bits_mt_list + [row_shifted_mt, row_upper_half_shifted_mt]
        #return result_mt, result_mt2
        #return lower_bits_mt, higher_bits_mt


top = TopLevel()    # instantiate the top level component
result = top(matrix1, matrix20, time=0)    # call into the instance of the top level, providing your inputs and time


iop_file = g.compile(base_name="matmul", result_tensor=result)
g.write_visualizer_data("matmul")
######################################################################################



# calculate the expected result with numpy
print('Calculating expected numpy result')

# create 16 bit input data
t1_data = (np.random.rand(1, dim)*2 - 1.0)
t2_data = np.random.rand(dim, dim)*2 - 1.0



# create 8 bit chunks
t1_data_8, exponent_t1_data = bch.divide_double_into_bitchunks(t1_data, num_of_chunks_result)

# recombine chunk into 64 floats
t1_data_double = bch.combine_bitchunks_into_double(t1_data_8, exponent_t1_data)

print('test t1_data: '+str(np.allclose(t1_data, t1_data_double, rtol=1e-1, atol=1e-1, equal_nan=True)))


# create 8 bit chunks
t2_data_8, exponent_t2_data = bch.divide_double_into_bitchunks(t2_data, bch.num_of_chunks)
#print( t2_data_8 )

# recombine chunk into 64 floats
t2_data_double = bch.combine_bitchunks_into_double(t2_data_8, exponent_t2_data)

print('test t2_data: '+str(np.allclose(t2_data, t2_data_double, rtol=1e-1, atol=1e-1, equal_nan=True)))

###########################################################################################

t0 = time.time()
# matmul with 32 bit arithmetics
mult_result = np.matmul(t1_data, t2_data.transpose())
print("numpy time: " + str( time.time()-t0) )

# multiplication with 8bit arithmetics
mult_result_8 = np.matmul( t1_data_8.reshape((num_of_chunks_result,dim)).astype(np.int64), t2_data_8.reshape((bch.num_of_chunks*dim,dim)).astype(np.int64).transpose() )
mult_result_exponent = exponent_t1_data + exponent_t2_data - bch.double_mantissa_bits # becouse of the multiplication the number of double mantissa bits in combining the 8bit results should be counted twice, hence here we offset the expoentn by the mantissa bit number


# reshape the test result according to the expected inputs of the bitchunks module
mult_result_8 = mult_result_8.reshape( (num_of_chunks_result, 1, bch.num_of_chunks, dim) )

# recombine chunk into 64bit floats
mult_result_double = bch.combine_bitchunks_into_double(mult_result_8, mult_result_exponent)

'''
# create 8 bit chunks
mult_result_8, mult_result_exponent = bch.divide_double_into_bitchunks(mult_result, bch.num_of_chunks)

# recombine chunk into 64 floats
mult_result_double = bch.combine_bitchunks_into_double(mult_result_8, mult_result_exponent)
'''

print('test matmul with numpy: '+str(np.allclose(mult_result, mult_result_double, rtol=1e-1, atol=1e-1, equal_nan=True)))
#print(mult_result_double)
#print(mult_result)


##########################################################################################################
# Run the compiled code on the Groq chip
print('Running the code on the Groq chip')


program = g.create_tsp_runner(iop_file)
t0 = time.time()
result = program(matrix1=t1_data_8.reshape((num_of_chunks_result,dim)), matrix20=t2_data_8.reshape((bch.num_of_chunks*dim,dim)))
print(result.keys())
groq_result_mm = result['result']
shifted_data = result['shifted_row_0']
shifted_data2 = result['shifted_data2']
print(shifted_data.shape)
print(result["lower_bits_0"])
print(' ')
print(shifted_data)
print(' ')
print(shifted_data2)


groq_result = np.zeros( (num_of_chunks_result, bch.num_of_chunks*dim), dtype=np.int8)
for idx in range(num_of_chunks_result):
    groq_result[idx,:] = result[f"lower_bits_{idx}"]
    
groq_result = groq_result.reshape((num_of_chunks_result,1,bch.num_of_chunks, dim))
#print( groq_result_tmp[:,0,0,59:70])


   

# reshape the test result according to the expected inputs of the bitchunks module
groq_result_mm = groq_result_mm.reshape( (num_of_chunks_result, 1, bch.num_of_chunks, dim) )
mult_result_exponent = exponent_t1_data + exponent_t2_data - bch.double_mantissa_bits # becouse of the multiplication the number of double mantissa bits in combining the 8bit results should be counted twice, hence here we offset the expoentn by the mantissa bit number


# recombine chunk into 64bit floats
t0b = time.time()
groq_result_double = bch.combine_bitchunks_into_double(groq_result_mm, mult_result_exponent)
print("Groq recombination time: " + str( time.time()-t0b) )

t0b = time.time()
groq_result_double2, mult_result_exponent_modified = bch.convert_groq_result_to_double(groq_result, mult_result_exponent)
groq_result_double2 = bch.combine_bitchunks_into_double(groq_result_double2, mult_result_exponent_modified)
print("Groq recombination time 2: " + str( time.time()-t0b) )

groq_result_double3, mult_result_exponent_modified3 = bch.convert_groq_result_to_double2(groq_result_mm, mult_result_exponent)
groq_result_double3 = bch.combine_bitchunks_into_double(groq_result_double3, mult_result_exponent_modified3)


print("Groq time: " + str( time.time()-t0) )

###########################################################################################################
# Check Result


print("Matrix Multiplication for input tensors of size {} x {}.  Results are: ".format(t1_data.shape, t2_data.shape))
print('Groq chip matmul: '+str(np.allclose(mult_result, groq_result_double, rtol=1e-10, atol=1e-15, equal_nan=True)))
print('Groq chip matmul: '+str(np.allclose(mult_result, groq_result_double2, rtol=1e-10, atol=1e-15, equal_nan=True)))
print('Groq chip matmul: '+str(np.allclose(mult_result, groq_result_double3, rtol=1e-10, atol=1e-15, equal_nan=True)))
print(groq_result_double[0,1:4])
print(groq_result_double2[0,1:4])
print(groq_result_double3[0,1:4])
print(mult_result[0,1:4])
#print(t1_data)


