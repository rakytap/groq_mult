import groq.api as g
import groq.api.nn as nn
import numpy as np
import bitchunks as bch
import time
print("Python packages imported successfully")

######################################################################################x
# Create and compile program for Groq chip

dim = 80
num_of_chunks_result = bch.num_of_chunks

matrix1 = g.input_tensor(shape=(num_of_chunks_result, dim), dtype=g.int8, name="matrix1", layout="H1(W), -1, S1(18)")
matrix20 = g.input_tensor(shape=(bch.num_of_chunks*dim, dim), dtype=g.int8, name="matrix20", layout="H1(W), -1, S16(23-41)")
#matrix21 = g.input_tensor(shape=(bch.num_of_chunks*dim, dim), dtype=g.int8, name="matrix21", layout="H1(E), -1, S16")




            
            
# component to tile 8bit chunked matrix-matrix product
class TiledMXM(nn.Component):  # Create our top level component


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        self.mm_list = []
        for idx in range(bch.num_of_chunks):
            self.mm_list.append( nn.MatMul(name=f"MyMatMul_{idx}", use_vxm_accum=False, planes=[1], num_planes=1, is_resource_scope=True ) )     #Matmul: using the nn.MatMul() component.
        
    
    
    def build(self, mat1_mt, mat20_mt, time=0):   #Provide input matrices and a default time    

            print('MXM scope')   
            #result_mt = self.mm(mat1_mt, mat2_mt, time=0).write(name="mm_result", layout="H1(W), -1, S4")  #recommended layout for the matmul result

            # split the input matrix into bitchunk memory tensors
            mat20_chunks_mt = g.split_vectors( mat20_mt,  [dim]*bch.num_of_chunks )

            #print( mat20_mt.physical_shape)
            mm_chunks_mt_list = []
            mm_chunks_mt_list_cp = []
            mm_chunks_mt_list_cp2 = []
            start_time = 0
            for idx in range(len(self.mm_list)):
                mm_st = self.mm_list[idx](mat1_mt, mat20_chunks_mt[idx], time=start_time+1)
                mm_mt = mm_st.write(name=f"mm_chunk_{idx}", layout=f"H1(W), A{bch.num_of_chunks}, S4(8-11)")
                if idx>0:
                    g.add_mem_constraints(mm_chunks_mt_list, [mm_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
                mm_chunks_mt_list.append( mm_mt )

                mm_mt_cp = mm_st.write(name=f"mm_chunk_cp_{idx}", layout=f"H1(W), A{bch.num_of_chunks}, S4(4-7)")
                if idx>0:
                    g.add_mem_constraints(mm_chunks_mt_list_cp, [mm_mt_cp], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
                mm_chunks_mt_list_cp.append( mm_mt_cp )

                mm_mt_cp2 = mm_st.write(name=f"mm_chunk_cp2_{idx}", layout=f"H1(W), A{bch.num_of_chunks}, S4(0-3)")
                if idx>0:
                    g.add_mem_constraints(mm_chunks_mt_list_cp2, [mm_mt_cp2], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
                mm_chunks_mt_list_cp2.append( mm_mt_cp2 )
                
                
                start_time = self.mm_list[idx].end_time

            print( mm_chunks_mt_list[0].physical_shape )
            result_mt = g.concat_inner_splits(mm_chunks_mt_list)
            result_mt.name = 'result'


            mm_chunks_mt_arr = np.ndarray( (num_of_chunks_result, bch.num_of_chunks), dtype='object')
            mm_chunks_mt_arr_cp = np.ndarray( (num_of_chunks_result, bch.num_of_chunks), dtype='object')
            mm_chunks_mt_arr_cp2 = np.ndarray( (num_of_chunks_result, bch.num_of_chunks), dtype='object')
            for col_idx in range(bch.num_of_chunks):

                row_chunks_list     =  g.split_vectors(  mm_chunks_mt_list[col_idx], [1]*num_of_chunks_result )
                row_chunks_list_cp  =  g.split_vectors(  mm_chunks_mt_list_cp[col_idx], [1]*num_of_chunks_result )
                row_chunks_list_cp2 =  g.split_vectors(  mm_chunks_mt_list_cp2[col_idx], [1]*num_of_chunks_result )

                for row_idx in range(num_of_chunks_result): 
                    #print( row_chunks_list[row_idx].physical_shape )
                    mm_chunks_mt_arr[row_idx, col_idx]     = row_chunks_list[row_idx]
                    mm_chunks_mt_arr_cp[row_idx, col_idx]  = row_chunks_list_cp[row_idx]
                    mm_chunks_mt_arr_cp2[row_idx, col_idx] = row_chunks_list_cp2[row_idx]
                    
                    
            element_num = bch.num_of_chunks*num_of_chunks_result
            return [result_mt] + list(mm_chunks_mt_arr.reshape(element_num)) + list(mm_chunks_mt_arr_cp.reshape(element_num)) + list(mm_chunks_mt_arr_cp2.reshape(element_num))       




            
# component to convert 32bit chunked matrix into 8bit chunked matrix
class Reduce32to8(nn.Component):  # Create our top level component


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        # ALU request for bitshift operations (The resource is reused within iterative calls)
        self.bitshift_alu_rq = g.tensor.create_alu_request(alus=[4])
        
        # ALU request for element-wise add operations (The resource is reused within iterative calls)
        self.add_alu_rq = g.tensor.create_alu_request(alus=[1])     
        
        # ALU request for element-wise and operations (The resource is reused within iterative calls)
        self.and_alu_rq = g.tensor.create_alu_request(alus=[8])    
        
    
    
    def build(self, mm_chunks_mt_arr, mm_chunks_mt_arr_cp, loop_length=26, time=0):   #Provide input matrices and a default time    

            # slices 16, 20,24 and 28 reserved for system and slice 38 on wes side
            
            chunk_num_in_rows = mm_chunks_mt_arr.shape[0]
            chunk_num_in_cols = mm_chunks_mt_arr.shape[1]
            
            
            # storage request for iteratively used memory space storing next_row tensors and its copy
            row_storreq    = g.tensor.create_storage_request(layout=f"H1(W), A{bch.num_of_chunks}, S4(0-3)")
            row_storreq_cp = g.tensor.create_storage_request(layout=f"H1(W), A{bch.num_of_chunks}, S4(8-11), B1(1)")            
                    
            next_row_mt = None
            next_row_mt_cp = None            
            extracted_bits_mt_list    = []    
            extracted_bits_mt_list_cp = []                
            
            # while calculationg multiplication, creating some constant data
            dtype = g.int32
            bitshift_shape = (1, 320*chunk_num_in_cols)
            bitshift_mt = g.constant_tensor(bitshift_shape, dtype, name="bitshift_tensor", layout=f"H1(W), A{bch.num_of_chunks}, S4(13-17)")
            bitshift_mt.data = np.ones(bitshift_shape, dtype=dtype.to_nptype()) * bch.bitchunk
            
            
            # number full of ones used to extract the least significant 7 bits of the chunks
            bits_extract = int(pow(2, bch.bitchunk))-1            
            
            # array to be used to extract lower 7 bits from a stream tensor
            dtype = g.int8
            array_extract_shape = bitshift_shape
            array_extract_mt = g.constant_tensor(array_extract_shape, dtype, name="bitextract_tensor", layout=f"H1(W), A{bch.num_of_chunks}, S1(12)")
            array_extract_mt.data = np.ones( array_extract_shape, dtype=np.int8 ) * bits_extract         
            
           
                                      
            
            #loop_length = 26 # cycles
            
            for row_idx in range(chunk_num_in_rows-1):
                #print('Iteration: row_idx='+str(row_idx))
                
                                   
                        
                #######  shift the row by 7 bits ###########
                if row_idx == 0:
                    row_mt = g.concat_inner_splits( list(mm_chunks_mt_arr[row_idx,:]) )       #row_result_mm_mt_list[0]
                    row_st = row_mt.read(streams=g.SG4_E[2], time=0)
                else:                
                    # use the rox created in the previous iteration
                    row_st = next_row_mt.read(streams=g.SG4_E[2], time=0+loop_length*row_idx) 
            

                bitshift_st = bitshift_mt.read(streams=g.SG4_E[3], time=None)               
                row_st = g.right_shift(row_st, bitshift_st, output_streams=g.SG4_E[2], alus=self.bitshift_alu_rq)
                #row_mt = row_st.write( name="bitshifted_row", layout="H1(E), A9, S
                
                
                
                #######  extract lower 7 bits of the row and store them into memory  ###########
                #print('extracting lower bits')
                if row_idx==0:

                    mm_chunks_mt_8_list = []
                    for jdx in range(chunk_num_in_cols):
                        mm_chunk = mm_chunks_mt_arr[row_idx,jdx]
                        mm_chunk_8 = g.reinterpret(mm_chunk, g.int8 )
                        mm_chunk_8 = g.split_vectors( mm_chunk_8,  [1]*4 ) # int32 separeted into int8 
                        mm_chunks_mt_8_list.append(mm_chunk_8[0])

                    lower_bits_mt = g.concat_inner_splits( mm_chunks_mt_8_list )
                    #row_mt_8 = g.reinterpret(row_mt, g.int8 )
                    #row_mt_bytes = g.split_vectors( row_mt_8,  [1]*4 ) # int32 separeted into int8                  
                    #lower_bits_mt = row_mt_bytes[0]                    
                           
                    lower_bits_st = lower_bits_mt.read(streams=g.SG4_E[4], time=bch.num_of_chunks)                    
                  
                else:             
                                    
                    next_row_chunks_8_list = g.split_inner_splits( next_row_mt_cp ) # sperate tensor into butchunks
                  
                    
                    lower_bits_chunks_list = []
                    for idx in range(len(next_row_chunks_8_list)):
                        tmp_mt = g.reinterpret(next_row_chunks_8_list[idx], g.int8 )
                        tmp_mt_bytes = g.split_vectors( tmp_mt,  [1]*4 ) # int32 separeted into int8   
                        lower_bits_chunks_list.append( tmp_mt_bytes[0] )
                        
                    lower_bits_mt = g.concat_inner_splits( lower_bits_chunks_list )
                    
                    #row_mt = g.reinterpret(next_row_mt, g.int8 )
                    #row_mt_bytes = g.split_vectors( row_mt,  [1]*4 ) # int32 separeted into int8                  
                    #lower_bits_mt = row_mt_bytes[0]
                            
                    lower_bits_st = lower_bits_mt.read(streams=g.SG4_E[4], time=3+loop_length*row_idx)
                                   
               
                
                array_extract_st = array_extract_mt.read(streams=g.SG4_E[5], time=None) 
            
                lower_bits_st = g.bitwise_and( lower_bits_st, array_extract_st, output_streams=g.SG4_W[4], alus=self.and_alu_rq )
                lower_bits_mt    = lower_bits_st.write(name=f"lower_bits_{row_idx}", layout=f"H1(W), A{bch.num_of_chunks}, S1(18)")
                lower_bits_mt_cp = lower_bits_st.write(name=f"lower_bits_cp_{row_idx}", layout=f"H1(W), A{bch.num_of_chunks}, S1(19)")                
                
                # add memory tensor exclusion excxeption for the used tensors
                g.add_mem_constraints(extracted_bits_mt_list, [lower_bits_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
                g.add_mem_constraints(extracted_bits_mt_list_cp, [lower_bits_mt_cp], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)                 
                #g.add_mem_constraints(extracted_bits_mt_list+row_result_mm_mt_list+[result_mt], [lower_bits_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            
                extracted_bits_mt_list.append( lower_bits_mt )               
                extracted_bits_mt_list_cp.append( lower_bits_mt_cp )                    
                
                
                #######  add the 7bit-shifet row to the next row  ###########
                #print('creating next row')
                next_row_mt = g.concat_inner_splits( list(mm_chunks_mt_arr_cp[row_idx+1,:]) ) #row_result_mm_mt_list[row_idx+1]  
                next_row_st = next_row_mt.read(streams=g.SG4_E[1], time=None)  
                next_row_st = g.add( row_st, next_row_st, output_streams=g.SG4_W[2], alus=self.add_alu_rq, time=None)
            
                # save the data of next_row = next_row + 2^(-7)*row into memory (2^(-7) is the right-bitshift operation)
                next_row_mt = next_row_st.write(name="next_row", storage_req=row_storreq)   
                next_row_mt_cp = next_row_st.write(name="next_row_cp", storage_req=row_storreq_cp)                  
                
            
            # excract the lower 8 bits from the last row (together with the sign bit)
            #print('extracting lower bits --- last row')
            
            next_row_chunks_8_list = g.split_inner_splits( next_row_mt_cp ) # sperate tensor into butchunks
                  
                    
            lower_bits_chunks_list = []
            for idx in range(len(next_row_chunks_8_list)):
                tmp_mt = g.reinterpret(next_row_chunks_8_list[idx], g.int8 )
                tmp_mt_bytes = g.split_vectors( tmp_mt,  [1]*4 ) # int32 separeted into int8   
                lower_bits_chunks_list.append( tmp_mt_bytes[0] )
                        
            lower_bits_mt = g.concat_inner_splits( lower_bits_chunks_list )
                    
            
            #row_mt = g.reinterpret(next_row_mt, g.int8 )
            #row_mt_bytes = g.split_vectors( row_mt,  [1]*4 ) # int32 separeted into int8                  
            #lower_bits_mt = row_mt_bytes[0]
            lower_bits_st = lower_bits_mt.read(streams=g.SG1_W[4], time=1+(chunk_num_in_rows-1)*loop_length)
            lower_bits_mt    = lower_bits_st.write(name=f"lower_bits_{chunk_num_in_rows-1}", layout=f"H1(W), A{bch.num_of_chunks}, S1(18)")
            lower_bits_mt_cp = lower_bits_st.write(name=f"lower_bits_cp_{chunk_num_in_rows-1}", layout=f"H1(W), A{bch.num_of_chunks}, S1(21)")            
            print( lower_bits_mt_cp.physical_shape )
            
            # add memory tensor exclusion excxeption for the used tensors
            g.add_mem_constraints(extracted_bits_mt_list, [lower_bits_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)       
            g.add_mem_constraints(extracted_bits_mt_list_cp, [lower_bits_mt_cp], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)                  
            
            extracted_bits_mt_list.append( lower_bits_mt )   
            extracted_bits_mt_list_cp.append( lower_bits_mt_cp )               
            
            g.add_mem_constraints(list(mm_chunks_mt_arr.reshape(mm_chunks_mt_arr.size)), [next_row_mt_cp], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)      


            return extracted_bits_mt_list + extracted_bits_mt_list_cp
            
            
            



# component to convert 32bit chunked matrix into 8bit chunked matrix
class ReduceMMResult(nn.Component):  # Create our top level component


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        # ALU request for bitshift operations (The resource is reused within iterative calls)
        self.bitshift_alu_rq = g.tensor.create_alu_request(alus=[4])
        
        # ALU request for element-wise add operations (The resource is reused within iterative calls)
        self.add_alu_rq = g.tensor.create_alu_request(alus=[1])    

        # ALU request for element-wise add operations (The resource is reused within iterative calls)
        self.add_alu_rq2 = g.tensor.create_alu_request(alus=[6])  
   
        
        # ALU request for element-wise and operations (The resource is reused within iterative calls)
        self.and_alu_rq = g.tensor.create_alu_request(alus=[8])    
        
    
    
    def build(self, mm_chunks_mt_arr, mm_chunks_mt_arr_cp, mm_chunks_mt_arr_cp2, loop_length=26, time=0):   #Provide input matrices and a default time    

            # slices 16, 20,24 and 28 reserved for system and slice 38 on wes side
            
            chunk_num_in_rows = mm_chunks_mt_arr.shape[0]
            chunk_num_in_cols = mm_chunks_mt_arr.shape[1]
            
            
            # storage request for iteratively used memory space storing next_row tensors and its copy
            row_storreq = g.tensor.create_storage_request(layout=f"H1(W), A{bch.num_of_chunks}, S4(0-3)")
            row_storreq_cp = g.tensor.create_storage_request(layout=f"H1(W), A{bch.num_of_chunks}, S4(8-11)")
                    
            next_row_mt = None
            next_row_mt_cp = None            
            extracted_bits_mt_list    = []    
            extracted_bits_mt_list_cp = []                
            
            # while calculationg multiplication, creating some constant data
            dtype = g.int32
            bitshift_shape = (1, 320*chunk_num_in_cols)
            bitshift_mt = g.constant_tensor(bitshift_shape, dtype, name="bitshift_tensor", layout=f"H1(W), A{bch.num_of_chunks}, S4(12-15)")
            bitshift_mt.data = np.ones(bitshift_shape, dtype=dtype.to_nptype()) * bch.bitchunk
            
            
            # number full of ones used to extract the least significant 7 bits of the chunks
            bits_extract = int(pow(2, bch.bitchunk))-1            
            
            # array to be used to extract lower 7 bits from a stream tensor
            dtype = g.int32
            array_extract_shape = bitshift_shape
            array_extract_mt = g.constant_tensor(array_extract_shape, dtype, name="bitextract_tensor", layout=f"H1(W), A{bch.num_of_chunks}, S4(17-21)")
            array_extract_mt.data = np.ones( array_extract_shape, dtype=np.int32 ) * bits_extract         
            
            g.add_mem_constraints([bitshift_mt], [array_extract_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)      


            # zero tensors to pad the shifted row by zeros
            dtype = g.int32
            zeros_shape = (1, dim)
            zeros_mt_pad32 = g.constant_tensor(zeros_shape, dtype, name="zeros_tensor_pad", layout="H1(W), A1, S4(0-3)")
            zeros_mt_pad32.data = np.zeros(zeros_shape, dtype=dtype.to_nptype())
            zeros_mt_pad32_cp      = g.constant_tensor(zeros_shape, dtype, name="zeros_tensor_pad_cp", layout="H1(W), A1, S4(8-11)")
            zeros_mt_pad32_cp.data = np.zeros(zeros_shape, dtype=dtype.to_nptype())
            g.add_mem_constraints(list(mm_chunks_mt_arr_cp2.reshape(-1)), [zeros_mt_pad32], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)    
            g.add_mem_constraints(list(mm_chunks_mt_arr.reshape(-1)), [zeros_mt_pad32_cp], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)                                            
                        
            for row_idx in range(chunk_num_in_rows-1):
                print('Iteration: row_idx='+str(row_idx))
                
                                   
                print('bitshift row')
                #######  shift the row by 7 bits ###########
                if row_idx == 0:
                    row_mt = g.concat_inner_splits( list(mm_chunks_mt_arr[row_idx,:]) )       #row_result_mm_mt_list[0]
                    row_st = row_mt.read(streams=g.SG4_E[2], time=0)
                else:                
                    # use the rox created in the previous iteration
                    row_st = next_row_mt.read(streams=g.SG4_E[2], time=loop_length*row_idx) 
            

                bitshift_st = bitshift_mt.read(streams=g.SG4_E[3], time=None)               
                row_st = g.right_shift(row_st, bitshift_st, output_streams=g.SG4_E[2], alus=self.bitshift_alu_rq)
                #row_mt = row_st.write( name="bitshifted_row", layout="H1(E), A9, S4" )  
                '''
                if row_idx==1: ######################################################################x
                    row_mt = row_st.write( name="bitshifted_row", layout="H1(E), -1, S4" )
                    break    
                '''

                
                #######  add the 7bit-shifet row to the next row  ###########
                print('calculating the sum of the bit-shifted and the next row')
                next_row_tmp_mt = g.concat_inner_splits( list(mm_chunks_mt_arr_cp[row_idx+1,:]) ) #row_result_mm_mt_list[row_idx+1]  
                next_row_tmp_st = next_row_tmp_mt.read(streams=g.SG4_E[1], time=None)  
                next_row_tmp_st = g.add( row_st, next_row_tmp_st, output_streams=g.SG4_E[2], alus=self.add_alu_rq, time=None)
            
          
                '''
                if row_idx==1: ######################################################################x
                    row_mt = row_st.write( name="bitshifted_row", layout="H1(E), -1, S4" )
                    next_row_mt = next_row_tmp_mt
                    break    
                '''


                                
                #######  extract lower 7 bits of the row and store them into memory  ###########
                print('extracting lower bits and add to the next row')
                if row_idx==0:
                    row_mt = g.concat_inner_splits( list(mm_chunks_mt_arr_cp2[row_idx,1:]) )       #row_result_mm_mt_list[0]
                    row_mt = g.concat_inner_splits( [row_mt,  zeros_mt_pad32] )    
                    row_st = row_mt.read(streams=g.SG4_E[4], time=None)#24)
                       
                else: 
                    next_row_chunks_8_list = g.split_inner_splits( next_row_mt_cp ) # sperate tensor into butchunks
                    row_mt = g.concat_inner_splits( next_row_chunks_8_list[1:] + [ zeros_mt_pad32_cp]) 
                    row_st = row_mt.read(streams=g.SG4_E[4], time=None)#loop_length*row_idx+23)
                    
                
                array_extract_st = array_extract_mt.read(streams=g.SG4_E[5], time=None)             
                lower_bits_st    = g.bitwise_and( row_st, array_extract_st, output_streams=g.SG4_E[4], alus=self.and_alu_rq )


                '''
                if row_idx==0:
                    next_row_mt = row_st.write( name="bitshifted_row", layout="H1(E), -1, S4" )
                    break   
                '''
            

                next_row_st = next_row_tmp_st#next_row_tmp_mt.read(streams=g.SG4_E[2])
                next_row_st = g.add( next_row_st, lower_bits_st, output_streams=g.SG4_W[2], alus=self.add_alu_rq2, time=None)
                next_row_mt    = next_row_st.write(name=f"next_row_{row_idx}", storage_req=row_storreq)   
                next_row_mt_cp = next_row_st.write(name=f"next_row_{row_idx}", storage_req=row_storreq_cp)   
              

                g.add_mem_constraints([next_row_mt], list(mm_chunks_mt_arr_cp2.reshape(-1))+[zeros_mt_pad32_cp, zeros_mt_pad32], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
                g.add_mem_constraints([next_row_mt_cp], list(mm_chunks_mt_arr.reshape(-1))+[zeros_mt_pad32_cp], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 

                                       

            g.add_mem_constraints([next_row_mt], list(mm_chunks_mt_arr_cp2.reshape(-1))+[zeros_mt_pad32_cp, zeros_mt_pad32], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
            g.add_mem_constraints([next_row_mt_cp], list(mm_chunks_mt_arr.reshape(-1))+[zeros_mt_pad32_cp], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
            g.add_mem_constraints([array_extract_mt], [next_row_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 


            return next_row_mt, next_row_mt_cp


class TopLevel(g.Component):  # Create our top level component
    def __init__(self):
        super().__init__()    

        # component to perform the int8 matrix multiplication, splitted into chunks. (TODO: parallelize using both MxM planes) 
        self.tiledmxm = TiledMXM( is_resource_scope=True ) 

        # component to reduce the bitchunk_num x bitchunk_num result matrix to a single vector of bitchunks. (Here overflow over int8 might still occur)
        self.reduceMMresult = ReduceMMResult( is_resource_scope=True )  

        # component to reduce the result of the previous component to int8 handling the overflows
        self.reduce32to8 = Reduce32to8( is_resource_scope=True ) 


    def build(self, mat1_mt, mat20_mt, time=0):   #Provide input matrices and a default time
    #def build(self, mat1_mt, mat20_mt, mat21_mt, time=0):   #Provide input matrices and a default time

        # component to perform the int8 matrix multiplication, splitted into chunks. (TODO: parallelize using both MxM planes) 
        with g.ResourceScope(name="mmscope", is_buffered=True, time=0) as mmscope :

            print('MXM scope')   
            mm_chunks_mt_list__ALL = self.tiledmxm(mat1_mt, mat20_mt)
            result_mt = mm_chunks_mt_list__ALL[0]
            mm_chunks_mt_list__ALL = mm_chunks_mt_list__ALL[1:]
           
            # split the result into np.arrays for further usage
            length = int(len(mm_chunks_mt_list__ALL)/3)
            mm_chunks_mt_list     = mm_chunks_mt_list__ALL[0:length]
            mm_chunks_mt_list_cp  = mm_chunks_mt_list__ALL[ length:2*length ]
            mm_chunks_mt_list_cp2 = mm_chunks_mt_list__ALL[ 2*length: ]
            
            mm_chunks_mt_arr     = np.array( mm_chunks_mt_list, dtype='object' ).reshape(num_of_chunks_result, bch.num_of_chunks)
            mm_chunks_mt_arr_cp  = np.array( mm_chunks_mt_list_cp, dtype='object' ).reshape(num_of_chunks_result, bch.num_of_chunks)
            mm_chunks_mt_arr_cp2 = np.array( mm_chunks_mt_list_cp2, dtype='object' ).reshape(num_of_chunks_result, bch.num_of_chunks)

        #return [result_mt] 

        # component to reduce the bitchunk_num x bitchunk_num result matrix to a single vector of bitchunks. (Here overflow over int8 might still occur)
        with g.ResourceScope(name="reducemmscope", is_buffered=True, time=None, predecessors=[mmscope]) as rmmscope :             
            
            
            print('***************** MxM reduce scope **********************')
            loop_length = 30+(bch.num_of_chunks-9)

            # keep in mind reduced_mm_mt <-> reduced_mm_mt_cp interchange compared to the component output
            reduced_mm_mt_cp, reduced_mm_mt = self.reduceMMresult(mm_chunks_mt_arr, mm_chunks_mt_arr_cp, mm_chunks_mt_arr_cp2, loop_length=loop_length)

            g.add_mem_constraints([reduced_mm_mt], [mat1_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

        # make copy of reduced_mm into memory slices expected by component Convert32to8      
        with g.ResourceScope(name="copyscope", is_buffered=True, time=None, predecessors=[rmmscope]) as cpscope :   


            # make copy of reduced_mm into memory slices expected by Convert32to8      
            reduced_mm_st_cp   = reduced_mm_mt_cp.read(streams=g.SG4_W[2], time=0)
            reduced_mm_mt_cp   = reduced_mm_st_cp.write(name=reduced_mm_mt_cp.name+"_cp", layout=f"H1(W), A{bch.num_of_chunks}, S4(4-7)")
            
            mm_chunks_mt_arr    =  np.array( g.split_inner_splits(  reduced_mm_mt ) ).reshape( (num_of_chunks_result, 1) )
            mm_chunks_mt_arr_cp =  np.array( g.split_inner_splits(  reduced_mm_mt_cp ) ).reshape( (num_of_chunks_result, 1) )


        # component to reduce the result of the previous component to int8 handling the overflows
        with g.ResourceScope(name="bitreducescope", is_buffered=True, time=None, predecessors=[cpscope]) as brscope :   

            
            
            print('***************** bitreduce scope 2**********************')
            loop_length = 17 + (bch.num_of_chunks-9)
            extracted_bits_mt_list__ALL = self.reduce32to8(mm_chunks_mt_arr, mm_chunks_mt_arr_cp, loop_length=loop_length)

            length = int(len(extracted_bits_mt_list__ALL)/2)
            extracted_bits_mt_list = extracted_bits_mt_list__ALL[0:length]
            extracted_bits_mt_list_cp = extracted_bits_mt_list__ALL[ length: ]    
            reduced_row_mt = g.concat_vectors(extracted_bits_mt_list, (num_of_chunks_result,dim) )
            reduced_row_mt.name = "reduced_row"
            print( reduced_row_mt.physical_shape)
            print( reduced_row_mt.shape)

            g.add_mem_constraints(extracted_bits_mt_list__ALL, [mat1_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            

        #g.resolve_storage_requests(brscope2)
 
        return [result_mt, reduced_mm_mt, reduced_row_mt] 
            
        
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

#t1_data = np.full((1, dim,), -((1 << 53)-1000000)/(1<<53))
#t2_data = np.full((dim, dim), -((1 << 53)-1000000)/(1<<53))



# create 8 bit chunks
t1_data_8, exponent_t1_data = bch.divide_double_into_bitchunks(t1_data, num_of_chunks_result)
print( 't1_data:')
print( t1_data_8[:,0,0,0:10])

# recombine chunk into 64 floats
t1_data_double = bch.combine_bitchunks_into_double(t1_data_8, exponent_t1_data)

print('test t1_data: '+str(np.allclose(t1_data, t1_data_double, rtol=1e-10, atol=1e-15, equal_nan=True)))


# create 8 bit chunks
t2_data_8, exponent_t2_data = bch.divide_double_into_bitchunks(t2_data, bch.num_of_chunks)
#print( t2_data_8 )

# recombine chunk into 64 floats
t2_data_double = bch.combine_bitchunks_into_double(t2_data_8, exponent_t2_data)

print('test t2_data: '+str(np.allclose(t2_data, t2_data_double, rtol=1e-10, atol=1e-15, equal_nan=True)))

###########################################################################################

t0 = time.time()
# matmul with 32 bit arithmetics
mult_result = np.matmul(t1_data.astype(np.longdouble), t2_data.astype(np.longdouble).transpose())
print("numpy time: " + str( time.time()-t0) )

# multiplication with 8bit arithmetics
mult_result_8 = np.matmul( t1_data_8.reshape((num_of_chunks_result,dim)).astype(np.int64), t2_data_8.reshape((bch.num_of_chunks*dim,dim)).astype(np.int64).transpose() )
mult_result_exponent = exponent_t1_data + exponent_t2_data - bch.double_mantissa_bits # becouse of the multiplication the number of double mantissa bits in combining the 8bit results should be counted twice, hence here we offset the expoentn by the mantissa bit number


# reshape the test result according to the expected inputs of the bitchunks module
mult_result_8 = mult_result_8.reshape( (num_of_chunks_result, 1, bch.num_of_chunks, dim) )

# recombine chunk into 64bit floats
mult_result_double = bch.combine_bitchunks_into_double(mult_result_8, mult_result_exponent)

print('test matmul with numpy: '+str(np.allclose(mult_result, mult_result_double, rtol=1e-1, atol=1e-1, equal_nan=True)))
#print(mult_result_double)
#print(mult_result)


##########################################################################################################
# Run the compiled code on the Groq chip
print(' ' )
print('Running the code on the Groq chip')


program = g.create_tsp_runner(iop_file)
t0 = time.time()
result = program(matrix1=t1_data_8.reshape((num_of_chunks_result,dim)), matrix20=t2_data_8.reshape((bch.num_of_chunks*dim,dim)))
groq_result_mm = result['result']

reduced_row = result['reduced_row']
reduced_row = reduced_row.reshape((bch.num_of_chunks, 1, 1, dim))
print( 'type of the result: '+str(reduced_row.dtype) )

# reshape the test result according to the expected inputs of the bitchunks module
groq_result_mm = groq_result_mm.reshape( (num_of_chunks_result, 1, bch.num_of_chunks, dim) )
mult_result_exponent = exponent_t1_data + exponent_t2_data - bch.double_mantissa_bits # because of the multiplication the number of double mantissa bits in combining the 8bit results should be counted twice, hence here we offset the expoentn by the mantissa bit number


t0b = time.time()
mult_result_exponent_modified3 = mult_result_exponent + bch.bitchunk*(bch.num_of_chunks-1)
groq_result_double3 = bch.combine_bitchunks_into_double(reduced_row, mult_result_exponent_modified3)

print("Groq bit-chunk recombination time: " + str( time.time()-t0b) )
print("Groq time full (including recombination): " + str( time.time()-t0) )


groq_result_double4, mult_result_exponent_modified4 = bch.convert_groq_result_to_double2(groq_result_mm, mult_result_exponent)
print('bit chunk recombination test: '+str(np.allclose(groq_result_double4.reshape(-1), reduced_row.reshape(-1), rtol=1e-10, atol=1e-15, equal_nan=True)))
groq_result_double4 = bch.combine_bitchunks_into_double(groq_result_double4, mult_result_exponent_modified4)


# recombine chunk into 64bit floats
groq_result_double = bch.combine_bitchunks_into_double(groq_result_mm, mult_result_exponent)








###########################################################################################################
# Check Result

#np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})
np.set_printoptions(precision=16)

print(' ')
print("Matrix Multiplication for input tensors of size {} x {}.  Results are: ".format(t1_data.shape, t2_data.shape))
print('test Groq chip matmul: '+str(np.allclose(mult_result, groq_result_double, rtol=1e-10, atol=1e-15, equal_nan=True)))
print('test Groq chip matmul: '+str(np.allclose(mult_result, groq_result_double3, rtol=1e-10, atol=1e-15, equal_nan=True)))
print('test Groq chip matmul: '+str(np.allclose(mult_result, groq_result_double4, rtol=1e-10, atol=1e-15, equal_nan=True)))
print(groq_result_double[0,1:4])
print(groq_result_double3[0,1:4])
print(groq_result_double4[0,1:4])
print(mult_result[0,1:4])
#print(t1_data)


