#!/usr/bin/env python

import underworld3 as uw
import numpy as np



def gather_data(val, bcast=False): 
    '''
    gather values on root (bcast=False) or all (bcast = True) processors
    Parameters:
        vals : Values to combine into a single array on the root or all processors
    
    returns:
        val_global : combination of values form all processors
    
    '''

    comm = uw.mpi.comm
    rank = uw.mpi.rank
    size = uw.mpi.size

    
    ### make sure all data comes in the same order
    with uw.mpi.call_pattern(pattern='sequential'):
        if len(val > 0):
            val_local = np.ascontiguousarray(val.copy())       
        else:
            val_local = np.array([np.nan], dtype='float64')
        
    comm.barrier()
        
            
    ### Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(val_local), root=0))
    
    
    if rank == 0:
        val_global = np.zeros((sum(sendcounts)), dtype='float64')
    else:
        val_global = None
        

    comm.barrier()
    

    ## gather x values, can't do them together
    comm.Gatherv(sendbuf=val_local, recvbuf=(val_global, sendcounts), root=0)
    
    comm.barrier()
    
    
    if uw.mpi.rank == 0:
        ### remove rows with NaN
        val_global = val_global[~np.isnan(val_global)]
    
        
    comm.barrier()
        
    if bcast == True:
        #### make swarm coords available on all processors
        val_global = comm.bcast(val_global, root=0)
        
    comm.barrier()

    return val_global







