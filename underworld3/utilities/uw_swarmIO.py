#!/usr/bin/env python

import h5py
import numpy as np
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank


def swarm_h5(swarm, timestep, fields=None, outputPath=''):
    '''
    Function to save swarm fields to h5 file for checkpointing purposes.
    
    swarm : The swarm that contains the coordinate data
    fields : UW swarm variables that contain the data to save. Should be passed as a list.
    timestep : timestep of the model to save
    outputPath : Folder to save the data, can be left undefined to save in the current working directory
    
    >>> 
    example usage:
    
    swarm_h5(swarm=swarm, fields=[material, strainRate], timestep=0)
    
    <<<<
    
    '''
    
    if h5py.h5.get_config().mpi == False and size > 1:
        import warnings
        warnings.warn("Collective IO not possible as h5py not available in parallel mode. Switching to sequential. This will be slow for models running on multiple processors")

    if not isinstance(fields, list):
        raise RuntimeError("`swarm_h5()` function parameter `fields` does not appear to be a list.")

        
    # elif h5py.h5.get_config().mpi == False and uw.mpi.size > 1:
    #     raise RuntimeError("Unable to save swarm in parallel due to hdf5 installed for serial IO only. To check, 'h5py.h5.get_config().mpi' should return 'True' for parallel IO")
        
    else:     
        ### save the swarm particle location
        swarm.save(filename=f'{outputPath}swarm-{timestep:04d}.h5')

        #### Generate a h5 file for each field
        if fields != None:
            for field in fields:
                field.save(filename=f'{outputPath}{field.name}-{timestep:04d}.h5')


def swarm_xdmf(timestep, fields=None, outputPath='', time=None):
    # Create an XDMF file
    '''
    Function to combine h5 files to a single xdmf to visualise the swarm in paraview. Should be run after the 'swarm_h5' function.
    
    fields : UW swarm variables that contain the data to save. Should be passed as a list.
    timestep : timestep of the model to save
    outputPath : Folder to save the data, can be left undefined to save in the current working directory. This directory should also contain the h5 files
    time       : Time of the model, can be left undefined and model output will be sequential but not contain the time data.
    
    >>> 
    example usage:
    
    swarm_xdmf(fields=[material, strainRate], timestep=0)
    
    <<<<
    
    '''
    if not isinstance(fields, list):
        raise RuntimeError("`swarm_xdmf()` function parameter `fields` does not appear to be a list.")
    
    for field in fields:
        if not os.path.exists(f'{outputPath}{field.name}-{timestep:04d}.h5'):
            raise RuntimeError(f"`swarm_xdmf()` could not find '{field.name}-{timestep:04d}.h5'.") 
            
    if rank == 0:
        ''' only need to combine the h5 to a single xdmf on one proc '''
        
        with open(f"{outputPath}swarm-{timestep:04d}.xmf", "w") as xdmf:
            # Write the XDMF header
            xdmf.write('<?xml version="1.0" ?>\n')
            xdmf.write('<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n')
            xdmf.write('<Domain>\n')
            xdmf.write(f'<Grid Name="swarm-{timestep:04d}" GridType="Uniform">\n')    


            if time != None:
                 xdmf.write(f'	<Time Value="{time}" />\n')



            # Write the grid element for the HDF5 dataset
            with h5py.File(f"{outputPath}swarm-{timestep:04}.h5", "r") as h5f:
                xdmf.write(f'	<Topology Type="POLYVERTEX" NodesPerElement="{h5f["coordinates"].shape[0]}"> </Topology>\n')
                if h5f['coordinates'].shape[1] == 2:
                    xdmf.write('		<Geometry Type="XY">\n')
                elif h5f['coordinates'].shape[1] == 3:
                    xdmf.write('		<Geometry Type="XYZ">\n') 
                xdmf.write(f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["coordinates"].shape[0]} {h5f["coordinates"].shape[1]}">{os.path.basename(h5f.filename)}:/coordinates</DataItem>\n')
                xdmf.write('		</Geometry>\n')

            # Write the attribute element for the field
            if fields != None:
                for field in fields:
                    with h5py.File(f'{outputPath}{field.name}-{timestep:04d}.h5', "r") as h5f:
                        if h5f['data'].dtype == np.int32:
                            xdmf.write(f'	<Attribute Type="Scalar" Center="Node" Name="{field.name}">\n')
                            xdmf.write(f'			<DataItem Format="HDF" NumberType="Int" Precision="4" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n')
                        elif h5f['data'].shape[1] == 1:
                            xdmf.write(f'	<Attribute Type="Scalar" Center="Node" Name="{field.name}">\n')
                            xdmf.write(f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n')
                        elif h5f['data'].shape[1] == 2 or h5f['data'].shape[1] == 3:
                            xdmf.write(f'	<Attribute Type="Vector" Center="Node" Name="{field.name}">\n')
                            xdmf.write(f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n')
                        else:
                            xdmf.write(f'	<Attribute Type="Tensor" Center="Node" Name="{field.name}">\n')
                            xdmf.write(f'			<DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{h5f["data"].shape[0]} {h5f["data"].shape[1]}">{os.path.basename(h5f.filename)}:/data</DataItem>\n')

                        xdmf.write('	</Attribute>\n')
            else:
                pass


            # Write the XDMF footer
            xdmf.write('</Grid>\n')
            xdmf.write('</Domain>\n')
            xdmf.write('</Xdmf>\n')







