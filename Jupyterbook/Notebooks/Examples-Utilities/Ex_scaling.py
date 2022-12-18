# %% [markdown]
# # Underworld scaling example
#
# How to utilise the scaling functionality that is included with UW to easily convert between dimensional and non-dimensional values

# %% [markdown]
#
#

# %%
import numpy as np
import underworld3 as uw


# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim = uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

# %%
### reference values
length         = 100.                      #* u.kilometer
kappa          = 1e-6                      #* u.meter**2/u.second
g              = 9.81                      #* u.meter/u.second**2
v              = 1                         # u.centimeter/u.year, velocity in cm/yr
alpha          = 3.0e-5                    # *1./u.kelvin
tempMax        = 1573.15                   # * u.kelvin
tempMin        = 273.15                    #* u.kelvin
rho0           = 3300.0                    #* u.kilogram / u.metre**3#  * 9.81 * u.meter / u.second**2
R              = 8.3145                    # [J/(K.mol)], gas constant

# %% [markdown]
# Create the fundamental values required to obtain scaling for all units

# %%
lengthScale   = length  * u.kilometer
surfaceTemp   = tempMin * u.degK
baseModelTemp = tempMax * u.degK
bodyforce     = rho0    * u.kilogram / u.metre**3 * g * u.meter / u.second**2

half_rate     = v * u.centimeter / u.year

KL = lengthScale.to_base_units()
Kt = (KL / half_rate).to_base_units()
KM = (bodyforce * KL**2 * Kt**2).to_base_units()
KT = (baseModelTemp - surfaceTemp).to_base_units()


# %%
scaling_coefficients                  = uw.scaling.get_coefficients()

scaling_coefficients["[length]"]      = KL.to_base_units()
scaling_coefficients["[time]"]        = Kt.to_base_units()
scaling_coefficients["[mass]"]        = KM.to_base_units()
scaling_coefficients["[temperature]"] = KT.to_base_units()


scaling_coefficients

# %%
### fundamental values
ref_length    = uw.scaling.dimensionalise(1., u.meter).magnitude

ref_length_km = uw.scaling.dimensionalise(1., u.kilometer).magnitude

ref_density   =  uw.scaling.dimensionalise(1., u.kilogram/u.meter**3).magnitude

ref_gravity   = uw.scaling.dimensionalise(1., u.meter/u.second**2).magnitude

ref_temp      = uw.scaling.dimensionalise(1., u.kelvin).magnitude

ref_velocity  = uw.scaling.dimensionalise(1., u.meter/u.second).magnitude

### derived values
ref_time      = ref_length / ref_velocity

ref_pressure  = ref_density * ref_gravity * ref_length

ref_stress    = ref_pressure

ref_viscosity = ref_pressure * ref_time

### Key ND values
ND_diffusivity = kappa        / (ref_length**2/ref_time)
ND_gravity     = g            / ref_gravity


# %%
if uw.mpi.rank == 0:
    print(f'time scaling: {ref_time/(60*60*24*365.25*1e6)} [Myr]')
    print(f'pressure scaling: {ref_pressure/1e6} [MPa]')
    print(f'viscosity scaling: {ref_viscosity} [Pa s]')
    print(f'velocity scaling: {ref_velocity*(1e2*60*60*24*365.25)} [cm/yr]')
    print(f'length scaling: {ref_length/1e3} [km]')

# %% [markdown]
# ### How to non-dimensionalise a value

# %%
cohesion = 10e6

# %%
### first way, using the UW non-dimensionalise function, have to provide the units
nd_cohesion = ndim(cohesion * u.pascal)
nd_cohesion

# %%
#### second way, just divide cohesion by the scaled reference value
cohesion/ref_stress

# %%
#### check if they are close
np.isclose(ndim(cohesion * u.pascal), (cohesion/ref_pressure), rtol=1e-10, atol=1e-10)

# %%
viscosity = 1e21

# %%
nd_visc = ndim(viscosity * u.pascal*u.second)

nd_visc

# %%
viscosity / ref_viscosity

# %%
np.isclose(ndim(viscosity * u.pascal*u.second), (viscosity / ref_viscosity), rtol=1e-10, atol=1e-10)

# %% [markdown]
# # How to dimensionalise

# %%
### can either use the inbuilt function, where you provide the units

dim(nd_visc, u.pascal*u.second)

# %%
#### or multiply by the ref value which has units of Pa s
nd_visc * ref_viscosity

# %%
### you can remove the units by using the magnitude function 
print(dim(nd_visc, u.pascal*u.second).magnitude)

#### or use .m (short for .magnitude)
print(dim(nd_visc, u.pascal*u.second).m)

# %%
#### check if they give the same answers
np.isclose(dim(nd_visc, u.pascal*u.second).m, nd_visc * ref_viscosity,  rtol=1e-10, atol=1e-10)

# %% [markdown]
# ### check mesh and swarm vars work
#
# Currently not supported

# %%
# Set the resolution.
res = 32
xmin, xmax = 0, 1
ymin, ymax = 0, 1

mesh = uw.meshing.StructuredQuadBox(elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))



# %%
dim(mesh.data, u.kilometer)

# %%
mesh.data

# %%
