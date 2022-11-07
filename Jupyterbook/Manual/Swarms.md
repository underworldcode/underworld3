# Particle Swarms

## Swarm Variables

Swarm variables are piecewise continuous functions that cannot be integrated directly
with the current pointwise-function requirements of PETScDS. Instead we map the
piecewise function to a continuous, mesh variable which we refer to as a *proxy* swarm
variable.

$$ \int_\Omega \phi u_n  = \int_\Omega \phi u_p $$

The left hand side of this equation can be assembled using the standard finite element
machinery of PETScDS, but, the right hand side must be integrated in a manner that accounts
for the piecewise nature of the swarm variable information.

One way in which we can build the right hand side is to use an inverse-distance weighted interpolation to an intermediary mesh variable (of comparable spatial density to the swarm, or higher polynomial degree). Alternatively, we can form a piecewise-pointwise function that matches the PETScDS template, although, strictly, this falls outside the restrictions imposed by that framework.

At first it can seem over-complicated to introduce another equation system that needs to be solved in order to use Lagrangian particle information interchangeably with mesh variables but there are some distinct advantages. In the traditional particle-in-cell or material-point finite element method, there is an ambiguity in the way in which we map Lagrangian variables to their mesh equivalents. We rely on an average over the path of a particle to cancel fluctuations that occur due to uneven particle distributions relative to integration points or nodal points. There is no consistent way for us to ensure that constraints are correctly applied, boundary conditions are satisfied or null-space modes are damped. However, these constraints can all be included in the projection equation to ensure that the proxy variables are compatible with the other mesh-variables used to build systems of equations.

In the code, we substitute proxy variables at the just-in-time compilation stage automatically.

*NOTE: - currently we don't really do that, but we probably should: swarm variable, boundary conditions, bespoke integration ... do all that in each update phase.*

