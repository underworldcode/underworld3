## Key — what each part solved in make_figures
*run started 2026-09-15T13:18:44+10:00*

### AdvDiffusion(T)
`SNES_AdvectionDiffusion_Composed`, unknown `T`, 2-D; recorded at step 0

Residual $\int F_0\,\phi + F_1 \cdot \nabla\phi = 0$ with

$$f_0(\phi) = \left[\begin{matrix}a^{\mathrm{AM}}_{0,18} \left({T}_{,0}(\mathbf{x}) {v}_{ 0 }(\mathbf{x}) + {T}_{,1}(\mathbf{x}) {v}_{ 1 }(\mathbf{x})\right) + a^{\mathrm{AM}}_{1,18} \left({v}_{ 0 }(\mathbf{x}) {{T}^{ * }}_{,0}(\mathbf{x}) + {v}_{ 1 }(\mathbf{x}) {{T}^{ * }}_{,1}(\mathbf{x})\right) + \frac{{T}(\mathbf{x}) - {{T}^{ * }}(\mathbf{x})}{\Delta t_{18}}\end{matrix}\right]$$
*Strong residual of the time scheme: time derivative, advection and source.*

where
- $\Delta t_{18}$ $= 1.147 \times 10^{-6}$ — DDt timestep
- $a^{\mathrm{AM}}_{0,18}$ $= 0.5$ — a^{\mathrm{AM}} coefficient 0 (DDt instance 18)
- $a^{\mathrm{AM}}_{1,18}$ $= 0.5$ — a^{\mathrm{AM}} coefficient 1 (DDt instance 18)

$$\mathbf{F}_1(\phi) = \left[\begin{matrix}\upkappa a^{\mathrm{AM}}_{0,18} {T}_{,0}(\mathbf{x}) + \upkappa a^{\mathrm{AM}}_{1,18} {{T}^{ * }}_{,0}(\mathbf{x}) + \frac{w^{\mathrm{SUPG}}_{18} \left({v}_{ 0 }^{ 2 }(\mathbf{x}) + {v}_{ 1 }^{ 2 }(\mathbf{x})\right) \left(a^{\mathrm{AM}}_{0,18} \left({T}_{,0}(\mathbf{x}) {v}_{ 0 }(\mathbf{x}) + {T}_{,1}(\mathbf{x}) {v}_{ 1 }(\mathbf{x})\right) + a^{\mathrm{AM}}_{1,18} \left({v}_{ 0 }(\mathbf{x}) {{T}^{ * }}_{,0}(\mathbf{x}) + {v}_{ 1 }(\mathbf{x}) {{T}^{ * }}_{,1}(\mathbf{x})\right) + \frac{{T}(\mathbf{x}) - {{T}^{ * }}(\mathbf{x})}{\Delta t_{18}}\right) {_h_cell}^{ 2 }(\mathbf{x}) {v}_{ 0 }(\mathbf{x})}{\left(64.0 \upkappa^{2} + \left({v}_{ 0 }^{ 2 }(\mathbf{x}) + {v}_{ 1 }^{ 2 }(\mathbf{x})\right) {_h_cell}^{ 2 }(\mathbf{x}) + 1.0 \cdot 10^{-30}\right) \sqrt{\frac{\upkappa^{2} \left(C^{\tau}_{\kappa,18}\right)^{2}}{{_h_cell}^{ 4 }(\mathbf{x})} + \frac{\left(C^{\tau}_{t,18}\right)^{2}}{\Delta t_{18}^{2}} + \frac{\left(C^{\tau}_{u,18}\right)^{2} \left({v}_{ 0 }^{ 2 }(\mathbf{x}) + {v}_{ 1 }^{ 2 }(\mathbf{x})\right)}{{_h_cell}^{ 2 }(\mathbf{x})} + 1.0 \cdot 10^{-30}}} & \upkappa a^{\mathrm{AM}}_{0,18} {T}_{,1}(\mathbf{x}) + \upkappa a^{\mathrm{AM}}_{1,18} {{T}^{ * }}_{,1}(\mathbf{x}) + \frac{w^{\mathrm{SUPG}}_{18} \left({v}_{ 0 }^{ 2 }(\mathbf{x}) + {v}_{ 1 }^{ 2 }(\mathbf{x})\right) \left(a^{\mathrm{AM}}_{0,18} \left({T}_{,0}(\mathbf{x}) {v}_{ 0 }(\mathbf{x}) + {T}_{,1}(\mathbf{x}) {v}_{ 1 }(\mathbf{x})\right) + a^{\mathrm{AM}}_{1,18} \left({v}_{ 0 }(\mathbf{x}) {{T}^{ * }}_{,0}(\mathbf{x}) + {v}_{ 1 }(\mathbf{x}) {{T}^{ * }}_{,1}(\mathbf{x})\right) + \frac{{T}(\mathbf{x}) - {{T}^{ * }}(\mathbf{x})}{\Delta t_{18}}\right) {_h_cell}^{ 2 }(\mathbf{x}) {v}_{ 1 }(\mathbf{x})}{\left(64.0 \upkappa^{2} + \left({v}_{ 0 }^{ 2 }(\mathbf{x}) + {v}_{ 1 }^{ 2 }(\mathbf{x})\right) {_h_cell}^{ 2 }(\mathbf{x}) + 1.0 \cdot 10^{-30}\right) \sqrt{\frac{\upkappa^{2} \left(C^{\tau}_{\kappa,18}\right)^{2}}{{_h_cell}^{ 4 }(\mathbf{x})} + \frac{\left(C^{\tau}_{t,18}\right)^{2}}{\Delta t_{18}^{2}} + \frac{\left(C^{\tau}_{u,18}\right)^{2} \left({v}_{ 0 }^{ 2 }(\mathbf{x}) + {v}_{ 1 }^{ 2 }(\mathbf{x})\right)}{{_h_cell}^{ 2 }(\mathbf{x})} + 1.0 \cdot 10^{-30}}}\end{matrix}\right]$$
*Diffusive flux of the time scheme plus the SUPG flux tau R u.*

where
- $C^{\tau}_{\kappa,18}$ $= 4$ — tau diffusive weight
- $C^{\tau}_{t,18}$ $= 2$ — tau transient weight
- $C^{\tau}_{u,18}$ $= 2$ — tau advective weight
- $\upkappa$ $= 10^{-6}\ \mathrm{m^{2}/s}$ — Diffusivity
- $w^{\mathrm{SUPG}}_{18}$ $= 1$ — SUPG term weight (0 = Galerkin)

Boundary conditions:
- essential on Lower: $1$
- essential on Upper: $0$

Given:
- `f` $= 0$ — volumetric source term
- `V_fn` $= \left[\begin{matrix}{v}_{ 0 }(\mathbf{x}) & {v}_{ 1 }(\mathbf{x})\end{matrix}\right]$ — advecting velocity
- `DiffusionModel.diffusivity` $= 10^{-6}\ \mathrm{m^{2}/s}$ — constitutive parameter

### Stokes(v)
`SNES_Stokes`, unknown `v`, 2-D; recorded at step 0

Residual $\int F_0\,\phi + F_1 \cdot \nabla\phi = 0$ with

$$\mathbf{f}_0\left( \mathbf{u} \right) = \left[\begin{matrix}\frac{\mathrm{x} \rho_0 \alpha g {T}(\mathbf{x})}{\sqrt{\mathrm{x}^{2} + \mathrm{y}^{2}}}\\\frac{\mathrm{y} \rho_0 \alpha g {T}(\mathbf{x})}{\sqrt{\mathrm{x}^{2} + \mathrm{y}^{2}}}\end{matrix}\right]$$
*Velocity equation body force term (pointwise).*

where
- $\rho_0 \alpha g$ $= 0.9712\ \mathrm{kg/K/m^{2}/s^{2}}$ — buoyancy coefficient: reference density x thermal expansivity x gravity

$$\mathbf{F}_1\left( \mathbf{u} \right) = \left[\begin{matrix}\eta \uplambda \left({v}_{ 0,0}(\mathbf{x}) + {v}_{ 1,1}(\mathbf{x})\right) + 2 \eta {v}_{ 0,0}(\mathbf{x}) - {p}(\mathbf{x}) & 2 \eta \left(\frac{{v}_{ 0,1}(\mathbf{x})}{2} + \frac{{v}_{ 1,0}(\mathbf{x})}{2}\right)\\2 \eta \left(\frac{{v}_{ 0,1}(\mathbf{x})}{2} + \frac{{v}_{ 1,0}(\mathbf{x})}{2}\right) & \eta \uplambda \left({v}_{ 0,0}(\mathbf{x}) + {v}_{ 1,1}(\mathbf{x})\right) + 2 \eta {v}_{ 1,1}(\mathbf{x}) - {p}(\mathbf{x})\end{matrix}\right]$$
*Velocity equation flux/stress term (pointwise).*

where
- $\eta$ $= 10^{22}\ \mathrm{Pa\cdot s}$ — Shear viscosity
- $\uplambda$ $= 0$ — Numerical Penalty

$$\mathbf{h}_0\left( \mathbf{p} \right) = \left[\begin{matrix}{v}_{ 0,0}(\mathbf{x}) + {v}_{ 1,1}(\mathbf{x})\end{matrix}\right]$$
*Pressure equation constraint term (continuity).*

Boundary conditions:
- rotated free-slip on Upper: $\mathbf{u}\cdot\hat{\mathbf{n}} = 0$
- rotated free-slip on Lower: $\mathbf{u}\cdot\hat{\mathbf{n}} = 0$

Given:
- `bodyforce` $= \left[\begin{matrix}- \frac{\mathrm{x} \rho_0 \alpha g {T}(\mathbf{x})}{\sqrt{\mathrm{x}^{2} + \mathrm{y}^{2}}}\\- \frac{\mathrm{y} \rho_0 \alpha g {T}(\mathbf{x})}{\sqrt{\mathrm{x}^{2} + \mathrm{y}^{2}}}\end{matrix}\right]$ — body force per unit volume; F0 is its negative
- `penalty` $= 0$ — augmented-Lagrangian grad-div penalty (0 = off)
- `ViscousFlowModel.shear_viscosity_0` $= 10^{22}\ \mathrm{Pa\cdot s}$ — constitutive parameter
