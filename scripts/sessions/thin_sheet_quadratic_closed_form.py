"""Second-order perturbation of the steady corotational Maxwell sheet on the unit
sphere driven by f = 4A rhat x grad_s(z^2) — the closed form behind
tests/test_1084_quadratic_stress_sphere.py.

eta = mu = L = 1, c = 4 (lambda = eta / mu = 1). Everything in embedding
(Cartesian) components, fields extended as polynomials, derivatives projected; the
problem is axisymmetric, so it is reduced on the meridian y = 0.

Steady first order: tau_1 = 2 eta E(u_1), u_1 = a rhat x grad_s(z^2), a = 1/2.
Second order: tau_2 = 2 eta E(u_2) - lambda P[(u_1 . grad) tau_1 - W_1 tau_1
+ tau_1 W_1] P, the plane-stress closure and the membrane balance solved for
u_2 = grad_s Phi + rhat x grad_s Psi with Phi, Psi polynomials in z.

Prints (2026-10-12): a = 1/2; -div_s u_2 = -3 (65 z^4 - 126 z^2 + 29) / 533,
i.e. P2 = 24/91, P4 = -24/287, all other degrees zero. Runs in a few minutes:

    pixi run -e dev python scripts/sessions/thin_sheet_quadratic_closed_form.py
"""
import sympy as sp
x, y, z = sp.symbols("x y z", real=True)
X = sp.Matrix([x, y, z])
eta, mu, c, L = 1, 1, 4, 1
lam = sp.Rational(eta, mu)
I3 = sp.eye(3)
P = I3 - X * X.T                     # on r = 1
S = sp.Symbol("S", positive=True)    # sin(theta); the problem is axisymmetric, so the
def onS(e):                          # meridian y = 0, x = S = sqrt(1 - z^2) is enough
    e = sp.expand(sp.sympify(e).subs({y: 0, x: S}))
    p = sp.Poly(e, S)
    out = 0
    for (k,), cf in p.terms():
        out += cf * (1 - z**2)**(k // 2) * S**(k % 2)
    return sp.expand(out)
def grad_s_vec(u):                   # (grad_s u)_ij = d_k u_i P_kj
    J = u.jacobian(X)
    return J * P
def grad_s_scalar(f):
    return P * sp.Matrix([sp.diff(f, v) for v in X])
def div_s_tensor(S):                 # (div_s S)_i = d_k S_ij P_jk
    out = sp.zeros(3, 1)
    for i in range(3):
        out[i] = sum(sp.diff(S[i, j], X[k]) * P[j, k] for j in range(3) for k in range(3))
    return out
def rcross(v):
    return X.cross(v)
def E_of(u):
    G = grad_s_vec(u)
    return P * (G + G.T) / 2 * P
def W_of(u):
    G = grad_s_vec(u)
    return P * (G - G.T) / 2 * P
def membrane(u, Q=None):
    """-P div_s(L sigma) + c u with sigma = tau + tr(tau) P, tau = 2 eta E(u) - Q."""
    tau = 2 * eta * E_of(u)
    if Q is not None:
        tau = tau - Q
    sig = tau + tau.trace() * P
    return -P * div_s_tensor(L * sig) + c * u

# first order (per unit A)
a = sp.Symbol("a")
f1 = 4 * rcross(grad_s_scalar(z**2))
u1 = a * rcross(grad_s_scalar(z**2))
res1 = (membrane(u1) - f1).applyfunc(onS); print("first order built", flush=True)
asol = sp.solve([e for e in res1 if e != 0][0], a)[0]
u1 = u1.subs(a, asol)
print("a =", asol, " residual:", (membrane(u1) - f1).applyfunc(onS).T)
tau1 = 2 * eta * E_of(u1)
W1 = W_of(u1)
adv = sp.zeros(3, 3)
for i in range(3):
    for j in range(3):
        adv[i, j] = sum(u1[k] * sp.diff(tau1[i, j], X[k]) for k in range(3))
Q = lam * P * (adv - W1 * tau1 + tau1 * W1) * P
# second order: u2 = grad_s Phi + rhat x grad_s Psi, Phi,Psi polynomials in z
cp = sp.symbols("p0:7"); cq = sp.symbols("q0:7")
Phi = sum(cp[k] * z**k for k in range(7)); Psi = sum(cq[k] * z**k for k in range(7))
u2 = grad_s_scalar(Phi) + rcross(grad_s_scalar(Psi))
res2 = (membrane(u2, Q)).applyfunc(onS)
eqs = []
for e in res2:
    eqs += sp.Poly(e, S, z).coeffs()
sol = sp.solve(eqs, list(cp[1:]) + list(cq[1:]), dict=True)
print("solutions:", sol)
s = sol[0]
Phi2 = Phi.subs(s).subs({k: 0 for k in cp + cq})
u2s = u2.subs(s).subs({k: 0 for k in cp + cq})
thin = onS(-sum(sp.diff(u2s[i], X[k]) * P[i, k] for i in range(3) for k in range(3)))
assert thin.free_symbols <= {z}
print("thinning rate / A^2 =", sp.factor(thin))
Pl = [sp.legendre(l, z) for l in range(6)]
for l in range(6):
    cl = sp.integrate(thin * Pl[l], (z, -1, 1)) / sp.integrate(Pl[l]**2, (z, -1, 1))
    print(f"P{l}: {sp.nsimplify(cl)} = {float(cl):.6f}")
