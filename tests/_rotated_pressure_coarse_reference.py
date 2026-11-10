"""80-digit tiny-component oracle; no production matrix/vector gathering."""
from decimal import Decimal, localcontext
import numpy as np

def audit_reference(A, V, r, inverse_diagonal, y, actual_C, actual_AV, actual_ATV, rs, s, operation):
    with localcontext() as ctx:
        ctx.prec = 80
        D = lambda x: Decimal.from_float(float(x))
        zero = Decimal(0)
        a = [[D(x) for x in row] for row in A]
        cols = [[D(x) for x in col] for col in V.T]
        dr = [D(x) for x in r]
        dy = [D(x) for x in y]
        inv = [D(x) for x in inverse_diagonal]
        dot = lambda x, z: sum((u * v for u, v in zip(x, z)), zero)
        mv = lambda M, x: [dot(row, x) for row in M]
        transpose = lambda M: list(map(list, zip(*M)))
        absolute = lambda M: [[abs(x) for x in row] for row in M]
        mat = lambda M: [[D(x) for x in row] for row in M]
        combine = lambda b: [cols[0][i] * b[0] + cols[1][i] * b[1] for i in range(11)]
        exact_AV = [mv(a, col) for col in cols]
        exact_ATV = [mv(transpose(a), col) for col in cols]
        exact_C = [[dot(left, right) for right in exact_AV] for left in cols]
        rounded_C = mat(actual_C)

        def solve(C, b):
            determinant = C[0][0] * C[1][1] - C[0][1] * C[1][0]
            assert determinant != 0
            return [(b[0] * C[1][1] - C[0][1] * b[1]) / determinant, (C[0][0] * b[1] - b[0] * C[1][0]) / determinant]

        def balanced(C):
            yc = combine(solve(C, [dot(col, dr) for col in cols]))
            Ayc = mv(a, yc)
            ys = [inv[i] * (dr[i] - Ayc[i]) for i in range(11)]
            Ays = mv(a, ys)
            correction = combine(solve(C, [dot(col, Ays) for col in cols]))
            return np.array([float(yc[i] + ys[i] - correction[i]) for i in range(11)])

        def ratios(errors, scales):
            return [float(abs(e) / scale) if scale else 0.0 if e == 0 else float('inf') for e, scale in zip(errors, scales)]
        cscale = [[dot([abs(x) for x in left], mv(absolute(a), [abs(x) for x in right])) for right in cols] for left in cols]
        ceta = [ratios([rounded_C[i][j] - exact_C[i][j] for j in range(2)], cscale[i]) for i in range(2)]
        residual = [value - dr[i] for i, value in enumerate(mv(a, dy))]
        scale = [value + abs(dr[i]) for i, value in enumerate(mv(absolute(a), [abs(x) for x in dy]))]
        eta = ratios([dot(col, residual) for col in cols], [dot([abs(x) for x in col], scale) for col in cols])
        av = mat(actual_AV)
        atv = mat(actual_ATV)
        drs = [D(x) for x in rs]
        ds = [D(x) for x in s]
        q = [D(x) for x in operation['q']]
        alpha = [D(x) for x in operation['alpha']]
        beta_rhs = [D(x) for x in operation['beta_rhs']]
        beta = [D(x) for x in operation['beta']]
        av_eta = []
        atv_eta = []
        for j, col in enumerate(cols):
            av_eta += ratios([av[i][j] - exact_AV[j][i] for i in range(11)], mv(absolute(a), [abs(x) for x in col]))
            atv_eta += ratios([atv[i][j] - exact_ATV[j][i] for i in range(11)], mv(transpose(absolute(a)), [abs(x) for x in col]))
        rs_ref = [dr[i] - dot(av[i], alpha) for i in range(11)]
        y_ref = [ds[i] + cols[0][i] * (alpha[0] - beta[0]) + cols[1][i] * (alpha[1] - beta[1]) for i in range(11)]
        component = dict(cached_AV_backward=max(av_eta), cached_ATV_backward=max(atv_eta), q_dot_backward=max(ratios([q[i] - dot(cols[i], dr) for i in range(2)], [dot([abs(x) for x in col], [abs(x) for x in dr]) for col in cols])), alpha_solve_backward=max(ratios([x - q[i] for i, x in enumerate(mv(rounded_C, alpha))], [x + abs(q[i]) for i, x in enumerate(mv(absolute(rounded_C), [abs(x) for x in alpha]))])), rs_axpy_backward=max(ratios([drs[i] - rs_ref[i] for i in range(11)], [abs(dr[i]) + dot([abs(x) for x in av[i]], [abs(x) for x in alpha]) for i in range(11)])), M0_apply_backward=max(ratios([ds[i] - inv[i] * drs[i] for i in range(11)], [abs(inv[i] * drs[i]) for i in range(11)])), beta_rhs_backward=max(ratios([beta_rhs[j] - dot([atv[i][j] for i in range(11)], ds) for j in range(2)], [dot([abs(atv[i][j]) for i in range(11)], [abs(x) for x in ds]) for j in range(2)])), beta_solve_backward=max(ratios([x - beta_rhs[i] for i, x in enumerate(mv(rounded_C, beta))], [x + abs(beta_rhs[i]) for i, x in enumerate(mv(absolute(rounded_C), [abs(x) for x in beta]))])), grouping_backward=max(ratios([dy[i] - y_ref[i] for i in range(11)], [abs(ds[i]) + sum((abs(cols[j][i]) * (abs(alpha[j]) + abs(beta[j])) for j in range(2)), zero) for i in range(11)])))
        metrics = dict(decimal80_C_assembly_backward=ceta, decimal80_C_assembly_scales=[[float(x) for x in row] for row in cscale], decimal80_actual_result_backward=eta, operation_component_backward=component, old_grouping_beta_lost=[float(operation['alpha'][i] - operation['beta'][i]) == operation['alpha'][i] and operation['beta'][i] != 0 for i in range(2)])
        return (balanced(rounded_C), balanced(exact_C), metrics)
