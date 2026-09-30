/*
  Velic SolKx: exact Stokes solution for an exponentially varying viscosity.

  Vendored verbatim from PETSc src/snes/tutorials/ex69.c (SolKxSolution), which
  carries the following notice. It is transcription source, not built code —
  underworld3.analytic.velic reads it to rebuild the solution in SymPy.

  Copyright (c) 1991-2025, UChicago Argonne, LLC and the PETSc Developers and
  Contributors. All rights reserved. Redistribution and use in source and binary
  forms, with or without modification, are permitted provided that the above
  copyright notice and this list of conditions are retained. THIS SOFTWARE IS
  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
  IMPLIED WARRANTIES ARE DISCLAIMED. (BSD-2-Clause; see the PETSc LICENSE.)

  The domain is the unit square with free slip everywhere. The forcing is
  fx = 0, fz = sigma*sin(km*z)*cos(kn*x) with km = m*Pi, kn = n*Pi and sigma = 1,
  and the viscosity is eta = exp(2*B*x).
*/

static PetscErrorCode SolKxSolution(const PetscReal pos[], PetscReal m, PetscInt n, PetscReal B, PetscScalar vel[], PetscScalar *p, PetscScalar s[], PetscScalar gamma[], PetscScalar *mu)
{
  PetscReal sigma = 1.0;
  PetscReal Z;
  PetscReal u1, u2, u3, u4, u5, u6;
  PetscReal sum1, sum2, sum3, sum4, sum5, sum6;
  PetscReal kn, km, x, z;
  PetscReal _PC1, _PC2, _PC3, _PC4;
  PetscReal Rp, UU, VV;
  PetscReal a, b, r, _aa, _bb, AA, BB, Rm;
  PetscReal num1, num2, num3, num4, den1;

  PetscReal t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
  PetscReal t11, t12, t13, t14, t15, t16, t17, t18, t19, t20;
  PetscReal t21, t22, t23, t24, t25, t26, t27, t28, t29, t30;
  PetscReal t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;
  PetscReal t41, t42, t43, t44, t45, t46, t47, t49, t51, t53;
  PetscReal t56, t58, t61, t62, t63, t64, t65, t66, t67, t68;
  PetscReal t69, t70, t71, t72, t73, t74, t75, t76, t77, t78;
  PetscReal t79, t80, t81, t82, t83, t84, t85, t86, t87, t88;
  PetscReal t89, t90, t91, t92, t93, t94, t95, t96, t97, t99;
  PetscReal t100, t101, t103, t104, t105, t106, t107, t108, t109, t110;
  PetscReal t111, t112, t113, t114, t115, t116, t117, t118, t119, t120;
  PetscReal t121, t124, t125, t126, t127, t129, t130, t132, t133, t135;
  PetscReal t136, t138, t140, t141, t142, t143, t152, t160, t162;

  PetscFunctionBegin;
  /*************************************************************************/
  /*************************************************************************/
  /* rho = -sin(km*z)*cos(kn*x) */
  x  = pos[0];
  z  = pos[1];
  Z  = PetscExpReal(2.0 * B * x);
  km = m * PETSC_PI; /* solution valid for km not zero -- should get trivial solution if km=0 */
  kn = (PetscReal)n * PETSC_PI;
  /*************************************************************************/
  /*************************************************************************/
  a  = B * B + km * km;
  b  = 2.0 * km * B;
  r  = PetscSqrtReal(a * a + b * b);
  Rp = PetscSqrtReal((r + a) / 2.0);
  Rm = PetscSqrtReal((r - a) / 2.0);
  UU = Rp - B;
  VV = Rp + B;

  sum1 = 0.0;
  sum2 = 0.0;
  sum3 = 0.0;
  sum4 = 0.0;
  sum5 = 0.0;
  sum6 = 0.0;
  /*sum7=0.0;*/

  /*******************************************/
  /*         calculate the constants         */
  /*******************************************/

  t1  = kn * kn;
  t4  = km * km;
  t5  = t4 + t1;
  t6  = t5 * t5;
  t8  = pow(km + kn, 0.2e1);
  t9  = B * B;
  t16 = pow(km - kn, 0.2e1);
  _aa = -0.4e1 * B * t1 * sigma * t5 / (t6 + 0.4e1 * t8 * t9) / (t6 + 0.4e1 * t16 * t9);

  t2  = km * km;
  t3  = kn * kn;
  t5  = pow(t2 + t3, 0.2e1);
  t6  = km - kn;
  t7  = km + kn;
  t9  = B * B;
  t13 = t7 * t7;
  t19 = t6 * t6;
  _bb = sigma * kn * (t5 + 0.4e1 * t6 * t7 * t9) / (t5 + 0.4e1 * t13 * t9) / (t5 + 0.4e1 * t19 * t9);

  AA = _aa;
  BB = _bb;

  /*******************************************/
  /*       calculate the velocities etc      */
  /*******************************************/
  t1   = Rm * Rm;
  t2   = B - Rp;
  t4   = Rp + B;
  t6   = UU * x;
  t9   = PetscExpReal(t6 - 0.4e1 * Rp);
  t13  = B * B;
  t16  = Rp * t1;
  t18  = Rp * Rp;
  t19  = B * t18;
  t20  = t13 * Rp;
  t22  = kn * kn;
  t24  = B * t1;
  t32  = 0.8e1 * t13 * BB * kn * Rp;
  t34  = 0.2e1 * Rm;
  t35  = PetscCosReal(t34);
  t37  = Rp * Rm;
  t49  = PetscSinReal(t34);
  t63  = PetscExpReal(t6 - 0.2e1 * Rp);
  t65  = Rm * t2;
  t67  = 0.2e1 * B * kn;
  t68  = B * Rm;
  t69  = t67 + t68 + t37;
  t73  = 0.3e1 * t13;
  t75  = 0.2e1 * B * Rp;
  t76  = t73 - t75 + t1 - t22 - t18;
  t78  = t65 * t76 * BB;
  t80  = Rm - kn;
  t81  = PetscCosReal(t80);
  t83  = -t67 + t68 + t37;
  t88  = Rm + kn;
  t89  = PetscCosReal(t88);
  t92  = t65 * t76 * AA;
  t97  = PetscSinReal(t80);
  t103 = PetscSinReal(t88);
  t108 = PetscExpReal(t6 - 0.3e1 * Rp - B);
  t110 = Rm * t4;
  t111 = t67 + t68 - t37;
  t115 = t73 + t75 + t1 - t22 - t18;
  t117 = t110 * t115 * BB;
  t120 = -t67 + t68 - t37;
  t127 = t110 * t115 * AA;
  t140 = PetscExpReal(t6 - Rp - B);
  num1 = -0.4e1 * t1 * t2 * t4 * AA * t9 + ((0.2e1 * Rp * (0.3e1 * t13 * B - 0.2e1 * t16 - t19 - 0.2e1 * t20 - B * t22 - t24) * AA - t32) * t35 + (0.2e1 * t37 * (t1 + 0.5e1 * t13 - t22 - t18) * AA - 0.8e1 * B * BB * kn * Rm * Rp) * t49 - 0.2e1 * B * (0.3e1 * t20 - Rp * t22 - t18 * Rp - 0.2e1 * t19 - t16 - 0.2e1 * t24) * AA + t32) * t63 + ((0.2e1 * t65 * t69 * AA + t78) * t81 + (0.2e1 * t65 * t83 * AA - t78) * t89 + (t92 - 0.2e1 * t65 * t69 * BB) * t97 + (t92 + 0.2e1 * t65 * t83 * BB) * t103) * t108 + ((-0.2e1 * t110 * t111 * AA - t117) * t81 + (-0.2e1 * t110 * t120 * AA + t117) * t89 + (-t127 + 0.2e1 * t110 * t111 * BB) * t97 + (-t127 - 0.2e1 * t110 * t120 * BB) * t103) * t140;

  t1   = Rp + B;
  t2   = Rm * t1;
  t3   = B * B;
  t4   = 0.3e1 * t3;
  t5   = B * Rp;
  t7   = Rm * Rm;
  t8   = kn * kn;
  t9   = Rp * Rp;
  t10  = t4 + 0.2e1 * t5 + t7 - t8 - t9;
  t12  = t2 * t10 * AA;
  t14  = B * Rm;
  t20  = UU * x;
  t23  = PetscExpReal(t20 - 0.4e1 * Rp);
  t25  = Rp * Rm;
  t32  = Rm * kn;
  t37  = 0.2e1 * Rm;
  t38  = PetscCosReal(t37);
  t40  = t3 * B;
  t44  = B * t9;
  t45  = t3 * Rp;
  t53  = t3 * BB;
  t58  = PetscSinReal(t37);
  t69  = PetscExpReal(t20 - 0.2e1 * Rp);
  t72  = 0.3e1 * t40 * Rm;
  t73  = t9 * Rp;
  t74  = t73 * Rm;
  t75  = t7 * Rm;
  t76  = B * t75;
  t77  = t14 * t8;
  t78  = Rp * t75;
  t80  = 0.8e1 * t45 * kn;
  t81  = t25 * t8;
  t83  = 0.5e1 * t45 * Rm;
  t84  = t44 * Rm;
  t85  = t72 - t74 + t76 - t77 + t78 + t80 - t81 + t83 + t84;
  t88  = 0.2e1 * t9 * t3;
  t90  = 0.3e1 * t40 * Rp;
  t91  = t7 * t3;
  t93  = 0.2e1 * t5 * t32;
  t94  = t5 * t7;
  t95  = t5 * t8;
  t96  = B * t73;
  t97  = t7 * t9;
  t100 = 0.2e1 * t3 * Rm * kn;
  t101 = -t88 + t90 - t91 - t93 - t94 - t95 - t96 - t97 - t100;
  t105 = Rm - kn;
  t106 = PetscCosReal(t105);
  t108 = t72 - t80 + t83 + t76 + t84 - t81 - t74 + t78 - t77;
  t110 = -t97 - t96 - t88 + t100 + t90 - t95 + t93 - t91 - t94;
  t114 = Rm + kn;
  t115 = PetscCosReal(t114);
  t121 = PetscSinReal(t105);
  t127 = PetscSinReal(t114);
  t132 = PetscExpReal(t20 - 0.3e1 * Rp - B);
  t135 = 0.2e1 * B * kn;
  t136 = t135 + t14 - t25;
  t142 = -t135 + t14 - t25;
  t152 = t2 * t10 * BB;
  t162 = PetscExpReal(t20 - Rp - B);
  num2 = (0.2e1 * t12 - 0.8e1 * t14 * kn * t1 * BB) * t23 + ((-0.2e1 * t25 * (t7 + 0.5e1 * t3 - t8 - t9) * AA + 0.8e1 * B * BB * t32 * Rp) * t38 + (0.2e1 * Rp * (0.3e1 * t40 - 0.2e1 * Rp * t7 - t44 - 0.2e1 * t45 - B * t8 - B * t7) * AA - 0.8e1 * t53 * kn * Rp) * t58 - 0.2e1 * t14 * (-t8 + t9 + t4 + t7) * AA + 0.8e1 * t53 * t32) * t69 + ((-t85 * AA - 0.2e1 * t101 * BB) * t106 + (-t108 * AA + 0.2e1 * t110 * BB) * t115 + (-0.2e1 * t101 * AA + t85 * BB) * t121 + (-0.2e1 * t110 * AA - t108 * BB) * t127) * t132 + ((t12 - 0.2e1 * t2 * t136 * BB) * t106 + (t12 + 0.2e1 * t2 * t142 * BB) * t115 + (-0.2e1 * t2 * t136 * AA - t152) * t121 + (-0.2e1 * t2 * t142 * AA + t152) * t127) * t162;

  t1   = Rm * Rm;
  t2   = B - Rp;
  t4   = Rp + B;
  t6   = VV * x;
  t7   = PetscExpReal(-t6);
  t11  = kn * kn;
  t13  = B * t1;
  t14  = Rp * Rp;
  t15  = B * t14;
  t16  = B * B;
  t17  = t16 * Rp;
  t21  = Rp * t1;
  t30  = 0.8e1 * t16 * BB * kn * Rp;
  t32  = 0.2e1 * Rm;
  t33  = PetscCosReal(t32);
  t35  = Rp * Rm;
  t47  = PetscSinReal(t32);
  t61  = PetscExpReal(-t6 - 0.2e1 * Rp);
  t63  = Rm * t2;
  t65  = 0.2e1 * B * kn;
  t66  = B * Rm;
  t67  = t65 + t66 + t35;
  t71  = 0.3e1 * t16;
  t73  = 0.2e1 * B * Rp;
  t74  = t71 - t73 + t1 - t11 - t14;
  t76  = t63 * t74 * BB;
  t78  = Rm - kn;
  t79  = PetscCosReal(t78);
  t81  = -t65 + t66 + t35;
  t86  = Rm + kn;
  t87  = PetscCosReal(t86);
  t90  = t63 * t74 * AA;
  t95  = PetscSinReal(t78);
  t101 = PetscSinReal(t86);
  t106 = PetscExpReal(-t6 - 0.3e1 * Rp - B);
  t108 = Rm * t4;
  t109 = t65 + t66 - t35;
  t113 = t71 + t73 + t1 - t11 - t14;
  t115 = t108 * t113 * BB;
  t118 = -t65 + t66 - t35;
  t125 = t108 * t113 * AA;
  t138 = PetscExpReal(-t6 - Rp - B);
  num3 = -0.4e1 * t1 * t2 * t4 * AA * t7 + ((-0.2e1 * Rp * (-B * t11 - t13 - t15 + 0.2e1 * t17 + 0.3e1 * t16 * B + 0.2e1 * t21) * AA + t30) * t33 + (-0.2e1 * t35 * (t1 + 0.5e1 * t16 - t11 - t14) * AA + 0.8e1 * B * BB * kn * Rm * Rp) * t47 + 0.2e1 * B * (0.3e1 * t17 - t21 + 0.2e1 * t15 + 0.2e1 * t13 - Rp * t11 - t14 * Rp) * AA - t30) * t61 + ((-0.2e1 * t63 * t67 * AA - t76) * t79 + (-0.2e1 * t63 * t81 * AA + t76) * t87 + (-t90 + 0.2e1 * t63 * t67 * BB) * t95 + (-t90 - 0.2e1 * t63 * t81 * BB) * t101) * t106 + ((0.2e1 * t108 * t109 * AA + t115) * t79 + (0.2e1 * t108 * t118 * AA - t115) * t87 + (t125 - 0.2e1 * t108 * t109 * BB) * t95 + (t125 + 0.2e1 * t108 * t118 * BB) * t101) * t138;

  t1   = B - Rp;
  t2   = Rm * t1;
  t3   = B * B;
  t4   = 0.3e1 * t3;
  t5   = B * Rp;
  t7   = Rm * Rm;
  t8   = kn * kn;
  t9   = Rp * Rp;
  t10  = t4 - 0.2e1 * t5 + t7 - t8 - t9;
  t12  = t2 * t10 * AA;
  t14  = B * Rm;
  t20  = VV * x;
  t21  = PetscExpReal(-t20);
  t23  = Rp * Rm;
  t30  = Rm * kn;
  t35  = 0.2e1 * Rm;
  t36  = PetscCosReal(t35);
  t40  = B * t9;
  t41  = t3 * Rp;
  t43  = t3 * B;
  t51  = t3 * BB;
  t56  = PetscSinReal(t35);
  t67  = PetscExpReal(-t20 - 0.2e1 * Rp);
  t70  = 0.2e1 * B * kn;
  t71  = t70 + t14 + t23;
  t76  = Rm - kn;
  t77  = PetscCosReal(t76);
  t79  = -t70 + t14 + t23;
  t84  = Rm + kn;
  t85  = PetscCosReal(t84);
  t91  = t2 * t10 * BB;
  t93  = PetscSinReal(t76);
  t99  = PetscSinReal(t84);
  t104 = PetscExpReal(-t20 - 0.3e1 * Rp - B);
  t107 = 0.3e1 * t43 * Rm;
  t108 = t9 * Rp;
  t109 = t108 * Rm;
  t110 = t7 * Rm;
  t111 = B * t110;
  t112 = t14 * t8;
  t113 = Rp * t110;
  t115 = 0.8e1 * t41 * kn;
  t116 = t23 * t8;
  t118 = 0.5e1 * t41 * Rm;
  t119 = t40 * Rm;
  t120 = t107 + t109 + t111 - t112 - t113 - t115 + t116 - t118 + t119;
  t124 = 0.2e1 * t3 * Rm * kn;
  t125 = t5 * t8;
  t126 = B * t108;
  t127 = t7 * t9;
  t129 = 0.2e1 * t9 * t3;
  t130 = t5 * t7;
  t132 = 0.3e1 * t43 * Rp;
  t133 = t7 * t3;
  t135 = 0.2e1 * t5 * t30;
  t136 = t124 - t125 - t126 + t127 + t129 - t130 + t132 + t133 - t135;
  t141 = t107 + t115 - t118 + t111 + t119 + t116 + t109 - t113 - t112;
  t143 = t132 + t129 - t125 + t133 + t127 - t124 - t130 - t126 + t135;
  t160 = PetscExpReal(-t20 - Rp - B);
  num4 = (0.2e1 * t12 - 0.8e1 * t14 * kn * t1 * BB) * t21 + ((0.2e1 * t23 * (t7 + 0.5e1 * t3 - t8 - t9) * AA - 0.8e1 * B * BB * t30 * Rp) * t36 + (-0.2e1 * Rp * (-B * t8 - B * t7 - t40 + 0.2e1 * t41 + 0.3e1 * t43 + 0.2e1 * Rp * t7) * AA + 0.8e1 * t51 * kn * Rp) * t56 - 0.2e1 * t14 * (-t8 + t9 + t4 + t7) * AA + 0.8e1 * t51 * t30) * t67 + ((t12 - 0.2e1 * t2 * t71 * BB) * t77 + (t12 + 0.2e1 * t2 * t79 * BB) * t85 + (-0.2e1 * t2 * t71 * AA - t91) * t93 + (-0.2e1 * t2 * t79 * AA + t91) * t99) * t104 + ((-t120 * AA + 0.2e1 * t136 * BB) * t77 + (-t141 * AA - 0.2e1 * t143 * BB) * t85 + (0.2e1 * t136 * AA + t120 * BB) * t93 + (0.2e1 * t143 * AA - t141 * BB) * t99) * t160;

  t1   = Rm * Rm;
  t2   = Rp * Rp;
  t3   = t1 * t2;
  t4   = B * B;
  t5   = t1 * t4;
  t9   = PetscExpReal(-0.4e1 * Rp);
  t15  = PetscCosReal(0.2e1 * Rm);
  t22  = PetscExpReal(-0.2e1 * Rp);
  den1 = (-0.4e1 * t3 + 0.4e1 * t5) * t9 + ((0.8e1 * t1 + 0.8e1 * t4) * t2 * t15 - 0.8e1 * t5 - 0.8e1 * t2 * t4) * t22 - 0.4e1 * t3 + 0.4e1 * t5;

  _PC1 = num1 / den1;
  _PC2 = num2 / den1;
  _PC3 = num3 / den1;
  _PC4 = num4 / den1;

  t1  = Rm * x;
  t2  = PetscCosReal(t1);
  t4  = PetscSinReal(t1);
  t10 = PetscExpReal(-0.2e1 * x * B);
  t12 = kn * x;
  t13 = PetscCosReal(t12);
  t16 = PetscSinReal(t12);
  u1  = -km * (_PC1 * t2 + _PC2 * t4 + _PC3 * t2 + _PC4 * t4 + t10 * AA * t13 + t10 * BB * t16);

  t2  = Rm * x;
  t3  = PetscCosReal(t2);
  t6  = PetscSinReal(t2);
  t22 = PetscExpReal(-0.2e1 * x * B);
  t23 = B * t22;
  t24 = kn * x;
  t25 = PetscCosReal(t24);
  t29 = PetscSinReal(t24);
  u2  = UU * _PC1 * t3 + UU * _PC2 * t6 - _PC1 * t6 * Rm + _PC2 * t3 * Rm - VV * _PC3 * t3 - VV * _PC4 * t6 - _PC3 * t6 * Rm + _PC4 * t3 * Rm - 0.2e1 * t23 * AA * t25 - 0.2e1 * t23 * BB * t29 - t22 * AA * t29 * kn + t22 * BB * t25 * kn;

  t3   = PetscExpReal(0.2e1 * x * B);
  t4   = t3 * B;
  t8   = km * km;
  t9   = t3 * t8;
  t11  = 0.3e1 * t9 * Rm;
  t12  = Rm * Rm;
  t14  = t3 * t12 * Rm;
  t15  = UU * UU;
  t19  = 0.4e1 * t4 * UU * Rm - t11 - t14 + 0.3e1 * t3 * t15 * Rm;
  t20  = Rm * x;
  t21  = PetscSinReal(t20);
  t27  = 0.2e1 * B * t9;
  t33  = 0.2e1 * t4 * t12;
  t36  = 0.3e1 * t3 * UU * t12 - t27 - 0.2e1 * t4 * t15 + 0.3e1 * t9 * UU + t33 - t3 * t15 * UU;
  t37  = PetscCosReal(t20);
  t49  = VV * VV;
  t53  = -0.4e1 * t4 * VV * Rm - t11 + 0.3e1 * t3 * t49 * Rm - t14;
  t64  = t3 * t49 * VV + t33 - 0.3e1 * t9 * VV - 0.2e1 * t4 * t49 - t27 - 0.3e1 * t3 * VV * t12;
  t76  = B * t8;
  t80  = kn * kn;
  t83  = B * B;
  t87  = t80 * kn;
  t90  = kn * x;
  t91  = PetscSinReal(t90);
  t106 = PetscCosReal(t90);
  u3 = -((t19 * t21 + t36 * t37) * _PC1 + (t36 * t21 - t19 * t37) * _PC2 + (t53 * t21 + t64 * t37) * _PC3 + (t64 * t21 - t53 * t37) * _PC4 + (-0.3e1 * t8 * AA * kn - 0.8e1 * t76 * BB - 0.4e1 * BB * B * t80 + 0.4e1 * AA * t83 * kn - AA * t87) * t91 + (-0.4e1 * AA * t80 * B - 0.4e1 * t83 * BB * kn + 0.3e1 * t8 * BB * kn - sigma + BB * t87 - 0.8e1 * t76 * AA) * t106) / km;

  t3  = PetscExpReal(0.2e1 * x * B);
  t4  = km * km;
  t5  = t3 * t4;
  t6  = Rm * x;
  t7  = PetscCosReal(t6);
  t8  = _PC1 * t7;
  t10 = PetscSinReal(t6);
  t11 = _PC2 * t10;
  t13 = _PC3 * t7;
  t15 = _PC4 * t10;
  t18 = kn * x;
  t19 = PetscCosReal(t18);
  t22 = PetscSinReal(t18);
  t24 = UU * UU;
  t25 = t3 * t24;
  t28 = t3 * UU;
  t38 = Rm * Rm;
  t39 = t7 * t38;
  t42 = t10 * t38;
  t44 = t5 * t8 + t5 * t11 + t5 * t13 + t5 * t15 + t4 * AA * t19 + t4 * BB * t22 + t25 * t8 + t25 * t11 - 0.2e1 * t28 * _PC1 * t10 * Rm + 0.2e1 * t28 * _PC2 * t7 * Rm - t3 * _PC1 * t39 - t3 * _PC2 * t42;
  t45 = VV * VV;
  t46 = t3 * t45;
  t49 = t3 * VV;
  t62 = B * B;
  t78 = kn * kn;
  t82 = t46 * t13 + t46 * t15 + 0.2e1 * t49 * _PC3 * t10 * Rm - 0.2e1 * t49 * _PC4 * t7 * Rm - t3 * _PC3 * t39 - t3 * _PC4 * t42 + 0.4e1 * t62 * AA * t19 + 0.4e1 * t62 * BB * t22 + 0.4e1 * B * AA * t22 * kn - 0.4e1 * B * BB * t19 * kn - AA * t19 * t78 - BB * t22 * t78;
  u4 = t44 + t82;

  t3   = PetscExpReal(0.2e1 * x * B);
  t4   = t3 * B;
  t8   = km * km;
  t9   = t3 * t8;
  t10  = t9 * Rm;
  t11  = Rm * Rm;
  t13  = t3 * t11 * Rm;
  t14  = UU * UU;
  t18  = 0.4e1 * t4 * UU * Rm - t10 - t13 + 0.3e1 * t3 * t14 * Rm;
  t19  = Rm * x;
  t20  = PetscSinReal(t19);
  t26  = 0.2e1 * B * t9;
  t31  = 0.2e1 * t4 * t11;
  t34  = 0.3e1 * t3 * UU * t11 - t26 - 0.2e1 * t4 * t14 + t9 * UU + t31 - t3 * t14 * UU;
  t35  = PetscCosReal(t19);
  t47  = VV * VV;
  t51  = -0.4e1 * t4 * VV * Rm - t10 + 0.3e1 * t3 * t47 * Rm - t13;
  t61  = t3 * t47 * VV + t31 - t9 * VV - 0.2e1 * t4 * t47 - t26 - 0.3e1 * t3 * VV * t11;
  t72  = B * t8;
  t76  = kn * kn;
  t79  = B * B;
  t83  = t76 * kn;
  t86  = kn * x;
  t87  = PetscSinReal(t86);
  t101 = PetscCosReal(t86);
  u5 = ((t18 * t20 + t34 * t35) * _PC1 + (t34 * t20 - t18 * t35) * _PC2 + (t51 * t20 + t61 * t35) * _PC3 + (t61 * t20 - t51 * t35) * _PC4 + (-t8 * AA * kn - 0.4e1 * t72 * BB - 0.4e1 * BB * B * t76 + 0.4e1 * AA * t79 * kn - AA * t83) * t87 + (-0.4e1 * AA * t76 * B - 0.4e1 * t79 * BB * kn + t8 * BB * kn - sigma + BB * t83 - 0.4e1 * t72 * AA) * t101) / km;

  t3  = PetscExpReal(0.2e1 * x * B);
  t4  = UU * UU;
  t8  = km * km;
  t9  = t3 * t8;
  t10 = t9 * Rm;
  t11 = Rm * Rm;
  t13 = t3 * t11 * Rm;
  t14 = t3 * B;
  t18 = 0.3e1 * t3 * t4 * Rm + t10 - t13 + 0.4e1 * t14 * UU * Rm;
  t19 = Rm * x;
  t20 = PetscSinReal(t19);
  t28 = 0.2e1 * B * t9;
  t33 = 0.2e1 * t14 * t11;
  t34 = -0.2e1 * t4 * t14 + 0.3e1 * t3 * UU * t11 - t28 - t3 * t4 * UU - t9 * UU + t33;
  t35 = PetscCosReal(t19);
  t47 = VV * VV;
  t51 = -0.4e1 * t14 * VV * Rm - t13 + t10 + 0.3e1 * t3 * t47 * Rm;
  t61 = -0.3e1 * t3 * VV * t11 + t33 + t3 * t47 * VV + t9 * VV - 0.2e1 * t14 * t47 - t28;
  t71 = kn * kn;
  t74 = B * B;
  t80 = t71 * kn;
  t83 = kn * x;
  t84 = PetscSinReal(t83);
  t96 = PetscCosReal(t83);
  u6 = -((t18 * t20 + t34 * t35) * _PC1 + (t34 * t20 - t18 * t35) * _PC2 + (t51 * t20 + t61 * t35) * _PC3 + (t61 * t20 - t51 * t35) * _PC4 + (-0.4e1 * BB * B * t71 + 0.4e1 * AA * t74 * kn + t8 * AA * kn - AA * t80) * t84 + (-0.4e1 * AA * t71 * B - t8 * BB * kn - 0.4e1 * t74 * BB * kn - sigma + BB * t80) * t96) / km;

  /* SS = sin(km*z)*(exp(UU*x)*(_PC1*cos(Rm*x)+_PC2*sin(Rm*x)) + exp(-VV*x)*(_PC3*cos(Rm*x)+_PC4*sin(Rm*x)) + exp(-2*x*B)*(AA*cos(kn*x)+BB*sin(kn*x))); */

  /* u1 = Vx, u2 = Vz, u3 = txx, u4 = tzx, u5 = pressure, u6 = tzz */

  sum5 += u5 * PetscCosReal(km * z); /* pressure */
  sum6 += u6 * PetscCosReal(km * z); /* zz total stress */

  u1 *= PetscCosReal(km * z); /* x velocity */
  sum1 += u1;
  u2 *= PetscSinReal(km * z); /* z velocity */
  sum2 += u2;

  u3 *= PetscCosReal(km * z); /* xx total stress */
  sum3 += u3;
  u4 *= PetscSinReal(km * z); /* zx stress */
  sum4 += u4;

  /* rho = -sigma*sin(km*z)*cos(kn*x); */ /* density */
  /* sum7 += rho; */

  /* Output */
  if (mu) *mu = Z;
  if (vel) {
    vel[0] = sum1;
    vel[1] = sum2;
  }
  if (p) (*p) = sum5;
  if (s) {
    s[0] = sum3;
    s[1] = sum4;
    s[2] = sum6;
  }
  if (gamma) {
    /* sigma = tau - p, tau = sigma + p, tau[] = 2*eta*gamma[] */
    gamma[0] = (sum3 + sum5) / (2.0 * Z);
    gamma[1] = (sum4) / (2.0 * Z);
    gamma[2] = (sum6 + sum5) / (2.0 * Z);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
