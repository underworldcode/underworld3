/*
  Newest-vertex bisection (NVB) as a native DMPlexTransform — UW extension.

  Route B of docs/developer/design/NVB_GRADED_ADAPT.md: a graded, parallel,
  SF-preserving simplicial refinement transform registered into PETSc from UW
  (no PETSc rebuild). NVB is a small delta from PETSc's SBR transform
  (src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c): identical
  bisection geometry + the same DMLabelPropagate cross-rank closure; the only
  NVB-specific change is the edge chosen to split (newest-vertex refinement edge
  vs SBR's geometric longest edge).

  STAGE 2a (this file): a SELF-CONTAINED clone of SBR, registered as "uwnvb". It
  reproduces `refine_sbr` byte-for-byte on a serial mesh (validated: identical
  triangulation for every marking pattern incl. full uniform refine) and matches
  it under the parallel closure. It depends only on:
    - exported libpetsc symbols (DMPlexPointQueue*, DMLabelPropagate*, the
      Identity cell-transform / orientation helpers, registration, plus the
      ordinary DMPlex / DMLabel / PetscSection API);
    - static-inline header helpers (DMPolytopeTypeGetArrangement,
      DMPolytopeTypeComposeOrientation, DMPlex_DistD_Internal);
    - four PETSc helpers that are NOT exported from libpetsc and are therefore
      reimplemented verbatim below (SetDimensions, MapCoordinatesBarycenter, and
      the SEGMENT+TRIANGLE cases of the regular cell-refine / subcell-orientation
      — the only polytopes SBR uses in 2D).

  The newest-vertex edge choice (Stage 2b) is a graded, multi-pass single-
  bisection transform layered on this SBR-equivalent base; see the design note.

  Reference (read-only) copy of the PETSc 3.25.0 sources this clones:
    src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c
    src/dm/impls/plex/transform/impls/refine/regular/plexrefregular.c
    src/dm/impls/plex/transform/interface/plextransform.c (the two _Internal helpers)
*/
#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/
#include <petscsf.h>

/* Same payload as PETSc's DMPlexRefine_SBR (private, so we declare our own). */
typedef struct {
  DMLabel      splitPoints; /* edges to bisect (1) and triangles to divide (2) */
  PetscSection secEdgeLen;  /* section for the edge-length field */
  PetscReal   *edgeLen;     /* lazily-computed edge lengths */
} DMPlexRefine_UWNVB;

/* ======================================================================== */
/* Reimplemented PETSc helpers that libpetsc does not export.               */
/* Byte-identical to the 3.25.0 originals (verified against source).        */
/* ======================================================================== */

/* == DMPlexTransformSetDimensions_Internal == */
static PetscErrorCode UWNVBSetDimensions(DMPlexTransform tr, DM dm, DM tdm)
{
  PetscInt dim, cdim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(tdm, dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMSetCoordinateDim(tdm, cdim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* == DMPlexTransformMapCoordinatesBarycenter_Internal == */
static PetscErrorCode UWNVBMapCoordinatesBarycenter(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  PetscInt v, d;

  PetscFunctionBeginHot;
  PetscCheck(ct == DM_POLYTOPE_POINT, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for refined point type %s", DMPolytopeTypes[ct]);
  for (d = 0; d < dE; ++d) out[d] = 0.0;
  for (v = 0; v < Nv; ++v)
    for (d = 0; d < dE; ++d) out[d] += in[v * dE + d];
  for (d = 0; d < dE; ++d) out[d] /= Nv;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* == DMPlexTransformCellRefine_Regular, SEGMENT + TRIANGLE cases only ==
   (regular 1->2 edge split and 1->4 triangle split — the uniform refinement
   SBR delegates to for fully-split cells and edges). */
static PetscErrorCode UWNVBRegularCellRefine(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  /* Split an edge with a new midpoint vertex, making two new edges:
       0--0--0--1--1 */
  static DMPolytopeType segT[] = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT};
  static PetscInt       segS[] = {1, 2};
  static PetscInt       segC[] = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0};
  static PetscInt       segO[] = {0, 0, 0, 0};
  /* Add 3 edges inside every triangle, making 4 new triangles. */
  static DMPolytopeType triT[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS[] = {3, 4};
  static PetscInt       triC[] = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0, 2, DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 0, 2};
  static PetscInt       triO[] = {0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0};

  PetscFunctionBegin;
  if (rt) *rt = 0;
  switch (source) {
  case DM_POLYTOPE_SEGMENT:
    *Nt     = 2;
    *target = segT;
    *size   = segS;
    *cone   = segC;
    *ornt   = segO;
    break;
  case DM_POLYTOPE_TRIANGLE:
    *Nt     = 2;
    *target = triT;
    *size   = triS;
    *cone   = triC;
    *ornt   = triO;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "UWNVB regular refine only handles SEGMENT/TRIANGLE, not %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* == DMPlexTransformGetSubcellOrientation_Regular, SEGMENT + TRIANGLE cases == */
static PetscErrorCode UWNVBRegularGetSubcellOrientation(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  static PetscInt seg_seg[] = {1, -1, 0, -1, 0, 0, 1, 0};
  static PetscInt tri_seg[] = {2, -1, 1, -1, 0, -1, 1, -1, 0, -1, 2, -1, 0, -1, 2, -1, 1, -1, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0};
  static PetscInt tri_tri[] = {1, -3, 0, -3, 2, -3, 3, -2, 0, -2, 2, -2, 1, -2, 3, -1, 2, -1, 1, -1, 0, -1, 3, -3, 0, 0, 1, 0, 2, 0, 3, 0, 1, 1, 2, 1, 0, 1, 3, 1, 2, 2, 0, 2, 1, 2, 3, 2};

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = o;
  if (!so) PetscFunctionReturn(PETSC_SUCCESS);
  switch (sct) {
  case DM_POLYTOPE_POINT:
    break;
  case DM_POLYTOPE_SEGMENT:
    switch (tct) {
    case DM_POLYTOPE_POINT:
      break;
    case DM_POLYTOPE_SEGMENT:
      *rnew = seg_seg[(so + 1) * 4 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, seg_seg[(so + 1) * 4 + r * 2 + 1]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
  case DM_POLYTOPE_TRIANGLE:
    switch (tct) {
    case DM_POLYTOPE_SEGMENT:
      *rnew = tri_seg[(so + 3) * 6 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_seg[(so + 3) * 6 + r * 2 + 1]);
      break;
    case DM_POLYTOPE_TRIANGLE:
      *rnew = tri_tri[(so + 3) * 8 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_tri[(so + 3) * 8 + r * 2 + 1]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "UWNVB regular orientation only handles SEGMENT/TRIANGLE, not %s", DMPolytopeTypes[sct]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ======================================================================== */
/* SBR algorithm, cloned verbatim and renamed UWNVB.                        */
/* ======================================================================== */

static PetscErrorCode UWNVBGetEdgeLen(DMPlexTransform tr, PetscInt edge, PetscReal *len)
{
  DMPlexRefine_UWNVB *nvb = (DMPlexRefine_UWNVB *)tr->data;
  DM                  dm;
  PetscInt            off;

  PetscFunctionBeginHot;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(PetscSectionGetOffset(nvb->secEdgeLen, edge, &off));
  if (nvb->edgeLen[off] <= 0.0) {
    DM                 cdm;
    Vec                coordsLocal;
    const PetscScalar *coords;
    const PetscInt    *cone;
    PetscScalar       *cA, *cB;
    PetscInt           coneSize, cdim;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMPlexGetCone(dm, edge, &cone));
    PetscCall(DMPlexGetConeSize(dm, edge, &coneSize));
    PetscCheck(coneSize == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Edge %" PetscInt_FMT " cone size must be 2, not %" PetscInt_FMT, edge, coneSize);
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCall(DMGetCoordinatesLocalNoncollective(dm, &coordsLocal));
    PetscCall(VecGetArrayRead(coordsLocal, &coords));
    PetscCall(DMPlexPointLocalRead(cdm, cone[0], coords, &cA));
    PetscCall(DMPlexPointLocalRead(cdm, cone[1], coords, &cB));
    nvb->edgeLen[off] = DMPlex_DistD_Internal(cdim, cA, cB);
    PetscCall(VecRestoreArrayRead(coordsLocal, &coords));
  }
  *len = nvb->edgeLen[off];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Mark local edges that should be split.
   STAGE 2a: pick the longest edge of each marked cell (== SBR). STAGE 2b picks
   the newest-vertex refinement edge instead. */
static PetscErrorCode UWNVBSplitLocalEdges(DMPlexTransform tr, DMPlexPointQueue queue)
{
  DMPlexRefine_UWNVB *nvb = (DMPlexRefine_UWNVB *)tr->data;
  DM                  dm;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  while (!DMPlexPointQueueEmpty(queue)) {
    PetscInt        p = -1;
    const PetscInt *support;
    PetscInt        supportSize, s;

    PetscCall(DMPlexPointQueueDequeue(queue, &p));
    PetscCall(DMPlexGetSupport(dm, p, &support));
    PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
    for (s = 0; s < supportSize; ++s) {
      const PetscInt  cell = support[s];
      const PetscInt *cone;
      PetscInt        coneSize, c;
      PetscInt        cval, eval, maxedge;
      PetscReal       len, maxlen;

      PetscCall(DMLabelGetValue(nvb->splitPoints, cell, &cval));
      if (cval == 2) continue;
      PetscCall(DMPlexGetCone(dm, cell, &cone));
      PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
      PetscCall(UWNVBGetEdgeLen(tr, cone[0], &maxlen));
      maxedge = cone[0];
      for (c = 1; c < coneSize; ++c) {
        PetscCall(UWNVBGetEdgeLen(tr, cone[c], &len));
        if (len > maxlen) {
          maxlen  = len;
          maxedge = cone[c];
        }
      }
      PetscCall(DMLabelGetValue(nvb->splitPoints, maxedge, &eval));
      if (eval != 1) {
        PetscCall(DMLabelSetValue(nvb->splitPoints, maxedge, 1));
        PetscCall(DMPlexPointQueueEnqueue(queue, maxedge));
      }
      PetscCall(DMLabelSetValue(nvb->splitPoints, cell, 2));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UWNVBSplitPoint(PETSC_UNUSED DMLabel label, PetscInt p, PETSC_UNUSED PetscInt val, void *ctx)
{
  DMPlexPointQueue queue = (DMPlexPointQueue)ctx;

  PetscFunctionBegin;
  PetscCall(DMPlexPointQueueEnqueue(queue, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The 'splitPoints' label marks mesh points to be divided: edges with 1,
  triangles with 2. The refinement type is then:

    vertex:                   0   (RT_VERTEX)
    edge unsplit:             1   (RT_EDGE)
    edge split:               2   (RT_EDGE_SPLIT)
    triangle unsplit:         3   (RT_TRIANGLE)
    triangle split all edges: 4   (RT_TRIANGLE_SPLIT)
    triangle split edges 0 1: 5   (RT_TRIANGLE_SPLIT_01)  ... etc
*/
typedef enum {
  RT_VERTEX,
  RT_EDGE,
  RT_EDGE_SPLIT,
  RT_TRIANGLE,
  RT_TRIANGLE_SPLIT,
  RT_TRIANGLE_SPLIT_01,
  RT_TRIANGLE_SPLIT_10,
  RT_TRIANGLE_SPLIT_12,
  RT_TRIANGLE_SPLIT_21,
  RT_TRIANGLE_SPLIT_20,
  RT_TRIANGLE_SPLIT_02,
  RT_TRIANGLE_SPLIT_0,
  RT_TRIANGLE_SPLIT_1,
  RT_TRIANGLE_SPLIT_2
} RefinementType;

static PetscErrorCode DMPlexTransformSetUp_UWNVB(DMPlexTransform tr)
{
  DMPlexRefine_UWNVB *nvb = (DMPlexRefine_UWNVB *)tr->data;
  DM                  dm;
  DMLabel             active;
  PetscSF             pointSF;
  DMPlexPointQueue    queue = NULL;
  IS                  refineIS;
  const PetscInt     *refineCells;
  PetscInt            pStart, pEnd, p, eStart, eEnd, e, edgeLenSize, Nc, c;
  PetscBool           empty;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Split Points", &nvb->splitPoints));
  /* Create edge lengths */
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &nvb->secEdgeLen));
  PetscCall(PetscSectionSetChart(nvb->secEdgeLen, eStart, eEnd));
  for (e = eStart; e < eEnd; ++e) PetscCall(PetscSectionSetDof(nvb->secEdgeLen, e, 1));
  PetscCall(PetscSectionSetUp(nvb->secEdgeLen));
  PetscCall(PetscSectionGetStorageSize(nvb->secEdgeLen, &edgeLenSize));
  PetscCall(PetscCalloc1(edgeLenSize, &nvb->edgeLen));
  /* Add edges of cells that are marked for refinement to edge queue */
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCheck(active, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_WRONGSTATE, "DMPlexTransform must have an adaptation label in order to use UWNVB algorithm");
  PetscCall(DMPlexPointQueueCreate(1024, &queue));
  PetscCall(DMLabelGetStratumIS(active, DM_ADAPT_REFINE, &refineIS));
  PetscCall(DMLabelGetStratumSize(active, DM_ADAPT_REFINE, &Nc));
  if (refineIS) PetscCall(ISGetIndices(refineIS, &refineCells));
  for (c = 0; c < Nc; ++c) {
    const PetscInt cell = refineCells[c];
    PetscInt       depth;

    PetscCall(DMPlexGetPointDepth(dm, cell, &depth));
    if (depth == 1) {
      PetscCall(DMLabelSetValue(nvb->splitPoints, cell, 1));
      PetscCall(DMPlexPointQueueEnqueue(queue, cell));
    } else {
      PetscInt *closure = NULL;
      PetscInt  Ncl, cl;

      PetscCall(DMLabelSetValue(nvb->splitPoints, cell, depth));
      PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
      for (cl = 0; cl < Ncl; cl += 2) {
        const PetscInt edge = closure[cl];

        if (edge >= eStart && edge < eEnd) {
          PetscCall(DMLabelSetValue(nvb->splitPoints, edge, 1));
          PetscCall(DMPlexPointQueueEnqueue(queue, edge));
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    }
  }
  if (refineIS) PetscCall(ISRestoreIndices(refineIS, &refineCells));
  PetscCall(ISDestroy(&refineIS));
  /* Setup communication */
  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCall(DMLabelPropagateBegin(nvb->splitPoints, pointSF));
  /* While edge queue is not empty (collective): split locally, then push the
     newly-marked edges across the point SF, repeat until globally empty. */
  PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  while (!empty) {
    PetscCall(UWNVBSplitLocalEdges(tr, queue));
    PetscCall(DMLabelPropagatePush(nvb->splitPoints, pointSF, UWNVBSplitPoint, queue));
    PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  }
  PetscCall(DMLabelPropagateEnd(nvb->splitPoints, pointSF));
  /* Calculate refineType for each cell */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    DMLabel        trType = tr->trType;
    DMPolytopeType ct;
    PetscInt       val;

    PetscCall(DMPlexGetCellType(dm, p, &ct));
    switch (ct) {
    case DM_POLYTOPE_POINT:
      PetscCall(DMLabelSetValue(trType, p, RT_VERTEX));
      break;
    case DM_POLYTOPE_SEGMENT:
      PetscCall(DMLabelGetValue(nvb->splitPoints, p, &val));
      if (val == 1) PetscCall(DMLabelSetValue(trType, p, RT_EDGE_SPLIT));
      else PetscCall(DMLabelSetValue(trType, p, RT_EDGE));
      break;
    case DM_POLYTOPE_TRIANGLE:
      PetscCall(DMLabelGetValue(nvb->splitPoints, p, &val));
      if (val == 2) {
        const PetscInt *cone;
        PetscReal       lens[3];
        PetscInt        vals[3], i;

        PetscCall(DMPlexGetCone(dm, p, &cone));
        for (i = 0; i < 3; ++i) {
          PetscCall(DMLabelGetValue(nvb->splitPoints, cone[i], &vals[i]));
          vals[i] = vals[i] < 0 ? 0 : vals[i];
          PetscCall(UWNVBGetEdgeLen(tr, cone[i], &lens[i]));
        }
        if (vals[0] && vals[1] && vals[2]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT));
        else if (vals[0] && vals[1]) PetscCall(DMLabelSetValue(trType, p, lens[0] > lens[1] ? RT_TRIANGLE_SPLIT_01 : RT_TRIANGLE_SPLIT_10));
        else if (vals[1] && vals[2]) PetscCall(DMLabelSetValue(trType, p, lens[1] > lens[2] ? RT_TRIANGLE_SPLIT_12 : RT_TRIANGLE_SPLIT_21));
        else if (vals[2] && vals[0]) PetscCall(DMLabelSetValue(trType, p, lens[2] > lens[0] ? RT_TRIANGLE_SPLIT_20 : RT_TRIANGLE_SPLIT_02));
        else if (vals[0]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_0));
        else if (vals[1]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_1));
        else if (vals[2]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_2));
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " does not fit any refinement type (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")", p, vals[0], vals[1], vals[2]);
      } else PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
    PetscCall(DMLabelGetValue(nvb->splitPoints, p, &val));
  }
  /* Cleanup */
  PetscCall(DMPlexPointQueueDestroy(&queue));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_UWNVB(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscInt rt;

  PetscFunctionBeginHot;
  PetscCall(DMLabelGetValue(tr->trType, sp, &rt));
  *rnew = r;
  *onew = o;
  switch (rt) {
  case RT_TRIANGLE_SPLIT_01:
  case RT_TRIANGLE_SPLIT_10:
  case RT_TRIANGLE_SPLIT_12:
  case RT_TRIANGLE_SPLIT_21:
  case RT_TRIANGLE_SPLIT_20:
  case RT_TRIANGLE_SPLIT_02:
    break;
  case RT_TRIANGLE_SPLIT_0:
  case RT_TRIANGLE_SPLIT_1:
  case RT_TRIANGLE_SPLIT_2:
    switch (tct) {
    case DM_POLYTOPE_SEGMENT:
      break;
    case DM_POLYTOPE_TRIANGLE:
      *onew = so < 0 ? -(o + 1) : o;
      *rnew = so < 0 ? (r + 1) % 2 : r;
      break;
    default:
      break;
    }
    break;
  case RT_EDGE_SPLIT:
  case RT_TRIANGLE_SPLIT:
    PetscCall(UWNVBRegularGetSubcellOrientation(tr, sct, sp, so, tct, r, o, rnew, onew));
    break;
  default:
    PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add 1 edge inside this triangle, making 2 new triangles (single bisection). */
static PetscErrorCode UWNVBGetTriangleSplitSingle(PetscInt o, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  const PetscInt       *arr     = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, o);
  static DMPolytopeType triT1[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS1[] = {1, 2};
  static PetscInt       triC1[] = {DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                   DM_POLYTOPE_SEGMENT, 0, 0};
  static PetscInt       triO1[] = {0, 0, 0, 0, -1, 0, 0, 0};

  PetscFunctionBeginHot;
  /* To get the other divisions, we reorient the triangle */
  triC1[2]  = arr[0 * 2];
  triC1[7]  = arr[1 * 2];
  triC1[11] = arr[0 * 2];
  triC1[15] = arr[1 * 2];
  triC1[22] = arr[1 * 2];
  triC1[26] = arr[2 * 2];
  *Nt       = 2;
  *target   = triT1;
  *size     = triS1;
  *cone     = triC1;
  *ornt     = triO1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add 2 edges inside this triangle, making 3 new triangles (double bisection). */
static PetscErrorCode UWNVBGetTriangleSplitDouble(PetscInt o, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  PetscInt              e0, e1;
  const PetscInt       *arr     = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, o);
  static DMPolytopeType triT2[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS2[] = {2, 3};
  static PetscInt       triC2[] = {DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 1};
  static PetscInt       triO2[] = {0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0};

  PetscFunctionBeginHot;
  /* To get the other divisions, we reorient the triangle */
  triC2[2]  = arr[0 * 2];
  triC2[3]  = arr[0 * 2 + 1] ? 1 : 0;
  triC2[7]  = arr[1 * 2];
  triC2[11] = arr[1 * 2];
  triC2[15] = arr[2 * 2];
  /* Swap the first two edges if the triangle is reversed */
  e0            = o < 0 ? 23 : 19;
  e1            = o < 0 ? 19 : 23;
  triC2[e0]     = arr[0 * 2];
  triC2[e0 + 1] = 0;
  triC2[e1]     = arr[1 * 2];
  triC2[e1 + 1] = o < 0 ? 1 : 0;
  triO2[6]      = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, -1, arr[2 * 2 + 1]);
  /* Swap the first two edges if the triangle is reversed */
  e0            = o < 0 ? 34 : 30;
  e1            = o < 0 ? 30 : 34;
  triC2[e0]     = arr[1 * 2];
  triC2[e0 + 1] = o < 0 ? 0 : 1;
  triC2[e1]     = arr[2 * 2];
  triC2[e1 + 1] = o < 0 ? 1 : 0;
  triO2[9]      = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, -1, arr[2 * 2 + 1]);
  /* Swap the last two edges if the triangle is reversed */
  triC2[41] = arr[2 * 2];
  triC2[42] = o < 0 ? 0 : 1;
  triC2[45] = o < 0 ? 1 : 0;
  triC2[48] = o < 0 ? 0 : 1;
  triO2[11] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, 0, arr[1 * 2 + 1]);
  triO2[12] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, 0, arr[2 * 2 + 1]);
  *Nt       = 2;
  *target   = triT2;
  *size     = triS2;
  *cone     = triC2;
  *ornt     = triO2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellTransform_UWNVB(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMLabel  trType = tr->trType;
  PetscInt val;

  PetscFunctionBeginHot;
  PetscCheck(p >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point argument is invalid");
  PetscCall(DMLabelGetValue(trType, p, &val));
  if (rt) *rt = val;
  switch (source) {
  case DM_POLYTOPE_POINT:
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
  case DM_POLYTOPE_TETRAHEDRON:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_TRI_PRISM:
  case DM_POLYTOPE_TRI_PRISM_TENSOR:
  case DM_POLYTOPE_QUAD_PRISM_TENSOR:
  case DM_POLYTOPE_PYRAMID:
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    break;
  case DM_POLYTOPE_SEGMENT:
    if (val == RT_EDGE) PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    else PetscCall(UWNVBRegularCellRefine(tr, source, p, NULL, Nt, target, size, cone, ornt));
    break;
  case DM_POLYTOPE_TRIANGLE:
    switch (val) {
    case RT_TRIANGLE_SPLIT_0:
      PetscCall(UWNVBGetTriangleSplitSingle(2, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_1:
      PetscCall(UWNVBGetTriangleSplitSingle(0, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_2:
      PetscCall(UWNVBGetTriangleSplitSingle(1, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_21:
      PetscCall(UWNVBGetTriangleSplitDouble(-3, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_10:
      PetscCall(UWNVBGetTriangleSplitDouble(-2, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_02:
      PetscCall(UWNVBGetTriangleSplitDouble(-1, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_12:
      PetscCall(UWNVBGetTriangleSplitDouble(0, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_20:
      PetscCall(UWNVBGetTriangleSplitDouble(1, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_01:
      PetscCall(UWNVBGetTriangleSplitDouble(2, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT:
      PetscCall(UWNVBRegularCellRefine(tr, source, p, NULL, Nt, target, size, cone, ornt));
      break;
    default:
      PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetFromOptions_UWNVB(DMPlexTransform tr, PetscOptionItems PetscOptionsObject)
{
  PetscInt  cells[256], n = 256, i;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlex Options");
  PetscCall(PetscOptionsIntArray("-dm_plex_transform_uwnvb_ref_cell", "Mark cells for refinement", "", cells, &n, &flg));
  if (flg) {
    DMLabel active;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active));
    for (i = 0; i < n; ++i) PetscCall(DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE));
    PetscCall(DMPlexTransformSetActive(tr, active));
    PetscCall(DMLabelDestroy(&active));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformView_UWNVB(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "UWNVB refinement %s\n", name ? name : ""));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscCall(DMLabelView(tr->trType, viewer));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_UWNVB(DMPlexTransform tr)
{
  DMPlexRefine_UWNVB *nvb = (DMPlexRefine_UWNVB *)tr->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(nvb->edgeLen));
  PetscCall(PetscSectionDestroy(&nvb->secEdgeLen));
  PetscCall(DMLabelDestroy(&nvb->splitPoints));
  PetscCall(PetscFree(tr->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_UWNVB(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_UWNVB;
  tr->ops->setfromoptions        = DMPlexTransformSetFromOptions_UWNVB;
  tr->ops->setup                 = DMPlexTransformSetUp_UWNVB;
  tr->ops->destroy               = DMPlexTransformDestroy_UWNVB;
  tr->ops->setdimensions         = UWNVBSetDimensions;
  tr->ops->celltransform         = DMPlexTransformCellTransform_UWNVB;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_UWNVB;
  tr->ops->mapcoordinates        = UWNVBMapCoordinatesBarycenter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_UWNVB(DMPlexTransform tr)
{
  DMPlexRefine_UWNVB *nvb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&nvb));
  tr->data = nvb;
  PetscCall(DMPlexTransformInitialize_UWNVB(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Idempotent registration entry called from the Cython module on import. */
PetscErrorCode UWNVBTransformRegister(void)
{
  PetscFunctionBegin;
  PetscCall(DMPlexTransformRegister("uwnvb", DMPlexTransformCreate_UWNVB));
  PetscFunctionReturn(PETSC_SUCCESS);
}
