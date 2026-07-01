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

/* ======================================================================== */
/* Stage 2b — graded NVB by single-bisection multi-pass.                    */
/*                                                                          */
/* The `uwnvb` transform above is the SBR-equivalent (longest-edge, one-pass */
/* with green/blue closure) used as a validated native-SBR. Grading needs a */
/* different execution: each triangle carries an explicit refinement-edge    */
/* SLOT (cone index 0/1/2), and refinement proceeds as repeated CONFORMING   */
/* single bisections of *compatible* refinement edges (an edge that is the   */
/* refinement edge of every incident cell). A one-pass conforming transform  */
/* is forced into green/blue multi-edge splits, which corrupt the refedge    */
/* structure and defeat grading (see NVB_GRADED_ADAPT.md, Stage 2b finding); */
/* single bisections keep child slots trivially correct (opposite the new    */
/* midpoint) so the closure stays bounded.                                   */
/*                                                                          */
/* `uwnvb_bisect` is the per-sub-pass engine: given an edge label marking a   */
/* set of pairwise-independent-per-cell edges (each the refedge of its cells) */
/* it single-splits exactly those edges. The C driver UWNVBRefine() below    */
/* computes the closure + drains it over uwnvb_bisect sub-passes, maintaining */
/* the slot label — all in C so the SF-sensitive parallel logic stays in one */
/* place.                                                                    */
/* ======================================================================== */
#define UWNVB_SLOT_LABEL   "uwnvb_refedge"       /* per-triangle refinement-edge cone slot 0/1/2 */
#define UWNVB_BISECT_LABEL "uwnvb_bisect_edges"  /* per-edge: 1 => bisect this sub-pass */

/* The vertex of a triangle opposite edge slot `i`, given the three edges'
   endpoint pairs (the vertex shared by the other two edges). */
static inline PetscInt UWNVBOppVertex(const PetscInt everts[3][2], PetscInt i)
{
  PetscInt j = (i + 1) % 3, k = (i + 2) % 3, a, b;
  for (a = 0; a < 2; ++a)
    for (b = 0; b < 2; ++b)
      if (everts[j][a] == everts[k][b]) return everts[j][a];
  return -1;
}

/* SetUp for uwnvb_bisect: mark the edges named by UWNVB_BISECT_LABEL and set the
   refinement type of each incident triangle from how many of its edges are marked
   (1 => single bisection on that edge; 2/3 => the green/blue split types, kept as
   a safety net — the driver only ever feeds pairwise-independent single edges). */
static PetscErrorCode DMPlexTransformSetUp_UWNVBBisect(DMPlexTransform tr)
{
  DMPlexRefine_UWNVB *nvb = (DMPlexRefine_UWNVB *)tr->data;
  DM                  dm;
  DMLabel             bisect;
  IS                  bisectIS;
  const PetscInt     *bisectEdges;
  PetscInt            pStart, pEnd, p, eStart, eEnd, e, edgeLenSize, Nb, i;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Split Points", &nvb->splitPoints));
  /* edge lengths (for the 2/3-marked disambiguation only) */
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &nvb->secEdgeLen));
  PetscCall(PetscSectionSetChart(nvb->secEdgeLen, eStart, eEnd));
  for (e = eStart; e < eEnd; ++e) PetscCall(PetscSectionSetDof(nvb->secEdgeLen, e, 1));
  PetscCall(PetscSectionSetUp(nvb->secEdgeLen));
  PetscCall(PetscSectionGetStorageSize(nvb->secEdgeLen, &edgeLenSize));
  PetscCall(PetscCalloc1(edgeLenSize, &nvb->edgeLen));
  /* mark the requested edges + their incident triangles */
  PetscCall(DMGetLabel(dm, UWNVB_BISECT_LABEL, &bisect));
  PetscCheck(bisect, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_WRONGSTATE, "uwnvb_bisect needs the '%s' edge label", UWNVB_BISECT_LABEL);
  PetscCall(DMLabelGetStratumIS(bisect, 1, &bisectIS));
  PetscCall(DMLabelGetStratumSize(bisect, 1, &Nb));
  if (bisectIS) {
    PetscCall(ISGetIndices(bisectIS, &bisectEdges));
    for (i = 0; i < Nb; ++i) {
      const PetscInt  edge = bisectEdges[i];
      const PetscInt *supp;
      PetscInt        suppSize, s;

      PetscCall(DMLabelSetValue(nvb->splitPoints, edge, 1));
      PetscCall(DMPlexGetSupport(dm, edge, &supp));
      PetscCall(DMPlexGetSupportSize(dm, edge, &suppSize));
      for (s = 0; s < suppSize; ++s) PetscCall(DMLabelSetValue(nvb->splitPoints, supp[s], 2));
    }
    PetscCall(ISRestoreIndices(bisectIS, &bisectEdges));
  }
  PetscCall(ISDestroy(&bisectIS));
  /* refine type per point (no closure: exactly the marked edges are split) */
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
      PetscCall(DMLabelSetValue(trType, p, val == 1 ? RT_EDGE_SPLIT : RT_EDGE));
      break;
    case DM_POLYTOPE_TRIANGLE:
      PetscCall(DMLabelGetValue(nvb->splitPoints, p, &val));
      if (val == 2) {
        const PetscInt *cone;
        PetscReal       lens[3];
        PetscInt        vals[3], k;

        PetscCall(DMPlexGetCone(dm, p, &cone));
        for (k = 0; k < 3; ++k) {
          PetscCall(DMLabelGetValue(nvb->splitPoints, cone[k], &vals[k]));
          vals[k] = vals[k] == 1 ? 1 : 0;
          PetscCall(UWNVBGetEdgeLen(tr, cone[k], &lens[k]));
        }
        if (vals[0] && vals[1] && vals[2]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT));
        else if (vals[0] && vals[1]) PetscCall(DMLabelSetValue(trType, p, lens[0] > lens[1] ? RT_TRIANGLE_SPLIT_01 : RT_TRIANGLE_SPLIT_10));
        else if (vals[1] && vals[2]) PetscCall(DMLabelSetValue(trType, p, lens[1] > lens[2] ? RT_TRIANGLE_SPLIT_12 : RT_TRIANGLE_SPLIT_21));
        else if (vals[2] && vals[0]) PetscCall(DMLabelSetValue(trType, p, lens[2] > lens[0] ? RT_TRIANGLE_SPLIT_20 : RT_TRIANGLE_SPLIT_02));
        else if (vals[0]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_0));
        else if (vals[1]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_1));
        else PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_2));
      } else PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_UWNVBBisect(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_UWNVB;
  tr->ops->setup                 = DMPlexTransformSetUp_UWNVBBisect;
  tr->ops->destroy               = DMPlexTransformDestroy_UWNVB;
  tr->ops->setdimensions         = UWNVBSetDimensions;
  tr->ops->celltransform         = DMPlexTransformCellTransform_UWNVB;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_UWNVB;
  tr->ops->mapcoordinates        = UWNVBMapCoordinatesBarycenter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_UWNVBBisect(DMPlexTransform tr)
{
  DMPlexRefine_UWNVB *nvb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&nvb));
  tr->data = nvb;
  PetscCall(DMPlexTransformInitialize_UWNVBBisect(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---- driver helpers ---------------------------------------------------- */

/* Length of edge `e` from the local coordinates (standalone, no transform cache). */
static PetscErrorCode uwnvb_edge_len(DM dm, PetscInt e, PetscReal *len)
{
  DM                 cdm;
  Vec                coordsLocal;
  const PetscScalar *coords;
  const PetscInt    *cone;
  PetscScalar       *cA, *cB;
  PetscInt           cdim;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetCone(dm, e, &cone));
  PetscCall(DMGetCoordinatesLocalNoncollective(dm, &coordsLocal));
  PetscCall(VecGetArrayRead(coordsLocal, &coords));
  PetscCall(DMPlexPointLocalRead(cdm, cone[0], coords, &cA));
  PetscCall(DMPlexPointLocalRead(cdm, cone[1], coords, &cB));
  *len = DMPlex_DistD_Internal(cdim, cA, cB);
  PetscCall(VecRestoreArrayRead(coordsLocal, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Refinement edge of triangle `cell`: the stored cone slot, or the longest-edge
   seed when no slot is set (base mesh / uninitialised cell). */
static PetscErrorCode uwnvb_refedge(DM dm, DMLabel slot, PetscInt cell, PetscInt *refedge, PetscInt *refslot)
{
  const PetscInt *cone;
  PetscInt        s = -1, k;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCone(dm, cell, &cone));
  if (slot) PetscCall(DMLabelGetValue(slot, cell, &s));
  if (s >= 0 && s < 3) {
    *refslot = s;
    *refedge = cone[s];
  } else {
    PetscReal len, maxlen = -1.0;
    PetscInt  best = 0;
    for (k = 0; k < 3; ++k) {
      PetscCall(uwnvb_edge_len(dm, cone[k], &len));
      if (len > maxlen) { maxlen = len; best = k; }
    }
    *refslot = best;
    *refedge = cone[best];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Make a point-indexed 0/1 array globally consistent under logical-AND across
   the point SF (serial: no-op). After this, a shared point's value is the AND of
   every rank's local value — used so an edge is "agreed" only if all incident
   cells (on any rank) accept it as their refinement edge. */
static PetscErrorCode uwnvb_sf_land(DM dm, PetscInt *val)
{
  PetscSF  sf;
  PetscInt nroots;

  PetscFunctionBegin;
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  if (nroots < 0) PetscFunctionReturn(PETSC_SUCCESS); /* no SF (serial) */
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, val, val, MPI_LAND));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, val, val, MPI_LAND));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, val, val, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, val, val, MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---- driver: graded NVB refinement of `dm` ----------------------------- */
/*
  Refine `dm` so that every cell flagged in the `wantName` adaptation label
  (values DM_ADAPT_REFINE) is bisected once, with the bounded NVB conforming
  closure, and return the refined DM carrying an updated `uwnvb_refedge` slot.

  Two phases (see NVB_GRADED_ADAPT.md):
   1. CLOSURE — grow C (cells that must bisect) by adding, for each C cell whose
      refinement edge is blocked by a neighbour with a different refinement edge,
      that neighbour (LEPP; bounded).
   2. DRAIN — repeatedly single-split the *compatible* refinement edges (refedge
      of all incident cells) via the uwnvb_bisect transform, until C is empty.
      Each pass is conforming and touches each cell at most once (single split),
      so child slots are trivial (edge opposite the new midpoint).

  NOTE: the CLOSURE currently walks on-rank neighbours only; the SF-reconciled
  cross-rank closure (DMLabelPropagate of requested edges) is the remaining
  parallel step. Compatibility (DRAIN) is already SF-reconciled via uwnvb_sf_land.
*/
PETSC_EXTERN PetscErrorCode UWNVBRefine(DM dm, const char *wantName, DM *rdm)
{
  MPI_Comm comm;
  DM       work;
  DMLabel  want, C, slot = NULL;
  PetscInt cStart, cEnd, c, guard, nC;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(rdm, 3);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetLabel(dm, wantName, &want));
  PetscCheck(want, comm, PETSC_ERR_ARG_WRONGSTATE, "Adaptation label '%s' not found on DM", wantName);

  work = dm;
  PetscCall(PetscObjectReference((PetscObject)work));
  PetscCall(DMGetLabel(work, UWNVB_SLOT_LABEL, &slot)); /* NULL on the base */

  /* --- 1. closure: C = want, grown by blocking neighbours across refinement
     edges. Blockers are discovered by propagating REQUESTED refinement edges over
     the point SF (edges are shared SF points; cells are not) — the same
     DMLabelPropagate mechanism SBR uses, so the closure is cross-rank complete and
     bounded (LEPP; Rivara/Stevenson). Each rank derives its own C cells locally
     from the requested edges incident to them. --- */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "uwnvb_C", &C));
  {
    IS               wantIS;
    const PetscInt  *wantCells;
    DMLabel          req;
    DMPlexPointQueue queue;
    PetscSF          pointSF;
    PetscInt         nWant, i;
    PetscBool        empty;

    PetscCall(DMPlexGetHeightStratum(work, 0, &cStart, &cEnd));
    PetscCall(DMGetPointSF(work, &pointSF));
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "uwnvb_req", &req));
    PetscCall(DMPlexPointQueueCreate(1024, &queue));
    /* seed: each want triangle joins C and requests its refinement edge */
    PetscCall(DMLabelGetStratumIS(want, DM_ADAPT_REFINE, &wantIS));
    PetscCall(DMLabelGetStratumSize(want, DM_ADAPT_REFINE, &nWant));
    if (wantIS) {
      PetscCall(ISGetIndices(wantIS, &wantCells));
      for (i = 0; i < nWant; ++i) {
        DMPolytopeType ct;
        PetscInt       refedge, refslot, rv;
        /* Ignore non-triangle marks — the adaptation label can carry stale
           entries transferred from a previous refine onto non-cell points. */
        if (wantCells[i] < cStart || wantCells[i] >= cEnd) continue;
        PetscCall(DMPlexGetCellType(work, wantCells[i], &ct));
        if (ct != DM_POLYTOPE_TRIANGLE) continue;
        PetscCall(DMLabelSetValue(C, wantCells[i], 1));
        PetscCall(uwnvb_refedge(work, slot, wantCells[i], &refedge, &refslot));
        PetscCall(DMLabelGetValue(req, refedge, &rv));
        if (rv != 1) {
          PetscCall(DMLabelSetValue(req, refedge, 1));
          PetscCall(DMPlexPointQueueEnqueue(queue, refedge));
        }
      }
      PetscCall(ISRestoreIndices(wantIS, &wantCells));
    }
    PetscCall(ISDestroy(&wantIS));
    PetscCall(DMLabelPropagateBegin(req, pointSF));
    PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)work, queue, &empty));
    while (!empty) {
      while (!DMPlexPointQueueEmpty(queue)) {
        PetscInt        e = -1, ss, s;
        const PetscInt *supp;
        PetscCall(DMPlexPointQueueDequeue(queue, &e));
        PetscCall(DMPlexGetSupport(work, e, &supp));
        PetscCall(DMPlexGetSupportSize(work, e, &ss));
        for (s = 0; s < ss; ++s) {
          PetscInt d = supp[s], de, ds, cv, rv;
          PetscCall(uwnvb_refedge(work, slot, d, &de, &ds));
          if (de != e) { /* d blocks a request on e: it must bisect its own refedge first */
            PetscCall(DMLabelGetValue(C, d, &cv));
            if (cv != 1) PetscCall(DMLabelSetValue(C, d, 1));
            PetscCall(DMLabelGetValue(req, de, &rv));
            if (rv != 1) {
              PetscCall(DMLabelSetValue(req, de, 1));
              PetscCall(DMPlexPointQueueEnqueue(queue, de));
            }
          }
        }
      }
      /* exchange newly-requested edges across ranks; the callback enqueues any
         that arrive here so their incident cells are processed next round */
      PetscCall(DMLabelPropagatePush(req, pointSF, UWNVBSplitPoint, queue));
      PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)work, queue, &empty));
    }
    PetscCall(DMLabelPropagateEnd(req, pointSF));
    PetscCall(DMLabelDestroy(&req));
    PetscCall(DMPlexPointQueueDestroy(&queue));
  }

  /* --- 2. drain: single-split compatible refedges of C until C empty --- */
  PetscCall(DMLabelGetStratumSize(C, 1, &nC));
  { /* GLOBAL count: ranks with no local C must still enter the collective loop */
    PetscInt nCG = nC;
    PetscCallMPI(MPIU_Allreduce(&nC, &nCG, 1, MPIU_INT, MPI_SUM, comm));
    nC = nCG;
  }
  guard = 0;
  while (nC > 0) {
    DMPlexTransform tr;
    DM              out;
    DMLabel         bisect, slotOut, Cout;
    PetscInt       *agree, pStart, pEnd, p, e, eStart, eEnd, nready = 0;

    PetscCheck(++guard < 100000, comm, PETSC_ERR_PLIB, "UWNVB drain did not converge");

    /* agree[e] = 1 iff e is the refinement edge of all its incident cells */
    PetscCall(DMPlexGetChart(work, &pStart, &pEnd));
    PetscCall(PetscMalloc1(pEnd - pStart, &agree));
    for (p = 0; p < pEnd - pStart; ++p) agree[p] = 1;
    PetscCall(DMPlexGetHeightStratum(work, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      DMPolytopeType  ct;
      const PetscInt *cone;
      PetscInt        refedge, refslot, k;
      PetscCall(DMPlexGetCellType(work, c, &ct));
      if (ct != DM_POLYTOPE_TRIANGLE) continue;
      PetscCall(uwnvb_refedge(work, slot, c, &refedge, &refslot));
      PetscCall(DMPlexGetCone(work, c, &cone));
      for (k = 0; k < 3; ++k)
        if (cone[k] != refedge) agree[cone[k] - pStart] = 0; /* c vetoes non-refedges */
    }
    PetscCall(uwnvb_sf_land(work, agree));

    /* ready edges = refedge of a C cell that is agreed; mark them for bisection */
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, UWNVB_BISECT_LABEL, &bisect));
    {
      IS              cis;
      const PetscInt *ccells;
      PetscInt        n, i;
      PetscCall(DMLabelGetStratumIS(C, 1, &cis));
      PetscCall(DMLabelGetStratumSize(C, 1, &n));
      if (cis) {
        PetscCall(ISGetIndices(cis, &ccells));
        for (i = 0; i < n; ++i) {
          PetscInt refedge, refslot, bv;
          PetscCall(uwnvb_refedge(work, slot, ccells[i], &refedge, &refslot));
          if (agree[refedge - pStart]) {
            PetscCall(DMLabelGetValue(bisect, refedge, &bv));
            if (bv != 1) { PetscCall(DMLabelSetValue(bisect, refedge, 1)); ++nready; }
          }
        }
        PetscCall(ISRestoreIndices(cis, &ccells));
      }
      PetscCall(ISDestroy(&cis));
    }
    PetscCall(PetscFree(agree));
    {
      PetscInt nreadyG = nready;
      PetscCallMPI(MPIU_Allreduce(&nready, &nreadyG, 1, MPIU_INT, MPI_SUM, comm));
      PetscCheck(nreadyG > 0, comm, PETSC_ERR_PLIB, "UWNVB drain stalled: C non-empty but no compatible edge");
    }

    /* attach the bisect label to `work` and single-split those edges. `work` may
       be the caller's dm on the first pass, so the label is removed again below. */
    PetscCall(DMAddLabel(work, bisect));
    PetscCall(DMLabelDestroy(&bisect)); /* work holds the ref now */
    PetscCall(DMPlexTransformCreate(comm, &tr));
    PetscCall(DMPlexTransformSetType(tr, "uwnvb_bisect"));
    PetscCall(DMPlexTransformSetDM(tr, work));
    PetscCall(DMPlexTransformSetUp(tr));
    PetscCall(DMPlexTransformApply(tr, work, &out));
    { /* clean the transient bisect label off both work (esp. if == caller dm) and out */
      DMLabel tmp = NULL;
      PetscCall(DMRemoveLabel(work, UWNVB_BISECT_LABEL, &tmp));
      PetscCall(DMLabelDestroy(&tmp));
      tmp = NULL;
      PetscCall(DMRemoveLabel(out, UWNVB_BISECT_LABEL, &tmp));
      PetscCall(DMLabelDestroy(&tmp));
    }

    /* rebuild the slot label on `out` (single split => child refedge is the edge
       opposite the new midpoint; unsplit cells copy their slot) */
    {
      DMLabel staleS = NULL;
      PetscCall(DMGetLabel(out, UWNVB_SLOT_LABEL, &staleS));
      if (staleS) { PetscCall(DMRemoveLabel(out, UWNVB_SLOT_LABEL, &staleS)); PetscCall(DMLabelDestroy(&staleS)); }
    }
    PetscCall(DMCreateLabel(out, UWNVB_SLOT_LABEL));
    PetscCall(DMGetLabel(out, UWNVB_SLOT_LABEL, &slotOut));
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "uwnvb_C", &Cout));

    PetscCall(DMPlexGetHeightStratum(work, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      DMPolytopeType ct;
      PetscInt       rt, refedge, refslot, cval, pNew;
      PetscCall(DMPlexGetCellType(work, c, &ct));
      if (ct != DM_POLYTOPE_TRIANGLE) continue;
      PetscCall(DMLabelGetValue(tr->trType, c, &rt));
      PetscCall(uwnvb_refedge(work, slot, c, &refedge, &refslot));
      PetscCall(DMLabelGetValue(C, c, &cval));

      if (rt == RT_TRIANGLE) {
        /* Unsplit: the refinement edge is unchanged, but the identity transform may
           renumber the cone — so map the parent refinement EDGE to its child edge
           and record THAT slot (copying the index would corrupt the labelling). */
        const PetscInt *ccone;
        PetscInt        childRefEdge, j, slotval = 0;
        PetscCall(DMPlexTransformGetTargetPoint(tr, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_TRIANGLE, c, 0, &pNew));
        PetscCall(DMPlexTransformGetTargetPoint(tr, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_SEGMENT, refedge, 0, &childRefEdge));
        PetscCall(DMPlexGetCone(out, pNew, &ccone));
        for (j = 0; j < 3; ++j)
          if (ccone[j] == childRefEdge) { slotval = j; break; }
        PetscCall(DMLabelSetValue(slotOut, pNew, slotval));
        if (cval == 1) PetscCall(DMLabelSetValue(Cout, pNew, 1));
      } else {
        /* single split on refedge: two children, peak = midpoint m */
        PetscInt m, r, j;
        PetscCheck(rt == RT_TRIANGLE_SPLIT_0 || rt == RT_TRIANGLE_SPLIT_1 || rt == RT_TRIANGLE_SPLIT_2,
                   comm, PETSC_ERR_PLIB, "UWNVB drain produced a non-single split (rt=%" PetscInt_FMT ")", rt);
        PetscCall(DMPlexTransformGetTargetPoint(tr, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_POINT, refedge, 0, &m));
        for (r = 0; r < 2; ++r) {
          const PetscInt *ccone;
          PetscInt        cev[3][2], slotval = 0;
          PetscCall(DMPlexTransformGetTargetPoint(tr, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_TRIANGLE, c, r, &pNew));
          PetscCall(DMPlexGetCone(out, pNew, &ccone));
          for (j = 0; j < 3; ++j) {
            const PetscInt *ec;
            PetscCall(DMPlexGetCone(out, ccone[j], &ec));
            cev[j][0] = ec[0];
            cev[j][1] = ec[1];
          }
          for (j = 0; j < 3; ++j)
            if (UWNVBOppVertex(cev, j) == m) { slotval = j; break; }
          PetscCall(DMLabelSetValue(slotOut, pNew, slotval));
          /* children of a bisected cell are done -> NOT added to Cout */
        }
      }
    }

    PetscCall(DMPlexTransformDestroy(&tr));
    PetscCall(DMDestroy(&work));
    work = out;
    slot = slotOut;
    PetscCall(DMLabelDestroy(&C));
    C    = Cout;
    PetscCall(DMLabelGetStratumSize(C, 1, &nC));
    {
      PetscInt nCG = nC;
      PetscCallMPI(MPIU_Allreduce(&nC, &nCG, 1, MPIU_INT, MPI_SUM, comm));
      nC = nCG;
    }
  }

  PetscCall(DMLabelDestroy(&C));
  /* Drop the transferred adaptation label so repeated refines don't accumulate
     stale marks (the transform copies user labels onto every output). */
  {
    DMLabel staleW = NULL;
    PetscCall(DMGetLabel(work, wantName, &staleW));
    if (staleW) { PetscCall(DMRemoveLabel(work, wantName, &staleW)); PetscCall(DMLabelDestroy(&staleW)); }
  }
  *rdm = work;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Idempotent registration entry called from the Cython module on import. */
PetscErrorCode UWNVBTransformRegister(void)
{
  PetscFunctionBegin;
  PetscCall(DMPlexTransformRegister("uwnvb", DMPlexTransformCreate_UWNVB));
  PetscCall(DMPlexTransformRegister("uwnvb_bisect", DMPlexTransformCreate_UWNVBBisect));
  PetscFunctionReturn(PETSC_SUCCESS);
}
