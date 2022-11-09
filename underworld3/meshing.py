from typing import Optional, Tuple
from enum import Enum

import tempfile
import numpy as np
from petsc4py import PETSc

from underworld3.discretisation import Mesh, MeshVariable
from underworld3 import VarType
from underworld3.coordinates import CoordinateSystemType
import sympy


def UnstructuredSimplexBox(
    minCoords: Tuple = (0.0, 0.0),
    maxCoords: Tuple = (1.0, 1.0),
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    regular: bool = False,
    filename=None,
):

    """
    Generates a 2 or 3-dimensional box mesh.

    Parameters
    ----------
    minCoord:
        Tuple specifying minimum mesh location.
    maxCoord:
        Tuple specifying maximum mesh location.

    regular option works in 2D but not (currently) in 3D
    """

    boundaries = {"Bottom": 1, "Top": 2, "Right": 3, "Left": 4, "Front": 5, "Back": 6}

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Box")

    # Create Box Geometry
    dim = len(minCoords)

    if dim == 2:

        xmin, ymin = minCoords
        xmax, ymax = maxCoords

        p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
        p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
        p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
        p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

        l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
        l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
        l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
        l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])

        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        surface = gmsh.model.geo.add_plane_surface([cl])

        gmsh.model.geo.synchronize()

        # Add Physical groups
        for name, tag in boundaries.items():
            gmsh.model.add_physical_group(1, [tag], tag)
            gmsh.model.set_physical_name(1, tag, name)

        gmsh.model.addPhysicalGroup(2, [surface], surface)
        gmsh.model.setPhysicalName(2, surface, "Elements")

        if regular:
            gmsh.model.mesh.set_transfinite_surface(
                surface, cornerTags=[p1, p2, p3, p4]
            )

    else:

        xmin, ymin, zmin = minCoords
        xmax, ymax, zmax = maxCoords

        p1 = gmsh.model.geo.add_point(xmin, ymin, zmin, meshSize=cellSize)
        p2 = gmsh.model.geo.add_point(xmax, ymin, zmin, meshSize=cellSize)
        p3 = gmsh.model.geo.add_point(xmin, ymax, zmin, meshSize=cellSize)
        p4 = gmsh.model.geo.add_point(xmax, ymax, zmin, meshSize=cellSize)
        p5 = gmsh.model.geo.add_point(xmin, ymin, zmax, meshSize=cellSize)
        p6 = gmsh.model.geo.add_point(xmax, ymin, zmax, meshSize=cellSize)
        p7 = gmsh.model.geo.add_point(xmin, ymax, zmax, meshSize=cellSize)
        p8 = gmsh.model.geo.add_point(xmax, ymax, zmax, meshSize=cellSize)

        l1 = gmsh.model.geo.add_line(p1, p2)
        l2 = gmsh.model.geo.add_line(p2, p4)
        l3 = gmsh.model.geo.add_line(p4, p3)
        l4 = gmsh.model.geo.add_line(p3, p1)
        l5 = gmsh.model.geo.add_line(p5, p6)
        l6 = gmsh.model.geo.add_line(p6, p8)
        l7 = gmsh.model.geo.add_line(p8, p7)
        l8 = gmsh.model.geo.add_line(p7, p5)
        l9 = gmsh.model.geo.add_line(p5, p1)
        l10 = gmsh.model.geo.add_line(p2, p6)
        l11 = gmsh.model.geo.add_line(p7, p3)
        l12 = gmsh.model.geo.add_line(p4, p8)

        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        bottom = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Bottom"])

        cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
        top = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Top"])

        cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
        right = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Right"])

        cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
        left = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Left"])

        cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
        front = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Front"])

        cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
        back = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Back"])

        sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
        volume = gmsh.model.geo.add_volume([sloop])

        gmsh.model.geo.synchronize()

        # Add Physical groups
        for name, tag in boundaries.items():
            gmsh.model.add_physical_group(2, [tag], tag)
            gmsh.model.set_physical_name(2, tag, name)

        gmsh.model.addPhysicalGroup(3, [volume], volume)
        gmsh.model.setPhysicalName(3, volume, "Elements")

    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(dim)
        gmsh.write(fp.name)
        if filename:
            gmsh.write(filename)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)

    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")

    return Mesh(
        plex,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        filename=filename,
    )


def StructuredQuadBox(
    elementRes: Optional[Tuple[int, int, int]] = (16, 16),
    minCoords: Optional[Tuple[float, float, float]] = None,
    maxCoords: Optional[Tuple[float, float, float]] = None,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
):

    """
    Generates a 2 or 3-dimensional box mesh.

    Parameters
    ----------
    elementRes:
        Tuple specifying number of elements in each axis direction.
    minCoord:
        Optional. Tuple specifying minimum mesh location.
    maxCoord:
        Optional. Tuple specifying maximum mesh location.
    """
    if minCoords == None:
        minCoords = len(elementRes) * (0.0,)
    if maxCoords == None:
        maxCoords = len(elementRes) * (1.0,)

    import gmsh

    boundaries = {"Bottom": 1, "Top": 2, "Right": 3, "Left": 4, "Front": 5, "Back": 6}

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Box")

    # Create Box Geometry
    dim = len(minCoords)

    if dim == 2:

        xmin, ymin = minCoords
        xmax, ymax = maxCoords

        p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, tag=1)
        p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, tag=2)
        p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, tag=3)
        p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, tag=4)

        l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
        l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
        l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
        l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])

        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        surface = gmsh.model.geo.add_plane_surface([cl])

        gmsh.model.geo.synchronize()

        # Add Physical groups
        gmsh.model.add_physical_group(1, [l1], l1)
        gmsh.model.set_physical_name(1, l1, "Bottom")
        gmsh.model.add_physical_group(1, [l2], l2)
        gmsh.model.set_physical_name(1, l2, "Right")
        gmsh.model.add_physical_group(1, [l3], l3)
        gmsh.model.set_physical_name(1, l3, "Top")
        gmsh.model.add_physical_group(1, [l4], l4)
        gmsh.model.set_physical_name(1, l4, "Left")

        gmsh.model.add_physical_group(2, [surface], surface)
        gmsh.model.set_physical_name(2, surface, "Elements")

        nx, ny = elementRes

        gmsh.model.mesh.set_transfinite_curve(
            tag=l1, numNodes=nx + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            tag=l2, numNodes=ny + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            tag=l3, numNodes=nx + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            tag=l4, numNodes=ny + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_surface(
            tag=surface, arrangement="Left", cornerTags=[p1, p2, p3, p4]
        )
        gmsh.model.mesh.set_recombine(2, surface)

    else:

        xmin, ymin, zmin = minCoords
        xmax, ymax, zmax = maxCoords

        p1 = gmsh.model.geo.add_point(xmin, ymin, zmin)
        p2 = gmsh.model.geo.add_point(xmax, ymin, zmin)
        p3 = gmsh.model.geo.add_point(xmin, ymax, zmin)
        p4 = gmsh.model.geo.add_point(xmax, ymax, zmin)
        p5 = gmsh.model.geo.add_point(xmin, ymin, zmax)
        p6 = gmsh.model.geo.add_point(xmax, ymin, zmax)
        p7 = gmsh.model.geo.add_point(xmin, ymax, zmax)
        p8 = gmsh.model.geo.add_point(xmax, ymax, zmax)

        l1 = gmsh.model.geo.add_line(p1, p2)
        l2 = gmsh.model.geo.add_line(p2, p4)
        l3 = gmsh.model.geo.add_line(p4, p3)
        l4 = gmsh.model.geo.add_line(p3, p1)
        l5 = gmsh.model.geo.add_line(p5, p6)
        l6 = gmsh.model.geo.add_line(p6, p8)
        l7 = gmsh.model.geo.add_line(p8, p7)
        l8 = gmsh.model.geo.add_line(p7, p5)
        l9 = gmsh.model.geo.add_line(p5, p1)
        l10 = gmsh.model.geo.add_line(p2, p6)
        l11 = gmsh.model.geo.add_line(p7, p3)
        l12 = gmsh.model.geo.add_line(p4, p8)

        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        bottom = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Bottom"])

        cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
        top = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Top"])

        cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
        right = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Right"])

        cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
        left = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Left"])

        cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
        front = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Front"])

        cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
        back = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Back"])

        sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
        volume = gmsh.model.geo.add_volume([sloop])

        gmsh.model.geo.synchronize()

        nx, ny, nz = elementRes

        gmsh.model.mesh.set_transfinite_curve(
            l1, numNodes=nx + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l2, numNodes=ny + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l3, numNodes=nx + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l4, numNodes=ny + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l5, numNodes=nx + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l6, numNodes=ny + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l7, numNodes=nx + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l8, numNodes=ny + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l9, numNodes=nz + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l10, numNodes=nz + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l11, numNodes=nz + 1, meshType="Progression", coef=1.0
        )
        gmsh.model.mesh.set_transfinite_curve(
            l12, numNodes=nz + 1, meshType="Progression", coef=1.0
        )

        gmsh.model.mesh.set_transfinite_surface(
            tag=bottom, arrangement="Left", cornerTags=[p1, p2, p4, p3]
        )
        gmsh.model.mesh.set_transfinite_surface(
            tag=top, arrangement="Left", cornerTags=[p5, p6, p8, p7]
        )
        gmsh.model.mesh.set_transfinite_surface(
            tag=front, arrangement="Left", cornerTags=[p1, p2, p6, p5]
        )
        gmsh.model.mesh.set_transfinite_surface(
            tag=back, arrangement="Left", cornerTags=[p3, p4, p8, p7]
        )
        gmsh.model.mesh.set_transfinite_surface(
            tag=right, arrangement="Left", cornerTags=[p2, p6, p8, p4]
        )
        gmsh.model.mesh.set_transfinite_surface(
            tag=left, arrangement="Left", cornerTags=[p5, p1, p3, p7]
        )

        gmsh.model.mesh.set_recombine(2, front)
        gmsh.model.mesh.set_recombine(2, back)
        gmsh.model.mesh.set_recombine(2, bottom)
        gmsh.model.mesh.set_recombine(2, top)
        gmsh.model.mesh.set_recombine(2, right)
        gmsh.model.mesh.set_recombine(2, left)

        gmsh.model.mesh.set_transfinite_volume(
            volume, cornerTags=[p1, p2, p4, p3, p5, p6, p8, p7]
        )

        # Add Physical groups
        for name, tag in boundaries.items():
            gmsh.model.add_physical_group(2, [tag], tag)
            gmsh.model.set_physical_name(2, tag, name)

        gmsh.model.addPhysicalGroup(3, [volume], volume)
        gmsh.model.setPhysicalName(3, volume, "Elements")

    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(dim)
        gmsh.write(fp.name)
        if filename:
            gmsh.write(filename)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)

    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")

    return Mesh(
        plex,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        filename=filename,
    )


def SphericalShell(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.1,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
):

    boundaries = {"Lower": 1, "Upper": 2}

    vertices = {"Centre": 1}

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Sphere")

    p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize)

    ball1_tag = gmsh.model.occ.addSphere(0, 0, 0, radiusOuter)

    if radiusInner > 0.0:
        ball2_tag = gmsh.model.occ.addSphere(0, 0, 0, radiusInner)
        gmsh.model.occ.cut(
            [(3, ball1_tag)], [(3, ball2_tag)], removeObject=True, removeTool=True
        )

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cellSize)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    volume = gmsh.model.getEntities(3)[0]

    if radiusInner > 0.0:
        outerSurface, innerSurface = surfaces
        gmsh.model.addPhysicalGroup(
            innerSurface[0], [innerSurface[1]], boundaries["Lower"]
        )
        gmsh.model.setPhysicalName(innerSurface[1], boundaries["Lower"], "Lower")
        gmsh.model.addPhysicalGroup(
            outerSurface[0], [outerSurface[1]], boundaries["Upper"]
        )
        gmsh.model.setPhysicalName(outerSurface[1], boundaries["Upper"], "Upper")
        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], volume[1])
        gmsh.model.setPhysicalName(volume[1], volume[1], "Elements")

    else:
        outerSurface = surfaces[0]
        gmsh.model.addPhysicalGroup(
            outerSurface[0], [outerSurface[1]], boundaries["Upper"]
        )
        gmsh.model.setPhysicalName(outerSurface[1], boundaries["Upper"], "Upper")
        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], volume[1])
        gmsh.model.setPhysicalName(volume[1], volume[1], "Elements")
        gmsh.model.addPhysicalGroup(0, [p1], tag=vertices["Centre"])
        gmsh.model.setPhysicalName(0, vertices["Centre"], "Centre")

    gmsh.model.occ.synchronize()

    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(3)
        gmsh.write(fp.name)
        if filename:
            gmsh.write(filename)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)

    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")

    # This seems not to work any longer ?? 3.17.4
    for name, tag in vertices.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Vertex Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Vertex Sets")

    return Mesh(
        plex,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        filename=filename,
    )


def Annulus(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.3,
    cellSize: float = 0.1,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
):

    boundaries = {"Lower": 1, "Upper": 2, "FixedStars": 3}

    vertices = {"Centre": 10}

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("Annulus")

    p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize)

    loops = []

    if radiusInner > 0.0:
        p2 = gmsh.model.geo.add_point(radiusInner, 0.0, 0.0, meshSize=cellSize)
        p3 = gmsh.model.geo.add_point(-radiusInner, 0.0, 0.0, meshSize=cellSize)

        c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
        c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

        cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries["Lower"])

        loops = [cl1] + loops

    p4 = gmsh.model.geo.add_point(radiusOuter, 0.0, 0.0, meshSize=cellSize)
    p5 = gmsh.model.geo.add_point(-radiusOuter, 0.0, 0.0, meshSize=cellSize)

    c3 = gmsh.model.geo.add_circle_arc(p4, p1, p5)
    c4 = gmsh.model.geo.add_circle_arc(p5, p1, p4)

    # l1 = gmsh.model.geo.add_line(p5, p4)

    cl2 = gmsh.model.geo.add_curve_loop([c3, c4], tag=boundaries["Upper"])

    loops = [cl2] + loops

    s = gmsh.model.geo.add_plane_surface(loops)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(0, [p1], 2, s)
    # gmsh.model.mesh.embed(1, [l1], 2, s)

    if radiusInner > 0.0:
        gmsh.model.addPhysicalGroup(1, [c1, c2], boundaries["Lower"], name="Lower")
    else:
        gmsh.model.addPhysicalGroup(0, [p1], tag=vertices["Centre"], name="Centre")

    gmsh.model.addPhysicalGroup(1, [c3, c4], boundaries["Upper"], name="Upper")

    gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

    gmsh.model.geo.synchronize()

    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(2)
        gmsh.write(fp.name)
        if filename:
            gmsh.write(filename)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)

    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    for name, tag in vertices.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Vertex Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    return Mesh(
        plex,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        filename=filename,
    )


def AnnulusFixedStars(
    radiusFixedStars: float = 1.5,
    radiusOuter: float = 1.0,
    radiusInner: float = 0.5,
    cellSize: float = 0.1,
    cellSize_FS: float = 0.2,
    centre: bool = False,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
):

    boundaries = {"Lower": 1, "Upper": 2, "FixedStars": 3}
    vertices = {"Centre": 10}

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("AnnulusFS")

    p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize)

    loops = []

    if radiusInner > 0.0:
        p2 = gmsh.model.geo.add_point(radiusInner, 0.0, 0.0, meshSize=cellSize)
        p3 = gmsh.model.geo.add_point(-radiusInner, 0.0, 0.0, meshSize=cellSize)

        c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
        c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

        cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries["Lower"])

        loops = [cl1] + loops

    p4 = gmsh.model.geo.add_point(radiusOuter, 0.0, 0.0, meshSize=cellSize)
    p5 = gmsh.model.geo.add_point(-radiusOuter, 0.0, 0.0, meshSize=cellSize)

    c3 = gmsh.model.geo.add_circle_arc(p4, p1, p5)
    c4 = gmsh.model.geo.add_circle_arc(p5, p1, p4)

    # Fixed Stars

    p6 = gmsh.model.geo.add_point(radiusFixedStars, 0.0, 0.0, meshSize=cellSize_FS)
    p7 = gmsh.model.geo.add_point(-radiusFixedStars, 0.0, 0.0, meshSize=cellSize_FS)

    c5 = gmsh.model.geo.add_circle_arc(p6, p1, p7)
    c6 = gmsh.model.geo.add_circle_arc(p7, p1, p6)

    cl2 = gmsh.model.geo.add_curve_loop([c3, c4], tag=boundaries["Upper"])
    cl3 = gmsh.model.geo.add_curve_loop([c5, c6], tag=boundaries["FixedStars"])

    loops = [cl3] + loops

    s = gmsh.model.geo.add_plane_surface(loops)

    gmsh.model.geo.synchronize()

    if radiusInner == 0.0:
        gmsh.model.mesh.embed(0, [p1], 2, s)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [c3, c4], 2, s)

    gmsh.model.geo.synchronize()

    if radiusInner > 0.0:
        gmsh.model.addPhysicalGroup(1, [c1, c2], boundaries["Lower"], name="Lower")
    else:
        gmsh.model.addPhysicalGroup(0, [p1], tag=vertices["Centre"], name="Centre")

    gmsh.model.addPhysicalGroup(
        1,
        [c3, c4],
        boundaries["Upper"],
        name="Upper",
    )
    gmsh.model.addPhysicalGroup(
        1,
        [c5, c6],
        boundaries["FixedStars"],
        name="FixedStars",
    )

    gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

    gmsh.model.geo.synchronize()

    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(2)
        gmsh.write(fp.name)
        if filename:
            gmsh.write(filename)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)

    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    for name, tag in vertices.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Vertex Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    return Mesh(
        plex,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        filename=filename,
    )


def CubedSphere(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.3,
    numElements: int = 5,
    degree: int = 1,
    qdegree: int = 2,
    simplex: bool = False,
    filename=None,
):

    """Cubed Sphere mesh in hexahedra (which can be left uncombined to produce a simplex-based mesh
    The number of elements is the edge of each cube"""

    boundaries = {"Lower": 1, "Upper": 2}

    r1 = radiusInner / np.sqrt(3)
    r2 = radiusOuter / np.sqrt(3)

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Cubed Sphere")

    center_point = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, tag=1)

    gmsh.model.geo.addPoint(r2, r2, -r2, tag=2)
    gmsh.model.geo.addPoint(-r2, r2, -r2, tag=3)
    gmsh.model.geo.addPoint(-r2, -r2, -r2, tag=4)
    gmsh.model.geo.addPoint(r2, -r2, -r2, tag=5)

    gmsh.model.geo.addCircleArc(3, 1, 2, tag=1)
    gmsh.model.geo.addCircleArc(2, 1, 5, tag=2)
    gmsh.model.geo.addCircleArc(5, 1, 4, tag=3)
    gmsh.model.geo.addCircleArc(4, 1, 3, tag=4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)
    gmsh.model.geo.addSurfaceFilling([1], tag=1, sphereCenterTag=1)

    gmsh.model.geo.addPoint(r1, r1, -r1, tag=6)
    gmsh.model.geo.addPoint(-r1, r1, -r1, tag=7)
    gmsh.model.geo.addPoint(-r1, -r1, -r1, tag=8)
    gmsh.model.geo.addPoint(r1, -r1, -r1, tag=9)

    gmsh.model.geo.addCircleArc(7, 1, 6, tag=5)
    gmsh.model.geo.addCircleArc(6, 1, 9, tag=6)
    gmsh.model.geo.addCircleArc(9, 1, 8, tag=7)
    gmsh.model.geo.addCircleArc(8, 1, 7, tag=8)

    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], tag=2)
    gmsh.model.geo.addSurfaceFilling([2], tag=2, sphereCenterTag=1)

    gmsh.model.geo.addLine(2, 6, tag=9)
    gmsh.model.geo.addLine(3, 7, tag=10)
    gmsh.model.geo.addLine(5, 9, tag=11)
    gmsh.model.geo.addLine(4, 8, tag=12)

    gmsh.model.geo.addCurveLoop([3, 12, -7, -11], tag=3)
    gmsh.model.geo.addSurfaceFilling([3], tag=3)

    gmsh.model.geo.addCurveLoop([10, 5, -9, -1], tag=4)
    gmsh.model.geo.addSurfaceFilling([4], tag=4)

    gmsh.model.geo.addCurveLoop([9, 6, -11, -2], tag=5)
    gmsh.model.geo.addSurfaceFilling([5], tag=5)

    gmsh.model.geo.addCurveLoop([12, 8, -10, -4], tag=6)
    gmsh.model.geo.addSurfaceFilling([6], tag=6)

    gmsh.model.geo.addSurfaceLoop([2, 4, 6, 3, 1, 5], tag=1)
    gmsh.model.geo.addVolume([1], tag=1)

    # Make copies
    gmsh.model.geo.rotate(
        gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.pi / 2.0
    )
    gmsh.model.geo.rotate(
        gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.pi
    )
    gmsh.model.geo.rotate(
        gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0 * np.pi / 2.0
    )
    gmsh.model.geo.rotate(
        gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.pi / 2.0
    )
    gmsh.model.geo.rotate(
        gmsh.model.geo.copy([(3, 1)]), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -np.pi / 2.0
    )

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [1, 34, 61, 88, 115, 137], boundaries["Upper"])
    gmsh.model.setPhysicalName(2, boundaries["Upper"], "Upper")
    gmsh.model.addPhysicalGroup(2, [2, 14, 41, 68, 95, 117], boundaries["Lower"])
    gmsh.model.setPhysicalName(2, boundaries["Lower"], "Lower")

    gmsh.model.addPhysicalGroup(3, [1, 13, 40, 67, 94, 116], 1)
    gmsh.model.setPhysicalName(3, 1, "Elements")

    for _, line in gmsh.model.get_entities(1):
        gmsh.model.mesh.setTransfiniteCurve(line, numNodes=numElements + 1)

    for _, surface in gmsh.model.get_entities(2):
        gmsh.model.mesh.setTransfiniteSurface(surface)
        if not simplex:
            gmsh.model.mesh.set_recombine(2, surface)

    if not simplex:
        for _, volume in gmsh.model.get_entities(3):
            gmsh.model.mesh.set_transfinite_volume(volume)

    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(3)
        gmsh.write(fp.name)
        if filename:
            gmsh.write(filename)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)

    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")

    return Mesh(
        plex,
        degree=degree,
        qdegree=qdegree,
        coordinate_system_type=CoordinateSystemType.SPHERICAL,
        filename=filename,
    )
