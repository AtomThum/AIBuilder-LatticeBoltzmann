import numpy as np
import math
import itertools as itr

np.seterr(divide=None, invalid=None)


class WallBoundary:
    def __init__(self, yResolution: int, xResolution: int, invert: bool = False):
        self.yResolution = yResolution
        self.xResolution = xResolution
        self.invert = invert  # True if boundary, False if not
        self.boundary = np.full((yResolution, xResolution), invert)
        self.invertedBoundary = np.invert(self.boundary)
        self.boundaryIndex = None

    def updateInvertedBoundary(self):
        self.invertedBoundary = np.invert(self.boundary)

    def generateIndex(self):
        self.boundaryIndex = []
        for i, j in itr.product(range(self.yResolution), range(self.xResolution)):
            if self.boundary[i, j] != self.invert:
                self.boundaryIndex.append((i, j))
            else:
                pass
        self.updateInvertedBoundary()

    def cylindricalWall(self, cylinderCenter: list, cylinderRadius: float):
        for yIndex, xIndex in itr.product(
            range(self.yResolution), range(self.xResolution)
        ):
            if math.dist(cylinderCenter, [yIndex, xIndex]) <= cylinderRadius:
                self.boundary[yIndex, xIndex] = not self.invert
        self.updateInvertedBoundary()

    def filledStraightRectangularWall(self, cornerCoord1: tuple, cornerCoord2: tuple):
        maxY = max(cornerCoord1[0], cornerCoord2[0])
        minY = min(cornerCoord1[0], cornerCoord2[0])
        maxX = max(cornerCoord1[1], cornerCoord2[1])
        minX = min(cornerCoord1[1], cornerCoord2[1])

        for yIndex, xIndex in itr.product(
            range(self.yResolution), range(self.xResolution)
        ):
            if (
                (xIndex <= maxX)
                and (xIndex >= minX)
                and (yIndex <= maxY)
                and (yIndex >= minY)
            ):
                self.boundary[yIndex, xIndex] = not self.invert
        self.updateInvertedBoundary()

    # Border around the simulation
    # Thickness will be implemented later.
    def borderWall(self, thickness: int = 1):
        self.boundary[0 : self.yResolution, -1 + thickness] = not self.invert
        self.boundary[0 : self.yResolution, self.xResolution - thickness] = (
            not self.invert
        )
        self.boundary[-1 + thickness, 0 : self.xResolution] = not self.invert
        self.boundary[self.yResolution - thickness, 0 : self.xResolution] = (
            not self.invert
        )
        self.updateInvertedBoundary()

    # By numpy's convention, it's (0, 1)
    def dotWalls(self, *args: tuple):
        for position in args:
            self.boundary[position[0], position[1]] = not self.invert


class PressureBoundary:
    indices = [[1, 8, 5], [2, 5, 6], [3, 6, 7], [4, 7, 8]]

    def __init__(self, y: int, x: int, ux, uy, direction: int):
        self.y = y
        self.x = x
        self.uy = uy
        self.ux = ux
        self.direction = direction
        if direction in [3, 4]:
            reflectIndex = direction - 2
        else:
            reflectIndex = direction + 2

        self.mainVelocity = ux if direction in [1, 3] else uy
        self.minorVelocity = uy if direction in [1, 3] else ux
        self.setIndices = PressureBoundary.indices[direction - 1]
        self.getIndices = PressureBoundary.indices[reflectIndex - 1]


class VelocityBoundary:
    def __init__(self, y: int, x: int, magnitude: tuple, direction: tuple):
        self.y = y
        self.x = x
        self.magnitude = magnitude # Follows the e conventions
        self.direction = np.array(direction)
