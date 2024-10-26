import numpy as np
import math
import itertools as itr

np.seterr(divide=None, invalid=None)


class WallBoundary:
    def __init__(self, xResolution: int, yResolution: int, invert: bool = False):
        self.xResolution = xResolution
        self.yResolution = yResolution
        self.invert = invert  # True if boundary, False if not
        self.boundary = np.full((xResolution, yResolution), invert)
        self.invertedBoundary = np.invert(self.boundary)
        self.boundaryIndex = None

    def updateInvertedBoundary(self):
        self.invertedBoundary = np.invert(self.boundary)

    def generateIndex(self):
        self.boundaryIndex = []
        for i, j in itr.product(range(self.xResolution), range(self.yResolution)):
            if self.boundary[i, j] != self.invert:
                self.boundaryIndex.append((i, j))
            else:
                pass
        self.updateInvertedBoundary()

    def cylindricalWall(self, cylinderCenter: list, cylinderRadius: float):
        for xIndex in range(self.xResolution):
            for yIndex in range(self.yResolution):
                if math.dist(cylinderCenter, [xIndex, yIndex]) <= cylinderRadius:
                    self.boundary[xIndex, yIndex] = not self.invert
        self.updateInvertedBoundary()

    def filledStraightRectangularWall(self, cornerCoord1: list, cornerCoord2: list):
        maxX = max(cornerCoord1[0], cornerCoord2[0])
        minX = min(cornerCoord1[0], cornerCoord2[0])

        maxY = max(cornerCoord1[1], cornerCoord2[1])
        minY = min(cornerCoord1[1], cornerCoord2[1])
        for xIndex in range(self.xResolution):
            for yIndex in range(self.yResolution):
                if (
                    (xIndex < maxX)
                    and (xIndex > minX)
                    and (yIndex < maxY)
                    and (yIndex > minY)
                ):
                    self.boundary[xIndex, yIndex] = not self.invert
        self.updateInvertedBoundary()

    # Border around the simulation
    # Thickness will be implemented later.
    def borderWall(self, thickness: int = 1):
        self.boundary[0 : self.xResolution, -1 + thickness] = not self.invert
        self.boundary[0 : self.xResolution, self.yResolution - thickness] = (
            not self.invert
        )
        self.boundary[-1 + thickness, 0 : self.yResolution] = not self.invert
        self.boundary[self.xResolution - thickness, 0 : self.yResolution] = (
            not self.invert
        )
        self.updateInvertedBoundary()
    
    def dotWall(self, *args: tuple):
        for position in args:
            self.boundary[position] = not self.invert


class PressureBoundary:
    indices = [[1, 8, 5], [2, 5, 6], [3, 6, 7], [4, 7, 8]]

    def __init__(self, x: int, y: int, ux, uy, direction: int):
        self.x = x
        self.y = y
        self.ux = ux
        self.uy = uy
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
    def __init__(self, x: int, y: int, magnitude: list, direction: list):
        self.x = x
        self.y = y
        self.magnitude = magnitude
        self.direction = np.array(direction)
