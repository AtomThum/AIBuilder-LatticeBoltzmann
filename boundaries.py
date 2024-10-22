import numpy as np
import math

indices = [[1,8,5],[2,5,6],[3,6,7],[4,7,8]]

class WallBoundary:
    def __init__(self, xResolution: int, yResolution: int, invert: bool = False):
        self.xResolution = xResolution
        self.yResolution = yResolution
        self.invert = invert
        self.boundary = np.full((xResolution, yResolution), invert)

    def cylindricalWall(self, cylinderCenter: list, cylinderRadius):
        for xIndex in range(self.xResolution):
            for yIndex in range(self.yResolution):
                if math.dist(cylinderCenter, [xIndex, yIndex]) <= cylinderRadius:
                    self.boundary[xIndex, yIndex] = not self.invert

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

class PressureBoundary:
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
        self.mainvelocity = ux if direction in [1,3] else uy
        self.minorvelocity = uy if direction in [1,3] else ux
        self.setindices = indices[direction - 1]
        self.getindices = indices[reflectIndex - 1]