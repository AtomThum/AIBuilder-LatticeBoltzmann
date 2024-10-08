import numpy as np
import math


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