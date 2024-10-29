import numpy as np
import copy
import math
import itertools as itr
import random
from scipy.ndimage import convolve

np.seterr(divide=None, invalid=None)


def numericalInverse(n):
    return int(n != 1)


def addTuple(a, b):
    return tuple(i + j for i, j in zip(a, b))


def angleToPosition(magnitude: float, angle: float):
    return magnitude * np.array([[np.sin(angle), np.cos(angle)]])


class WallBoundary:
    unitVect = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    unitX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    unitY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    mineSweeper = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    directions = [(1, -1), (1, 1), (1, 1), (-1, 1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    unitVect = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )

    def __init__(self, yResolution: int, xResolution: int, invert: bool = False):
        self.yResolution = yResolution
        self.xResolution = xResolution
        self.invert = invert  # True if boundary, False if not
        self.boundary = np.full((yResolution, xResolution), invert)
        self.invertedBoundary = np.invert(self.boundary)
        self.boundaryIndex = []
        self.invertedBoundaryIndex = []

        self.possibleACPos = np.full((yResolution, xResolution), False)
        self.possibleACIndex = []
        self.possibleACDirections = None

        self.cutPositionsX = []
        self.cutPositionsY = []
        self.cutTypes = []
        self.cutSizesX = []
        self.cutSizesY = []

        self.possiblePositions = [
            (int(self.yResolution / 3), self.xResolution - 1),
            (0, int(self.xResolution / 3)),
            (int(self.yResolution / 3), 0),
            (self.yResolution - 1, int(self.xResolution / 3)),
            (0, self.xResolution - 1),
            (0, 0),
            (self.yResolution - 1, 0),
            (self.yResolution - 1, self.xResolution - 1),
        ]

    def updateInvertedBoundary(self):
        self.invertedBoundary = np.invert(self.boundary)

    def generateIndex(self):
        self.boundaryIndex = []
        self.invertedBoundaryIndex = []
        for i, j in itr.product(range(self.yResolution), range(self.xResolution)):
            if self.boundary[i, j] != self.invert:
                self.boundaryIndex.append((i, j))
            else:
                self.invertedBoundaryIndex.append((i, j))
        self.updateInvertedBoundary()

    def generateRoom(self):
        for i in random.sample(range(8), k=random.randint(1, 8)):
            wallPos = self.possiblePositions[i]
            maxSize = int(min(self.yResolution, self.xResolution) * 0.4)
            minSize = int(min(self.yResolution, self.xResolution) * 0.2)
            sizeX = random.randint(minSize, maxSize)
            sizeY = random.randint(minSize, maxSize)
            if random.random() < 0.5:
                self.cylindricalWall(wallPos, sizeX)
                self.cutTypes.append(0)
                self.cutPositionsX.append(wallPos[1])
                self.cutPositionsY.append(wallPos[0])
                self.cutSizesX.append(sizeX)
                self.cutSizesY.append(sizeX)

            else:
                endPos = (
                    wallPos[0] + (sizeY * WallBoundary.directions[i][0]),
                    wallPos[1] + (sizeX * WallBoundary.directions[i][1]),
                )
                self.filledStraightRectangularWall(wallPos, endPos)
                self.cutTypes.append(1)
                self.cutPositionsX.append(wallPos[1])
                self.cutPositionsY.append(wallPos[0])
                self.cutSizesX.append(sizeX)
                self.cutSizesY.append(sizeY)

        return {
            "SizeX": self.xResolution,
            "SizeY": self.yResolution,
            "NumberOfCuts": len(self.cutTypes),
            "TypesOfCuts": self.cutTypes,
            "CutPositionsX": self.cutPositionsX,
            "CutPositionsY": self.cutPositionsY,
            "CutSizesX": self.cutSizesX,
            "CutSizesY": self.cutSizesY,
        }

    def generateACPosandDirections(self):
        self.possibleACPos = []
        self.possibleACDirections = []
        applyall = np.vectorize(numericalInverse)
        inverted = applyall(self.boundary.astype(int))
        padded_array = np.pad(inverted, pad_width=1, mode="constant", constant_values=0)
        edgeSwept = convolve(padded_array, WallBoundary.mineSweeper)[1:-1, 1:-1]
        edgesProcessed = np.logical_and(edgeSwept >= 2, edgeSwept <= 5)

        for i in self.boundaryIndex:
            if edgesProcessed[i[0], i[1]]:
                self.possibleACPos.append(i)
                possibleACDirection = []
                for index, j in enumerate(WallBoundary.unitVect):
                    if (
                        i[0] + j[0] >= 0
                        and i[0] + j[0] < self.yResolution
                        and i[1] + j[1] >= 0
                        and i[1] + j[1] < self.xResolution
                    ):
                        isWallThere = self.boundary[i[0] + j[0], i[1] + j[1]]
                        if not isWallThere:
                            possibleACDirection.append(index)
                self.possibleACDirections.append(possibleACDirection)

    def generateACDirections(self):
        for shiftIndex, axisIndex in itr.product([-1, 1], [1, 0]):
            shiftedBoundary = np.roll(self.boundary, shift=shiftIndex, axis=axisIndex)
            self.possibleACPos = np.logical_or(
                self.possibleACPos, self.invertedBoundary & shiftedBoundary
            )

    def indexPossibleACPos(self, clear: bool = False):
        if clear:
            self.possibleACIndex = []
        else:
            pass

        testArray = copy.deepcopy(self.possibleACPos)
        currentIndex = tuple()
        for yIndex, xIndex in itr.product(
            range(self.yResolution), range(self.xResolution)
        ):
            if testArray[yIndex, xIndex]:
                currentIndex = (yIndex, xIndex)
                break

        while testArray[currentIndex]:
            for latticeIndex in [1, 2, 3, 4, 5, 6, 7, 8, 0]:
                nextIndex = addTuple(
                    currentIndex,
                    (
                        WallBoundary.unitX[latticeIndex],
                        WallBoundary.unitY[latticeIndex],
                    ),
                )
                if testArray[nextIndex]:
                    self.possibleACIndex.append(nextIndex)
                    testArray[currentIndex] = 0
                    currentIndex = nextIndex
                    break
                else:
                    pass

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
    def __init__(self, y: int, x: int, magnitude, direction):
        self.y = y
        self.x = x
        self.magnitude = magnitude  # Follows the e conventions
        self.direction = np.array(direction)


class AC:
    unitVect = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    unitVectInv = np.transpose(np.linalg.pinv(unitVect))

    def __init__(
        self, y: int, x: int, magnitude: float, angle: float, threshold: float = 0.4
    ):
        self.y = y
        self.x = x
        self.magnitude = magnitude
        self.angle = angle
        self.threshold = threshold

        self.velocityBoundaries = []

        testVect = angleToPosition(self.angle, self.magnitude)
        result = np.matmul(AC.unitVectInv, np.transpose(testVect))
        result = result.reshape(9)

        for i in range(9):
            if result[i] > self.threshold:
                self.velocityBoundaries.append(
                    VelocityBoundary(self.y, self.x, result[i], i)
                )
            else:
                pass
