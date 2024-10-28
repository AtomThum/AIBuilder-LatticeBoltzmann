import numpy as np
import itertools as itr
import random
from boundaries import WallBoundary


class WallGenerators:
    def __init__(self, wallBoundary: WallBoundary):
        self.wallBoundary = wallBoundary
        self.xResolution = wallBoundary.xResolution
        self.yResolution = wallBoundary.yResolution

    def generateRandomBlocks(self, blockAmount: int, blockSize: int):
        xBlockSize = np.random.randint(1, blockSize, size=blockAmount)
        yBlockSize = np.random.randint(1, blockSize, size=blockAmount)

        x1Corner = np.random.randint(0, self.xResolution, size=blockAmount)
        y1Corner = np.random.randint(0, self.yResolution, size=blockAmount)

        x2Corner = x1Corner + xBlockSize
        y2Corner = y1Corner + yBlockSize

        for _ in range(blockAmount):
            self.wallBoundary.filledStraightRectangularWall(
                [x1Corner[_], y1Corner[_]], [x2Corner[_], y2Corner[_]]
            )

    def generateRoomContour(self):
        for i in random.sample(range(8), k=random.randint(1, 8)):
            possiblePositions = [
                (int(self.yResolution / 3), self.xResolution - 1),
                (0, int(self.xResolution / 3)),
                (int(self.yResolution / 3), 0),
                (self.yResolution - 1, int(self.xResolution / 3)),
                (0, self.xResolution - 1),
                (0, 0),
                (self.yResolution - 1, 0),
                (self.yResolution - 1, self.xResolution - 1),
            ]

            wallPos = possiblePositions[i]
            maxSize = int(min(self.yResolution, self.xResolution) * 0.4)
            minSize = int(min(self.yResolution, self.xResolution) * 0.2)
            if random.random() < 0.5:
                self.wallBoundary.cylindricalWall(
                    wallPos, random.randint(minSize, maxSize)
                )
            else:
                directions = [
                    (1, -1),
                    (1, 1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (-1, 1),
                    (-1, -1),
                ]
                endPos = (
                    wallPos[0] + (random.randint(minSize, maxSize) * directions[i, 0]),
                    wallPos[1] + (random.randint(minSize, maxSize) * directions[i, 1]),
                )
                self.wallBoundary.filledStraightRectangularWall(wallPos, endPos)


boundary = WallBoundary(100, 100)
generator = WallGenerators(boundary)
