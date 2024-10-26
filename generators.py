import numpy as np
import itertools as itr
from boundaries import WallBoundary

class WallGenerators:
    def __init__(self, wallBoundary:WallBoundary):
        self.wallBoundary = wallBoundary
        self.xResolution = wallBoundary.xResolution
        self.yResolution = wallBoundary.yResolution
    
    def generateRandomBlocks(self, blockAmount:int, blockSize:int):
        xBlockSize = np.random.randint(1, blockSize, size = blockAmount)
        yBlockSize = np.random.randint(1, blockSize, size = blockAmount)
        
        x1Corner = np.random.randint(0, self.xResolution, size = blockAmount)
        y1Corner = np.random.randint(0, self.yResolution, size = blockAmount)

        x2Corner = x1Corner + xBlockSize
        y2Corner = y1Corner + yBlockSize
        
        for _ in range(blockAmount):
            self.wallBoundary.filledStraightRectangularWall([x1Corner[_], y1Corner[_]], [x2Corner[_], y2Corner[_]])

boundary = WallBoundary(100, 100)
generator = WallGenerators(boundary)
