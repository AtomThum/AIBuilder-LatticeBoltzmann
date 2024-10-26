import numpy as np
from boundaries import WallBoundary, PressureBoundary, VelocityBoundary

latticeSize = 9
xResolution = 40
yResolution = 40
relaxationTime = 0.809 # Best: 0.809
# Weights
unitVect = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
)
unitX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
unitY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
weight = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

class Simulation:
    unitVect = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    unitX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    unitY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weight = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
    def __init__(
        self,
        xResolution: int,
        yResolution: int,
        initCondition: np.array,
        wallBoundary: WallBoundary,
        pressureBoundary: list = [],
        velocityBoundary: list = [],
        relaxationTime: float = 0.809,
        initialStep: int = 0;
    ):
        self.xResolution = xResolution
        self.yResolution = yResolution
        self.fluid = initCondition
        self.relaxationTime = relaxationTime
        self.wallBoundary = wallBoundary
        self.pressureBoundaries = pressureBoundary
        self.velocityBoundary = velocityBoundary
        self.step = initialStep
        
        self.fluid[self.wallBoundary.boundary, :] = 0
        self.initialCondition = initCondition.copy()

        self.density = np.sum(self.fluid, axis = 2)
        self.momentumX = np.sum(self.fluid * unitX, axis = 2)
        self.momentumY = np.sum(self.fluid * unitY, axis = 2)
        self.speedX = self.momentumX / self.density
        self.speedY = self.momentumY / self.density
        self.speedX = np.nan_to_num(self.speedX, posinf=0, neginf=0, nan=0)
        self.speedY = np.nan_to_num(self.speedY, posinf=0, neginf=0, nan=0)
    
    def updateDensity(self):
        self.density = np.sum(self.fluid, axis=2)
    
    def updateMomentum(self):
        self.momentumX = np.sum(self.fluid * unitX, axis = 2)
        self.momentumY = np.sum(self.fluid * unitY, axis = 2)
    
    def updateSpeed(self):
        self.speedX = self.momentumX / self.density
        self.speedY = self.momentumY / self.density
        self.speedX = np.nan_to_num(self.speedX, posinf=0, neginf=0, nan=0)
        self.speedY = np.nan_to_num(self.speedY, posinf=0, neginf=0, nan=0)

    def stepFluid(self):
        fluid = self.fluid
        
        # Stabilizing Step
        density = np.sum(fluid, axis=2)
        speedX = np.sum(fluid * unitX, axis=2) / density
        speedY = np.sum(fluid * unitY, axis=2) / density
        speedX = np.nan_to_num(speedX, posinf=0, neginf=0, nan=0)
        speedY = np.nan_to_num(speedY, posinf=0, neginf=0, nan=0)

        # Equilizing step
        fluidEquilibrium = np.zeros(self.fluid.shape)
        for latticeIndex, cx, cy, w in zip(range(latticeSize), unitX, unitY, weight):
            fluidEquilibrium[:, :, latticeIndex] = (
                density
                * w
                * (
                    1
                    + 3 * (cx * speedX + cy * speedY)
                    + 9 * (cx * speedX + cy * speedY) ** 2 / 2
                    - 3 * (speedX**2 + speedY**2) / 2
                )
            )

        # fluid[boundary.invertedBoundary, :] -= (1 / relaxationTime) * (fluid[boundary.invertedBoundary, :] - fluidEquilibrium[boundary.invertedBoundary, :])
        fluid -= (1 / relaxationTime) * (fluid - fluidEquilibrium)

        for latticeIndex, shiftX, shiftY in zip(range(latticeSize), unitX, unitY):
            fluid[:, :, latticeIndex] = np.roll(fluid[:, :, latticeIndex], shiftX, axis=1)
            fluid[:, :, latticeIndex] = np.roll(fluid[:, :, latticeIndex], shiftY, axis=0)

        fluidBoundary = fluid[self.wallBoundary.boundary, :]
        fluidBoundary = fluidBoundary[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
        fluid[self.wallBoundary.boundary, :] = fluidBoundary

        for pressureBoundary in self.pressureBoundaries:
            for latticeIndex in range(latticeSize):
                fluid[pressureBoundary.x, pressureBoundary.y, latticeIndex] = 0
            densityAtIndex = density[pressureBoundary.x, pressureBoundary.y]
            fluid[pressureBoundary.x , pressureBoundary.y, pressureBoundary.setIndices[0]] = fluid[pressureBoundary.x , pressureBoundary.y, pressureBoundary.getIndices[0]] + (2/3)*(pressureBoundary.mainVelocity)
            fluid[pressureBoundary.x , pressureBoundary.y, pressureBoundary.setIndices[1]] = fluid[pressureBoundary.x , pressureBoundary.y, pressureBoundary.getIndices[1]] - (0.5 * (fluid[pressureBoundary.x , pressureBoundary.y, (4 if pressureBoundary.direction - 1 == 0 else pressureBoundary.direction - 1)] - fluid[pressureBoundary.x , pressureBoundary.y, (1 if pressureBoundary.direction + 1 == 5 else pressureBoundary.direction + 1)])) + (0.5 * densityAtIndex * pressureBoundary.minorVelocity) + (1/6 * densityAtIndex * pressureBoundary.mainVelocity)
            fluid[pressureBoundary.x , pressureBoundary.y, pressureBoundary.setIndices[2]] = fluid[pressureBoundary.x , pressureBoundary.y, pressureBoundary.getIndices[2]] + (0.5 * (fluid[pressureBoundary.x , pressureBoundary.y, (4 if pressureBoundary.direction - 1 == 0 else pressureBoundary.direction - 1)] - fluid[pressureBoundary.x , pressureBoundary.y, (1 if pressureBoundary.direction + 1 == 5 else pressureBoundary.direction + 1)])) - (0.5 * densityAtIndex * pressureBoundary.minorVelocity) + (1/6 * densityAtIndex * pressureBoundary.mainVelocity)

        self.fluid = fluid
        self.step += 1
        return fluid
    
    def simulateFluid(self, step: int):
        [self.stepFluid(self) for _ in range(step)]