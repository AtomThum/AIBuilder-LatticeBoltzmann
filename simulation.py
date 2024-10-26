import numpy as np
import copy
from boundaries import WallBoundary, PressureBoundary, VelocityBoundary

latticeSize = 9
xResolution = 40
yResolution = 40


class Simulation:
    unitVect = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    unitX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    unitY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weight = np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )
    latticeSize = 9

    def __init__(
        self,
        xResolution: int,
        yResolution: int,
        initCondition: np.array,
        wallBoundary: WallBoundary,
        pressureBoundary: list = [],
        velocityBoundary: list = [],
        relaxationTime: float = 0.809,
        initialStep: int = 0,
    ):
        self.xResolution = xResolution
        self.yResolution = yResolution
        self.xIndex = np.arange(xResolution)
        self.yIndex = np.arange(yResolution)

        self.fluid = initCondition  # Boundary is imposed by default.
        self.relaxationTime = relaxationTime
        self.wallBoundary = wallBoundary
        self.pressureBoundaries = pressureBoundary
        self.velocityBoundary = velocityBoundary
        self.step = initialStep

        self.fluid[self.wallBoundary.boundary, :] = 0
        self.initialCondition = copy.deepcopy(self.fluid)

        # Initial calculation from the initial condition. These variables will be updated as time goes on
        self.density = np.sum(self.fluid, axis=2)
        self.momentumX = np.sum(self.fluid * Simulation.unitX, axis=2)
        self.momentumY = np.sum(self.fluid * Simulation.unitY, axis=2)
        self.speedX = self.momentumX / self.density
        self.speedY = self.momentumY / self.density
        self.speedX = np.nan_to_num(self.speedX, posinf=0, neginf=0, nan=0)
        self.speedY = np.nan_to_num(self.speedY, posinf=0, neginf=0, nan=0)

        self.initialDensity = copy.deepcopy(self.density)
        self.initialMomentumX = copy.deepcopy(self.momentumX)
        self.initialMomentumY = copy.deepcopy(self.momentumY)
        self.initialSpeedX = copy.deepcopy(self.speedX)
        self.initialSpeedY = copy.deepcopy(self.speedY)

    def updateDensity(self):
        self.density = np.sum(self.fluid, axis=2)

    def updateMomentum(self):
        self.momentumX = np.sum(self.fluid * Simulation.unitX, axis=2)
        self.momentumY = np.sum(self.fluid * Simulation.unitY, axis=2)

    # If this function is called, you don't have to update density and momentum
    def updateSpeed(self):
        self.updateDensity()
        self.updateMomentum()

        self.speedX = self.momentumX / self.density
        self.speedY = self.momentumY / self.density
        self.speedX = np.nan_to_num(self.speedX, posinf=0, neginf=0, nan=0)
        self.speedY = np.nan_to_num(self.speedY, posinf=0, neginf=0, nan=0)

    def stepFluid(self):
        # Equilizing step
        fluidEquilibrium = np.zeros(self.fluid.shape)
        for latticeIndex, cx, cy, w in zip(
            range(latticeSize), Simulation.unitX, Simulation.unitY, Simulation.weight
        ):
            fluidEquilibrium[:, :, latticeIndex] = (
                self.density
                * w
                * (
                    1
                    + 3 * (cx * self.speedX + cy * self.speedY)
                    + 9 * (cx * self.speedX + cy * self.speedY) ** 2 / 2
                    - 3 * (self.speedX**2 + self.speedY**2) / 2
                )
            )

        # self.fluid[self.wallBoundary.invertedBoundary, :] -= (
        #     self.fluid[self.wallBoundary.invertedBoundary, :]
        #     - fluidEquilibrium[self.wallBoundary.invertedBoundary, :]
        #     / self.relaxationTime
        # )
        self.fluid -= (self.fluid - fluidEquilibrium) / self.relaxationTime

        for latticeIndex, shiftX, shiftY in zip(
            range(latticeSize), Simulation.unitX, Simulation.unitY
        ):
            self.fluid[:, :, latticeIndex] = np.roll(
                self.fluid[:, :, latticeIndex], shiftX, axis=1
            )
            self.fluid[:, :, latticeIndex] = np.roll(
                self.fluid[:, :, latticeIndex], shiftY, axis=0
            )

        fluidBoundary = self.fluid[self.wallBoundary.boundary, :]
        fluidBoundary = fluidBoundary[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
        self.fluid[self.wallBoundary.boundary, :] = fluidBoundary

        for pressureBoundary in self.pressureBoundaries:
            for latticeIndex in range(latticeSize):
                self.fluid[pressureBoundary.x, pressureBoundary.y, latticeIndex] = 0
            densityAtIndex = self.density[pressureBoundary.x, pressureBoundary.y]
            self.fluid[
                pressureBoundary.x, pressureBoundary.y, pressureBoundary.setIndices[0]
            ] = self.fluid[
                pressureBoundary.x, pressureBoundary.y, pressureBoundary.getIndices[0]
            ] + (
                2 / 3
            ) * (
                pressureBoundary.mainVelocity
            )
            self.fluid[
                pressureBoundary.x, pressureBoundary.y, pressureBoundary.setIndices[1]
            ] = (
                self.fluid[
                    pressureBoundary.x,
                    pressureBoundary.y,
                    pressureBoundary.getIndices[1],
                ]
                - (
                    0.5
                    * (
                        self.fluid[
                            pressureBoundary.x,
                            pressureBoundary.y,
                            (
                                4
                                if pressureBoundary.direction - 1 == 0
                                else pressureBoundary.direction - 1
                            ),
                        ]
                        - self.fluid[
                            pressureBoundary.x,
                            pressureBoundary.y,
                            (
                                1
                                if pressureBoundary.direction + 1 == 5
                                else pressureBoundary.direction + 1
                            ),
                        ]
                    )
                )
                + (0.5 * densityAtIndex * pressureBoundary.minorVelocity)
                + (1 / 6 * densityAtIndex * pressureBoundary.mainVelocity)
            )
            self.fluid[
                pressureBoundary.x, pressureBoundary.y, pressureBoundary.setIndices[2]
            ] = (
                self.feluid[
                    pressureBoundary.x,
                    pressureBoundary.y,
                    pressureBoundary.getIndices[2],
                ]
                + (
                    0.5
                    * (
                        self.fluid[
                            pressureBoundary.x,
                            pressureBoundary.y,
                            (
                                4
                                if pressureBoundary.direction - 1 == 0
                                else pressureBoundary.direction - 1
                            ),
                        ]
                        - self.fluid[
                            pressureBoundary.x,
                            pressureBoundary.y,
                            (
                                1
                                if pressureBoundary.direction + 1 == 5
                                else pressureBoundary.direction + 1
                            ),
                        ]
                    )
                )
                - (0.5 * densityAtIndex * pressureBoundary.minorVelocity)
                + (1 / 6 * densityAtIndex * pressureBoundary.mainVelocity)
            )

        self.updateSpeed()
        self.step += 1
        return self.fluid

    def simulateFluid(self, step: int):
        [self.stepFluid() for _ in range(step)]

    # Uses mean absolute error to determine whether the fluid is in equilibrium
    def isAtDensityEquilibirum(self, threshold: float):
        meanDensity = np.mean(self.density)
        error = np.sum(np.abs(self.density - meanDensity)) / (xResolution * yResolution)
        if error > threshold:
            return False
        else:
            return True
