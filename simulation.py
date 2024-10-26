import numpy as np
import copy
import itertools as itr
from boundaries import WallBoundary, PressureBoundary, VelocityBoundary


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
    reflectIndices = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

    def __init__(
        self,
        yResolution: int,
        xResolution: int,
        initCondition: np.array,
        wallBoundary: WallBoundary,
        pressureBoundaries: list = [],
        velocityBoundaries: list = [],
        relaxationTime: float = 0.809,
        initialStep: int = 0,
    ):
        # By python's convention, yResolution must come first.
        # The array must be of shape (yResolution, xResolution, Simulation.latticeSize)
        self.yResolution = yResolution
        self.xResolution = xResolution
        self.yIndex = np.arange(yResolution)
        self.xIndex = np.arange(xResolution)

        self.initCondition = initCondition

        self.fluid = initCondition
        self.lastStepFluid = initCondition
        self.relaxationTime = relaxationTime
        self.wallBoundary = wallBoundary
        self.wallBoundary.generateIndex()
        self.fluid[self.wallBoundary.boundary, :] = 0
        self.pressureBoundaries = pressureBoundaries
        self.velocityBoundaries = velocityBoundaries
        self.step = initialStep

        self.density = np.sum(self.fluid, axis=2)
        self.momentumY = np.sum(self.fluid * Simulation.unitY, axis=2)
        self.momentumX = np.sum(self.fluid * Simulation.unitX, axis=2)
        self.speedY = self.momentumY / self.density
        self.speedX = self.momentumX / self.density
        self.speedY = np.nan_to_num(self.speedY, posinf=0, neginf=0, nan=0)
        self.speedX = np.nan_to_num(self.speedX, posinf=0, neginf=0, nan=0)

    def updateDensity(self):
        self.density = np.sum(self.fluid, axis=2)

    def updateMomentum(self):
        self.momentumY = np.sum(self.fluid * Simulation.unitY, axis=2)
        self.momentumX = np.sum(self.fluid * Simulation.unitX, axis=2)

    # If this function is called, you don't have to update density and momentum
    def updateSpeed(self):
        self.updateDensity()
        self.updateMomentum()

        self.speedY = self.momentumY / self.density
        self.speedX = self.momentumX / self.density
        self.speedY = np.nan_to_num(self.speedY, posinf=0, neginf=0, nan=0)
        self.speedX = np.nan_to_num(self.speedX, posinf=0, neginf=0, nan=0)

    def streamFluid(self):
        for latticeIndex, shiftY, shiftX in zip(
            range(Simulation.latticeSize), Simulation.unitY, Simulation.unitX
        ):
            self.fluid[:, :, latticeIndex] = np.roll(
                self.fluid[:, :, latticeIndex], shiftY, axis=0
            )
            self.fluid[:, :, latticeIndex] = np.roll(
                self.fluid[:, :, latticeIndex], shiftX, axis=1
            )
        self.updateSpeed()

    def bounceBackFluid(self):
        for y, x in self.wallBoundary.boundaryIndex:
            for latticeIndex in range(Simulation.latticeSize):
                if self.fluid[y, x, latticeIndex] != 0:
                    self.fluid[
                        y - Simulation.unitY[latticeIndex],
                        x - Simulation.unitX[latticeIndex],
                        Simulation.reflectIndices[latticeIndex],
                    ] = self.fluid[y, x, latticeIndex]
                    self.fluid[y, x, latticeIndex] = 0
        self.updateSpeed()

    def collideFluid(self):
        fluidEquilibrium = np.zeros(self.fluid.shape)
        for latticeIndex, cy, cx, w in zip(
            range(Simulation.latticeSize),
            Simulation.unitY,
            Simulation.unitX,
            Simulation.weight,
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
        self.fluid += (fluidEquilibrium - self.fluid) / self.relaxationTime
        self.updateSpeed()

    def imposeVelocityBoundaryCondition(self):
        for velocityBoundary in self.velocityBoundaries:
            self.fluid[
                velocityBoundary.y, velocityBoundary.x, velocityBoundary.direction
            ] = velocityBoundary.magnitude
        self.updateSpeed()

    def stepSimulation(self):
        self.lastStepFluid = self.fluid
        self.streamFluid()
        self.bounceBackFluid()
        self.collideFluid()
        self.imposeVelocityBoundaryCondition()

    def simulate(self, step: int = 1):
        [self.stepSimulation() for _ in range(step)]

    def isAtDensityEquilibirum(self, threshold: float):
        error = np.sum(np.abs(self.lastStepFluid - self.fluid)) / (
            self.xResolution * self.yResolution
        )
        if error > threshold:
            return False
        else:
            return True

    def simulateUntilEquilibrium(self, limit: int = 5000, threshold: float = 0.5):
        for _ in range(limit):
            self.stepSimulation()
            if self.isAtDensityEquilibirum(threshold):
                break
            else:
                pass
            step += 1
