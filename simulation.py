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
        
        self.vorticity = None

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
    
    def updateVorticity(self):
        diffY = np.gradient(self.speedY, axis = 1)
        diffX = np.gradient(self.speedX, axis = 0)
        self.vorticity = diffX - diffY

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

    def bounceBackFluid(self):
        for y, x in self.wallBoundary.boundaryIndex:
            for latticeIndex in range(Simulation.latticeSize):
                if self.fluid[y, x, latticeIndex] != 0:
                    bounceIndexY = y - Simulation.unitY[latticeIndex]
                    bounceIndexX = x - Simulation.unitX[latticeIndex]
                    if (bounceIndexY >= 0 and bounceIndexY < self.yResolution) and (bounceIndexX >= 0 and bounceIndexX < self.xResolution):
                        self.fluid[
                            bounceIndexY,
                            bounceIndexX,
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

    def imposeVelocityBoundaryCondition(self):
        for velocityBoundary in self.velocityBoundaries:
            self.fluid[
                velocityBoundary.y, velocityBoundary.x, velocityBoundary.direction
            ] = velocityBoundary.magnitude
        self.updateSpeed()

    def imposePressureBoundaryCondition(self):
        for pressureBoundary in self.pressureBoundaries:
            for latticeIndex in range(Simulation.latticeSize):
                self.fluid[pressureBoundary.y, pressureBoundary.x, latticeIndex] = 0
            densityAtIndex = self.density[pressureBoundary.y, pressureBoundary.x]
            self.fluid[
                pressureBoundary.y, pressureBoundary.x, pressureBoundary.setIndices[0]
            ] = self.fluid[
                pressureBoundary.y, pressureBoundary.x, pressureBoundary.getIndices[0]
            ] + (
                2 / 3
            ) * (
                pressureBoundary.mainVelocity
            )
            self.fluid[
                pressureBoundary.y, pressureBoundary.x, pressureBoundary.setIndices[1]
            ] = (
                self.fluid[
                    pressureBoundary.y,
                    pressureBoundary.x,
                    pressureBoundary.getIndices[1],
                ]
                - (
                    0.5
                    * (
                        self.fluid[
                            pressureBoundary.y,
                            pressureBoundary.x,
                            (
                                4
                                if pressureBoundary.direction - 1 == 0
                                else pressureBoundary.direction - 1
                            ),
                        ]
                        - self.fluid[
                            pressureBoundary.y,
                            pressureBoundary.x,
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
                pressureBoundary.y, pressureBoundary.x, pressureBoundary.setIndices[2]
            ] = (
                self.fluid[
                    pressureBoundary.y,
                    pressureBoundary.x,
                    pressureBoundary.getIndices[2],
                ]
                + (
                    0.5
                    * (
                        self.fluid[
                            pressureBoundary.y,
                            pressureBoundary.x,
                            (
                                4
                                if pressureBoundary.direction - 1 == 0
                                else pressureBoundary.direction - 1
                            ),
                        ]
                        - self.fluid[
                            pressureBoundary.y,
                            pressureBoundary.x,
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

    def stepSimulation(self):
        self.lastStepFluid = copy.deepcopy(self.fluid)
        self.streamFluid()
        self.bounceBackFluid()
        self.collideFluid()
        self.imposeVelocityBoundaryCondition()
        self.imposePressureBoundaryCondition()

    def simulate(self, step: int = 1):
        [self.stepSimulation() for _ in range(step)]

    def isAtDensityEquilibirum(self, threshold: float = 0.5):
        error = np.sum(np.abs(self.lastStepFluid - self.fluid))
        #print(error)
        if error > threshold:
            return False
        else:
            return True
        

    def simulateUntilEquilibrium(self, limit: int = 5000, equilTreshold: float = 0.5, explodeTreshold: float = 400):
        step = 0
        isStable = True
        for _ in range(limit):
            self.stepSimulation()
            step += 1
            if np.average(self.fluid) > explodeTreshold:
                isStable = False
                break
            if self.isAtDensityEquilibirum(equilTreshold):
                break

        return step, isStable
