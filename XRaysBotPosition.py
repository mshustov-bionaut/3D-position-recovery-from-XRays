"""
This module contains class to reconstruct bot's position in the animal-associated coordinate
system by 2 X-Rays images

Shustov. Bionaut labs. Jan.2023
"""

import numpy as np
import scipy.optimize
import random
import stl
import matplotlib.pyplot as plt


class XRays3DRecovery():
    """ class to recover 3D position of the bot using XRays images"""

    def __init__(self):

        # create matrices of transformation and biases
        self.M = np.zeros((3,3))
        self.N = np.zeros((3,3))
        self.b = np.zeros((3,1))
        self.c = np.zeros((3, 1))

        # boolean flags of initialization
        self.ini_1 = False
        self.ini_2 = False
        self.ini_stl = False


    def projection_calibration(self,projection_number, real_fiducial_coords, screen_fiducial_coords):
        """ calibration for one of XRays projections
        :param projection_number - number of the processed image (1 or 2)
        :param real_fiducial_coords - list (len =4) of coordinates (X_Y_Z) of 4 fiducials in
                the animal-associated coordinates [mm]
        :param screen_fiducial_coords - list (len=4) of coordinates (x'-y') of fiducials
                on the XRays projection [pixels]
        """

        if len(list(real_fiducial_coords)) != 4 or len(list(screen_fiducial_coords)) != 4:
            print('Incorrect shape of input coordinates')
            return 2

        if int(projection_number) == 1:
            self.M, self.b = self.__find_transform_matrix(real_fiducial_coords,
                                                          screen_fiducial_coords)
            self.ini_1 = True
            return 0
        elif int(projection_number) == 2:
            self.N, self.c = self.__find_transform_matrix(real_fiducial_coords,
                                                          screen_fiducial_coords)
            self.ini_2 = True
            return 0
        else:
            print('Invalid number of XRays projection')
            return 1


    def __find_transform_matrix(self,real_fiducial_coords, screen_fiducial_coords):
        """ find transformation matrix for one XRays projection
        :param real_fiducial_coords - list of 4 tuples with real coordinates X-Y-Z
        :param screen_fiducial_coords - list of 4 tuples with screen coordinates x'-y'
        """

        real_fiducial_coords = list(real_fiducial_coords)
        screen_fiducial_coords = list(screen_fiducial_coords)

        xr1 = real_fiducial_coords[0]
        xs1 = screen_fiducial_coords[0]
        xr2 = real_fiducial_coords[1]
        xs2 = screen_fiducial_coords[1]
        xr3 = real_fiducial_coords[2]
        xs3 = screen_fiducial_coords[2]
        xr4 = real_fiducial_coords[3]
        xs4 = screen_fiducial_coords[3]

        def fun(variables):
            """function to create system of equations"""

            m11, m12, m13, m21, m22, m23, b1, b2 = variables

            eqn_1 = m11 * xr1[0] + m12 * xr1[1] + m13 * xr1[2] + b1 - xs1[0]
            eqn_2 = m21 * xr1[0] + m22 * xr1[1] + m23 * xr1[2] + b2 - xs1[1]
            eqn_3 = m11 * xr2[0] + m12 * xr2[1] + m13 * xr2[2] + b1 - xs2[0]
            eqn_4 = m21 * xr2[0] + m22 * xr2[1] + m23 * xr2[2] + b2 - xs2[1]
            eqn_5 = m11 * xr3[0] + m12 * xr3[1] + m13 * xr3[2] + b1 - xs3[0]
            eqn_6 = m21 * xr3[0] + m22 * xr3[1] + m23 * xr3[2] + b2 - xs3[1]
            eqn_7 = m11 * xr4[0] + m12 * xr4[1] + m13 * xr4[2] + b1 - xs4[0]
            eqn_8 = m21 * xr4[0] + m22 * xr4[1] + m23 * xr4[2] + b2 - xs4[1]

            return [eqn_1, eqn_2, eqn_3, eqn_4, eqn_5, eqn_6, eqn_7, eqn_8]

        # create initial approximations
        r = list()
        for i in range(8):
            r.append(random.uniform(-10, 10))

        # solve the system to get matrix elements
        matr = np.zeros((3,3))
        bias = np.zeros((2,1))

        res = scipy.optimize.fsolve(fun, r)
        matr[0,0], matr[0,1], matr[0,2], matr[1,0], matr[1,1], matr[1,2], bias[0], bias[1] = res

        return matr,bias

    def bot_position(self,bot_screen_coord_1=(0,0), bot_screen_coord_2=(0,0)):
        """ find bot position in 3D
        :param bot_screen_coord_1, bot_screen_coord_2 - screen coordinates of
                    the bot on the XRays projection #1 and #2
        """

        if self.ini_1 and self.ini_2:
            matrA = np.zeros((3,3))
            matrB = np.zeros((3,1))

            matrA = self.M.copy()
            matrA[2,0] = self.N[0,0]
            matrA[2,1] = self.N[0,1]
            matrA[2,2] = self.N[0,2]

            matrB[0] = bot_screen_coord_1[0] - self.b[0]
            matrB[1] = bot_screen_coord_1[1] - self.b[1]
            matrB[2] = bot_screen_coord_2[0] - self.c[0]

            return np.linalg.solve(matrA,matrB)

        else:
            print('Coordinates transformer was not fully initialized')
            return(0,0,0)

    def load_stl(self,filename='andrewbrain_1.stl'):
        """ Load the .stl file with head (brain) geometry"""

        from stl import mesh
        from mpl_toolkits import mplot3d
        from matplotlib import pyplot as plt

        #figure = plt.figure()
        #axes = mplot3d.Axes3D(figure)

        self.my_mesh = mesh.Mesh.from_file(filename)
        #axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.my_mesh.vectors))

        #scale = self.my_mesh.points.flatten()
        #axes.auto_scale_xyz(scale,scale,scale)

        #axes.scatter(-75,-75,0)
        #plt.show()




    def show_stl(self):
        """ show .stl file with 3D image and place a bot on it"""

        if self.ini_stl:
            pass

        else:
            print('.stl file is not loaded')