from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
import matplotlib
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

#from path_group import PathGroup
from track_setup import Track


class TestCase(unittest.TestCase):

    # Track matrix
    # 0) (1)Line (2)Curve
    # 1) lenght or horizontal-radius
    # 2) vertical-axis radius
    # 3) arc(degrees)
    # 4) width (====WIP====)
    # 5) points per section (====WIP====)
    track_input=np.array([[0, 50, 0, 0,   5, 5],
                        [1, 60, 30, -90, 5, 10],
                        [0, 50, 0, 0,   5, 5],
                        [1, 60, 30, -90, 5, 10],
                        [0, 50, 0, 0,   5, 5],
                        [1, 60, 30, -90, 5, 10],
                        [0, 50, 0, 0,   5, 5],
                        [1, 60, 30, -90, 5, 10],
                        ])

    # Create the track where the path will be optimized
    track = Track()
    track.create(track_input)
    
    #prob = Problem(model=PathGroup(centre_line=track.centre_line, angle=track.angle, width=track.width, num_elements=track.num_elements))
    
    #prob.driver = ScipyOptimizeDriver()
    #prob.driver.options['optimizer'] = 'SLSQP'
    #prob.driver.options['tol'] = 1e-3
    #prob.driver.options['disp'] = True
    #prob.driver.options['maxiter'] = 1000

    #prob.setup()

    #prob.run_driver()

    #print(prob['inputs_comp.z'])

    # Unknown Variable

    #z = (np.random.rand(track.num_elements)*2-1)
    
    z = (0)
    miu = 1
    g = 9.81
    vmax = 100

    x,y = np.array([track.centre_line[:,0], track.centre_line[:,1]]) + np.array([ np.sin(track.angle)*z*track.width , -np.cos(track.angle)*z*track.width ])

    # Gradient needs to be redefined
    dydx = np.gradient(y, x)
    # wrong dydx = np.nan_to_num(dydx, nan=9.9999999e89, posinf=9.9999999e89, neginf=-9.9999999e89)
    # dydx[dydx > 1e5] = 9.9999999e89
    # dydx[dydx < -1e5] = -9.9999999e89
    # dydx[abs(dydx) < 1e-10] = 0

    d2ydx2 = np.gradient(dydx, x)
    # d2ydx2 = np.nan_to_num(d2ydx2, nan=9.9999999e89, posinf=9.9999999e89, neginf=-9.9999999e89)
    # d2ydx2[d2ydx2 > 1e5] = 9.9999999e89
    # d2ydx2[d2ydx2 < -1e5] = -9.9999999e89
    # d2ydx2[abs(d2ydx2) < 1e-10] = 0


    print('\nx,y = ',x,y)
    print('\ndydx = ',dydx)
    print('\nd2ydx2 = ',d2ydx2)

    k = d2ydx2/(1+dydx**2)**(3/2)
    # k = np.nan_to_num(k, nan=9.9999999e89, posinf=9.9999999e89, neginf=-9.9999999e89)
    # k[k > 1e5] = 9.9999999e89
    # k[k < -1e5] = -9.9999999e89
    # k[abs(k) < 1e-10] = 0

    
    print('k=',k)

    print('r=',1/k)

    v = np.sqrt(miu*g/abs(k))
    v = np.nan_to_num(v, nan=99999999, posinf=99999999, neginf=-99999999)

    v[v > vmax] = vmax



    print('v = ',v)

    #1/v>=(k/ug)**.5
    #v<=vmax
    #
    ds = np.zeros(track.num_elements)

    for i in range(0,track.num_elements):
        ds[i] = ((x[i]-x[i-1])**2+(y[i]-y[i-1])**2)**.5

    E = np.sum(ds*(abs(k)/miu/g)**.5)

    print('\nds=',ds)
    print('\nE=',E)

    # Track Plotting
    fig, ax = plt.subplots()

    plt.plot( track.centre_line[:,0] , track.centre_line[:,1] )
    plt.grid(color='lightgray',linestyle='--')

    plt.plot( track.line1[:,0] , track.line1[:,1] )
    plt.plot( track.line2[:,0] , track.line2[:,1] )

    ## Path Plotting
    plt.plot(x,y,'bo')


    ## Velocity Plotting

    plt.quiver(x, y, v, np.zeros(np.size(v)))

    ## Plot Comp
    plt.axis("equal")
    plt.show()
if __name__ == "__main__":
    unittest.main()
