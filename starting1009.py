import numpy as np
import matplotlib
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from openmdao.api import Problem, ScipyOptimizeDriver, ExecComp, IndepVarComp, Group

# Course matrix
# 0) (1)Line (2)Curve
# 1) lenght or horizontal-radius
# 2) vertical-axis radius
# 3) arc(degrees)
# 4) width (====WIP====)
# 5) points per section (====WIP====)
course_input=np.array([[0, 5, 0, 0,   .1, 3],
                       [1, 3, 3, -90, .1, 5],
                       [0, 5, 0, 0,   .1, 3],
                       [1, 3, 3, -90, .1, 5],
                       [0, 5, 0, 0,   .1, 3],
                       [1, 3, 3, -90, .1, 5],
                       [0, 5, 0, 0,   .1, 3],
                       [1, 3, 3, -90, .1, 5],
                       ])

def make_line(l,w,p):
    # Creates a line portion of the points required for the plotting of the course
    # l - length
    # w - width
    # p - number of points

    # Centre line
    centre_line = np.transpose(np.array([np.linspace(0,l,p), np.linspace(0,0,p), np.linspace(0,0,p), np.linspace(w/2,w/2,p)]))

    # Plot lines
    line1 = np.array([[centre_line[0,0] , centre_line[0,1] - w/2 ],
                      [centre_line[-1,0], centre_line[-1,1] - w/2]])

    line2 = np.array([[centre_line[0,0] , centre_line[0,1] + w/2 ],
                      [centre_line[-1,0], centre_line[-1,1] + w/2]])

    return centre_line, line1, line2


def make_curve(h,v,a,w,p):
    # Creates an ellipse portion of the points required for the plotting of the course
    # h - horizontal-axis
    # v - vertical-axis
    # a - angle in degrees
    # w - width
    # p - number of points

    if a < 0:
        # Centre line
        centre_line = np.transpose(np.array([h * np.sin(np.radians(np.linspace(0,-a,p))) ,
                                             v * (np.cos(np.radians(np.linspace(0,-a,p))) - 1) ,
                                             np.radians(np.linspace(0,a,p)) ,
                                             np.linspace(w/2,w/2,p)] ))

        # Plot lines
        line1 = np.transpose(np.array([(h-w/2) * np.sin(np.radians(np.linspace(0,-a))) ,
                                       (v-w/2) * (np.cos(np.radians(np.linspace(0,-a))) - 1) - w/2 ] ))


        line2 = np.transpose(np.array([(h+w/2) * np.sin(np.radians(np.linspace(0,-a))) ,
                                       (v+w/2) * (np.cos(np.radians(np.linspace(0,-a))) - 1) + w/2 ] ))
    else:
        # Centre line
        centre_line = np.transpose(np.array([h * np.sin(np.radians(np.linspace(0,a,p))) ,
                                             -v * (np.cos(np.radians(np.linspace(0,a,p))) - 1) ,
                                             np.radians(np.linspace(0,a,p)) ,
                                             np.linspace(w/2,w/2,p)] ))

        # Plot lines
        line1 = np.transpose(np.array([(h+w/2) * np.sin(np.radians(np.linspace(0,a))) ,
                                       (-v-w/2) * (np.cos(np.radians(np.linspace(0,a))) - 1) - w/2 ] ))


        line2 = np.transpose(np.array([(h-w/2) * np.sin(np.radians(np.linspace(0,a))) ,
                                       (-v+w/2) * (np.cos(np.radians(np.linspace(0,a))) - 1) + w/2 ] ))

    return centre_line, line1, line2




def polar2xy(z):
    return z[0] * np.cos(z[1]) , z[0] * np.sin(z[1])

def xy2polar(z):
    return ( np.sqrt(z[0]**2+z[1]**2), np.angle(z[0]+1j*z[1]) )

def rotation_matrix(a):
    # Rotate x,y coordinates in radians
    rotation = np.array([[np.cos(a), -np.sin(a)],
                         [np.sin(a), np.cos(a) ]])
    return rotation

def iz2xy(centre_line,z):
    # Transform position in terms of the referential of the centre_line's direction (course transversal line) and z (position along the line)
    # to (x,y) coordinates

    return centre_line[[0,1]] + np.dot(np.array([ np.sin(centre_line[2]) , -np.cos(centre_line[2]) ]),z)

def iv2xy(centre_line,v):
    # Transforms a velocity vector in the referential of the centre_line's direction
    # to (x,y) coordiantes
    return np.dot(rotation_matrix(centre_line[2]),np.transpose(v))

def iva2xy(centre_line,v,a):
    # Transforms an acceleration vector in the velocity's referential
    # to (x,y) coordiantes
    alpha = xy2polar(v)[1] + centre_line[2]

    return np.dot(rotation_matrix(alpha),np.transpose(a))


class Course:

    def __init__(self,course_input):
        self.course_input = course_input
        self.centre_line = []
        self.line1 = []
        self.line2 = []
        self.ilength = []

    def create(self,course_input):

        for i in range(0,len(course_input)):
            if course_input[i,0]==0: # If segment is a line
                temp_centre_line, temp_line1, temp_line2 = make_line(course_input[i,1],course_input[i,4],course_input[i,5])
            elif course_input[i,0]==1: # If segment is a curve
                temp_centre_line, temp_line1, temp_line2 = make_curve(course_input[i,1],course_input[i,2],course_input[i,3],course_input[i,4],course_input[i,5])
            else:
                print('Invalid course section {}.'.format(i))


            if i == 0: #Coordinate transformation of the segment

                self.centre_line = temp_centre_line
                self.line1 = temp_line1
                self.line2 = temp_line2

            else:
                # Segment Rotation Matrix
                segment_rotation = rotation_matrix(self.centre_line[-1,2])

                # Centre Line
                ## Rotation
                temp_centre_line[:,[0,1]] = np.transpose(np.dot(segment_rotation,np.transpose(temp_centre_line[:,[0,1]])))

                ## Translocation
                temp_centre_line[:,[0,1]] = temp_centre_line[:,[0,1]] + self.centre_line[-1,[0,1]]

                ## Angle Addition
                temp_centre_line[:,2] = temp_centre_line[:,2] + self.centre_line[-1,2]

                # Limit Lines
                ## Rotation
                temp_line1[:,[0,1]] = np.transpose(np.dot(segment_rotation,np.transpose(temp_line1[:,[0,1]])))
                temp_line2[:,[0,1]] = np.transpose(np.dot(segment_rotation,np.transpose(temp_line2[:,[0,1]])))

                ## Translocation
                temp_line1[:,[0,1]] = temp_line1[:,[0,1]] + self.centre_line[-1,[0,1]]
                temp_line2[:,[0,1]] = temp_line2[:,[0,1]] + self.centre_line[-1,[0,1]]

                # Add to final matrices
                self.centre_line = np.append(self.centre_line, temp_centre_line[1:,:], axis=0)
                self.line1 = np.append(self.line1, temp_line1[1:,:], axis=0)
                self.line2 = np.append(self.line2, temp_line2[1:,:], axis=0)
        self.ilength = len(self.centre_line)

# Create the course where the path will be optimized
course = Course(course_input)
course.create(course_input)
#print(course.centre_line)


# Unknown Variable

z = (np.random.rand(course.ilength)*2-1)
miu = 1
g = 9.81


#print('z',z)

x,y = np.array([course.centre_line[:,0], course.centre_line[:,1]]) + np.array([ np.sin(course.centre_line[:,2])*z*course.centre_line[:,3] , -np.cos(course.centre_line[:,2])*z*course.centre_line[:,3] ])

#print('x',x)

dydx = np.gradient(y, x)
d2ydx2 = np.gradient(dydx, x)


#print('dydx',dydx)

#print('d2ydx2',d2ydx2)

k = d2ydx2/(1+dydx**2)**(3/2)

print('1/k = ',k)

#1/v>=(k/ug)**.5
#v<=vmax
#
ds = np.zeros(course.ilength)

for i in range(0,course.ilength):
    ds[i] = ((x[i]-x[i-1])**2+(y[i]-y[i-1])**2)**.5

E = np.sum(ds*(abs(k)/miu/g)**.5)

print('ds = ', ds)
print('E = ', E)

# Plotting
fig, ax = plt.subplots()

plt.plot( course.centre_line[:,0] , course.centre_line[:,1] )
plt.grid(color='lightgray',linestyle='--')

plt.plot( course.line1[:,0] , course.line1[:,1] )
plt.plot( course.line2[:,0] , course.line2[:,1] )

## Curve Plotting



plt.plot(x,y,'bo')
#plt.plot(course.centre_line[:,0],course.centre_line[:,1],'bo')

plt.axis("equal")
plt.show()