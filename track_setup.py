
import numpy as np


def make_line(l,w,p):
    # Creates a line portion of the points required for the plotting of the track
    # l - length
    # w - width
    # p - number of points

    # Centre line
    centre_line = np.transpose(np.array([np.linspace(0,l,p), np.linspace(0,0,p)]))

    angle = np.transpose(np.array(np.linspace(0,0,p)))

    width = np.transpose(np.array(np.linspace(w/2,w/2,p)))

    # Plot lines
    line1 = np.array([[centre_line[0,0] , centre_line[0,1] - w/2 ],
                      [centre_line[-1,0], centre_line[-1,1] - w/2]])

    line2 = np.array([[centre_line[0,0] , centre_line[0,1] + w/2 ],
                      [centre_line[-1,0], centre_line[-1,1] + w/2]])

    return centre_line, angle, width, line1, line2


def make_curve(h,v,a,w,p):
    # Creates an ellipse portion of the points required for the plotting of the track
    # h - horizontal-axis
    # v - vertical-axis
    # a - angle in degrees
    # w - width
    # p - number of points

    if a < 0:
        # Centre line
        centre_line = np.transpose(np.array([h * np.sin(np.radians(np.linspace(0,-a,p))) ,
                                             v * (np.cos(np.radians(np.linspace(0,-a,p))) - 1) ] ))

        angle = np.transpose(np.array(np.radians(np.linspace(0,a,p))))

        width = np.transpose(np.array(np.linspace(w/2,w/2,p)))

        # Plot lines
        line1 = np.transpose(np.array([(h-w/2) * np.sin(np.radians(np.linspace(0,-a))) ,
                                       (v-w/2) * (np.cos(np.radians(np.linspace(0,-a))) - 1) - w/2 ] ))


        line2 = np.transpose(np.array([(h+w/2) * np.sin(np.radians(np.linspace(0,-a))) ,
                                       (v+w/2) * (np.cos(np.radians(np.linspace(0,-a))) - 1) + w/2 ] ))
    else:
        # Centre line
        centre_line = np.transpose(np.array([h * np.sin(np.radians(np.linspace(0,a,p))) ,
                                             -v * (np.cos(np.radians(np.linspace(0,a,p))) - 1) ] ))

        angle = np.transpose(np.array(np.radians(np.linspace(0,a,p))))

        width = np.transpose(np.array(np.linspace(w/2,w/2,p)))

        # Plot lines
        line1 = np.transpose(np.array([(h+w/2) * np.sin(np.radians(np.linspace(0,a))) ,
                                       (-v-w/2) * (np.cos(np.radians(np.linspace(0,a))) - 1) - w/2 ] ))


        line2 = np.transpose(np.array([(h-w/2) * np.sin(np.radians(np.linspace(0,a))) ,
                                       (-v+w/2) * (np.cos(np.radians(np.linspace(0,a))) - 1) + w/2 ] ))

    return centre_line, angle, width, line1, line2




def polar2xy(z):
    return z[0] * np.cos(z[1]) , z[0] * np.sin(z[1])

def xy2polar(z):
    return ( np.sqrt(z[0]**2+z[1]**2), np.angle(z[0]+1j*z[1]) )

def rotation_matrix(a):
    # Rotate x,y coordinates in a radians
    rotation = np.array([[np.cos(a), -np.sin(a)],
                         [np.sin(a), np.cos(a) ]])
    return rotation

def iz2xy(centre_line,z):
    # Transform position in terms of the referential of the centre_line's direction (track transversal line) and z (position along the line)
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


class Track:

    def __init__(self):
        self.centre_line = []
        self.angle = []
        self.width = []
        self.line1 = []
        self.line2 = []
        self.num_elements = []

    def create(self,track_input):

        for i in range(0,len(track_input)):
            if track_input[i,0]==0: # If segment is a line
                temp_centre_line, temp_angle, temp_width, temp_line1, temp_line2 = make_line(track_input[i,1],track_input[i,4],track_input[i,5])
            elif track_input[i,0]==1: # If segment is a curve
                temp_centre_line, temp_angle, temp_width,  temp_line1, temp_line2 = make_curve(track_input[i,1],track_input[i,2],track_input[i,3],track_input[i,4],track_input[i,5])
            else:
                print('Invalid track section {}.'.format(i))


            if i == 0: #Coordinate transformation of the segment

                self.centre_line = temp_centre_line
                self.angle = temp_angle
                self.width = temp_width
                self.line1 = temp_line1
                self.line2 = temp_line2

            else:
                # Segment Rotation Matrix
                segment_rotation = rotation_matrix(self.angle[-1])

                # Centre Line
                ## Rotation
                temp_centre_line[:,[0,1]] = np.transpose(np.dot(segment_rotation,np.transpose(temp_centre_line[:,[0,1]])))

                ## Translocation
                temp_centre_line[:,[0,1]] = temp_centre_line[:,[0,1]] + self.centre_line[-1,[0,1]]

                ## Angle Addition
                temp_angle = temp_angle + self.angle[-1]

                # Limit Lines
                ## Rotation
                temp_line1[:,[0,1]] = np.transpose(np.dot(segment_rotation,np.transpose(temp_line1[:,[0,1]])))
                temp_line2[:,[0,1]] = np.transpose(np.dot(segment_rotation,np.transpose(temp_line2[:,[0,1]])))

                ## Translocation
                temp_line1[:,[0,1]] = temp_line1[:,[0,1]] + self.centre_line[-1,[0,1]]
                temp_line2[:,[0,1]] = temp_line2[:,[0,1]] + self.centre_line[-1,[0,1]]

                # Add to final matrices
                self.centre_line = np.append(self.centre_line, temp_centre_line[1:,:], axis=0)
                self.angle = np.append(self.angle, temp_angle[1:], axis=0)
                self.width = np.append(self.width, temp_width[1:], axis=0)
                self.line1 = np.append(self.line1, temp_line1[1:,:], axis=0)
                self.line2 = np.append(self.line2, temp_line2[1:,:], axis=0)
        self.num_elements = len(self.centre_line)
