#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIS CODE WAS NOT MADE BY ME

IT WAS PROVIDED FOR EAE 127 APPLIED AERODYNAMICS

IT WAS AUTHORIZED FOR USE IN THIS CLASS ONLY
"""

# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

###############################################################################
### GEOMETRY MODIFICATION FUNCTIONS ###########################################
###############################################################################

def ReadXfoilGeometry(ifile):
    """Reads two coulmn xfoil output files. Either geometry or cp distributions
    ifile --> path of input file (string)
    """
    xgeom, ygeom = np.loadtxt(ifile, skiprows=1, unpack=True)
    return xgeom, ygeom

def FindLE_top(X):
    """Return index dividing upper and lower surface given MSES geometry.
    Search along upper surface until LE.
    MSES files start at rear of airfoil, and x diminishes until the leading
    edge, where it then increases back to the trailing edge.  This code finds
    the transition where x goes from decreasing to increasing.
    X --> MSES x coordinates
    """
    xold = X[0]
    for i, x in enumerate(X[1:]):
        if x >= xold:
            #If current x greater/equal to prev x, x is increasing (lower surf)
            return i #return index of Leading Edge (divides upper/lower surfs)
        else:
            #If current x less than prev x, x still diminishing (upper surf)
            xold = x

def FindLE_bot(X):
    """Return index dividing upper and lower surface given MSES geometry.
    Search along lower surface until LE.
    MSES files start at rear of airfoil, and x diminishes until the leading
    edge, where it then increases back to the trailing edge.  This code finds
    the transition where x goes from decreasing to increasing.
    X --> MSES x coordinates
    """
    Xreverse = X[::-1]
    xold = Xreverse[0]
    for i, x in enumerate(Xreverse[1:]):
        if x >= xold:
            #If current x greater/equal to prev x, x is increasing (on upper surf)
            return len(X) - 1 - i #return index of Leading Edge (divides upper/lower surfs)
        else:
            #If current x less than prev x, x still diminishing (still on lower surf)
            xold = x

def MsesSplit(x, y):
    """Split MSES format into upper and lower surfaces.
    Find LE from MSES x geometry coordinates,
    Split y at this index(s).
    If LE point is at y=0, include in both sets of data.
    Return y split into upper/lower surfaces, with LE overlapping
    x --> MSES x coordinates
    y --> Any other MSES parameter (e.g. x/c, z/c, Cp, etc)
    """
    #FIND LE FROM BOTH SIDES (DETECT SHARED LE POINT)
    #Get index of leading edge starting from upper surface TE
    iLE_top = FindLE_top(x)
    #Get index of leading edge starting from lower surface TE
    iLE_bot = FindLE_bot(x)
    #Split upper and lower surface, reverse order upper surface
    up = y[iLE_top::-1]
    lo = y[iLE_bot:]
    return up, lo

def MsesInterp(xout, xmses, ymses):
    """Split MSES format data into upper and lower surfaces.  Then
    interpolate data to match given xout vector.
    xout  --> desired x locations
    xmses --> original x MSES data
    ymses --> original x/c, z/c, Cp, etc MSES data
    """
    xup_mses, xlo_mses = MsesSplit(xmses, xmses)
    yup_mses, ylo_mses = MsesSplit(xmses, ymses)
    yup = np.interp(xout, xup_mses, yup_mses)
    ylo = np.interp(xout, xlo_mses, ylo_mses)
    return yup, ylo

def MsesMerge(xlo, xup, ylo, yup):
    """ Merge separate upper and lower surface data into single MSES set.
    If LE point is shared by both sides, drop LE from lower set to avoid overlap
    xlo, xup --> lower/upper surface x coordinates to merge
    ylo, yup --> lower/upper surface y OR surface Cp values to merge
    """
    #drop LE point of lower surface if coincident with upper surface
    if xlo[0] == xup[0] and ylo[0] == yup[0]:
    # if xlo[0] == xup[0] and ylo[0] == 0 and yup[0] == 0:
        xlo = xlo[1:]
        ylo = ylo[1:]
    n1 = len(xup)     #number of upper surface points
    n = n1 + len(xlo) #number of upper AND lower surface points
    x, y = np.zeros(n), np.zeros(n)
    #reverse direction of upper surface coordinates
    x[:n1], y[:n1] = xup[-1::-1], yup[-1::-1]
    #append lower surface coordinates as they are
    x[n1:], y[n1:] = xlo, ylo
    return x, y

###############################################################################
### PANEL DISCRETIZATION FUNCTIONS ############################################
###############################################################################

def MakePanels(xgeom, ygeom, n_panel, method):
    """Discretize geometry into panels using various methods.
    Return array of panels.
    n_panel --> number of panels
    method --> panel discretization method:
                'constant' --> interpolate points with constant spacing in x
                'circle' --> map circle points to airfoil,
                'given' --> use given distribution
    """

    def ConstantSpacing(xgeom, ygeom, n_panel, frac=0.25):
        """Creates airfoil panel points with uniform x-spacing.
        (Finer spacing at TE to enforce kutta).
        x, y    --> geometry coordinates (MSES format)
        n_panel --> number of panels to distretize into
        frac    --> Ratio of TE panel length to uniform spacing length
        """

        c = max(xgeom) - min(xgeom) #chord

        #Number of Panels On One Surface
        if n_panel%2 != 0:
            #odd number of panels
            Nsurf = int((n_panel + 1) / 2)
            xoffset = 1 #offset LE so we have flat LE
            # print('\n\nConstant spacing with ODD # panels (Flat, vertical LE)\n\n')
            #                      _ _ _ _
            #                     /        \
            #              LE--> |          >  <--TE
            #                     \_ _ _ _ /
        else:
            #even number of panels
            Nsurf = int((n_panel + 2) / 2)
            xoffset = 0 #do not offset LE
            # print('\n\nConstant spacing with EVEN # panels (Wedge-shape LE)\n\n')
            #                      _ _ _ _
            #                     /        \
            #              LE--> <          >  <--TE
            #                     \_ _ _ _ /


        #Trailing Edge Length
        TE_length = (frac * c) / (Nsurf + (frac - 1))

        #MAKE UNIFORM SPACING VECTOR FOR NON-TE PANELS
        #Start at some non-zero x for panel normal to flow in front at LE
        xLE = min(xgeom) + 0.001 * c * xoffset
        #End is where TE panels start
        xTE = max(xgeom) - TE_length
        #create x points
            #(number of points is half (one surface) of amount needed to end up
            #with n_panels after accounting for TE panels
        xnew = np.linspace(xLE, xTE, Nsurf-1)

        #ADD TE POINT (distance from xnew[-2] to xnew[-1] is TE_length)
        xnew = np.append(xnew, max(xgeom))

        #SPLIT AND INTERPOLATE GEOMETRY FOR NEW X-SPACING
        ynewup, ynewlo = MsesInterp(xnew, xgeom, ygeom)

        #MANAGE ODD/EVEN PANELS
            #Odd number of panels has vertical LE face
            #Even number of panels has pointy wedge LE face
        if xoffset == 0:
            #Only one point at LE, average y postition
            yLE = (ynewup[0] + ynewlo[0])/2
            #make upper surface have new LE point
            ynewup[0] = yLE
            #remove identical LE point from lower surface
            ynewlo = ynewlo[1:]
            xnewlo = xnew[1:]
        else:
            #lower x is the same as upper for odd number of panels
            xnewlo = np.array(xnew)

        #Merge new coordinates into one set
        xends, yends = MsesMerge(xnewlo, xnew, ynewlo, ynewup)

        return xends, yends
    

    def CircleMethod(xgeom, ygeom, n_panel):
        """Create x-spacing based on a circle, then map y-points onto body
        geometry using linear interpolation.  Some panel spacings cause
        interpolation to break, so check the resulting geometry for each panel#
        x, y --> geometry coordinates
        n_panel --> number of panels to distretize into
        """

        #Make circle with diameter equal to chord of airfoil
        R = (xgeom.max() - xgeom.min()) / 2
        center = (xgeom.max() + xgeom.min()) / 2
        #x-coordinates of circle (also new x-coordinates of airfoil)
            #(n+1 because first and last points are the same)
        xends = center + R*np.cos(np.linspace(0, 2*np.pi, n_panel+1))
        #for each new x, find pair of original geometry points surrounding it
        #and linear interpolate to get new y

        #append first point of original geometry to end so that i+1
        #at last point works out
        xgeom, ygeom = np.append(xgeom, xgeom[0]), np.append(ygeom, ygeom[0])

        #Get index of split in circle x
        i = 0
        while i < len(xends):
            if (xends[i] - xends[i-1] < 0) and (xends[i+1] - xends[i] > 0):
                isplit = i
                break
            i += 1
        #Split upper and lower surfaces for interpolation
        xnewup = xends[isplit::-1]
        xnewlo = xends[isplit+1:]
        xoldup, yoldup, xoldlo, yoldlo = MsesSplit(xgeom, ygeom)
        #linear interpolate new y points
        ynewup = np.interp(xnewup, xoldup, yoldup)
        ynewlo = np.interp(xnewlo, xoldlo, yoldlo)
        #Merge new coordinates into one set
        yends = MsesMerge(xnewup, ynewup, xnewlo, ynewlo)[1]
        # yends[n_panel] = yends[0]

        #***************************might need to make trailing edge one point*
        return xends, yends

    if method == 'constant':
        #Constant spacing method Method
        xends, yends = ConstantSpacing(xgeom, ygeom, n_panel)
    elif method == 'circle':
        #Circle Method
        xends, yends = CircleMethod(xgeom, ygeom, n_panel)
    elif method == 'given':
        #use points as given
        xends, yends = xgeom, ygeom
        n_panel = len(xends) - 1

    # print('npanel', n_panel)
    # print('len ends', len(xends))
    # print('Upper/Lower y-coordinate at TE: (', xends[0],',', yends[0], ') ; (',
    #                                            xends[0],',', yends[-1], ')')
    #assign panels
    panels = np.zeros(n_panel, dtype=object)
    for i in range(0, n_panel):
        panels[i] = Panel(xends[i], yends[i], xends[i+1], yends[i+1])

    return panels

def PlotPanels(xgeom, ygeom, panels, name, show_plot=1):
    #dont plot if show_plot=0
    if show_plot==0:
        return

    edgespace = 0.1 * (xgeom.max()-xgeom.min())
    xmin, xmax = xgeom.min()-edgespace, xgeom.max()+edgespace
    ymin, ymax = ygeom.min()-edgespace, ygeom.max()+edgespace
    factor = 12
    size = (factor*(xmax-xmin), factor*(ymax-ymin))
    plt.figure(figsize = size)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(str(len(panels)) + ' Panel Discretization of ' + name)
    plt.xlabel(r'$\frac{x}{c}$')
    plt.ylabel(r'$\frac{y}{c}$')
    # plt.grid(True)
    plt.plot(xgeom, ygeom, '--b', label = 'Original Geometry', linewidth=2)
    plt.plot(np.append([p.xa for p in panels], panels[-1].xb),
                        np.append([p.ya for p in panels], panels[-1].yb),
                            'g', label = 'Panel', linewidth=2)
    plt.plot([p.xa for p in panels], [p.ya for p in panels],
                    'go', label = 'End Points', linewidth=1,)
    plt.plot([p.xc for p in panels], [p.yc for p in panels],
                 'k^', label = 'Center Points', linewidth=1, )
    plt.legend(loc='best', framealpha=0.75)
    # plt.axis('equal')
    plt.show()

def VortexPanelNormalIntegral(pi, pj):
    """Evaluate contribution of panel j at control point of panel i in
    panel i's normal direction.
    pi --> panel which is being contributed to
    pj --> panel which is contributing to other panel's normal velocity
    """
    #function to integrate
    def func(s):
        #s is the line of points from a panel start point to end point.
        #We need to integrate along s:

        #x-coordinates of s-vector along panel
        xjsj = pj.xa - np.sin(pj.beta)*s
        #y-coordinates of s-vector along panel
        yjsj = pj.ya + np.cos(pj.beta)*s
        #We need to calculate contribution of each x,y along panel j.
        #that's what xjsj yjsj are: the points along panel j.

        #The partial derivative of our integral with respect to the normal
        #direction produces these partial derivative terms, which can be
        #calculated using simple trigonometry from the angle between the
        #unit vectors.
        dydn = np.sin(pi.beta)
        dxdn = np.cos(pi.beta)

        #Now, let's make a vector that is the equation for the normal velocity
        #produced by each x,y pair along panel j and panel i's control point
        return (-1/(2 * np.pi) * ( (pi.xc - xjsj)*dydn - (pi.yc - yjsj)*dxdn ) / ( (pi.xc - xjsj) ** 2 + (pi.yc - yjsj) ** 2) )
        #pi.xc, pi.yc are the control points of panel i
        #xjsj, yjsj are the points along panel j
        #gamma isnt in this equation because we will multiply it in later with
            #our linear system of equations

    #Integrate along length of panel
    return integrate.quad(lambda s:func(s), 0., pj.length)[0]

def GeneralIntegral(x, y, pj, dir):
    """Evaluate contribution of panel at center of another panel,
    in x,y cartesian directions. (use to calculate field velocity vectors)
    Insure flow is tangent to surface at control point of each panel.
    x,y --> point where contribution is calculated
    pj --> panel from which contribution is calculated
    direction of component and type of contribution --> 'x source',
                                                        'y vortex',
                                                        etc
    """
    if dir=='x source':
        #derivatives with respect to: x direction for sources contribution
        dxd_, dyd_ = 1, 0
        coeff = 1
    elif dir=='x vortex':
        #derivatives with respect to: x direction for vortex contribution
            #(opposite of source)
            #so to be explicitly clear: the vairables dxd_, dyd_ are switched,
                #so this is technically incorrect notation
        dxd_, dyd_ = 0, 1
        coeff = -1
    elif dir=='y source':
        #derivatives with respect to: y direction for sources
        dxd_, dyd_ = 0, 1
        coeff = 1
    elif dir=='y vortex':
        #derivatives with respect to: y direction for vorticity
        #(again, signs are technically incorrect to make funciton work)
        dxd_, dyd_ = 1, 0
        coeff = -1

    #function to integrate
    def func(s):
        #x-coord of s-vector along panel
        xjsj = pj.xa - np.sin(pj.beta)*s
        #y-coord of s-vector along panel
        yjsj = pj.ya + np.cos(pj.beta)*s
        return (coeff*1./(2.*np.pi) * ((x - xjsj)*dxd_ + coeff*(y - yjsj)*dyd_)
                / ((x - xjsj) ** 2 + (y - yjsj) ** 2))

    #Integrate along length of panel
    return integrate.quad(lambda s:func(s), 0., pj.length)[0]

def GetVelocity(x, y, panels, Vinf, alpha):
    """Calculate velocity in x and y directions for (x,y) of any shape
    (Works for 2D meshgrid, 1D Numpy array, or single values of x,y).
    Generalized version of 'GetVelocityField'
    x,y --> coordinate(s) at which to calculate u,v
    """
    #Vectorize General Integral function so that it works for any shape (x,y)
    GeneralIntegralVectorized = np.vectorize(GeneralIntegral)

    #x velocity contribution at all points from all panel vorticity dists
    u = ( Vinf * np.cos(alpha)
        +sum([p.gamma*GeneralIntegralVectorized(x, y, p, 'x vortex')
                                            for p in panels]) )
    #y velocity contribution at all points from all panel vorticity dists
    v = ( Vinf * np.sin(alpha)
        +sum([p.gamma*GeneralIntegralVectorized(x, y, p, 'y vortex')
                                            for p in panels]) )
    return u, v

class Panel:
    def __init__(self, xa, ya, xb, yb):
        """Initialize panel
        (xa, ya) --> first end-point of panel
        (xb, yb) --> second end-point of panel
        """
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        #CONTROL-POINT (center-point)
        self.xc, self.yc = (xa + xb) / 2, ( ya + yb ) / 2
        #LENGTH OF THE PANEL
        self.length = ((xb - xa)**2 + (yb - ya)**2) ** 0.5

        #INITIALIZE:
        #VORTEX STRENGTH DISTRIBUTION
        self.gamma = 0.
        #TANGENTIAL VELOCITY AT PANEL SURFACE (CONTROL POINT)
        self.vt = 0.
        #PRESSURE COEFFICIENT AT PANEL SURFACE (CONTROL POINT)
        self.Cp = 0.

        #ORIENTATION OF THE PANEL (angle between x-axis and panel's normal)
        if xb-xa <= 0.:
            self.beta = np.arccos((yb-ya)/self.length)
        elif xb-xa > 0.:
            self.beta = np.pi + np.arccos(-(yb-ya)/self.length)

        #PANEL ON UPPER OR LOWER SURFACE (for plotting surface pressure)
        if self.beta <= np.pi:
            self.surf = 'upper'
        else:
            self.surf = 'lower'
            
def SolveVorticity(panels, ihole, Vinf, alpha):
    """Solve for the source strength distributions and vortex sheet such that
    the tangency condition and the Kutta condition are satified.
    """

    def KuttaCondition(panels):
        """
        KUTTA CONDITION:
            Tangential velocities of upper TE (Panel 1) and lower TE (Panel N) are
        the same assuming bernoulli and same pressure because shared point (TE).
            Vorticity of a vortex sheet: gamma = utop - ubot (Anderson eqn 4.8).
        Since utop = ubot for TE, ------> gammaTE = 0 <--------- that's the K.C.
        """

        n_panel = len(panels)
        #To make vorticity of TE top and TE bottom panel add to zero,
        #row entry will be array of zeros with first and last entry as 1's,
        #thus only accounting for TE top and bottom.  set equal to zero for KC
        x = np.zeros(n_panel)
        x[0], x[-1] = 1, 1
        return x

    def TangencyCondition(panels, ihole):
        """Array accounting for vortex contribution from each panel.
        panels --> array of panels
        ihole  --> panel index to apply kutta condition at ("hole")
        """
        n_panel = len(panels)
        #Populate matrix of system of equations for each panel
        size = (n_panel, n_panel)
        A = np.zeros(size)
        #Fill diagonal with term for each panel's contribution to itself
        np.fill_diagonal(A, 0.)
        #Fill matrix with each panel's contribution to every other panel
        for i, pi in enumerate(panels):
            for j, pj in enumerate(panels):
                if i != j:
                    A[i,j] = VortexPanelNormalIntegral(pi, pj)

        #Replace Missing Panel row with Kutta Condition
        A[ihole,:] = KuttaCondition(panels)

        return A

    #Make system to solve

    """
          |   vortex contributions    |  vorticies  | freestream normal |
          |                           |             |    contribution   |

           j=1 j=2 . .  . . j=N-1 j=N
    i=1   [ 0                        ]   [ gamma1  ]   [ -Vinf,n1 ]
          [                vortex    ]   [         ]   [ -Vinf,n2 ]
    i=2   [     0         contrib    ]   [ gamma2  ]   [     .    ]
     .    [       .                  ]   [    .    ]   [     .    ]
     .    [                          ]   [    .    ]   [          ]
    i=ik  [ 1   0  . .  . .  0   1   ] * [ gammaik ] = [     0    ]<--K.C.
     .    [                          ]   [    .    ]   [          ]
     .    [                .         ]   [    .    ]   [          ]
    i=N-1 [   vortex          0      ]   [ gammaN-1]   [     .    ]
          [   contrib                ]   [         ]   [     .    ]
    i=N   [                        0 ]   [ gammaN  ]   [ -Vinf,nN ]


                    [A]           *       [gam]      =        [b]

            NOTE:  ik is index of missing panel to enforce kutta condition
    """

    n_panel = len(panels)
    #CONTRIBUTION OF EACH PANEL TO FLOW TANGENCY OF EACH CONTROL POINT
    A = TangencyCondition(panels, ihole)
    #Right Hand Side
    b = np.empty(n_panel, dtype=float)
    for i, p in enumerate(panels):
        #FREESTREAM FLOW IN NORMAL DIR, MUST BE CANCELED BY VORTEX CONTRIBUTIONS
        b[i] = -Vinf * np.cos(alpha - p.beta)
    #KUTTA CONDITION (gamTE = 0)
    b[ihole] = 0

    #SOLVE SYSTEM
    gam = np.linalg.solve(A, b)

    #assign variables
    for i, p in enumerate(panels):
        p.gamma = gam[i]

    #Check for Kutta Condition
    # print('gamma1 =', panels[0].gamma,
    #       'gammaN =', panels[-1].gamma,
    #       'gamma1 + gammaN =', panels[0].gamma + panels[-1].gamma)

    return panels

def TangencyCheck(panels, Vinf, alpha, title='', show_plot=1):
    """Calculate velocity normal to body surface, should be zero.

    panels --> array of panels
    method --> 'integrate' (integrate contributions from all panels),
                'gamma' (panel surface velocity equals panel vorticity)
    """
    #dont plot if show_plot=0
    if show_plot==0:
        return

    n_panel = len(panels)

    #Populate matrix of system of equations for each panel
    size = (n_panel, n_panel)
    A = np.zeros(size)
    #Fill diagonal with term for each panel's contribution to itself
    np.fill_diagonal(A, 0.)
    #Fill matrix with each panel's contribution to every other panel
    for i, pi in enumerate(panels):
        for j, pj in enumerate(panels):
            if i != j:
                #tangential vel. contrib. at all points from all vortex dists.
                A[i,j] = VortexPanelNormalIntegral(pi, pj)

    #Populate b vector
    b = Vinf * np.cos([alpha - p.beta for p in panels])
    #vector of vorticity distribution (all panel gammas are the same)
    gam = [p.gamma for p in panels]
    #solve system for normal velocities
    vn = np.dot(A, gam) + b

    #Plot Surface Normal Velocity
    size = 8
    plt.figure(figsize = (size,size))
    # plt.xlim(0, 360)
    plt.title('Normal Velocity Over ' + title)
    plt.xlabel(r'$\frac{x}{c}$')
    plt.ylabel('$V_N$')
    plt.plot([p.xc for p in panels], vn)
#     plt.legend(loc='best')
    plt.show()
