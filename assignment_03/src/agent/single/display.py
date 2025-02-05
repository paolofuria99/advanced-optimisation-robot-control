import os
import subprocess
import time

import gepetto.corbaserver
import pinocchio as pin


class Visual:
    """
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
        * the name of the 3D objects inside Gepetto viewer.
        * the ID of the joint in the kinematic tree to which the body is attached.
        * the placement of the body with respect to the joint frame.
    """

    def __init__(self, name, jointParent, placement):
        self.name = name  # Name in gepetto viewer
        self.jointParent = jointParent  # ID (int) of the joint
        self.placement = placement  # placement of the body wrt joint, i.e. bodyMjoint

    def place(self, display, oMjoint):
        oMbody = oMjoint * self.placement
        display.place(self.name, oMbody, False)


class Display:
    """
    A class implementing a client for the Gepetto-viewer server. The main
    method of the class is 'place', that sets the position/rotation of a 3D visual object in a scene.
    """

    def __init__(self, windowName="pinocchio"):
        """
        This function connect with the Gepetto-viewer server and open a window with the given name.
        If the window already exists, it is kept in the current state. Otherwise, the newly-created
        window is set up with a scene named 'world'.
        """
        l = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
        if int(l[1]) == 0:
            os.system('gepetto-gui &')
        time.sleep(2)

        # Create the client and connect it with the display server.
        try:
            self.viewer = gepetto.corbaserver.Client()
        except:
            print("Error while starting the viewer client. ")
            print("Check whether Gepetto-viewer is properly started")

        # Open a window for displaying your agent.
        try:
            # If the window already exists, do not do anything.
            windowID = self.viewer.gui.getWindowID(windowName)
            print("Warning: window '" + windowName + "' already created.")
            print("The previously created objects will not be destroyed and do not have to be created again.")
        except:
            # Otherwise, create the empty window.
            windowID = self.viewer.gui.createWindow(windowName)
            # Start a new "scene" in this window, named "world", with just a floor.
            self.viewer.gui.createScene("world")
            self.viewer.gui.addSceneToWindow("world", windowID)

        # Finally, refresh the layout to obtain your first rendering.
        self.viewer.gui.refresh()
        self.viewer.gui.setCameraTransform(
            windowName, [0.027320027351379395,
                         -5.775243759155273,
                         0.08012843132019043,
                         0.7071067690849304,
                         0.0,
                         0.0,
                         0.7071067690849304]
        )

    def place(self, objName, M, refresh=True):
        """
        This function places (ie changes both translation and rotation) of the object
        names "objName" in place given by the SE3 object "M". By default, immediately refresh
        the layout. If multiple objects have to be placed at the same time, do the refresh
        only at the end of the list.
        """
        self.viewer.gui.applyConfiguration(
            objName,
            pin.SE3ToXYZQUATtuple(M)
        )
        if refresh: self.viewer.gui.refresh()
