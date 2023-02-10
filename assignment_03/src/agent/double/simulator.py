import os
import subprocess
import time

import gepetto.corbaserver
import pinocchio as se3


class RobotSimulator:

    def __init__(self, time_step, robot):
        self.robot = robot

        self.frame_axes = []  # list of frames whose axes must be displayed in viewer

        self.DISPLAY_T = time_step  # refresh period for viewer

        # for gepetto viewer
        try:
            prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
            if int(prompt[1]) == 0:
                os.system('gepetto-gui &')
            time.sleep(1)
        except:
            pass
        gepetto.corbaserver.Client()
        self.robot.initViewer(loadModel=False)
        self.gui = self.robot.viewer.gui

        self.robot.loadViewerModel()
        self.robot.displayCollisions(False)
        self.robot.displayVisuals(True)

        self.gui.setCameraTransform(
            "python-pinocchio", [1.4622807502746582,
                                 -0.1106378510594368,
                                 0.15224549174308777,
                                 0.49773091077804565,
                                 0.46019819378852844,
                                 0.49909576773643494,
                                 0.5398008823394775]
        )

    def display(self, q):
        for frame in self.frame_axes:
            frame_id = self.robot.model.getFrameId(frame)
            H = self.robot.framePlacement(q, frame_id)
            self.robot.applyConfiguration("world/axes-" + frame, se3.SE3ToXYZQUATtuple(H))

        self.robot.display(q)
