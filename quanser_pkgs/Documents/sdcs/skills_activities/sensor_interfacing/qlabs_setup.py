import os
import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

def setup(
        initialPosition=[16.6, 2, 0.000],
        initialOrientation=[0,0,np.pi/2],
        rtModel=rtmodels.QCAR
    ):

    # Try to connect to Qlabs
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost")
        print("Connected to QLabs")
    except:
        print("Unable to connect to QLabs")
        quit()

    # Delete any previous QCar instances and stop any running spawn models
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # Spawn a QCar at the given initial pose
    hqcar = QLabsQCar(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        waitForConfirmation=True
    )

    # Create a new camera view and attach it to the QCar
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([8.484, 1.973, 12.209], [-0, 0.748, 0.792])
    hqcar.possess()


    # Start spawn model
    QLabsRealTime().start_real_time_model(rtModel)

def terminate():
    QLabsRealTime().terminate_real_time_model("QCar_Workspace")

if __name__ == '__main__':
    # XXX Add processing of command line arguments
    setup()