import os
import pathlib
import argparse
import shutil
import sys
import json
from datetime import datetime

from TrainingController import TrainingController

if __name__ == '__main__':
    if not os.path.isfile(os.path.join(pathlib.Path(__file__).parent.resolve(), "training_parameters.json")):
        print(f"training_parameters.json is not here! exiting...")
        sys.exit()
    print()
    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "training_parameters.json"), 'r') as of:
        training_paramsjson = json.load(of)

    for item in training_paramsjson.items():
        print(item)
    logs_filePath = os.path.join(pathlib.Path(__file__).parent.resolve(), "logs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default=0, type=int)
    parser.add_argument("--overwrite_logs", default=True, type=bool)
    parser.add_argument("--debug_mode", default=True, type=bool)
    args = parser.parse_args()

    # cache some parameters
    experiment_id = args.exp_id
    overwrite2logs_flag = args.overwrite_logs
    debug_mode = args.debug_mode
    print(args.__dict__)

    # check if a log folder exists and create if not
    if not os.path.isdir(logs_filePath):
        os.mkdir(logs_filePath)
        print(f"logs file is created under: {logs_filePath}")
    logs_filePath = os.path.join(logs_filePath, str(experiment_id) + "_" + str(datetime.today().strftime('%d-%m-%Y'
                                                                                                         '-%H-%M-%S')))
    if not os.path.isdir(logs_filePath):
        os.mkdir(logs_filePath)
    else:
        if overwrite2logs_flag:
            shutil.rmtree(logs_filePath)
            os.mkdir(logs_filePath)
            print(f"deleted and created new folder to: {logs_filePath}")
        else:
            print(f"there is a folder named:{logs_filePath} and overwrite is disabled, exiting...:)")
            sys.exit()

    # save current_reward_setting
    dummy1 = os.path.join(pathlib.Path(__file__).parent.resolve(), "mSimulationCar.py")
    dummy2 = os.path.join(logs_filePath, "r2.py")
    shutil.copy2(dummy1, dummy2)
    # create a log file for training_logs, and one for training parameters
    training_log_file = open(os.path.join(logs_filePath, f"logs.txt"), mode="w")
    parameters_l_file = open(os.path.join(logs_filePath, f"training_params.txt"), mode="w")
    parameters_l_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
    parameters_l_file.write(str(training_paramsjson).replace(",", "\n"))
    parameters_l_file.flush()

    # start the training part
    training_path = pathlib.Path(__file__).parent.resolve()
    training_object = TrainingController(training_path=training_path,
                                         logs_file_path=logs_filePath,
                                         log_files=[training_log_file, parameters_l_file],
                                         training_params=training_paramsjson,
                                         debugMode=debug_mode)

    training_log_file.flush()
    parameters_l_file.flush()
    training_log_file.close()
    parameters_l_file.close()
