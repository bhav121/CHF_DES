"""Docstring for the example.py file. 

This file demonstrates how to use the package.
"""

## Imports. 
import datetime
import warnings
import os
import sys
from pathlib import Path

# Add chf_des directory to the path.
sys.path.insert(0, os.getcwd() + "/chf_des/")


from chf_des.experimenter import Experimenter

warnings.filterwarnings("ignore")

# Set simulation parameters.
simulation_start_date = datetime.datetime.strptime("01/01/2018", "%d/%m/%Y")
simulation_end_date = datetime.datetime.strptime("01/12/2022", "%d/%m/%Y")
simulation_time = (simulation_end_date - simulation_start_date).days
interarrival_time = 1/5
echo_test_time =(2/3)/24
inpatient_test_capacity = 1
outpatient_test_capacity = 1
confidence_interval = 0.9 

# Set default rule set.
default_rule_set =  {
                     "start_date": simulation_start_date,
                     "end_date": simulation_end_date,
                     "outpatient_weekday_supply_limit": 7,
                     "outpatient_saturday_supply_limit": 7,
                     "outpatient_sunday_supply_limit": 7,
                     "inpatient_weekday_supply_limit": 3,
                     "inpatient_saturday_supply_limit": 3,
                     "inpatient_sunday_supply_limit": 3,
                     "lockdown": False,
                     "label": "Default"
                  }

# Set intervention rule sets.
intervention_rule_sets = [ 
                           {
                              "start_date": datetime.datetime.strptime("23/03/2020", "%d/%m/%Y"),
                              "end_date": datetime.datetime.strptime("15/06/2020", "%d/%m/%Y"),
                              "outpatient_weekday_supply_limit": 0,
                              "outpatient_saturday_supply_limit": 0,
                              "outpatient_sunday_supply_limit": 0,
                              "inpatient_weekday_supply_limit": 3,
                              "inpatient_saturday_supply_limit": 3,
                              "inpatient_sunday_supply_limit": 3,
                              "lockdown": True,
                              "label": "Lockdown restrictions 1"
                        }
                        ,
                           {
                              "start_date": datetime.datetime.strptime("25/09/2020", "%d/%m/%Y"),
                              "end_date": datetime.datetime.strptime("12/04/2021", "%d/%m/%Y"),
                              "outpatient_weekday_supply_limit": 5,
                              "outpatient_saturday_supply_limit": 5,
                              "outpatient_sunday_supply_limit": 5,
                              "inpatient_weekday_supply_limit": 3,
                              "inpatient_saturday_supply_limit": 3,
                              "inpatient_sunday_supply_limit": 3,
                              "lockdown": True,
                              "label": "Lockdown restrictions 2"
                           }
                        ,
                           {
                              "start_date": datetime.datetime.strptime("01/05/2021", "%d/%m/%Y"),
                              "end_date": datetime.datetime.strptime("01/12/2021", "%d/%m/%Y"),
                              "outpatient_weekday_supply_limit": 10,
                              "outpatient_saturday_supply_limit": 10,
                              "outpatient_sunday_supply_limit": 10,
                              "inpatient_weekday_supply_limit": 3,
                              "inpatient_saturday_supply_limit": 3,
                              "inpatient_sunday_supply_limit": 3,
                              "lockdown": False,
                              "label": "Double outpatient capacity intervention"
                        }
                     ]

# Collect simulation parameters together. 
simulation_parameters = {
                           "intervention_rule_sets": intervention_rule_sets,
                           "default_rule_set": default_rule_set,
                           "start_date": simulation_start_date,
                           "end_date": simulation_end_date,
                           "interarrival_time": interarrival_time,
                           "simulation_time": simulation_time,
                           "echo_test_time": echo_test_time,
                           "inpatient_test_capacity": inpatient_test_capacity,
                           "outpatient_test_capacity": outpatient_test_capacity,
                           "confidence_interval": confidence_interval
                        }

# Get current directory.
current_dir = os.getcwd()

# Get results directory.
results_dir_path = current_dir + "/results/"

# Set experiment results path
experiment_results_path = results_dir_path + "experiment_" + str(1) + "/"

# Create experiment results directory if it doesn't already exist.
try:
   os.mkdir(experiment_results_path)
except:
   print("Directory already exists.")


# Set intervention colour dictionary.
intervention_colour_dictionary = {
                                    "Lockdown restrictions 1": "r",
                                    "Lockdown restrictions 2": "r",
                                    "Double outpatient capacity intervention": "b"
                                 }

# Set plotting start and end dates.
plotting_start_date = datetime.datetime.strptime("01/01/2020", "%d/%m/%Y").date()
plotting_end_date = simulation_end_date.date()

# Set plotting parameters.
plotting_parameters = {
                        "plotting_start_date": plotting_start_date,
                        "plotting_end_date": plotting_end_date,
                        "plotting_path": experiment_results_path, 
                        "patient_attribute": "age",
                        "save_flag": False,
                        "show_flag": False,
                        "intervention_colour_dictionary": intervention_colour_dictionary
                     }

# Initialise experimenter object.
experimenter = Experimenter(simulation_parameters)

# Set the number of simulation trials.
number_of_simulation_trials = 3

# Run experiment.
print("Started running simulations")
experimenter.run_experiment(number_of_simulation_trials, development_flag=False)

# Analyse experiment.
print("Started analysing simulations.")
experimenter.analyse_experiment()

# Plot experiment.
experimenter.plot_experiment(plotting_parameters)
