"""

This module contains utility functions for all other modules. 

"""
## Imports 

import datetime
import random
import functools
import operator
import time
from typing import Callable
import simpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from simulator import Simulator
from scipy import random
import matplotlib.dates as mdates
from matplotlib import figure
from numpy.random import default_rng 

## Global variables.


"""
A note on parameters:

These were the parameters from the original code. 

death_parameters = [2000, -0.0001, -0.01, 10]
hospital_admission_parameters = [2000, -0.0003, -0.01, 10]

The new parameters are just examples. This part of the simulation 
is a work in progress.
"""

# Set parameters for distributions. 
age_group_probabilities = [0.1, 0.14, 0.19, 0.29, 0.24, 0.04]
age_groups = [(20, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
pro_BNP_parameters = [5.2235, 0.07, 0.0009, 69.555, 2]
hospital_admission_parameters = [0.05, 1e-3, 10, 10]
death_parameters = [0.05, 3e-3, 20, 10]
gp_appointment_multiplier = 1/100
gp_lockdown_delay_parameters = [2, 7]
hospital_admissions_lockdown_delay_parameters = [2, 7]

# Set random number generator. 
# Note from Numpy docs: By default, Generator uses bits provided by PCG64 which has 
# better statistical properties than the legacy MT19937 used in RandomState.
rng = default_rng()

## Functions

def check_datetime_range_membership(
        start_datetime: datetime.datetime, 
        end_datetime: datetime.datetime, 
        current_datetime: datetime.datetime
    ) -> bool:
    """Checks if a datetime is within a specified range.

    This function is used by get_current_rule_set to determine the 
    current rule set for the simulation parameters. 

    Parameters
    ----------
        start_datetime : datetime.datetime 
            Start datetime of datetime range.
        end_datetime : datetime.datetime 
            End datetime of datetime range.
        current_datetime : datetime.datetime 
            Current datetime. 

    Returns
    ----------
        datetime_in_range : bool 
            Boolean identifying if datetime is within datetime range.
    """
    # Check if datetime is within current date range. 
    datetime_in_range = start_datetime < current_datetime and current_datetime < end_datetime
   
    return datetime_in_range


def determine_current_rule_set(
        current_datetime: datetime.datetime,
        intervention_rule_sets: list,
        default_rule_set: dict
    ) -> dict:
    """ Determines the current rule set. 

    This function determines the current rule set. The rule set is a set of simulation parameters
    such as the inpatient and outpatient echo supply limits. These rule sets can be defined to 
    simulate lockdown conditions and system interventions. 

    Parameters
    ----------
        current_datetime : datetime.datetime
            Current datetime. 
        intervention_rule_sets : list
            List of dictionarities containing the rule sets that specify some of the simulation 
            parameters. 
        default_rule_set : dict
            Dictionary containing the default rule set. 
    
    Returns
    ----------
        current_rule_set : dict 
            Dictionary containing the current rule set. 
    """  
    # Determine the current day of the week. 5 = Saturday, 6 = Sunday 
    week_day = current_datetime.weekday()

    # Find the appropriate rule set by checking if the current date falls within 
    # one of the specified date ranges.

    try:
        current_rule_set = next(rule_set for rule_set in intervention_rule_sets if 
                                (check_datetime_range_membership(rule_set["start_date"], 
                                                                rule_set["end_date"],  
                                                                current_datetime)))
    except:
        current_rule_set = default_rule_set

    return current_rule_set


def generate_arrival_time(interarrival_time: float) -> float:
    """ Generates patient arrival time. 

    This function generates the time that the patient arrives
    in the CHF system. The arrival process is modelled as a
    Poisson point process. The arrival times are sampled from an
    exponential distribution with rate parameter specified by the
    variable interarrival_time. 

    Parameters
    ----------
        interarrival_time : float 
            Interarrival time. 

    Returns
    ----------
        arrival_time : float 
            Time at which the patient arrives in the CHF system. 
    """
    # Sample patient arrival time from exponential distribution. 
    arrival_time = rng.exponential(interarrival_time)

    return arrival_time


def generate_gp_lockdown_delay() -> float:
    """ Generates gp lockdown delay. 

    This function generates the length of time that patients delay seeing
    their GP during the pandemic..
    
    Returns
    ----------
        gp_lockdown_delay : float 
            Time delay for seeing GP.
    """   
    # Generate lockdown delay with gp lockdown delay parameters. 
    gp_lockdown_delay =  generate_lockdown_delay(gp_lockdown_delay_parameters)
    
    return gp_lockdown_delay

    
def generate_hospital_admission_lockdown_delay() -> float:
    """ Generates hospital admission lockdown delay. 

    This function generates the length of time that patients delay seeing
    visiting hospital during the pandemic..
    
    Returns
    ----------
        hospital_admission_delay : float 
            Time delay for visiting hospital.
    """   
    # Generate lockdown delay with hospital admission lockdown delay parameters. 
    hospital_admission_delay =  generate_lockdown_delay(hospital_admissions_lockdown_delay_parameters)
    
    return hospital_admission_delay


def generate_lockdown_delay(lockdown_delay_parameters: list) -> float:
    """ Generates lockdown delay. 

    This function generates the gp or hospital admission lockdown delay 
    depending on the argument. The lockdown delays are sampled from 
    gamma distributions. 
    
    Returns
    ----------
        lockdown_delay : float 
            Time delay.
    """   
    # Generate lockdown delay with appropriate parameters.
    lockdown_delay = rng.gamma(shape=lockdown_delay_parameters[0],
                               scale=lockdown_delay_parameters[1])

    return lockdown_delay


def generate_age() -> int:
    """ Generates the age of the patient.

    This function randomly generates the age of a patient using
    predefined probabilities for predefined age groups.

    Returns
    ----------
    age : int 
        Age of patient. 
    """
    # Sample age group.
    selected_age_group_index = rng.choice(
                                            range(len(age_groups)),
                                            p=age_group_probabilities
                                        )

    selected_age_group = age_groups[selected_age_group_index]
        
    # Sample age from selected age group.
    age = int(rng.uniform(selected_age_group[0], selected_age_group[1]))

    return age


def generate_pro_BNP(patient_age: int) -> float:
    """ Generates  NT pro BNP result. 

    This function generates the NT pro BNP result for 
    a patient given their age. 
    
    Parameters
    ----------
        patient_age : int
            Age of the patient.

    Returns
    ---------- 
        pro_BNP : float 
            Patients NT pro BNP result.
    """
    # Generate pro BNP result from age.
    exp_component = np.exp(pro_BNP_parameters[0] 
                            + pro_BNP_parameters[1]*(patient_age-pro_BNP_parameters[3]) 
                            + pro_BNP_parameters[2]*(patient_age-pro_BNP_parameters[3]))
        
    noise_component = rng.normal(scale=pro_BNP_parameters[4])

    pro_BNP = np.power(exp_component, 2) + noise_component

    return pro_BNP


def generate_expected_gp_appointment_time(
        patient_arrival_time: float,
        patient_age: int,
        patient_pro_BNP: float,
        lockdown_status: bool
    ) -> float:
    """ Generates the expected time that a patient visits their GP.

    This function generates the expected time until a patient visits their GP. 
    Outside of lockdown periods, this time is sampled from an exponential 
    distribution. During lockdown periods, a lockdown delay is added. 
    
    Parameters
    ----------
        patient_arrival_time: float
            Time that the patient arrived in the CHF system. 
        patient_age : int 
            Age of patient.
        patient_pro_BNP : float
            Patients NT pro BNP result.
        lockdown_status : bool
            Flag indicating whether it is a currently a lockdown or not. 

    Returns
    ---------- 
        expected_gp_appointment_time : float 
            Expected time of patients gp appointment. 
    """
    # Compute GP appointment rate for exponential distribution.
    gp_appointment_rate = gp_appointment_multiplier*hospital_admission_parameters[0] * \
                              (hospital_admission_parameters[1]*patient_pro_BNP \
                              +hospital_admission_parameters[2]*patient_age)
                    
    # Sample expected time until GP appointment. 
    expected_time_until_gp_appointment = rng.exponential(gp_appointment_rate)

    # Compute expected time of GP appointment. 
    expected_gp_appointment_time = patient_arrival_time + expected_time_until_gp_appointment \
                                   + lockdown_status * generate_gp_lockdown_delay()

    return expected_gp_appointment_time


def generate_expected_hospital_admission_time(
        patient_arrival_time: float,
        patient_age: int,
        patient_pro_BNP: float,
        lockdown_status: bool
    ) -> float:
    """ Generates the expected time that a patient is admitted to hospital.

    This function generates the expected time that a patient is admitted to hospital.
    Outside of lockdown periods, this time is sampled from an exponential 
    distribution. During lockdown periods, a lockdown delay is added. 
    
    Parameters
    ----------
        patient_arrival_time: float
            Time that the patient arrived in the CHF system. 
        patient_age : int 
            Age of patient.
        patient_pro_BNP : float
            Patients NT pro BNP result.
        lockdown_status : bool
            Flag indicating whether it is a currently a lockdown or not. 

    Returns
    ---------- 
        expected_hospital_admission_appointment_time : float 
            Expected time of patients hospital admission. 
    """
    # Compute hospital admission rate for exponential distribution.
    hospital_admission_rate = hospital_admission_parameters[0] * \
                              (hospital_admission_parameters[1]*patient_pro_BNP \
                              +hospital_admission_parameters[2]*patient_age)
                    
    # Sample expected time until hospital admission. 
    expected_time_until_hospital_admission = rng.exponential(hospital_admission_rate)

    # Compute expected time of GP appointment. 
    expected_hospital_admission_appointment_time = patient_arrival_time \
                                                 + expected_time_until_hospital_admission \
                                                 + lockdown_status * generate_hospital_admission_lockdown_delay()

    return expected_hospital_admission_appointment_time


def generate_expected_death_time(
        patient_arrival_time: float,
        patient_age: int,
        patient_pro_BNP: float,
    ) -> float:
    """ Generates the expected time that a patient dies.

    This function generates the expected time that a patient dies.
    
    Parameters
    ----------
        patient_arrival_time: float
            Time that the patient arrived in the CHF system. 
        patient_age : int 
            Age of patient.
        patient_pro_BNP : float
            Patients NT pro BNP result.      

    Returns
    ---------- 
        expected_death_time : float 
            Expected time of patients death. 
    """
    # Compute death rate for exponential distribution.
    death_rate = death_parameters[0] * \
                 (death_parameters[1]*patient_pro_BNP \
                 +death_parameters[2]*patient_age)

    # Sample expected time until hospital admission. 
    expected_time_until_death = rng.exponential(death_rate)

    # Compute expected time of death. 
    expected_death_time = patient_arrival_time + expected_time_until_death
    
    return expected_death_time


def generate_echo_time_limit(patient_pro_BNP: float) -> float:
    """ Generate echo time limit.

    This function generates the time by which an outpatient should 
    receive their echo. This is used to set the priority of outpatients 
    joining the outpatient queue. The time limit is set as step
    function of the patients NT pro BNP result.

   Returns
    ---------- 
        echo_timie_limit : float 
            Expected time of patients death. 
    """ 
    # Set echo time limit as function of patients NT pro BNP result. 
    echo_time_limit = 2.0*7.0 if patient_pro_BNP > 2000 else 6.0*7.0
    
    return echo_time_limit


def schedule_gp_appointment(env: simpy.Environment, patient: object) -> simpy.Event:
    """ Schedules a gp appointment event. 

    This function schedules a SimPy timeout event for a gp appointment.
    
    Parameters
    ----------
        env : simpy.Environment
            Simpy environment.
        patient : object
            Patient.

    Returns
    ----------
        gp_app_event : simpy.event.timeout
            GP appointment event. 
    """
    # Schedule gp appointment event. 
    gp_app_event = env.timeout(patient.expected_gp_appointment_time - env.now,
                               value="gp_appointment")    

    return gp_app_event


def schedule_hospital_admission(env: simpy.Environment, patient: object) -> simpy.Event:
    """ Schedules a hospital admission event. 

    This function schedules a SimPy timeout event for a hospital admission.
    
    Parameters
    ----------
        env : simpy.Environment
            Simpy environment.
        patient : object
            Patient.

    Returns
    ----------
        hospital_admission_event : simpy.event.timeout
            Hospital admission event. 
    """
    # Schedule hospital admission event. 
    hospital_admission_event = env.timeout(patient.expected_hospital_admission_time - env.now,
                                           value="hospital_admission")    
                                
    return hospital_admission_event


def schedule_death(env: simpy.Environment, patient: object) -> simpy.Event:
    """ Schedules a death event.

    This function schedules a SimPy timeout event for a death.
    
    Parameters
    ----------
        env : simpy.Environment
            Simpy environment.
        patient : object
            Patient.

    Returns
    ----------
        death_event : simpy.event.timeout
            Death event. 
    """
    # Schedule death event. 
    death_event = env.timeout(patient.expected_death_time - env.now, value="death")    
                                
    return death_event


def compute_queue_time(
        current_time: float,
        request: simpy.resources.resource.PriorityRequest
    ) -> float:
    """ Compute time elapsed since joining queue.
    
    This function computes the time elapsed  since the patient joined the queue. 

    Parameters
    ----------
        current_time : float
            Current simulation time.
        request : simpy.resources.resource.PriorityRequest
            Simpy priority request that represents the patients action of joining the queue 
            and requesting access to the echo resource. 

    Returns
    ----------
        time_in_queue : float
            Time elapsed patient has been in the queue. 
    """
    # Compute time elapsed since patient joined the queue. 
    time_in_queue = current_time - request.time

    return time_in_queue 


def get_age(request: simpy.resources.resource.PriorityRequest) -> float:
    """ Get patients age.
    
    This function extracts the patients age from the request.

    Parameters
    ----------
        request : simpy.resources.resource.PriorityRequest
            Simpy priority request that represents the patients action of joining the queue 
            and requesting access to the echo resource. 

    Returns
    ----------
        patient_age : int
            Patients age.
    """
    # Get patients age. 
    patient_age = request.preempt["age"]

    return patient_age


def get_pro_BNP(request: simpy.resources.resource.PriorityRequest) -> float:
    """ Get patients NT pro BNP.
    
    This function extracts the patients NT pro BNP result from the request.

    Parameters
    ----------
        request : simpy.resources.resource.PriorityRequest
            Simpy priority request that represents the patients action of joining the queue 
            and requesting access to the echo resource. 

    Returns
    ----------
        patient_pro_BNP : float
            Patients NT pro BNP result.
             
    """
    # Get patients age. 
    patient_age = request.preempt["pro BNP"]

    return patient_age


def compute_queue_statistics(current_time: float, queue: simpy.Resource.GetQueue) -> dict:
    """ Computes echo queue statistics.

    This function computes the following statistics for an echo queue:
    
    -- queue length.
    -- average time in queue.
    -- average age of patients in queue.
    -- average NT pro BNP of patients in queue.

    It can be applied to both the outpatient and inpatient queues.
    
    Parameters
    ----------
        current_time : float
            Current simulation time.
        queue : simpy.Resource.GetQueue
            Echo queue.

    Returns
    ----------
        queue_statistics : dict
            Computed statistics for the queue. 
    """
    # Count the number of patients in the queue.
    queue_length = len(queue)

    # Create a list for mapping the time_in_queue function over each patient in the queue. 
    queue_time_list = [current_time]*queue_length
      
    # Compute the time that each patient has been waiting in the queue.
    queue_wait_times = list(map(compute_queue_time, queue_time_list, queue))  

    # Compute the average time patients have been waiting in the queue.
    avg_time_in_queue = functools.reduce(operator.add, queue_wait_times, 0)/np.float64(queue_length)
  
    # Compute the average age of patients in the queue.
    avg_age_in_queue = np.mean(list(map(get_age, queue)))

    # Compute the average NT pro BNP of patients in the queue.
    avg_pro_BNP_in_queue = np.mean(list(map(get_pro_BNP, queue))) 

    # Store statistics in a dictionary and return. 
    queue_statistics = {
                            "queue_length": queue_length,
                            "avg_time_in_queue": avg_time_in_queue,
                            "avg_age_in_queue": avg_age_in_queue,
                            "avg_pro_BNP_in_queue": avg_pro_BNP_in_queue
                        } 

    return queue_statistics


def compute_proportion(from_outpatient_queue: dict.values, total: dict.values) -> list:
    """ Computes proportion of patients being admitted to hospital from the outpatient queue.

    This function computes the proportion of patients being admitted to hospital from 
    the outpatient queue. Patients can go straight to hospital or be admitted whilst they 
    are waiting in the outpatient queue.
    
    Parameters
    ----------
        from_outpatient_queue : dict.values
            Daily count of patients admitted to hospital from the outpatient queue.
        total : dict.values
            Daily count of all patients admitted to hospital.

    Returns
    ----------
        proportion : list
            Daily proportion of patients admitted from outpatient queue.
    """
    # Compute proportion of patients admitted from outpatient queue. 

    proportion = list(
                        map(
                            lambda x, y: np.float64(x)/np.float64(y),
                            from_outpatient_queue,
                            total
                        )
                    ) 
    
    return proportion
    

def combine_and_compute_statistics(
        dataframes_list: list,
        confidence_interval: float
    ) -> pd.DataFrame:
    """ Combines dataframes from independent simulation trials and computes statistics.

    The results from each simulation are stored in a dataframe. This function 
    merges the dataframe from each simulation run and then computes mean
    and confidence interval statistics. 
    
    Parameters
    ----------
        dataframes_list : list
            List of dataframes from each simulation run.
        confidence_interval : float
            Percent to use for confidence interval computation.

    Returns
    ----------
        final_dataframe : pd.DataFrame
            Merged dataframe with statistics computed and stored. 
    """
    # Concatenate dataframes.
    concatened_dataframe = pd.concat(dataframes_list).groupby(level=0).agg(list)
   
    # Compute mean and confidence intervals and add them to the concatenated dataframe
    # and return it. 
    final_dataframe = add_median_and_confidence_intervals(concatened_dataframe, confidence_interval)

    return final_dataframe

   
def add_median_and_confidence_intervals(
        dataframe: pd.DataFrame,
        confidence_interval: float
    ) -> pd.DataFrame:
    """ Computes the mean and confidence intervals across simulation trials.

    This function computes the mean and confidence intervals for the simulations.
    
    Parameters
    ----------
        dataframe : pd.DataFrame
            Merged dataframe containing the results from all the simulations.
        confidence_interval : float
            Percent to use for confidence interval computation.

    Returns
    ----------
        dataframe : pd.DataFrame
            Merged dataframe with statistics computed and stored. 
    """
    # Compute upper bound
    ub = (confidence_interval+1)/2

    # Compute lower bound
    lb = (1-confidence_interval)/2
    
    # Loop through all columns.
    for column in dataframe.columns:
        
        # Convert dataframe column of lists into a list. 
        column_list = dataframe[column].tolist()

        # Compute median, lower and upper bounds.
        dataframe[column + " median"] = np.median(column_list, axis=1)    
        dataframe[column + " upper bound"] = np.quantile(column_list, ub, axis=1)
        dataframe[column + " lower bound"] = np.quantile(column_list, lb, axis=1)

        # Remove raw data.
        dataframe.drop(columns=column, inplace=True)

    return dataframe 


def initialise_dataframe(index: pd.DataFrame, columns: list) -> pd.DataFrame:
    """ Initialise dataframe with specified index and columns

    This function intialises a dataframe with a specified index and list of column
    headers.
    
    Parameters
    ----------
        index : pd.DataFrame
            Merged dataframe containing the results from all the simulations.
        columns : list
            List of column headers.

    Returns
    ----------
        dataframe : pd.DataFrame
            Initialised dataframe.
    """
    # Initialise dataframe
    dataframe = pd.DataFrame(index=index, columns=columns)

    return dataframe


def run_simulation(index: int, parameters: dict) -> dict:
    """ Run simulation.

    This function runs the simulator class with the given parameters.
    
    Parameters
    ----------
        index : int
            Index of the simulation run used for parallel processing.
        parameters : dict
            Dictionary of simulation parameters..

    Returns
    ----------
        simulation_result : dict
            Dictionary containing the results of the simulation
    """
    # Set random seed to ensure simulation results are different for each run. 
    np.random.seed(random.seed())

    # Initialise simulator object.
    simulator = create_simulator(parameters)
    
    # Run simulation.
    simulation_result = simulator.run_simulation()
    
    return simulation_result


def create_simulator(parameters: dict) -> object:
    """ Create simulator object..

    This function creates an instance of the simulator object.
    
    Parameters
    ----------
        parameters : dict
            Dictionary of simulation parameters..

    Returns
    ----------
        simulator : object
            Instance of the simulator class.
    """
    # Instantiate simulator object.
    simulator = Simulator(parameters)
                
    return simulator


def update_hospital_admissions_stats(
        hospital_admissions_total_all: dict,
        hospital_admissions_avg_pro_BNP_all: dict,
        hospital_admissions_avg_age_all: dict,
        hospital_admissions_total_from_outpatient_queue: dict,
        hospital_admissions_avg_pro_BNP_from_outpatient_queue: dict,
        hospital_admissions_avg_age_from_outpatient_queue: dict,
        patient: object
    ) -> None:
    """ Update hospital admission statistics.

    This function uses the patient data to update the hospital admission 
    statistics when a patient is admitted to hospital.
    
    Parameters
    ----------
        hospital_admissions_total_all : dict
            Dictionary for storing total count of hospital admissions.
        hospital_admissions_avg_pro_BNP_all : dict
            Dictionary for storing the average pro BNP of patients admitted to hospital.
        hospital_admissions_avg_age_all : dict
            Dictionary for storing the average age of patients admitted to hospital.
        hospital_admissions_total_from_outpatient_queue : dict
            Dictionary for storing the total count of hospital admissions from 
            the outpatient queue.
        hospital_admissions_avg_pro_BNP_from_outpatient_queue : dict
            Dictionary for storing the average pro BNP of patients admitted to hospital
            from the outpatient queue.
        hospital_admissions_avg_age_from_outpatient_queue : dict
            Dictionary for storing the average age of patients admitted to hospital 
            from the outpatient queue.
        patient : object
            Patient that was admitted to hospital.

    Note
    ----------
    The averages are computed once the simulation has finished. We accumulate the statistics 
    and then use the total count to compute the averages. 
    """
    # Retrieve the current date for indexing the dictionaries.
    current_date = patient.hospital_admission_datetime.date()
    
    # Update dictionary data. 
    hospital_admissions_total_all[current_date] += 1
    hospital_admissions_avg_pro_BNP_all[current_date] += patient.pro_BNP
    hospital_admissions_avg_age_all[current_date] += patient.age
    
    # If patient outpatient arrival datetime is recorded then the patient has 
    # come from the outpatient queue.
    outpatient_queue_flag = bool(patient.outpatient_queue_arrival_datetime)
    
    # Update hospital admissions from the outpatient queue dictionary data. 
    hospital_admissions_total_from_outpatient_queue[current_date] += outpatient_queue_flag
    hospital_admissions_avg_pro_BNP_from_outpatient_queue[current_date] += outpatient_queue_flag*patient.pro_BNP
    hospital_admissions_avg_age_from_outpatient_queue[current_date] += outpatient_queue_flag*patient.age


def update_community_deaths_stats(
        community_deaths_total: dict,
        community_deaths_avg_pro_BNP: dict,
        community_deaths_avg_age: dict,
        patient: object
    ) -> None:
    """ Update community death statistics.

    This function uses the patient data to update the community death statistics when 
    a patient dies. If they die before they have a GP appointment then they contribute 
    to the community death statistics.  
    
    Parameters
    ----------
        community_deaths_total : dict
            Dictionary for storing total count of community deaths.
        community_deaths_avg_pro_BNP : dict
            Dictionary for storing the average pro BNP of patients dying in the community.
        community_deaths_avg_age : dict
            Dictionary for storing the average age of patients dying in the community.
        patient : object
            Patient that died.

    Note
    ----------
    The averages are computed once the simulation has finished. We accumulate the statistics 
    and then use the total count to compute the averages. 
    """
    # Retrieve the current date for indexing the dictionaries.
    current_date = patient.death_datetime.date()
    
    # Update dictionary data.
    # Note: patient.died_in_community is a Boolean that evaluates to True 
    #       if the patient died in the community. It acts like a mask here.
    community_deaths_total[current_date] += patient.died_in_community
    community_deaths_avg_pro_BNP[current_date] += patient.died_in_community*patient.pro_BNP
    community_deaths_avg_age[current_date] += patient.died_in_community*patient.age


def update_outpatient_queue_deaths_stats(
        outpatient_queue_deaths_total: dict,
        outpatient_queue_deaths_avg_pro_BNP: dict,
        outpatient_queue_deaths_avg_age: dict,  
        patient: object
    ) -> None:
    """ Update outpatient queue death statistics.

    This function uses the patient data to update the outpatient queue death statistics when 
    a patient dies. If they die before whilst they are waiting in the outpatient echo queue
    then they contribute to the outpatient queue death statistics.  
    
    Parameters
    ----------
        outpatient_queue_deaths_total : dict
            Dictionary for storing total count of outpatient queue deaths.
        outpatient_queue_deaths_avg_pro_BNP : dict
            Dictionary for storing the average pro BNP of patients dying in the outpatient
            queue.
        outpatient_queue_deaths_avg_age : dict
            Dictionary for storing the average age of patients dying in the outpatient queue.
        patient : object
            Patient that died.

    Note
    ----------
    The averages are computed once the simulation has finished. We accumulate the statistics 
    and then use the total count to compute the averages. 
    """
    # Retrieve the current date for indexing the dictionaries.
    current_date = patient.death_datetime.date()

    # Update dictionary data.
    # Note: patient.died_in_outpatient_queue is a Boolean that evaluates to True 
    #       if the patient died in the outpatient queue. It acts like a mask here.
    outpatient_queue_deaths_total[current_date] += patient.died_in_outpatient_queue
    outpatient_queue_deaths_avg_pro_BNP[current_date] += patient.died_in_outpatient_queue*patient.pro_BNP
    outpatient_queue_deaths_avg_age[current_date] += patient.died_in_outpatient_queue*patient.age


def update_inpatient_queue_deaths_stats(
        inpatient_queue_deaths_total: dict,
        inpatient_queue_deaths_avg_pro_BNP: dict,
        inpatient_queue_deaths_avg_age: dict,
        patient: object
    ) -> None:
    """ Update inpatient queue death statistics.

    This function uses the patient data to update the inpatient queue death statistics when 
    a patient dies. If they die before whilst they are waiting in the inpatient echo queue
    then they contribute to the inpatient queue death statistics.  
    
    Parameters
    ----------
        inpatient_queue_deaths_total : dict
            Dictionary for storing total count of inpatient queue deaths.
        inpatient_queue_deaths_avg_pro_BNP : dict
            Dictionary for storing the average pro BNP of patients dying in the inpatient
            queue.
        inpatient_queue_deaths_avg_age : dict
            Dictionary for storing the average age of patients dying in the inpatient queue.
        patient : object
            Patient that died.

    Note
    ----------
    The averages are computed once the simulation has finished. We accumulate the statistics 
    and then use the total count to compute the averages. 
    """              

    # Retrieve the current date for indexing the dictionaries.
    current_date = patient.death_datetime.date()
    
    # Update dictionary data.
    # Note: patient.died_in_inpatient_queue is a Boolean that evaluates to True 
    #       if the patient died in the inpatient queue. It acts like a mask here.
    inpatient_queue_deaths_total[current_date] += patient.died_in_inpatient_queue
    inpatient_queue_deaths_avg_pro_BNP[current_date] += patient.died_in_inpatient_queue*patient.pro_BNP
    inpatient_queue_deaths_avg_age[current_date] += patient.died_in_inpatient_queue*patient.age


def update_outpatient_wait_times_stats(
        outpatient_wait_times_total: dict,
        outpatient_wait_times_avg_time: dict,
        patient: object
    ) -> None:
    """ Update outpatient echo wait time statistics.

    This function uses the patient data to update the outpatient echo wait time statistics.
    These statistics track how long patients waited to receive their outpatient echo for 
    patients who have received an outpatient echo.  
    
    Parameters
    ----------
        outpatient_wait_times_total : dict
            Dictionary for storing the count of patients that received an outpatient echo.
        outpatient_wait_times_avg_time : dict
            Dictionary for storing the average time that patients waited in the outpatient queu
            before they received their echo.
        patient : object
            Patient that died.

    Note
    ----------
    The averages are computed once the simulation has finished. We accumulate the statistics 
    and then use the total count to compute the averages. 
    """    
    # Retrieve the current date for indexing the dictionaries.
    current_date = patient.echo_datetime.date()

    # Update dictionary data. 
    outpatient_wait_times_total[current_date] += patient.outpatient_echo
    outpatient_wait_times_avg_time[current_date] += patient.echo_wait_time


def update_inpatient_wait_times_stats(
        inpatient_wait_times_total: dict,
        inpatient_wait_times_avg_time: dict,
        patient: object
    ) -> None:
    """ Update inpatient echo wait time statistics.

    This function uses the patient data to update the inpatient echo wait time statistics.
    These statistics track how long patients waited to receive their inpatient echo for 
    patients who have received an inpatient echo.  
    
    Parameters
    ----------
        inpatient_wait_times_total : dict
            Dictionary for storing the count of patients that received an inpatient echo.
        inpatient_wait_times_avg_time : dict
            Dictionary for storing the average time that patients waited in the inpatient queu
            before they received their echo.
        patient : object
            Patient that died.

    Note
    ----------
    The averages are computed once the simulation has finished. We accumulate the statistics 
    and then use the total count to compute the averages. 
    """    
    # Retrieve the current date for indexing the dictionaries.
    current_date = patient.echo_datetime.date()

    # Update dictionary data. 
    inpatient_wait_times_total[current_date] += patient.inpatient_echo
    inpatient_wait_times_avg_time[current_date] += patient.echo_wait_time


def compute_averages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Compute averages for the statistics.

    This function uses the total counts to compute the average statistics.
    
    Parameters
    ----------
        dataframe : pd.DataFrame
            Dataframe of statistics to compute averages for.

    Returns
    ----------
        dataframe : pd.DataFrame
            Dataframe of statistics with averages computed.
    """    
    # Extract list of colums.
    columns_to_average = dataframe.columns.tolist()

    # Remove total column as this does not need to be averaged.
    columns_to_average.remove("Total")
    
    # Compute the mean.
    dataframe[columns_to_average] = dataframe[columns_to_average].div(dataframe["Total"], axis=0).fillna(0)
   
    return dataframe


def compute_hospital_admissions_averages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Compute averages for the hospital admission statistics..

    This function uses the total counts to compute the average statistics for
    the hospital admissions. It is a seperate function as the hospital admission 
    statistics dataframe has a different layout to the others.
    
    Parameters
    ----------
        dataframe : pd.DataFrame
            Dataframe of statistics to compute averages for.

    Returns
    ----------
        dataframe : pd.DataFrame
            Dataframe of statistics with averages computed.
    """    
    # Extract list of colums.
    columns_to_average = dataframe.columns.tolist()

    # Remove total column as this does not need to be averaged.
    columns_to_average.remove("Total all")
    columns_to_average.remove("Total from outpatient queue")
  
    # Compute the means.
    dataframe[columns_to_average[0:2]] = dataframe[columns_to_average[0:2]].div(dataframe["Total all"], axis=0).fillna(0)
    dataframe[columns_to_average[2:4]] = dataframe[columns_to_average[2:4]].div(dataframe["Total from outpatient queue"], axis=0).fillna(0)

    return dataframe


def generic_plot(
        dataframe: pd.DataFrame,
        axes: plt.axes,
        patient_statistic: str,
        confidence_interval: float,
        title: str,
        x_label: str,
        y_label: str) -> plt.axes:
    """ Plot patient statistics.

    This function is a generic plotting function that is used to 
    plot the median, upper and lower bounds for the patient statistics. 
    
    Parameters
    ----------
        dataframe : pd.DataFrame
            Dataframe of statistics to plot.. 
        axes: plt.axes
            Axes object to plot the data on.
        patient_outcome: str
            Patient outcome to plot.
        confidence_interval: float
            Confidence interval used to compute the bounds.
        title: str
            Title for the plot.
        x_label: str
            Label for the x-axis.
        y_label: str
            Label for the y-axis.

    Returns
    ----------
        axes: plt.axes,
            Axes with data plotted.
    """ 
    # Plot median.
    dataframe.plot(
                    y=patient_statistic + " median",
                    ax=axes,
                    color="b",
                    label="Median"
                )

    # Plot upper bound.
    dataframe.plot(
                    y=patient_statistic + " upper bound",
                    ax=axes,
                    color="b",
                    alpha=0.2,
                    label="{}% CI".format(confidence_interval)
                )

    # Plot lower bound.
    dataframe.plot(
                    y=patient_statistic + " lower bound",
                    ax=axes,
                    color="b",
                    alpha=0.2,
                    label="_nolabel_"
                )
    # Shade confidence interval region.
    axes.fill_between(  
                        dataframe.index,
                        dataframe[patient_statistic + " upper bound"],
                        dataframe[patient_statistic + " lower bound"],
                        alpha=0.2
                    )                                   

    # Set title and axis labels. 
    axes.set_title(title, fontsize=20)
    axes.set_xlabel(x_label, fontsize=20)
    axes.set_ylabel(y_label, fontsize=20)

    return axes


def format_figure(
        fig: figure.Figure,
        lines: list,
        labels: list,
        figure_name: str,
        save_flag: bool,
        show_flag: bool
    ) -> None:
    """ Plot patient outcomes.

    This function is a generic plotting function that is used to 
    plot the median, upper and lower bounds for the patient outcomes.  
    
    Parameters
    ----------
        fig : fig
            Figure of plot to save.
        lines : plt.lines
            Intervention lines of the plot.
        labels : plt.labels
            Labels of the plot.
        figure_name : str
            Name to save the figure with.
        save_flag : bool
            Flag indicating whether or not to save the figure.
        show_flag : bool
            Flag indicating whether or not to show the figure. 
    """ 
    # Format the dates.
    fig.autofmt_xdate()
    
    # Delete spare copies of the intervention lines.
    del lines[1::2]
    
    # Add legend.
    fig.legend(lines,  labels, loc="lower center", ncol=3, prop={"size":15})
    
    # Format the figure for saving.
    manager = plt.get_current_fig_manager()
    
    manager.resize(*manager.window.maxsize())
    
    plt.tight_layout()

    fig = plt.gcf()
    
    fig.set_size_inches(    
                            manager.window.maxsize()[0]/plt.gcf().dpi,
                            manager.window.maxsize()[1]/plt.gcf().dpi,
                            forward=False
                        )

    # Save the figure if requested.
    if save_flag:
        # Save figure.
        fig.savefig(figure_name + ".svg", dpi=plt.gcf().dpi) 
        
    # Show the figure if requested.
    if show_flag:
        # Show figure.
        plt.show()

def add_intervention_lines(
        intervention_rule_sets: list,
        intervention_colour_dictionary: dict,
        axes: plt.axes
    ) -> list:
    """ Add intervention lines to the plot.

    This function adds the intervention lines to the plot. These could be 
    the lockdown lines and any interventions like increases in capacity. 
    By lines, we mean lines on the plot that indicate the time interval over
    which the intervention applies.
    
    Parameters
    ----------
        intervention_rule_sets : list
            List containing the intervention rule sets dictionaries.
        intervention_colour_dictionary : list
            List containing the intervention colour dictionaries. These specify
            what colour to use for the intervention lines.
        axes: plt.axes,
            Axes with data plotted.
  
    Returns
    ----------
        plot_data : list
            List containing the axes, lines and labels.

    """ 
    # Initialise empty lists.
    lines = []
    labels = []
            
    # Loop through the intervention rule sets.
    for rule_set in intervention_rule_sets:
            
        # Add the lines to the plot with the metadata specified in the intervention 
        # rule sets and colour dictionary.
        lines.append(
                        axes.axvline(
                                        pd.to_datetime(rule_set["start_date"]),
                                        color=intervention_colour_dictionary[rule_set["label"]],
                                        lw=1,
                                        label=rule_set["label"]
                                    )
                    )
        # Add the lines to the plot with the metadata specified in the intervention 
        # rule sets and colour dictionary.
        lines.append(
                        axes.axvline(
                                        pd.to_datetime(rule_set["end_date"]),
                                        color=intervention_colour_dictionary[rule_set["label"]],
                                        lw=1,
                                        label=rule_set["label"]                                       
                                    )
                    )
        # Add the intervention label.
        labels.append(rule_set["label"])
    
    # Create axes object.
    axes = plt.gca()

    # Format axes object.
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    axes.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
    # Store data in a list and return.
    plot_data = [axes, lines, labels]

    return plot_data


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Clean data for plotting.

    This function cleans the data for plotting. It currently just replaces
    nans with zeros. It may be worth thinking about implementing some interpolation
    or smoothing function here...

    Parameters
    ----------
        dataframe : pd.DataFrame
            Dataframe of statistics to clean. 
  
    """    
    # Fill nans with zeros.
    dataframe.fillna(0, inplace=True)
    

def timer(func: Callable) -> Callable:
    """ Print runtime of function.

    Decorator for printing the runtime of a function.

    Parameters
    ----------
        func : function
            Function to compute the runtime of.
  
    """       
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        
        start_time = time.perf_counter()  
        func(*args, **kwargs)
        end_time = time.perf_counter() 
        
        run_time = end_time - start_time  
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
            
    return wrapper_timer