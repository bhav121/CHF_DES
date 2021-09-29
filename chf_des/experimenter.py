"""

This module runs experiments. This involves:

    - Initiating simulation trials. 

    - Collecting simulation results. 
    
    - Passing experiment results to plotter object to plot.
    
"""
## Imports 
import utilities 
import multiprocessing as mp
from plotter import Plotter

## Global variables
#  Note: This global dictionary is used to store the results from
#        the asynchronous computation of independent simulation trials.
global experiment_results

# Initialise experiment results dictionary.
experiment_results = {
                        "Hospital admissions": [],
                        "Community deaths": [],
                        "Outpatient queue deaths": [],
                        "Inpatient queue deaths": [],
                        "Outpatient wait times": [],
                        "Inpatient wait times": [],
                        "Echo queues": []
                    }

## Class
class Experimenter():

    def __init__(self, simulation_parameters: dict) -> None:
        """ Manages experiments. 

        This class contains methods for running and plotting the results of experiments.

        Parameters
        ----------
            simulation_parameters : dict
                Parameters needed to run the simulation.
        """
        # Store simulation parameters.
        self.parameters = simulation_parameters
    
        # Initialise variables to store results of experiment.
        self.experiment_results = {
                                    "Hospital admissions": [],
                                    "Community deaths": [],
                                    "Outpatient queue deaths": [],
                                    "Inpatient queue deaths": [],
                                    "Outpatient wait times": [],
                                    "Inpatient wait times": [],
                                    "Echo queues": []
                                }

        # Store number of simulation trials. 
        self.number_of_simulation_trials = None 

        # Store confidence interval.
        self.confidence_interval = self.parameters["confidence_interval"]

        # Store intervention rule sets.
        self.intervention_rule_sets = self.parameters["intervention_rule_sets"]
    
    # Decorator for timing the method.
    @utilities.timer
    def run_experiment(
            self, 
            number_simulation_trials: int,
            development_flag: bool
        ) -> None:
        """ Run experiment.

        This method runs the indepedent simulation trials and collects the results.
        There is a development flag that enables or disables asynchronous and parallelised
        computation of the simulation trials. Set development_flag to True if you want
        to change something in the simulation and receive helpful error messages... 


        Parameters
        ----------
            number_simulation_trials : int
                Number of simulation trials.
            development_flag : bool
                Flag to indicate whether or not to run computation asynchronously and in parallel.

        See also
        ----------
            utilities.run_simulation
        """
        
        # If development flag on then implement a normal for loop.
        if development_flag:

            for index in range(number_simulation_trials):

                # Run simulation trial.
                simulation_result = utilities.run_simulation(index, self.parameters)
                
                # Append results to the experiment results dictionary.
                self.experiment_results = get_result_standard(self.experiment_results,
                                                              simulation_result)
        
        # If development flag not on then run asynchronously and in parallel.
        else:

            # Create a pool of CPU workers.          
            pool = mp.Pool(mp.cpu_count())

            # Run async and paralell for loop.
            for index in range(number_simulation_trials):
                pool.apply_async(utilities.run_simulation,
                                args=(index, self.parameters),
                                callback=get_result_async)
      
            pool.close()
            pool.join()   
        
            # Store results.
            self.experiment_results = experiment_results
        
        self.number_of_simulation_trials = number_simulation_trials      

    @utilities.timer     
    def analyse_experiment(self) -> None:
        """ Analyse experiment.

        This method combines the results of all the simulation trials and computes
        the median, lower bound and upper bounds for the statistics of interest.


        See also
        ----------
            utilities.combine_and_compute_statistics
        """
        # Create list of confidence intervals for mapping purposes.
        confidence_interval_list = [self.confidence_interval for x in self.experiment_results]

        # Use map to apply the combine and compute statistics function for all statistics.
        combined_results = list(
                                    map(
                                        utilities.combine_and_compute_statistics,
                                        list(self.experiment_results.values()),
                                        confidence_interval_list
                                    )
                                )  
        # Store the results in the dictionary.
        self.experiment_results = dict(zip(self.experiment_results.keys(), combined_results))

    @utilities.timer
    def plot_experiment(self, plotting_parameters: dict) -> None:
        """ Plot results of experiment.

        This method instantiates the Plotter object and proceeds to plot 
        the results of the experiment. 

        Parameters
        ----------
            plotting_parameters : dict
                Parameters used for plotting. 

        See also
        ----------
            Plotter    
        """
        # Add remaining necessary data to plotting parameters dictionary. 
        plotting_parameters["experiment_results"] = self.experiment_results
        plotting_parameters["confidence_interval"] = self.confidence_interval
        plotting_parameters["intervention_rule_sets"] = self.intervention_rule_sets

        # Initialise plotter object.
        plotter = Plotter(plotting_parameters)
        
        # Plot statistics.
        plotter.plot_queue_wait_times(
                                        save_flag=plotting_parameters["save_flag"],
                                        show_flag=plotting_parameters["show_flag"] 
                                    )
                 
        plotter.plot_deaths_combined(
                                        save_flag=plotting_parameters["save_flag"],
                                        show_flag=plotting_parameters["show_flag"] 
                                    )

        plotter.plot_wait_times(
                                    save_flag=plotting_parameters["save_flag"],
                                    show_flag=plotting_parameters["show_flag"] 
                                )

        plotter.plot_queue_lengths_combined(
                                                save_flag=plotting_parameters["save_flag"],
                                                show_flag=plotting_parameters["show_flag"] 
                                            )

        plotter.plot_deaths_separate(
                                        death_location="Community",
                                        patient_attribute=plotting_parameters["patient_attribute"],
                                        save_flag=plotting_parameters["save_flag"],
                                        show_flag=plotting_parameters["show_flag"]
                                    ) 
                                        
        plotter.plot_deaths_separate(
                                        death_location="Outpatient queue",
                                        patient_attribute=plotting_parameters["patient_attribute"],
                                        save_flag=plotting_parameters["save_flag"],
                                        show_flag=plotting_parameters["show_flag"]
                                    ) 

        plotter.plot_deaths_separate(
                                        death_location="Inpatient queue",
                                        patient_attribute=plotting_parameters["patient_attribute"],
                                        save_flag=plotting_parameters["save_flag"],
                                        show_flag=plotting_parameters["show_flag"]
                                    ) 

        plotter.plot_hospital_admissions_all( 
                                                patient_attribute=plotting_parameters["patient_attribute"],
                                                save_flag=plotting_parameters["save_flag"],
                                                show_flag=plotting_parameters["show_flag"]
                                            ) 

        plotter.plot_hospital_admissions_from_outpatient_queue( 
                                                                patient_attribute=plotting_parameters["patient_attribute"],
                                                                save_flag=plotting_parameters["save_flag"],
                                                                show_flag=plotting_parameters["show_flag"]
                                                            ) 

        plotter.plot_queue_lengths_seperate(
                                                queue_location="Outpatient",
                                                patient_attribute=plotting_parameters["patient_attribute"],
                                                save_flag=plotting_parameters["save_flag"],
                                                show_flag=plotting_parameters["show_flag"]
                                            )

        plotter.plot_queue_lengths_seperate(
                                                queue_location="Inpatient",
                                                patient_attribute=plotting_parameters["patient_attribute"],
                                                save_flag=plotting_parameters["save_flag"],
                                                show_flag=plotting_parameters["show_flag"]
                                            )        
        
def get_result_async(simulation_result: dict):
    """ Append simulation trial results to experiment results.

    This function appends the results from the simulation to the experiment results 
    dictionary for later use.

    Parameters
    ----------
        simulation_result : dict
            Dictionary containing the results of the simulation
    """
    experiment_results["Hospital admissions"].append(simulation_result["Hospital admissions"])
    experiment_results["Community deaths"].append(simulation_result["Community deaths"])
    experiment_results["Outpatient queue deaths"].append(simulation_result["Outpatient queue deaths"])
    experiment_results["Inpatient queue deaths"].append(simulation_result["Inpatient queue deaths"]) 
    experiment_results["Outpatient wait times"].append(simulation_result["Outpatient wait times"])
    experiment_results["Inpatient wait times"].append(simulation_result["Inpatient wait times"])
    experiment_results["Echo queues"].append(simulation_result["Echo queues"])
    
def get_result_standard(experiment_results: dict, simulation_result: dict) -> dict:
    """ Append simulation trial results to experiment results.

    This function appends the results from the simulation to the experiment results 
    dictionary for later use.

    Parameters
    ----------
        simulation_result : dict
            Dictionary containing the results of the simulation.
        experiment_results : dict
            Dictionary containing the results of the experiment.

    Returns
    ----------
        experiment_results : dict
            Dictionary containing the results of the experiment.

    """
    experiment_results["Hospital admissions"].append(simulation_result["Hospital admissions"])
    experiment_results["Community deaths"].append(simulation_result["Community deaths"])
    experiment_results["Outpatient queue deaths"].append(simulation_result["Outpatient queue deaths"])
    experiment_results["Inpatient queue deaths"].append(simulation_result["Inpatient queue deaths"]) 
    experiment_results["Outpatient wait times"].append(simulation_result["Outpatient wait times"])
    experiment_results["Inpatient wait times"].append(simulation_result["Inpatient wait times"])
    experiment_results["Echo queues"].append(simulation_result["Echo queues"])
    
    return experiment_results