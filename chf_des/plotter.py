"""

This module plots the results of an experiment.

"""
## Imports 
import utilities
from matplotlib import pyplot as plt 

## Class
class Plotter():
    def __init__(self, plotting_parameters: dict) -> None:
        """ Manages plotting. 

        This class includes methods for plotting the results of the experiments.

        Parameters
        ----------
            plotting_parameters : dict
                Parameters needed to run the simulation.

        """
        # Store the plotting parameters.
        self.plotting_parameters = plotting_parameters

        # Unpack experiment results dictionary into separate data frames.
        self.hospital_admissions = plotting_parameters["experiment_results"]["Hospital admissions"] \
                                   [plotting_parameters["plotting_start_date"]:plotting_parameters["plotting_end_date"]]

        self.community_deaths =  plotting_parameters["experiment_results"]["Community deaths"] \
                                 [plotting_parameters["plotting_start_date"]:plotting_parameters["plotting_end_date"]]

        self.outpatient_queue_deaths =  plotting_parameters["experiment_results"]["Outpatient queue deaths"] \
                                        [plotting_parameters["plotting_start_date"]:plotting_parameters["plotting_end_date"]]
        
        self.inpatient_queue_deaths = plotting_parameters["experiment_results"]["Inpatient queue deaths"] \
                                      [plotting_parameters["plotting_start_date"]:plotting_parameters["plotting_end_date"]]
        
        self.outpatient_wait_times = plotting_parameters["experiment_results"]["Outpatient wait times"] \
                                     [plotting_parameters["plotting_start_date"]:plotting_parameters["plotting_end_date"]]
        
        self.inpatient_wait_times = plotting_parameters["experiment_results"]["Inpatient wait times"] \
                                    [plotting_parameters["plotting_start_date"]:plotting_parameters["plotting_end_date"]]
        
        self.echo_queues =  plotting_parameters["experiment_results"]["Echo queues"] \
                            [plotting_parameters["plotting_start_date"]:plotting_parameters["plotting_end_date"]]
        
        # Clean the data.
        self.clean_data()
    
        # Store death statistic dataframes in a dictionary.
        self.death_statistics = {
                                    "Community": self.community_deaths,
                                    "Outpatient queue": self.outpatient_queue_deaths,
                                    "Inpatient queue": self.inpatient_queue_deaths
                                }
        
        # Store confidence interval. 
        self.confidence_interval = plotting_parameters["confidence_interval"]

        # Store intervention rule sets.
        self.intervention_rule_sets = plotting_parameters["intervention_rule_sets"]

        # Store colour dictionary to different between different interventions.
        self.intervention_colour_dictionary = plotting_parameters["intervention_colour_dictionary"]

        # Set start and end dates for plotting.
        self.plotting_start_date =  plotting_parameters["plotting_start_date"]

        self.plotting_end_date =  plotting_parameters["plotting_end_date"]

        # Set path to store plots
        self.plotting_path =  plotting_parameters["plotting_path"]

    def clean_data(self) -> None:
        """ Cleans data. 

        This method cleans data for plotting.

        See also
        ----------
            utilities.clean_data

        """
        utilities.clean_data(self.community_deaths)
        utilities.clean_data(self.hospital_admissions)
        utilities.clean_data(self.outpatient_queue_deaths)
        utilities.clean_data(self.inpatient_queue_deaths)
        utilities.clean_data(self.echo_queues)
        utilities.clean_data(self.outpatient_wait_times)
        utilities.clean_data(self.inpatient_wait_times)
        
    def plot_queue_lengths_combined(self, save_flag: bool, show_flag: bool) -> None:
        """ Plots echo queue lengths. 

        This method plots the outpatient and inpatient echocardiogram queue lengths 
        over time on the same plot. 

        Parameters
        ----------
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2) = plt.subplots(2, 1, sharex=True)
    
        # Plot the outpatient queue lengths.
        axes_1 = utilities.generic_plot(
                                            self.echo_queues,
                                            axes_1,
                                            "Outpatient queue length",
                                            self.confidence_interval,
                                            "Outpatient queue length",
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot the inpatient queue lengths.
        axes_2 = utilities.generic_plot(
                                            self.echo_queues,
                                            axes_2,
                                            "Inpatient queue length",
                                            self.confidence_interval,
                                            "Inpatient queue length",
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot intervention lines.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)

        # Set figure name.
        figure_name = self.plotting_path + "echo_queue_lengths"

        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)

    def plot_queue_lengths_seperate(
            self,
            queue_location: str,
            patient_attribute: str,
            save_flag: bool,
            show_flag: bool
        ) -> None:    
        """ Plots echo queue lengths. 

        This method plots either the outpatient or inpatient echocardiogram queue lengths.
        It also plots the average age or average NT pro BNP of patients in the queue as 
        requested.

        Parameters
        ----------
            queue_location : str
                Indicates whether to plot outpatient or inpatient data.
            patient_attribute : str
                Indicates whether to plot average age or average NT pro BNP.
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2) = plt.subplots(2, 1, sharex=True)
    
        # Plot queue length.
        axes_1 = utilities.generic_plot(
                                            self.echo_queues,
                                            axes_1,
                                            queue_location + " queue length",
                                            self.confidence_interval,
                                            queue_location + " queue length",
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot average patient attribute.
        axes_2 = utilities.generic_plot(
                                            self.echo_queues,
                                            axes_2,
                                            queue_location + " queue average " + patient_attribute,
                                            self.confidence_interval,
                                            "Average " + patient_attribute + " of patients in " + queue_location + " queue",
                                            "Time (days)",
                                            "Patient " + patient_attribute
                                        )
        
        # Plot intervention lines.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)

        # Set figure name.
        figure_name = self.plotting_path + queue_location + "_echo_queue_length_" + patient_attribute
        
        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)


    def plot_wait_times(self, save_flag: bool, show_flag: bool) -> None:
        """ Plots echo queue wait times. 

        This method plots the time that patients waited for outpatient and inpatient 
        echocardiograms.

        Parameters
        ----------
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2) = plt.subplots(2, 1, sharex=True)
    
        # Plot outpatient echo wait times.
        axes_1 = utilities.generic_plot(
                                            self.outpatient_wait_times,
                                            axes_1,
                                            "Average time",
                                            self.confidence_interval,
                                            "Average time patients waited for outpatient echo",
                                            "Time (days)",
                                            "Days"
                                        )
        
        # Plot inpatient echo wait times.
        axes_2 = utilities.generic_plot(
                                            self.inpatient_wait_times,
                                            axes_2,
                                            "Average time",
                                            self.confidence_interval,
                                            "Average time patients waited for inpatient echo",
                                            "Time (days)",
                                            "Days"
                                        )

        # Plot intervention lines.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)

        # Set figure name.
        figure_name = self.plotting_path + "echo_wait_lengths"
        
        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)



    def plot_queue_wait_times(self, save_flag: bool, show_flag: bool) -> None:
        """ Plots echo queue wait times. 

        This method plots the time that patients have been waiting in the outpatient 
        and inpatient queues.

        Parameters
        ----------
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2) = plt.subplots(2, 1, sharex=True)
    
        # Plot average time spent in the outpatient queue for patients who are yet to receive an echo.
        axes_1 = utilities.generic_plot(
                                            self.echo_queues,
                                            axes_1,
                                            "Average time spent in outpatient queue",
                                            self.confidence_interval,
                                            "Average time patients have been waiting in the outpatient queue",
                                            "Time (days)",
                                            "Days"
                                        )

        # Plot average time spent in the inpatient queue for patients who are yet to receive an echo.
        axes_2 = utilities.generic_plot(
                                            self.echo_queues,
                                            axes_2,
                                            "Average time spent in inpatient queue",
                                            self.confidence_interval,
                                            "Average time patients have been waiting in the inpatient queue",
                                            "Time (days)",
                                            "Days"
                                        )

        # Plot intervention lines.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)

        # Set figure name.
        figure_name = self.plotting_path + "echo_queue_wait_times"

        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)

    def plot_hospital_admissions_all(
            self,
            patient_attribute: str,
            save_flag: bool,
            show_flag: bool
        ) -> None:
        """ Plots total hospital admissions. 

        This method plots the total hospital admissions.It also plots the average age
        or average NT pro BNP of patients admitted to hospital.

        Parameters
        ----------
            patient_attribute : str
                Indicates whether to plot average age or average NT pro BNP.
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2) = plt.subplots(2, 1, sharex=True)
        
        # Plot all hospital admissions.
        axes_1 = utilities.generic_plot(
                                            self.hospital_admissions,
                                            axes_1,
                                            "Total all",
                                            self.confidence_interval,
                                            "Daily hospital admissions",
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot average patient attribute. 
        axes_2 = utilities.generic_plot(
                                            self.hospital_admissions,
                                            axes_2,
                                            "Average " + patient_attribute + " all",
                                            self.confidence_interval,
                                            "Average " + patient_attribute + " of patient admitted to hospital",
                                            "Time (days)",
                                            "Patient " + patient_attribute
                                        )
        
        # Plot intervention lines.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)

        # Set figure name.
        figure_name = self.plotting_path + "hospital_admissions_" + patient_attribute
 
        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)


    def plot_hospital_admissions_from_outpatient_queue(
            self,
            patient_attribute: str,
            save_flag: bool,
            show_flag: bool
        ) -> None:
        """ Plots hospital admissions from outpatient queue. 

        This method plots the total hospital admissions from the outpatient queue and 
        the proportion of all hospital admissions that come from the outpatient queue.
        It also plots the average age or average NT pro BNP of patients admitted to hospital.

        Parameters
        ----------
            patient_attribute : str
                Indicates whether to plot average age or average NT pro BNP.
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2, axes_3) = plt.subplots(3, 1, sharex=True)
        
        # Plot all admissions from the outpatient queue.
        axes_1 = utilities.generic_plot(
                                            self.hospital_admissions,
                                            axes_1,
                                            "Total from outpatient queue",
                                            self.confidence_interval,
                                            "Daily hospital admissions from outpatient queue",
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot proporition of all admissions from the outpatient queue.
        axes_2 = utilities.generic_plot(
                                            self.hospital_admissions,
                                            axes_2,
                                            "Proportion from outpatient queue",
                                            self.confidence_interval,
                                            "Proportion of daily hospital admissions from outpatient queue",
                                            "Time (days)",
                                            "Proportion"
                                        )
        # Plot average patient attribute.
        axes_3 = utilities.generic_plot(
                                            self.hospital_admissions,
                                            axes_3,
                                            "Average " + patient_attribute + " from outpatient queue",
                                            self.confidence_interval,
                                            "Average " + patient_attribute + " of patient admitted to hospital from outpatient queue",
                                            "Time (days)",
                                            "Patient " + patient_attribute
                                        )

        # Plot intervention rules.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)
        axes_3, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)

        # Set figure name.
        figure_name = self.plotting_path + "hospital_admissions_" + patient_attribute + "_from_outpatient_queue"
 
        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)


    def plot_deaths_separate(
            self,
            death_location: str,
            patient_attribute: str,
            save_flag: bool,
            show_flag: bool
        ) -> None:
        """ Plot patient deaths at specific location in the system. 

        This method plots patient deaths in either the community, the outpatient or inpatient queue.
        It also plots the average age or average NT pro BNP of patients in the queue as 
        requested.

        Parameters
        ----------
            death_location : str
                Indicates whether to plot coommunity, outpatient or inpatient data.
            patient_attribute : str
                Indicates whether to plot average age or average NT pro BNP.
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2) = plt.subplots(2, 1, sharex=True)

        # Plot number of patients dying in given location.
        axes_1 = utilities.generic_plot(
                                            self.death_statistics[death_location],
                                            axes_1,
                                            "Total",
                                            self.confidence_interval,
                                            "Daily deaths in the " + death_location.lower(),
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot the average patient attribute.
        axes_2 = utilities.generic_plot(
                                            self.death_statistics[death_location],
                                            axes_2,
                                            "Average " + patient_attribute,
                                            self.confidence_interval,
                                            "Average " + patient_attribute + " of patients dying in the " + death_location.lower(),
                                            "Time (days)",
                                            "Patient " + patient_attribute
                                        )

        # Plot intervention lines.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)

        # Set figure name.
        figure_name = self.plotting_path + death_location + "_deaths_" + patient_attribute

        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)


    def plot_deaths_combined(self, save_flag: bool, show_flag: bool):
        """ Plot all patient deaths. 

        This method plots all patient deaths.

        Parameters
        ----------
            save_flag : bool
                Flag indicating whether or not the plot should be saved.
            show_flag : bool
                Flag indicating whether or not the plot should be shown to the user.
    
        See also
        ----------
            utilities.generic_plot
            utilities.add_intervention_lines
            utilities.save_figure
        """
        # Initialise plot.
        fig, (axes_1, axes_2, axes_3) = plt.subplots(3, 1, sharex=True)
        
        # Plot deaths in the community.
        axes_1 = utilities.generic_plot(
                                            self.death_statistics["Community"],
                                            axes_1,
                                            "Total",
                                            self.confidence_interval,
                                            "Daily deaths in the community",
                                            "Time (days)",
                                            "Patients"
                                        )
        
        # Plot deaths in the outpatient queue.
        axes_2 = utilities.generic_plot(
                                            self.death_statistics["Outpatient queue"],
                                            axes_2,
                                            "Total",
                                            self.confidence_interval,
                                            "Daily deaths in the outpatient queue",
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot deaths in the inpatient queue.
        axes_3 = utilities.generic_plot(
                                            self.death_statistics["Inpatient queue"],
                                            axes_3,
                                            "Total",
                                            self.confidence_interval,
                                            "Daily deaths in the inpatient queue",
                                            "Time (days)",
                                            "Patients"
                                        )

        # Plot intervention lines.
        axes_1, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_1)
        axes_2, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_2)
        axes_3, lines, labels = utilities.add_intervention_lines(self.intervention_rule_sets, self.intervention_colour_dictionary, axes_3)

        # Set figure name.
        figure_name = self.plotting_path +"combined_deaths"

        # Format figure and save / show as requested.
        utilities.format_figure(fig, lines, labels, figure_name, save_flag, show_flag)



