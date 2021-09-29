"""

This module contains the echo system class. This class manages the 
outpatient and inpatient echocardiogram resources. It is responsible 
for:

    - Updating the daily echocardiogram supply limits. 

    - Refreshing the daily echocardiogram supply at the end of each day. 
    
    - Updating the global lockdown status. 
    
"""

## Imports 
import simpy
import datetime
import utilities

## Classes
class EchoSystem(object):

    def __init__(
            self, 
            env: simpy.Environment,
            simulation_parameters: dict
        ) -> None:
        """ Class to represent the echocardiogram resource system. 

        This class includes methods for managing the echocardiogram 
        resources. 

        Parameters
        ----------
            env : simpy.Environment
                Simpy environment in which the simulation occurs. 
            simulation_parameters : dict
                Dictionary containing the parameters needed to run the simulation.

        See also
        ----------
            utilities.determine_current_rule_set

        """
        # Set the simulation environment.
        self.env = env
        
        # Store start date. 
        self.start_date = simulation_parameters["start_date"] 

        # Store end date.
        self.end_date = simulation_parameters["end_date"] 

        # Set current date.
        self.current_datetime = self.start_date + datetime.timedelta(days=self.env.now)  
        
        # Store the intervention rule sets.
        self.intervention_rule_sets = simulation_parameters["intervention_rule_sets"]
        
        # Store the default rule set.
        self.default_rule_set = simulation_parameters["default_rule_set"]

        # Set current rule set.
        self.current_rule_set = utilities.determine_current_rule_set(self.current_datetime,
                                                                     self.intervention_rule_sets,
                                                                     self.default_rule_set)

        # Set lockdown status.
        self.lockdown_status = self.current_rule_set["lockdown"]
        
        # Initialise outpatient echo test capacity and supply. 
        self.outpatient_test = simpy.PriorityResource(env, capacity=simulation_parameters["outpatient_test_capacity"])
        self.outpatient_daily_supply = simpy.Container(env, init=self.current_rule_set["outpatient_weekday_supply_limit"])

        # Initialise inpatient echo test capacity and supply. 
        self.inpatient_test = simpy.PriorityResource(env, capacity=simulation_parameters["inpatient_test_capacity"])
        self.inpatient_daily_supply = simpy.Container(env, init=self.current_rule_set["inpatient_weekday_supply_limit"])

        # Initialise variables to store queue statistics. 
        self.outpatient_queue_length = None
        self.outpatient_queue_avg_pro_BNP = None
        self.outpatient_queue_avg_age = None
        self.avg_time_spent_in_outpatient_queue = None

        self.inpatient_queue_length = None
        self.inpatient_queue_avg_pro_BNP = None
        self.inpatient_queue_avg_age = None
        self.avg_time_spent_in_inpatient_queue = None 

        self.outpatient_supply_limit_dictionary = None
        self.inpatient_supply_limit_dictionary = None

        # Initialise dictionaries to store statistics.
        self.initialise_statistics_dictionaries()

        # Initialise echo supply limit dictionaries. 
        self.update_echo_supply_limit_dictionaries()

        # Initialise echo daily supply limits according to current rule set. 
        self.update_echo_daily_supply_limits()
        
    def initialise_statistics_dictionaries(self) -> None:
        """ Method to initialise dictionaries for storing statistics.

        This method initialises dictionaries for storing outpatient 
        and inpatient queue statistics. Dictionaries are used to store 
        statistics during the simulation as it they take less time to update
        then pandas dataframes. Once the simulation has finished, the dictionaries 
        are converted into a dataframe to make the analysis easier.
        """
        # Create the keys using the datetimes. 
        keys = [(self.start_date + datetime.timedelta(days=x)).date() for x in range((self.end_date - self.start_date).days)]

        # Create dictionaries with the keys.
        self.outpatient_queue_length = dict.fromkeys(keys, 0)
        self.outpatient_queue_avg_pro_BNP = dict.fromkeys(keys, 0)
        self.outpatient_queue_avg_age = dict.fromkeys(keys, 0)
        self.avg_time_spent_in_outpatient_queue = dict.fromkeys(keys, 0)

        self.inpatient_queue_length = dict.fromkeys(keys, 0)
        self.inpatient_queue_avg_pro_BNP = dict.fromkeys(keys, 0)
        self.inpatient_queue_avg_age = dict.fromkeys(keys, 0)
        self.avg_time_spent_in_inpatient_queue = dict.fromkeys(keys, 0)

    def update_echo_supply_limit_dictionaries(self) -> None:
        """ Method to update echo supply limit dictionaries.

        This method updates the outpatient and inpatient daily supply
        limits based on the current rule set. This enables the user to specify
        how many outpatient and inpatient echocardiograms can be performed 
        each day fo the week. This may change during lockdown or during user
        specified interventions. 
        """
        # Update the outpatient echo supply limit dictionary.
        self.outpatient_supply_limit_dictionary = {
                                                    0: self.current_rule_set["outpatient_weekday_supply_limit"],
                                                    1: self.current_rule_set["outpatient_weekday_supply_limit"],
                                                    2: self.current_rule_set["outpatient_weekday_supply_limit"],
                                                    3: self.current_rule_set["outpatient_weekday_supply_limit"],
                                                    4: self.current_rule_set["outpatient_weekday_supply_limit"],
                                                    5: self.current_rule_set["outpatient_saturday_supply_limit"],
                                                    6: self.current_rule_set["outpatient_sunday_supply_limit"]
                                                }
        
        # Update the inpatient echo supply limit dictionary.
        self.inpatient_supply_limit_dictionary = {
                                                    0: self.current_rule_set["inpatient_weekday_supply_limit"],
                                                    1: self.current_rule_set["inpatient_weekday_supply_limit"],
                                                    2: self.current_rule_set["inpatient_weekday_supply_limit"],
                                                    3: self.current_rule_set["inpatient_weekday_supply_limit"],
                                                    4: self.current_rule_set["inpatient_weekday_supply_limit"],
                                                    5: self.current_rule_set["inpatient_saturday_supply_limit"],
                                                    6: self.current_rule_set["inpatient_sunday_supply_limit"]
                                                }

    def update_lockdown_status(self) -> None:
        """ Method to update the lockdown status.

        This method updates the lockdown status depending on the current rule set.
        """
        self.lockdown_status = self.current_rule_set["lockdown"]

    def new_day(self) -> None:
        """  Method to implement the process of starting a new day.

        This method starts a new day in the simulation. This involves
        checking updating the current rule set, checking if a lockdown is 
        currently in effect and updating the echo supply resources.

        See also
        ----------
            utilities.determine_current_rule_set
        """

        # This continously runs throughout the simulation. 
        while True:
            
            # Wait for a day to pass. 
            yield self.env.timeout(1)
            
            # Get the current datetime
            self.current_datetime = self.start_date + datetime.timedelta(days=self.env.now)

            #  Get the current rule set. 
            self.current_rule_set = utilities.determine_current_rule_set(self.current_datetime,
                                                                         self.intervention_rule_sets,
                                                                         self.default_rule_set)

            # Update lockdown status.
            self.update_lockdown_status()

            # Update daily inpatient and outpatient echo supply limits.
            self.update_echo_daily_supply_limits()
            
            # Update outpatient queue data.
            self.update_outpatient_queue_stats()

            # Update inpatient queue data.
            self.update_inpatient_queue_stats()

            # Refresh outpatient echo daily supply to current daily supply limit.
            self.env.process(self.refresh_outpatient_supply())

            # Refresh inpatient echo daily supply to current daily supply limit.
            self.env.process(self.refresh_inpatient_supply())


    def update_echo_daily_supply_limits(self) -> None:
        """ Method to update the echo daily supply limits. 

        This method updates the echo daily supply limits depending
        on the daily supply rule set. 
        """
        # Update echo daily supply limit dictionaries.
        self.update_echo_supply_limit_dictionaries()

        # Set outpatient echo daily supply. 
        self.outpatient_daily_supply_limit = self.outpatient_supply_limit_dictionary[self.current_datetime.weekday()]
      
        # Set inpatient echo daily supply. 
        self.inpatient_daily_supply_limit = self.inpatient_supply_limit_dictionary[self.current_datetime.weekday()]

    def refresh_outpatient_supply(self) -> None:
        """ Method to refresh the outpatient echo supply.

        This method refreshes the outpatient supply at the beginning of every day. 

        Notes
        ----------
            Simpy has an annoying feature that you can't request 0 resource units from a container.
            This is why we have to empty the supply only if it is greater than 0.
        """
        # Empty the outpatient daily supply container. 
        if self.outpatient_daily_supply.level > 0:
            yield self.outpatient_daily_supply.get(self.outpatient_daily_supply.level)
        
        # Refresh the outpatient daily supply container with the current daily supply limit.
        if self.outpatient_daily_supply_limit > 0:
            yield self.outpatient_daily_supply.put(self.outpatient_daily_supply_limit)

    def refresh_inpatient_supply(self) -> None:
        """ Method to refresh the inpatient echo supply.

        This method refreshes the inpatient supply at the beginning of every day. 

        Notes
        ----------
            Simpy has an annoying feature that you can't request 0 resource units from a container.
            This is why we have to empty the supply only if it is greater than 0.
        """
        # Empty the inpatient daily supply container. 
        if self.inpatient_daily_supply.level > 0:
            yield self.inpatient_daily_supply.get(self.inpatient_daily_supply.level)
    
        # Refresh the inpatient daily supply container with the current daily supply limit.
        if self.inpatient_daily_supply_limit > 0:
            yield self.inpatient_daily_supply.put(self.inpatient_daily_supply_limit)


    def update_outpatient_queue_stats(self) -> None:
        """ Method to udpate the outpatient queue statistics.

        This method computes the outpatient queue statistics at the end of each day. It
        then adds these statistics to the appropriate dictionaries. 
        
        See also
        ----------
            utilities.compute_queue_statistics
        """
        # Compute outpatient queue statsitics.
        queue_statistics = utilities.compute_queue_statistics(float(self.env.now),
                                                              self.outpatient_test.queue)

        # Convert current date to datetime
        current_date = self.current_datetime.date()

        # Record outpatient queue length.
        self.outpatient_queue_length[current_date] = queue_statistics["queue_length"]

        # Record average time sppent in patient queue.
        self.avg_time_spent_in_outpatient_queue[current_date] = queue_statistics["avg_time_in_queue"]
        
        # Record average age of patients in outpatient queue.
        self.outpatient_queue_avg_age[current_date] = queue_statistics["avg_age_in_queue"]
    
        # Record average pro BNP of patients in outpatient queue.
        self.outpatient_queue_avg_pro_BNP[current_date] = queue_statistics["avg_pro_BNP_in_queue"]

    def update_inpatient_queue_stats(self) -> None:
        """ Method to udpate the outpatient queue statistics.

        This method computes the outpatient queue statistics at the end of each day. It
        then adds these statistics to the appropriate dictionaries. 

        See also
        ----------
            utilities.compute_queue_statistics
        """
        # Compute inpatient queue statsitics.
        queue_statistics = utilities.compute_queue_statistics(float(self.env.now),
                                                              self.inpatient_test.queue)

        # Convert current date to datetime
        current_date = self.current_datetime.date()

        # Record inpatient queue length.
        self.inpatient_queue_length[current_date] = queue_statistics["queue_length"]

        # Record average time sppent in patient queue.
        self.avg_time_spent_in_inpatient_queue[current_date] = queue_statistics["avg_time_in_queue"]
        
        # Record average age of patients in inpatient queue.
        self.inpatient_queue_avg_age[current_date] = queue_statistics["avg_age_in_queue"]
    
        # Record average pro BNP of patients in inpatient queue.
        self.inpatient_queue_avg_pro_BNP[current_date] = queue_statistics["avg_pro_BNP_in_queue"]
