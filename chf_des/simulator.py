"""

This module runs the simulation. 

"""
## Imports 
import simpy 
import datetime
import utilities
import pandas as pd
from patient import Patient
from system import EchoSystem

## Class

class Simulator():
    def __init__(self, simulation_parameters: dict) -> None:
        """ Manages simulation.

        This class implements the discrete event simulation for the 
        echocardiogram aspect of the CHF system. 

        Parameters
        ----------
            simulation_parameters : dict
                Dictionary containing the parameters needed to run the simulation.#

        Notes
        ----------
            The statistics are stored in dictionaries during the simulation and then 
            loaded into pandas data frames after the simulation has finished. This is 
            because indexed access is faster in dictionaries than data frames. 
        """
        # Create the simpy environment.
        self.env = simpy.Environment()

        # Store simulation parameters.
        self.interarrival_time = simulation_parameters["interarrival_time"]
        self.simulation_time = simulation_parameters["simulation_time"]
        self.echo_test_time = simulation_parameters["echo_test_time"]
        self.start_date = simulation_parameters["start_date"]
        self.end_date = simulation_parameters["end_date"]

        # Initialise statistics storage variables. 
        self.hospital_admissions = None
        self.community_deaths = None
        self.outpatient_queue_deaths = None  
        self.inpatient_queue_deaths = None
        self.outpatient_wait_times = None
        self.inpatient_wait_times = None

        self.hospital_admissions_total_all = None
        self.hospital_admissions_total_from_outpatient_queue = None

        self.hospital_admissions_avg_pro_BNP_all = None
        self.hospital_admissions_avg_pro_BNP_from_outpatient_queue = None

        self.hospital_admissions_avg_age_all = None
        self.hospital_admissions_avg_age_from_outpatient_queue = None

        self.community_deaths_total = None
        self.community_deaths_avg_pro_BNP = None
        self.community_deaths_avg_age = None

        self.outpatient_queue_deaths_total = None
        self.outpatient_queue_deaths_avg_pro_BNP = None
        self.outpatient_queue_deaths_avg_age = None

        self.inpatient_queue_deaths_total = None
        self.inpatient_queue_deaths_avg_pro_BNP = None
        self.inpatient_queue_deaths_avg_age = None

        self.outpatient_wait_times_total = None
        self.outpatient_wait_times_avg_time = None
        
        self.inpatient_wait_times_total = None
        self.inpatient_wait_times_avg_time = None

        self.outpatient_queue_length = None
        self.outpatient_queue_avg_pro_BNP = None
        self.outpatient_queue_avg_age = None
        self.avg_time_spent_in_outpatient_queue = None
        self.inpatient_queue_length = None
        self.inpatient_queue_avg_pro_BNP = None
        self.inpatient_queue_avg_age = None
        self.avg_time_spent_in_inpatient_queue = None

        # Create the column names used to create the pandas data frames for storing
        # all the statistics. 
        self.base_columns = ["Total", "Average pro BNP", "Average age"]

        self.wait_time_columns = ["Total", "Average time"]

        self.hospital_admission_columns = [
                                            "Total all",
                                            "Average pro BNP all",
                                            "Average age all",
                                            "Total from outpatient queue",
                                            "Average pro BNP from outpatient queue",
                                            "Average age from outpatient queue",
                                            "Proportion from outpatient queue"
                                        ]
        self.echo_queues_columns = [
                                        "Outpatient queue length",
                                        "Outpatient queue average age",
                                        "Outpatient queue average pro BNP",
                                        "Inpatient queue length",
                                        "Inpatient queue average age",
                                        "Inpatient queue average pro BNP",
                                        "Average time spent in outpatient queue",
                                        "Average time spent in inpatient queue"
                                ]

        # Initialise data frames.
        self.initialise_statistics_dataframes()

        # Initialise dictionaries. 
        self.initialise_statistics_dictionaries()

        # Initialise ech system object.
        self.echo_system = EchoSystem(self.env, simulation_parameters)
       
    def initialise_statistics_dictionaries(self) -> None:
        """ Initiailises dictionaries to store the statistics.

        This method initialises dictionaries for storing the statistics 
        of interest. These dictionaries are updated during the simulation. 
        """
        # Create the shared keys for indexing with current datetime..
        keys = [(self.start_date + datetime.timedelta(days=x)).date() for x in range((self.end_date - self.start_date).days)]

        # Create dictionaries using the shared keys.
        self.hospital_admissions_total_all = dict.fromkeys(keys, 0)
        self.hospital_admissions_total_from_outpatient_queue = dict.fromkeys(keys, 0)

        self.hospital_admissions_avg_pro_BNP_all = dict.fromkeys(keys, 0)
        self.hospital_admissions_avg_pro_BNP_from_outpatient_queue = dict.fromkeys(keys, 0)

        self.hospital_admissions_avg_age_all = dict.fromkeys(keys, 0)
        self.hospital_admissions_avg_age_from_outpatient_queue = dict.fromkeys(keys, 0)

        self.community_deaths_total = dict.fromkeys(keys, 0)
        self.community_deaths_avg_pro_BNP = dict.fromkeys(keys, 0)
        self.community_deaths_avg_age = dict.fromkeys(keys, 0)

        self.outpatient_queue_deaths_total = dict.fromkeys(keys, 0)
        self.outpatient_queue_deaths_avg_pro_BNP = dict.fromkeys(keys, 0)
        self.outpatient_queue_deaths_avg_age = dict.fromkeys(keys, 0)

        self.inpatient_queue_deaths_total = dict.fromkeys(keys, 0)
        self.inpatient_queue_deaths_avg_pro_BNP = dict.fromkeys(keys, 0)
        self.inpatient_queue_deaths_avg_age = dict.fromkeys(keys, 0)

        self.outpatient_wait_times_total = dict.fromkeys(keys, 0)
        self.outpatient_wait_times_avg_time = dict.fromkeys(keys, 0)
        
        self.inpatient_wait_times_total = dict.fromkeys(keys, 0)
        self.inpatient_wait_times_avg_time = dict.fromkeys(keys, 0)

    def initialise_statistics_dataframes(self) -> None:
        """ Initiailises data frames to store the statistics.

        This method initialises data frames for storing the statistics 
        of interest. These data frames are updated at the end of the simulation 
        using the data stored in the dictionaries during the simulation. 
        """
        # Create the shared index for indexing with current datetime..
        index = [(self.start_date + datetime.timedelta(days=x)).date() for x in range((self.end_date - self.start_date).days)]
      
        # Initialise the data frames with the shared index.
        self.hospital_admissions = pd.DataFrame(index=index, columns=self.hospital_admission_columns)
        self.community_deaths = pd.DataFrame(index=index, columns=self.base_columns)
        self.outpatient_queue_deaths = pd.DataFrame(index=index, columns=self.base_columns)
        self.inpatient_queue_deaths = pd.DataFrame(index=index, columns=self.base_columns)
        self.outpatient_wait_times = pd.DataFrame(index=index, columns=self.wait_time_columns)
        self.inpatient_wait_times = pd.DataFrame(index=index, columns=self.wait_time_columns)
        self.echo_queues = pd.DataFrame(index=index, columns=self.echo_queues_columns)

    def run_simulation(self) -> list:
        """ Run the simulation. 

        This method runs the simulation until the given simulation time. It 
        returns a list of dataframes with the stored statistics of interest.
         in the form of two data frames. 

        Returns
        ----------
            simulation_result : list
                List containing the data frames that contain the statistics recorded
                during the simulation.
        """
        # Initiate the new day process.
        self.env.process(self.echo_system.new_day())
        
        # Initiate teh run echo system process. 
        self.env.process(self.run_system())

        # Run simulation for allotted time. 
        self.env.run(until=self.simulation_time)

        # Load the data frames with the data stored in the dictionaries. 
        self.load_dataframes_with_dictionary_data()

        # Compute averages.
        self.compute_average_statistics()

        # Store data frames in a list for returning. 
        simulation_result = {
                                "Hospital admissions": self.hospital_admissions,
                                "Community deaths": self.community_deaths,
                                "Outpatient queue deaths": self.outpatient_queue_deaths,
                                "Inpatient queue deaths": self.inpatient_queue_deaths,
                                "Outpatient wait times": self.outpatient_wait_times,
                                "Inpatient wait times": self.inpatient_wait_times,
                                "Echo queues": self.echo_queues
                            }

        return simulation_result

    def run_system(self) -> None:
        """ Run system.

        This method generates new patients and moves them through the system.
        """

        # Initialise patient count. 
        patient_count = 0
        
        # Start running the system.
        while True:

            # Generate time that patient arrives. 
            arrival_time = utilities.generate_arrival_time(self.interarrival_time)

            # Wait for patient to arrive.
            yield self.env.timeout(arrival_time)
           
            # Increment patient count and use it for patient id. 
            patient_count += 1

            # Initialise patient.
            patient = Patient(patient_id=patient_count)

            # Store patients simulation time of arrival in the system.
            patient.arrival_time = self.env.now

            # Store patients calendar datetime of arrival in the system.
            patient.arrival_datetime = self.start_date + datetime.timedelta(days=self.env.now) 

            # Generate initial patient attributes.
            patient.generate_attributes(self.echo_system.lockdown_status)
            
            # Move patient through the system. 
            self.env.process(self.move_through_patient_pathway(patient))

    def move_through_patient_pathway(self, patient: object) -> None:
        """ Simulates the patient pathway through the system.

        This method simulates the process of a patient presenting to the
        system and proceeding through the various stages. 

        Parameters
        ----------
            patient : object
                Patient object. 
        """ 
        # Determine initial event that occurs for the patient.
        patient = yield self.env.process(self.determine_initial_event(patient))
     
        # If the initial event is a gp appointment then move the patient through
        # the gp appointment process. 
        if patient.result.events[0].value=="gp_appointment":

            self.move_patient_through_gp_appointment_process(patient)
       
        # If the initial event is a hospital admission then move the patient through
        # the hospital admissino process.
        elif patient.result.events[0].value=="hospital_admission":

            self.move_patient_through_hospital_admission_process(patient)

        # If the initial event is a death then move the patient through
        # the death process
        elif patient.result.events[0].value=="death":
            
            self.move_patient_through_death_process(patient)
        
    def determine_initial_event(self, patient: object) -> None:
        """ Determines initial event.

        This method determines the initial event that occurs for a given patient. 
        The expected time until each event occurs are recorded in the patient object.
        These expected times are used to determine which event occurs first. 

        Parameters
        ----------
            patient : object
                Patient object. 
        
        See also
        ----------
            utilities.schedule_gp_appointment
            utilities.schedule_hospital_admission
            utilities.schedule_death
        """ 
        # Schedule events.
        gp_app_event = utilities.schedule_gp_appointment(self.env, patient)
        hospital_admission_event = utilities.schedule_hospital_admission(self.env, patient)
        death_event = utilities.schedule_death(self.env, patient)
        
        # Determine which event occurs first. 
        patient.result = yield gp_app_event | hospital_admission_event | death_event

        return patient

    def move_patient_through_gp_appointment_process(self, patient: object) -> None:
        """ Move patient through GP appointment process.

        This method moves the patient through the GP appointment process. The patients 
        NT pro BNP result is stored in the patient object. If the result is less than 
        400, the patient exits the system. If the result is greater than 400, the patient
        is moved through the outpatient queue. 

        Parameters
        ----------
            patient : object
                Patient object. 
        """ 
        # Record the datetime of the gp appointment. 
        patient.gp_appointment_datetime = self.start_date + datetime.timedelta(days=self.env.now)

        # Check the patients pro BNP result. 
        if patient.pro_BNP < 400:
            
            # Patient does not have CHF and exits the system.
            pass

        else: 

            # The patient is moved through the outpatient queue process.
            outpatient_process = self.env.process(self.move_patient_through_outpatient_queue_process(patient))

            # Start the outpatient queue interruption process. 
            outpatient_interruption_process = self.env.process(self.interrupt_outpatient_process(patient,
                                                                                                 outpatient_process))


    def move_patient_through_outpatient_queue_process(self, patient: object) -> None:
        """ Move patient through outpatient queue process.

        This method moves the patient through the outpatient queue process. The patient
        is sent for an outpatient echocardiogram. The priority of the patient is determined 
        by their NT pro BNP result. 

        Parameters
        ----------
            patient : object
                Patient object. 

        See also
        ----------
            utilities.update_outpatient_wait_times_stats
        
        Notes
        ----------
            The patient can be admitted to hospital or die at any time during the outpatient queue
            process. This is modelled using the SimPy Interrupt class. An interrupt exception
            is generated if hospital admission or death occurs before the outpatient queue process
            has finished.
        """ 
        # We use a try except construction to catch the interruption if it occurs before 
        # the process has finished.  
        try: 
            # Generate priority score.
            patient_priority = patient.echo_time_limit + self.env.now 

            # Generate a request to access the outpatient echocardiogram resource wiht 
            # the given priority.     
            capacity_request = self.echo_system.outpatient_test.request(patient_priority)

            # Record the datetime that the patient joined the outpatient echo queue.
            patient.outpatient_queue_arrival_datetime =  self.start_date + datetime.timedelta(days=self.env.now)

            # Store patient attributes in the preempt attribute of the capacity request.
            # The preempt attribute is not needed for anything else and the patients
            # attributes are needed to compute queue statistics.
            capacity_request.preempt = {"age": patient.age, "pro BNP": patient.pro_BNP}

            with capacity_request:
                
                # When the outpatient echo resource becomes available the patient is at
                # the front of the queue. 
                yield capacity_request

                # Request 1 unit of the outpatient daily supply. 
                yield self.echo_system.outpatient_daily_supply.get(1)

                # Once supply is available, perform the outpatient echo. 
                # Patient receives outpatient echo. Update patient attributes.
                patient = self.update_patient_data_after_outpatient_echo(patient)

                # Update outpatient wait times statistics.
                utilities.update_outpatient_wait_times_stats(
                                                                self.outpatient_wait_times_total,
                                                                self.outpatient_wait_times_avg_time,
                                                                patient
                                                            )         
                
                # Advance simulation by outpatient echo time. 
                yield self.env.timeout(self.echo_test_time)

        except simpy.Interrupt as interrupt:

            # If the patient is admitted to hospital or dies before the patient 
            # receives an outpatient echo then we must interrupt the process. 
            if interrupt.cause=="hospital_admission":

                # If the outpatient is admitted to hospital move patient through hospital admission process.
                self.move_patient_through_hospital_admission_process(patient)
            
            else:
                
                # If the outpatient dies then move patient through the death process. 
                self.move_patient_through_death_process(patient)
    
    def interrupt_outpatient_process(
            self,
            patient: object, 
            outpatient_process: simpy.Process
        ) -> None:
        """ Interrupt outpatient queue process.

        This method generates an interrupt exception if the patient is admitted to hospital or dies
        any time in between joining the outpatient queue and receiving their echocardiogram.

        Parameters
        ----------
            patient : object
                Patient object. 
            outpatient_process : simpy.Process
                Outpatient echocardiogram process.

        See also
        ----------
            utilities.schedule_hospital_admission
            utilities.schedule_death
        """ 
        # Schedule hospital admission event.
        hospital_admission_event = utilities.schedule_hospital_admission(self.env, patient)

        # Schedule death event. 
        death_event = utilities.schedule_death(self.env, patient)

        # Yield the event which occurs first. 
        patient.result = yield hospital_admission_event | death_event
     
        # If the outpatient process is still running then interrupt the process.
        if outpatient_process.is_alive:
            
            # Interrupt the outpatient process with this event.
            outpatient_process.interrupt(patient.result.events[0].value)

    def move_patient_through_hospital_admission_process(self, patient: object) -> None:
        """ Move patient through hospital admission process.

        This method moves the patient through the hospital admission process.

        Parameters
        ----------
            patient : object
                Patient object. 

        See also
        ----------
            utilities.update_hospital_admissions_stats

        """ 
        # Record the hospital admission datetime.
        patient.hospital_admission_datetime = self.start_date + datetime.timedelta(days=self.env.now)

        # Update hospital admission statistics. 
        utilities.update_hospital_admissions_stats(
                                                    self.hospital_admissions_total_all,
                                                    self.hospital_admissions_avg_pro_BNP_all,
                                                    self.hospital_admissions_avg_age_all,
                                                    self.hospital_admissions_total_from_outpatient_queue,
                                                    self.hospital_admissions_avg_pro_BNP_from_outpatient_queue,
                                                    self.hospital_admissions_avg_age_from_outpatient_queue,
                                                    patient
                                                )         

        # The patient is moved throught the inpatient queue process.
        inpatient_process = self.env.process(self.move_patient_through_inpatient_queue_process(patient))

        # Start the inpatient queue interruption process. 
        inpatient_interruption_process = self.env.process(self.interrupt_inpatient_process(patient,
                                                                                           inpatient_process))

    def move_patient_through_inpatient_queue_process(self, patient: object) -> None:
        """ Move patient through inpatient queue process.

        This method moves the patient through the inpatient queue process. The patient
        is sent for an inpatient echocardiogram. The priority of the patient is determined 
        by the time they arrive in the queue.

        Parameters
        ----------
            patient : object
                Patient object. 

        See also
        ----------
            utilities.update_inpatient_wait_times_stats

        Notes
        ----------
            The patient can die at any time during the inpatient queue process. This is 
            modelled using the SimPy Interrupt class. An interrupt exception is generated
            death occurs before the inpatient queue process has finished.
        """ 
        # We use a try except construction to catch the interruption if it occurs before 
        # the process has finished.  
        try:
            # Generate priority score as time the patient joins the inpatient queue.
            patient_priority = self.env.now 
            
            # Request inpatient echocardiogram resource with priority.  
            capacity_request = self.echo_system.inpatient_test.request(patient_priority)
    
            # Record the datetime that the patient joined the inpatient echo queue.
            patient.inpatient_queue_arrival_datetime =  self.start_date + datetime.timedelta(days=self.env.now)    

            # Sneak in patient info through redundant preempt attribute.
            capacity_request.preempt = {"age": patient.age, "pro BNP": patient.pro_BNP}

            # Request inpatient echo resource. 
            with capacity_request:
                
                # When the inpatient echo resource becomes available the patient is at
                # the front of the queue. 
                yield capacity_request

                # Request 1 unit of the inpatient daily supply. 
                yield self.echo_system.inpatient_daily_supply.get(1)

                # Once supply is available, perform the inpatient echo. 
                # Patient receives inpatient echo. Update patient attributes.
                patient = self.update_patient_data_after_inpatient_echo(patient)
                        
                utilities.update_inpatient_wait_times_stats(
                                                                self.inpatient_wait_times_total,
                                                                self.inpatient_wait_times_avg_time,
                                                                patient
                                                            ) 
                
                # Advance simulation by outpatient echo time. 
                yield self.env.timeout(self.echo_test_time)

        except simpy.Interrupt as interrupt:
            
            # If the patient dies before the patient receives an outpatient echo then 
            # the inpatient process is interrupted.  

            # Move patient through death process.  
            self.move_patient_through_death_process(patient)

    def interrupt_inpatient_process(
            self, 
            patient: object, 
            inpatient_process: simpy.Process
        ) -> None:
        """ Interrupt inpatient queue process.

        This method generates an interrupt exception if the patient dies
        any time in between joining the inpatient queue and receiving
        their echocardiogram.

        Parameters
        ----------
            patient : object
                Patient object. 
            inpatient_process : simpy.Process
                Inpatient echocardiogram process.

        See also
        ----------
            utilities.schedule_death
        """ 
        # Schedule death event.
        death_event = utilities.schedule_death(self.env, patient)
  
        # Yield the event which occurs first. 
        patient.result = yield death_event

        # Interrupt the inpatient process with this event if it is still
        # active.        
        if inpatient_process.is_alive:
            inpatient_process.interrupt(patient.result)        
        
    def move_patient_through_death_process(self, patient: object) -> None:
        """ Move patient through death process.

        This method moves the patient through the death process.

        Parameters
        ----------
            patient : object
                Patient object. 

        See also
        ----------
            utilities.update_community_deaths_stats
            utilities.update_outpatient_queue_deaths_stats
            utilities.update_inpatient_queue_deaths_stats
        """ 
        # Update patient attributes.
        patient.death_datetime = self.start_date + datetime.timedelta(days=self.env.now)     
        patient.died_in_community = (not bool(patient.gp_appointment_datetime)) and (not bool(patient.hospital_admission_datetime))
        patient.died_in_outpatient_queue = bool(patient.gp_appointment_datetime) and (not bool(patient.hospital_admission_datetime))
        patient.died_in_inpatient_queue = bool(patient.hospital_admission_datetime)

        # Update community death statistics.
        utilities.update_community_deaths_stats(
                                                    self.community_deaths_total,
                                                    self.community_deaths_avg_pro_BNP,
                                                    self.community_deaths_avg_age,
                                                    patient
                                                )
        
        # Update outpatient queue death statistics.
        utilities.update_outpatient_queue_deaths_stats(
                                                        self.outpatient_queue_deaths_total,
                                                        self.outpatient_queue_deaths_avg_pro_BNP,
                                                        self.outpatient_queue_deaths_avg_age,
                                                        patient
                                                    )

        # Update inpatient death statistics.
        utilities.update_inpatient_queue_deaths_stats(
                                                        self.inpatient_queue_deaths_total,
                                                        self.inpatient_queue_deaths_avg_pro_BNP,
                                                        self.inpatient_queue_deaths_avg_age,
                                                        patient
                                                    )                                   
        
    def update_patient_data_after_outpatient_echo(self, patient: object) -> None:
        """ Update patient data after an outpatient echocardiogram.

        This method updates the patient data after they have received
        and outpatient echocardiogram.

        Parameters
        ----------
            patient : object
                Patient object. 
        """ 
        # Update patient data.
        patient.outpatient_queue_exit_datetime =  self.start_date + datetime.timedelta(days=self.env.now)                        
        patient.outpatient_echo = True
        patient.echo_datetime = self.start_date + datetime.timedelta(days=self.env.now)
        patient.hospital_admission_datetime = None
        patient.death_datetime = None
        patient.echo_wait_time = patient.outpatient_queue_exit_datetime - patient.outpatient_queue_arrival_datetime
        patient.echo_wait_time = patient.echo_wait_time/pd.to_timedelta(1, unit='D')

        return patient

    def update_patient_data_after_inpatient_echo(self, patient: object) -> None:
        """ Update patient data after an inpatient echocardiogram.

        This method updates the patient data after they have received
        and outpatient echocardiogram.

        Parameters
        ----------
            patient : object
                Patient object. 
        """ 
        # Update patient data.
        patient.inpatient_echo = True
        patient.inpatient_queue_exit_datetime =  self.start_date + datetime.timedelta(days=self.env.now)            
        patient.echo_datetime = self.start_date + datetime.timedelta(days=self.env.now)  
        patient.death_datetime = None
        patient.echo_wait_time = patient.inpatient_queue_exit_datetime - patient.inpatient_queue_arrival_datetime
        patient.echo_wait_time = patient.echo_wait_time/pd.to_timedelta(1, unit='D')

        return patient

    def load_dataframes_with_dictionary_data(self) -> None:
        """ Load dataframes with dictionary data.

        This method moves the statistics stored in the dictionaries into the data frames for
        later processing and plotting.
        """ 
        # Compute the proportion of hospital admissions that came from the outpatient queue.
        hospital_admissions_proportion_from_outpatient_queue = utilities.compute_proportion(self.hospital_admissions_total_from_outpatient_queue.values(), 
                                                                                            self.hospital_admissions_total_all.values())
        
        # Load data frames with appropriate dictionary data.
        self.hospital_admissions[self.hospital_admission_columns] = [
                                                                        self.hospital_admissions_total_all.values(),
                                                                        self.hospital_admissions_avg_pro_BNP_all.values(),
                                                                        self.hospital_admissions_avg_age_all.values(),
                                                                        self.hospital_admissions_total_from_outpatient_queue.values(),
                                                                        self.hospital_admissions_avg_pro_BNP_from_outpatient_queue.values(),
                                                                        self.hospital_admissions_avg_age_from_outpatient_queue.values(),
                                                                        hospital_admissions_proportion_from_outpatient_queue
                                                                    ]

        self.community_deaths[self.base_columns] = [
                                                    self.community_deaths_total.values(),
                                                    self.community_deaths_avg_pro_BNP.values(),
                                                    self.community_deaths_avg_age.values()
                                                ]

        self.outpatient_queue_deaths[self.base_columns] = [
                                                            self.outpatient_queue_deaths_total.values(),
                                                            self.outpatient_queue_deaths_avg_pro_BNP.values(),
                                                            self.outpatient_queue_deaths_avg_age.values(),
                                                        ]

        self.inpatient_queue_deaths[self.base_columns] = [
                                                            self.inpatient_queue_deaths_total.values(),
                                                            self.inpatient_queue_deaths_avg_pro_BNP.values(),
                                                            self.inpatient_queue_deaths_avg_age.values()
                                                        ]

        self.outpatient_wait_times[self.wait_time_columns] = [
                                                                self.outpatient_wait_times_total.values(),
                                                                self.outpatient_wait_times_avg_time.values(),
                                                            ]

        self.inpatient_wait_times[self.wait_time_columns] = [
                                                                self.inpatient_wait_times_total.values(),
                                                                self.inpatient_wait_times_avg_time.values(),
                                                            ]

        self.echo_queues[self.echo_queues_columns] = [
                                                        self.echo_system.outpatient_queue_length.values(),
                                                        self.echo_system.outpatient_queue_avg_age.values(),
                                                        self.echo_system.outpatient_queue_avg_pro_BNP.values(),
                                                        self.echo_system.inpatient_queue_length.values(),
                                                        self.echo_system.inpatient_queue_avg_age.values(),
                                                        self.echo_system.inpatient_queue_avg_pro_BNP.values(),
                                                        self.echo_system.avg_time_spent_in_outpatient_queue.values(),
                                                        self.echo_system.avg_time_spent_in_inpatient_queue.values()
                                                    ]

    def compute_average_statistics(self) -> None:
        """ Compute averages.

        This method computes the average age and average NT pro BNP of patients that;

            -- Were admitted to hospital.
            -- Died in the community.
            -- Died in the outpatient queue.
            -- Died in the inpatient queue.
            -- Waited in the outpatient queue.
            -- Waited in the inpatient queue.

        Notes
        ----------
            These computations are done after the simulation has finished for efficiency reasons.
        """ 
        # Compute averages.
        self.hospital_admissions = utilities.compute_hospital_admissions_averages(self.hospital_admissions)
        self.community_deaths = utilities.compute_averages(self.community_deaths)
        self.outpatient_queue_deaths = utilities.compute_averages(self.outpatient_queue_deaths)
        self.inpatient_queue_deaths = utilities.compute_averages(self.inpatient_queue_deaths)
        self.outpatient_wait_times = utilities.compute_averages(self.outpatient_wait_times)
        self.inpatient_wait_times = utilities.compute_averages(self.inpatient_wait_times)

