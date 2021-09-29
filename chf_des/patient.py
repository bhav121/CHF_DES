"""

This module contains the patient data class. 

"""

## Imports
import utilities
from dataclasses import dataclass
from typing import Any

## Class
@dataclass
class Patient:
    """ Dataclass to store patient data.

    This class instantiates a patient in the CHF system 
    with a given set of attributes. The attributes are 
    determined using functions defined in the utilities 
    module. 

    Parameters
    ----------
        patient_id : int
            Patient identification. 
        arrival_time : float 
            Time that the patient arrived in the CHF system (measured in simulation time).
        arrival_datetime : float 
            Date and time that the patient arrived in the CHF system (measured in calendar time).
        age : int
            Patients' age.
        pro_BNP : float  
            Patients' NT pro BNP result.
        echo_time_limit : float 
            Time by which the patient should receive an echocardiogram.
        expected_gp_appointment_time : float
            Expected time until patient sees their GP.
        expected_hospital_admission_time : float
            Expected time until patient is admitted to hospital.
        expected_death_time : float 
            Expected time until patient dies.
        outpatient_echo : bool
            Boolean variable to identify if the patient has an outpatient echo.
        inpatient_echo : bool
            Boolean variable to identify if the patient has an inpatient echo. 
        outpatient_queue_arrival_datetime: Any
            Date and time that the patient joined the outpatient echo queue. Default to None.
        outpatient_queue_exit_datetime: Any
            Date and time that the patient left the outpatient echo queue.Default to None.
        inpatient_queue_arrival_datetime: Any
            Date and time that the patient joined the inpatient echo queue. Default to None.
        inpatient_queue_exit_datetime: Any
            Date and time that the patient left the inpatient echo queue. Default to None.
        gp_appointment_datetime : Any
            Date and time that the patient saw their GP. Default to None.
        hospital_admission_datetime : Any
            Date and time that the patient was admitted to hospital. Default to None.
        death_datetime : Any
            Date and time that the patient died. Default to None.
        echo_datetime : Any
            Date and time that the patient received an echocardiogram. Default to None.
        echo_wait_time: Any
            Time that the patient spent waiting for their echocardiogram. Default to None.
        died_in_community: bool
            Time and date that the patient died in the community. Default to False.
        died_in_outpatient_queue: bool
            Time and date that the patient died in the outpatient queue. Default to False.
        died_in_inpatient_queue: bool
            Time and date that the patient died in the inpatient queue. Default to False.
        """
    patient_id: int=0
    arrival_time: float=0
    arrival_datetime: Any=None
    age: int=0
    pro_BNP: float=0 
    echo_time_limit: float=0
    expected_gp_appointment_time: float=0
    expected_death_time: float=0
    expected_hospital_admission_time: float=0
    outpatient_queue_arrival_datetime: Any=None
    outpatient_queue_exit_datetime: Any=None
    inpatient_queue_arrival_datetime: Any=None
    inpatient_queue_exit_datetime: Any=None
    outpatient_echo: bool=False
    inpatient_echo: bool=False
    hospital_admission_datetime: Any=None
    death_datetime: Any=None
    gp_appointment_datetime: Any=None
    echo_datetime: Any=None
    died_in_community: bool=False
    died_in_outpatient_queue: bool=False
    died_in_inpatient_queue: bool=False
    echo_wait_time: Any=None

    
    def generate_attributes(self, lockdown_status: bool) -> None:
        """ Generates patient attributes. 

        This method generates the following patient attributes:

        -- Age
        -- NT pro BNP result
        -- Expected time until the patient sees their GP.
        -- Expected time until the patient is admitted to hospital.
        -- Expected time until the patient dies. 

        Parameters
        ----------
            lockdown_status : bool
                Boolean variable indicating if it is currently a lockdown or not. 
        """  
        # Generate patient age. 
        self.generate_age()

        # Generate pro_BNP.
        self.generate_pro_BNP()

        # Generate expected event times.
        self.generate_expected_gp_appointment_time(lockdown_status)
        self.generate_expected_hospital_admission_time(lockdown_status)
        self.generate_expected_death_time()
    
    def generate_age(self) -> None:
        """ Generates patient age.

        This method generates the age of the patient. 

        See Also
        --------
            utilities.generate_age
        """
        self.age = utilities.generate_age()
    
    def generate_pro_BNP(self) -> None:
        """ Generates patient NT pro BNP result.

        This method generates the NT pro BNP result.

        See Also
        --------
            utilities.generate_pro_BNP
        """
        self.pro_BNP = utilities.generate_pro_BNP(self.age)

    def generate_expected_gp_appointment_time(self, lockdown_status: bool) -> None:
        """ Generates patient expected GP appointment time.

        This method generates the expected time until a patient sees their GP.

        See Also
        --------
            utilities.generate_expected_gp_appointment_time
        """
        self.expected_gp_appointment_time = utilities.generate_expected_gp_appointment_time(
                                                                                                self.arrival_time,
                                                                                                self.age,
                                                                                                self.pro_BNP,
                                                                                                lockdown_status
                                                                                            )


    def generate_expected_hospital_admission_time(self, lockdown_status: bool) -> None:
        """ Generates patient expected hospital admission time.

        This method generates the expected time until a patient is admitted to 
        hospital.

        See Also
        --------
            utilities.generate_expected_hospital_admission_time
        """
        self.expected_hospital_admission_time = utilities.generate_expected_hospital_admission_time(
                                                                                                    self.arrival_time,
                                                                                                    self.age,
                                                                                                    self.pro_BNP,
                                                                                                    lockdown_status
                                                                                                )

    def generate_expected_death_time(self) -> None:
        """ Generates patient expected death time.

        This method generates the expected time until a patient dies.

        See Also
        --------
            utilities.generate_expected_death_time
        """
        self.expected_death_time = utilities.generate_expected_death_time(
                                                                            self.arrival_time,
                                                                            self.age,
                                                                            self.pro_BNP
                                                                        )

    def generate_echo_time_limit(self) -> None:
        """ Generates echo time limit for patient.

        This method generates the time by which a patient should receive their echocardiogram.

        See Also
        --------
            utilities.generate_echo_time_limit
        """
        self.echo_time_limit = utilities.generate_echo_time_limit(self.pro_BNP)