import numpy as np
import openmdao.api as om
import openconcept.api as oc

from openmdao.api import BalanceComp
from openconcept.analysis.trajectories import TrajectoryGroup, PhaseGroup
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.analysis.performance.solver_phases_eVTOL import SteadyVerticalFlightPhase, MomentumTheoryVerticalFlightPhase, SteadyFlightPhaseForVTOLCruise, UnsteadyFlightPhaseForTiltrotorTransition, MomentumTheoryMultiRotorCruisePhase

#from util import ComputeSinCos, NetWeight

class SimpleVTOLMission(TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, and vertical descent.
    The user needs to set the duration and vertical speed (fltcond|vs) of each phase in the runscript.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
        hover = self.add_subsystem('hover', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='hover') , promotes_inputs=['ac|*'])
        descent1 = self.add_subsystem('descent1', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])
        descent2 = self.add_subsystem('descent2', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])

        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(climb, hover)
        self.link_phases(hover, descent1)
        self.link_phases(descent1, descent2)

class SimpleVTOLMissionWithCruise(TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, cruise, and vertical descent.
    The user needs to set the [duration, fltcond|vs (vertical speed)] for climb/hover/descent, and [duration, fltcond|vs, fltcond|Ueas (airspeed), fltcond|Tangle (thrust tilt angle)]
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
        cruise1 = self.add_subsystem('cruise1', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        hover = self.add_subsystem('hover', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='hover') , promotes_inputs=['ac|*'])
        cruise2 = self.add_subsystem('cruise2', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        descent1 = self.add_subsystem('descent1', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])
        descent2 = self.add_subsystem('descent2', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])

        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(climb, cruise1)
        self.link_phases(cruise1, hover)
        self.link_phases(hover, cruise2)
        self.link_phases(cruise2, descent1)
        self.link_phases(descent1, descent2)


class BasicSimpleVTOLMission(TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, cruise, and vertical descent.
    The user needs to set the [duration, fltcond|vs (vertical speed)] for climb/hover/descent, and [duration, fltcond|vs, fltcond|Ueas (airspeed), fltcond|Tangle (thrust tilt angle)]
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        mp = self.add_subsystem('missionparams', om.IndepVarComp(),promotes_outputs=['*'])
        mp.add_output('takeoff|h',val=0.,units='ft')
        mp.add_output('cruise|h0',val=1500.,units='ft')
        mp.add_output('mission_range',val=30.,units='mi')
        mp.add_output('payload',val=1000.,units='lbm') # ~ 45kg

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        phase1 = self.add_subsystem('takeoff', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='takeoff'), promotes_inputs=['ac|*'])
        phase2 = self.add_subsystem('climb', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
        phase3 = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        phase4 = self.add_subsystem('descent', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])
        phase5 = self.add_subsystem('landing', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='landing'), promotes_inputs=['ac|*'])
        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(phase1, phase2)
        self.link_phases(phase2, phase3)
        self.link_phases(phase3, phase4)
        self.link_phases(phase4, phase5)

class eVTOLMission_validation1_Hansman(TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, cruise, and vertical descent.
    The user needs to set the [duration, fltcond|vs (vertical speed)] for climb/hover/descent, and [duration, fltcond|vs, fltcond|Ueas (airspeed), fltcond|Tangle (thrust tilt angle)]
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        mp = self.add_subsystem('missionparams', om.IndepVarComp(),promotes_outputs=['*'])
        mp.add_output('takeoff|h',val=0.,units='ft')
        mp.add_output('cruise|h0',val=5000.,units='ft')
        mp.add_output('mission_range',val=30.,units='mi')
        mp.add_output('payload',val=1000.,units='lbm') 

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        phase1 = self.add_subsystem('takeoff', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='takeoff'), promotes_inputs=['ac|*'])
        phase2 = self.add_subsystem('climb', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
        phase3 = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        phase4 = self.add_subsystem('landing', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='landing'), promotes_inputs=['ac|*'])
        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(phase1, phase2)
        self.link_phases(phase2, phase3)
        self.link_phases(phase3, phase4)


class BasicSimpleVTOLMIssionTakeoffAndCruiseOnly(TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, cruise, and vertical descent.
    The user needs to set the [duration, fltcond|vs (vertical speed)] for climb/hover/descent, and [duration, fltcond|vs, fltcond|Ueas (airspeed), fltcond|Tangle (thrust tilt angle)]
    """
    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        mp = self.add_subsystem('missionparams', om.IndepVarComp(),promotes_outputs=['*'])
        mp.add_output('takeoff|h',val=0.,units='ft')
        mp.add_output('cruise|h0',val=1500.,units='ft')
        mp.add_output('mission_range',val=30.,units='mi')
        mp.add_output('payload',val=1000.,units='lbm')

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        phase1 = self.add_subsystem('takeoff', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='takeoff'), promotes_inputs=['ac|*'])
        phase2 = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        phase3 = self.add_subsystem('landing', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='landing'), promotes_inputs=['ac|*'])
        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(phase1, phase2)
        self.link_phases(phase2, phase3)

class BasicSimpleVTOLMissionMomentumTakeoffAndCruiseOnly(TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, cruise, and vertical descent.
    The user needs to set the [duration, fltcond|vs (vertical speed)] for climb/hover/descent, and [duration, fltcond|vs, fltcond|Ueas (airspeed), fltcond|Tangle (thrust tilt angle)]
    """
    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        mp = self.add_subsystem('missionparams', om.IndepVarComp(),promotes_outputs=['*'])
        mp.add_output('takeoff|h',val=0.,units='ft')
        mp.add_output('cruise|h0',val=1500.,units='ft')
        mp.add_output('mission_range',val=30.,units='mi')
        mp.add_output('payload',val=1000.,units='lbm')

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        phase1 = self.add_subsystem('takeoff', MomentumTheoryVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='takeoff'), promotes_inputs=['ac|*'])
        phase2 = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        phase3 = self.add_subsystem('landing', MomentumTheoryVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='landing'), promotes_inputs=['ac|*'])
        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(phase1, phase2)
        self.link_phases(phase2, phase3)

class BasicSimpleVTOLMultirotorMissionMomentumTakeoffAndCruiseOnly(TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, cruise, and vertical descent.
    The user needs to set the [duration, fltcond|vs (vertical speed)] for climb/hover/descent, and [duration, fltcond|vs, fltcond|Ueas (airspeed), fltcond|Tangle (thrust tilt angle)]
    """
    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        mp = self.add_subsystem('missionparams', om.IndepVarComp(),promotes_outputs=['*'])
        mp.add_output('takeoff|h',val=0.,units='ft')
        mp.add_output('cruise|h0',val=1500.,units='ft')
        mp.add_output('mission_range',val=30.,units='mi')
        mp.add_output('payload',val=1000.,units='lbm')

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        phase1 = self.add_subsystem('takeoff', MomentumTheoryVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='takeoff'), promotes_inputs=['ac|*'])
        phase2 = self.add_subsystem('cruise', MomentumTheoryMultiRotorCruisePhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        phase3 = self.add_subsystem('landing', MomentumTheoryVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='landing'), promotes_inputs=['ac|*'])
        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(phase1, phase2)
        self.link_phases(phase2, phase3)

class SimpleVTOLMissionWithTransition(oc.TrajectoryGroup):
    """
    VTOL mission, including vertical climb, transition1, cruise, transition2, and vertical descent.
    The user can to set the followings in runscript
        - in climb/hover/descent, [duration, fltcond|vs]
        - in cruise, [duration, fltcond|vs, fltcond|Ueas, Tangle]
        - in transition, [duration, fltcond|vs, fltcond|Ueas, accel_horiz_target, accel_vert_target]
        TODO: determine durations of each phase by target cruise altitude and range (and using BalanceComps)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")
        self.options.declare('mode', default='full', desc="full or takeoff or landing")
        #self.options.declare('nrotors', default=4, desc="Number of rotors")
        # full: vertical climb, transition1, cruise, transition2, and vertical descent.
        # takeoff: exclude transition 2
        # landing: exclude transition 1

    def setup(self):
        nn = self.options['num_nodes']
        #nr = self.options['nrotors']
        acmodelclass = self.options['aircraft_model']
        mode = self.options['mode']

        if mode == 'full':
            # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
            
            climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
            #tran1 = self.add_subsystem('transition1', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_climb'), promotes_inputs=['ac|*'])
            tran1 = self.add_subsystem('transition1', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition'), promotes_inputs=['ac|*'])
            cruise = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
            #tran2 = self.add_subsystem('transition2', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_descent'), promotes_inputs=['ac|*'])
            tran2 = self.add_subsystem('transition2', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition'), promotes_inputs=['ac|*'])
            descent = self.add_subsystem('descent', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])          

            # impose CL continuity between cruise and transitions by varying body geometric AoA.
            tran1.add_subsystem('CLcont1', BalanceComp('body_geom_alpha', val=5., units='deg', eq_units=None, lower=-15, upper=15, rhs_name='CL_transition1_end', lhs_name='CL_cruise_init'), promotes_outputs=['body_geom_alpha'])
            self.connect('transition1.fltcond|CL', 'transition1.CLcont1.CL_transition1_end', src_indices=-1)
            self.connect('cruise.fltcond|CL', 'transition1.CLcont1.CL_cruise_init', src_indices=0)

            tran2.add_subsystem('CLcont2', BalanceComp('body_geom_alpha', val=5., units='deg', eq_units=None, lower=-15, upper=15, rhs_name='CL_transition2_init', lhs_name='CL_cruise_end'), promotes_outputs=['body_geom_alpha'])
            self.connect('transition2.fltcond|CL', 'transition2.CLcont2.CL_transition2_init', src_indices=0)
            self.connect('cruise.fltcond|CL', 'transition2.CLcont2.CL_cruise_end', src_indices=-1)

            # connect bettery SOC, altitude, and mission_time of each segments
            self.link_phases(climb, tran1)
            self.link_phases(tran1, cruise)
            self.link_phases(cruise, tran2)
            self.link_phases(tran2, descent)

        elif mode == 'takeoff':
            # transition in takeoff only
            climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
            tran1 = self.add_subsystem('transition1', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_climb'), promotes_inputs=['ac|*'])
            cruise = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
            descent = self.add_subsystem('descent', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])

            # impose CL continuity between cruise and transitions by varying body geometric AoA.
            tran1.add_subsystem('CLcont1', BalanceComp('body_geom_alpha', val=5., units='deg', eq_units=None, lower=-15, upper=15, rhs_name='CL_transition1_end', lhs_name='CL_cruise_init'), promotes_outputs=['body_geom_alpha'])
            self.connect('transition1.fltcond|CL', 'transition1.CLcont1.CL_transition1_end', src_indices=-1)
            self.connect('cruise.fltcond|CL', 'transition1.CLcont1.CL_cruise_init', src_indices=0)

            self.link_phases(climb, tran1)
            self.link_phases(tran1, cruise)
            self.link_phases(cruise, descent)

        elif mode == 'landing':
            climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
            cruise = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
            tran2 = self.add_subsystem('transition2', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_descent'), promotes_inputs=['ac|*'])
            descent = self.add_subsystem('descent', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])

            # impose CL continuity between cruise and transitions by varying body geometric AoA.
            tran2.add_subsystem('CLcont2', BalanceComp('body_geom_alpha', val=5., units='deg', eq_units=None, lower=-15, upper=15, rhs_name='CL_transition2_init', lhs_name='CL_cruise_end'), promotes_outputs=['body_geom_alpha'])
            self.connect('transition2.fltcond|CL', 'transition2.CLcont2.CL_transition2_init', src_indices=0)
            self.connect('cruise.fltcond|CL', 'transition2.CLcont2.CL_cruise_end', src_indices=-1)

            self.link_phases(climb, cruise)
            self.link_phases(cruise, tran2)
            self.link_phases(tran2, descent)