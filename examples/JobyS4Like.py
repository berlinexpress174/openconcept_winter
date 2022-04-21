#!/usr/bin/env python3
from __future__ import division
import sys
import os
import logging
from typing_extensions import runtime
from matplotlib.pyplot import connect
import numpy as np
from openmdao.core.indepvarcomp import IndepVarComp
sys.path.insert(0, os.getcwd())
import openmdao.api as om
from openmdao.api import ExplicitComponent, ScipyOptimizeDriver, SqliteRecorder, ExecComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.utilities.linearinterp import LinearInterpolator
from openconcept.utilities.visualization import plot_trajectory
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math.Compute_SinCos import ComputeSinCos
from examples.propulsion_layouts.simple_eVTOL_multirotor import UserDefinedMultiRotorVTOLPropulsionSystem
from examples.methods.weights_eVTOL import eVTOLEmptyWeight
from openconcept.utilities.eVTOL_Aero_model import BodyDrag, MaxRPM, RpmResidual
from openconcept.utilities.VTOLPowerAndThrust import PowerAndThrustCal
from examples.aircraft_data.JobyS4 import data as acdata
from openconcept.analysis.performance.mission_profiles_eVTOL import BasicSimpleVTOLMission, eVTOLMission_validation1_Hansman, BasicSimpleVTOLMIssionTakeoffAndCruiseOnly
from openconcept.analysis.performance.mission_profiles_eVTOL import BasicSimpleVTOLMissionMomentumTakeoffAndCruiseOnly

#from pyoptsparse import SLSQP, SNOPT, Optimization

class AugmentedFBObjective(ExplicitComponent):
    """
    This objective function aims for a maximum payload and minimum MOTW.
    """
    def setup(self):
        self.add_input('ac|propulsion|motor|rating', units= 'kW')
        self.add_input('ac|weights|MTOW', units='kg')
        self.add_output('mixed_objective')
        self.declare_partials(['mixed_objective'], ['ac|propulsion|motor|rating'], val= 1 )
        self.declare_partials(['mixed_objective'], ['ac|weights|MTOW'], val=1 ) 
    def compute(self, inputs, outputs):
        outputs['mixed_objective'] = inputs['ac|propulsion|motor|rating'] + inputs['ac|weights|MTOW']

class LeastElecLoad(ExplicitComponent):
    def setup(self):
        self.add_input('elec_load', units= None)
        self.add_output('elec_load_objective', units=None)
        self.declare_partials(['elec_load_objective'], ['elec_load'], val=-1)
    def compute(self, inputs, outputs):
        outputs['elec_load_objective'] = -inputs['elec_load']

class LowestMTOW(ExplicitComponent):
    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb')
        self.add_output('lowestMTOW', units = 'lb')
        #self.declare_partials(['lowestMTOW'], ['ac|weights|MTOW'], val=1/1000)
        self.declare_partials(['lowestMTOW'], ['ac|weights|MTOW'], val=1/100000)
    def compute(self, inputs, outputs):
        #outputs['lowestMTOW'] = inputs['ac|weights|MTOW']/1000
        outputs['lowestMTOW'] = inputs['ac|weights|MTOW']/100000

class eVTOLModel(om.Group):
    """
    A custom model specific to a hexarotor UAV
    This class will be passed in to the mission analysis code.

    Inputs:
        fltcond|*, throttole, duration, ac|propulsion|*, ac|weights|*

    Outputs:
        thrusts, drag, weight
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('num_rotors', default=num_rotors[0])
        self.options.declare('flight_phase', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        nr = self.options['num_rotors']
        flight_phase = self.options['flight_phase']

        # any control variables other than throttle and braking need to be defined here
        self.add_subsystem('calmaxproprpm', MaxRPM(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=['*'])
        self.add_subsystem('proprpm_residual', RpmResidual(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=['*'])

        propulsion_promotes_outputs = ['thrust']
        propulsion_promotes_inputs = ["fltcond|*", "ac|propulsion|*",
                                      "throttle", "ac|weights|*", "duration"]

        if flight_phase in ['takeoff','climb', 'hover', 'descent', 'landing']:
            self.add_subsystem('propmodel',PowerAndThrustCal(num_nodes=nn),
                                promotes_inputs=['*'],
                                promotes_outputs=['thrust','throttle'])
        elif flight_phase in ['cruise']:
            self.add_subsystem('propmodel',UserDefinedMultiRotorVTOLPropulsionSystem(num_nodes=nn, nrotors = nr),
                                promotes_inputs=propulsion_promotes_inputs,
                                promotes_outputs=propulsion_promotes_outputs)
            motor_elec_power_rpm_list = []
            for i in range(1,nr+1):
                    motor_elec_power_rpm_list.append('propmodel.prop' + str(i) + '.rpm')
            self.connect('proprpm', motor_elec_power_rpm_list)
        else:
            raise RuntimeError('flight phase must be in [takeoff, climb, hover, descent, cruise, landing] to compute the drag in aircraft model')

        # drag in cruise
        self.add_subsystem('drag', PolarDrag(num_nodes=nn), promotes_inputs=['fltcond|CL', 'ac|geom|*', ('CD0', 'ac|aero|polar|CD0'), 'fltcond|q', ('e', 'ac|aero|polar|e')], promotes_outputs=['drag'])

        self.add_subsystem('OEW',eVTOLEmptyWeight(),
                        promotes_inputs=['*'],
                        promotes_outputs=['OEW'])

        # weight (MTOW): constant at ac|weights|MTOW
        self.add_subsystem('weight', LinearInterpolator(num_nodes=nn, units='kg'),
                           promotes_inputs=[('start_val', 'ac|weights|MTOW'),
                                             ('end_val', 'ac|weights|MTOW')],
                           promotes_outputs=[('vec', 'weight')])
        


class eVTOL_test01AnalysisGroup(om.Group):
    """This is an example of a climb-hover-cruise-descent mission of eVTOL.
    """
    def setup(self):
        nn = 11
        
        dv_comp = self.add_subsystem('dv_comp', DictIndepVarComp(acdata), promotes_outputs=["*"])

        # add eVTOL parameters
        dv_comp.add_output_from_dict('ac|aero|CD0_body')
        dv_comp.add_output_from_dict('ac|aero|polar|e')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_cruise')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0')

        dv_comp.add_output_from_dict('ac|geom|wing|equip')
        dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
        dv_comp.add_output_from_dict('ac|geom|wing|AR')
        dv_comp.add_output_from_dict('ac|geom|wing|c4sweep')
        dv_comp.add_output_from_dict('ac|geom|wing|taper')
        dv_comp.add_output_from_dict('ac|geom|wing|toverc')
        dv_comp.add_output_from_dict('ac|geom|hstab|equip')
        dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|vstab|equip')
        dv_comp.add_output_from_dict('ac|geom|vstab|S_ref')
        
        dv_comp.add_output_from_dict('ac|geom|fuselage|width')
        dv_comp.add_output_from_dict('ac|geom|fuselage|length')
        dv_comp.add_output_from_dict('ac|geom|fuselage|height')
        dv_comp.add_output_from_dict('ac|geom|nosegear|equip')
        dv_comp.add_output_from_dict('ac|geom|nosegear|length')
        dv_comp.add_output_from_dict('ac|geom|maingear|equip')
        dv_comp.add_output_from_dict('ac|geom|maingear|length')

        dv_comp.add_output_from_dict('ac|weights|MTOW')
        dv_comp.add_output_from_dict('ac|weights|W_battery')

        dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|proprpm')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|coaxialprop') 
        dv_comp.add_output_from_dict('ac|propulsion|propeller|FM') 
        dv_comp.add_output_from_dict('ac|propulsion|propeller|propeller_type') 

        dv_comp.add_output_from_dict('ac|propulsion|motor|rating')
        dv_comp.add_output_from_dict('ac|propulsion|motor|num_motors')
        dv_comp.add_output_from_dict('ac|propulsion|nacelle|equip')

        dv_comp.add_output('ac|propulsion|propeller|num_rotors', val = num_rotors, units = None)
        dv_comp.add_output('ac|propulsion|battery|specific_energy',val=300,units='W*h/kg')
        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|eVTOL_type_tiltrotor')
        dv_comp.add_output_from_dict('ac|eVTOL_type_multirotor')
        dv_comp.add_output_from_dict('ac|payload')
        
        analysis = self.add_subsystem('analysis', BasicSimpleVTOLMissionMomentumTakeoffAndCruiseOnly(num_nodes=nn, aircraft_model=eVTOLModel),
                                        promotes_inputs=['*'], promotes_outputs=['*'])
        
        margins = self.add_subsystem('margins',ExecComp('MTOW_margin = MTOW - OEW - W_battery - payload',
                                                        MTOW_margin={'units':'lbm'},
                                                        MTOW={'units':'lb'},
                                                        OEW={'units':'lb'},
                                                        W_battery={'units':'lb'},
                                                        payload={'units':'lb'}))
                                                       
        self.connect('takeoff.OEW','margins.OEW')
        self.connect('ac|weights|MTOW','margins.MTOW')
        self.connect('ac|weights|W_battery','margins.W_battery')
        self.connect('ac|payload','margins.payload')

        augobj = self.add_subsystem('aug_obj', AugmentedFBObjective(), promotes_outputs=['mixed_objective'])
        self.connect('ac|weights|MTOW','aug_obj.ac|weights|MTOW')
        self.connect('ac|propulsion|motor|rating','aug_obj.ac|propulsion|motor|rating')
        
        #leastele = self.add_subsystem('least_elelaod', LeastElecLoad(), promotes_outputs=['elec_load_objective'])
        #self.connect('landing.propmodel.batt1.SOC_final','least_elelaod.elec_load')

        lowestMOTW = self.add_subsystem('lowest_MOTW', LowestMTOW(), promotes_outputs=['lowestMTOW'])
        self.connect('ac|weights|MTOW','lowest_MOTW.ac|weights|MTOW')
       
        


def configure_problem():
    prob = om.Problem()
    prob.model = eVTOL_test01AnalysisGroup()

    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    #prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='wall', print_bound_enforce=False)
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=False)
    return prob


def set_values(prob, num_nodes, num_rotors, spec_energy, payload, cruise_speed, design_range):
    # set some (required) mission parameters. Each phase needs a vertical speed and duration

    prob.set_val('ac|propulsion|battery|specific_energy', spec_energy, units = 'W*h/kg')
    prob.set_val('ac|propulsion|propeller|num_rotors', num_rotors, units = None)
    prob.set_val('ac|payload', payload, units = 'lb')

    # Vertical Takeoff
    prob.set_val('takeoff.thrust', np.ones((num_nodes,))*1000., units='lbf')
    prob.set_val('takeoff.fltcond|vs', np.ones((num_nodes,))*500., units='ft/min')
    prob.set_val('takeoff.duration', 180, units='s')

    cruise_duration = 60*(design_range/cruise_speed)
    # cruise : use lift to compensate 100% of weight
    prob.set_val('cruise.thrust', np.ones((num_nodes,))*1000., units='lbf')
    prob.set_val('cruise.fltcond|Ueas', np.ones((num_nodes,))*cruise_speed, units='mi/h') # Target to 150 mph.
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,))*0.001, units='ft/min')
    prob.set_val('cruise.Tangle', np.ones((num_nodes,))*0.001, units='deg')
    prob.set_val('cruise.duration', cruise_duration, units='min')
    
    # Vertical Landing
    prob.set_val('landing.thrust', np.ones((num_nodes,))*1000., units='lbf')
    prob.set_val('landing.fltcond|vs', np.ones((num_nodes,))*(-500), units='ft/min')
    #prob.set_val('landing.Vs', np.ones((num_nodes,))*(-500), units='ft/min')
    prob.set_val('landing.duration', 180, units='s')
    prob.set_val('cruise|h0',1500,units='ft')

    

def show_outputs(prob):
    # print some outputs

    vars_list = ['ac|weights|MTOW', 'takeoff.OEW','ac|payload',
                 'takeoff.OEW.W_wing','takeoff.OEW.W_fuselage','takeoff.OEW.W_empennage','takeoff.OEW.W_nacelle',
                 'takeoff.OEW.W_gear', 'takeoff.OEW.W_equipment','takeoff.OEW.W_Prop_hub','takeoff.W_Motor','takeoff.W_Motor_Controller',
                 'takeoff.OEW.W_structure_fudged',
                 #'takeoff.OEW.P_TO','takeoff.thrust',
                 'landing.propmodel.batt1.SOC_final', 'landing.ode_integ.mission_time_final',
                 'landing.ode_integ.fltcond|h_final', 'landing.ode_integ.range_final', 'ac|weights|W_battery','margins.MTOW_margin',
                 'ac|propulsion|motor|rating','ac|propulsion|propeller|diameter',
                 'ac|geom|wing|S_ref','ac|geom|wing|AR','ac|geom|hstab|S_ref','ac|geom|vstab|S_ref','ac|geom|fuselage|length',
                 #'climb.Tangle','descent.Tangle',
                 'cruise.Tangle',
                 'ac|propulsion|battery|specific_energy','takeoff.throttle']
    units = ['lb', 'lb', 'lb',
             'lb', 'lb', 'lb', 'lb',
             'lb', 'lb', 'lb', 'lb','lb',
             'lb',
             #'kW','N',
             None, 'min', 
             'ft', 'mi','lb', 'lb', 
             'kW','ft',
             'm**2',None,'ft**2','ft**2','ft',
             #'deg', 'deg',
             'deg',
             'W*h/kg',None]
    nice_print_names = ['MTOW', 'Takeoff OEW', 'Payload',
                        'Wing OEW', 'Fuselage OEW', 'Empennage OEW', 'Nacelle OEW',
                        'Gear OEW', 'Equipemnt OEW', 'Propeller weight OEW', 'Total Motor OEW', 'Total Mortor controller OEW',
                        'Structural fudge OEW',
                        #'Takeoff power (One motor)','Takeoff Thrust (Total thrust)',
                        'Final battery state of charge', 'Total mission time',
                        'Final altitude', 'Final range','Battery weight','MTOW margin', 'Motor rating','Rotor Disk diameter',
                        'Wing area','AR','Hstab_S_ref','Vstab_S_ref','Fuselage length',
                        #'Climb Thrust angle', 'Descent Thrust angle',
                        'Cruise Tangle',
                        'Battery specific energies','Takeoff thrust state']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+str(units[i]))

    # plot time histories of some entries
    plots = False
    if plots:
        
        
        # we will call prob.get_value(phase + x_var/y_var)
        x_var = 'mission_time'
        x_unit = 'min'
        y_vars = ['fltcond|h', 'throttle', 'fltcond|vs', 'propmodel.batt1.SOC', 'thrust', 'lift', 'range']
        y_units = ['ft', None, 'ft/min', None, 'N', 'N', 'mi']
        x_label = 'Time (min)'
        y_labels = ['Altitude (ft) (integrated)', 'Throttle (state)', 'Vertical speed (m/s) (input)', 'Battery SOC (integrated)', 'Thrust (N)', 'Wing Lift (N)', 'Range (mi)']
        #phases = ['takeoff','climb', 'cruise', 'descent', 'landing']
        phases = ['takeoff','cruise', 'landing']
        plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels, marker= '-',
                        plot_title=None)
                        #plot_title='eVTOL flight history')

        # we will call prob.get_value(phase + x_var/y_var)
        x_var = 'range'
        x_unit = 'mi'
        y_vars = ['fltcond|h']
        y_units = ['ft']
        x_label = 'mission_range (mi)'
        y_label = ['mission_altitude (ft)']
        #phases = ['takeoff','climb', 'cruise', 'descent', 'landing']
        phases = ['takeoff','cruise', 'landing']
        plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_label, marker= '-',
                        plot_title=None)
                        #plot_title='eVTOL flight distance history')


def run_eVTOL_type1_analysis(plots=False):
    num_nodes = 11
    prob = configure_problem()
    prob.setup(check=False, mode='fwd')
    set_values(prob, num_nodes, num_rotors, specific_energy, payload, cruise_speed, design_range)  # Specific energy can be changed from input
    prob.run_model()
    om.n2(prob)
    #prob.check_partials(compact_print=True)
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    #for run type choose optimization, comp_sizing, or analysis
    #run_type = 'example'
    #run_type = 'optimization'
    #run_type = 'comp_sizing'
    run_type = 'extreme_study'
    #run_type = 'analysis'
    num_nodes = 11
    

    if run_type == 'example':
        # runs a default analysis-only mission (no optimization)
        num_rotors = [6]
        design_range = 30
        specific_energy = 300
        payload = 960
        cruise_speed = 150
        
        run_eVTOL_type1_analysis(plots=False)
    
    else:
        # can run a sweep of design range and spec energy (not tested) 
        # design_ranges = [5,10,15,20,25,30] #miles 
        # specific_energies = [100,150,200,250,300]
        
        # or a single point
        
        #design_ranges = [30,60,120,240,480]#,960,1920]
        #payloads = [960]
        #cruise_speeds = [100,200,300]
        design_ranges = [60]
        payloads = [240,480,720,960,1200,1440]
        cruise_speeds = [100,150,200,300]
        specific_energies = [300]
        num_rotors = [6]

        write_logs = True
        if write_logs:
            logging.basicConfig(filename='opt.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        # run a sweep of cases at various specific energies and ranges
        for this_design_range in design_ranges:
            for this_cruise_speed in cruise_speeds:
                for this_payload in payloads:
                    for this_spec_energy in specific_energies:
                        try: 
                            prob = configure_problem()
                            design_range = this_design_range
                            spec_energy = this_spec_energy
                            payload = this_payload
                            cruise_speed = this_cruise_speed
                            if run_type == 'optimization':
                                print('======Performing Multidisciplinary Design Optimization===========')
                                prob.model.add_design_var('ac|weights|MTOW', lower=2500, upper=5000)
                                #prob.model.add_design_var('ac|geom|wing|S_ref',lower=15,upper=40, units='m**2')
                                #prob.model.add_design_var('ac|propulsion|motor|rating',lower=1,upper=300)
                                prob.model.add_design_var('ac|propulsion|propeller|diameter',lower=2,upper=10, units='ft')
                                prob.model.add_design_var('ac|weights|W_battery',lower=20,upper=2250, units= 'lb') # units: lb
                
                                prob.model.add_constraint('margins.MTOW_margin',lower=0.0)
                                prob.model.add_constraint('landing.propmodel.batt1.SOC_final',lower=0.0)
                                
                                prob.model.add_constraint('takeoff.throttle',upper=1.2*np.ones(num_nodes)) #maximum throttle 1.2
                                prob.model.add_constraint('takeoff.propmodel.motor1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                                prob.model.add_constraint('takeoff.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                                #prob.model.add_constraint('cruise.fltcond|Ueas', lower = np.ones((num_nodes,))*60.,upper = np.ones((num_nodes,))*100., units='mi/h')
                                prob.model.add_constraint('cruise.propmodel.motor1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                                #prob.model.add_constraint('cruise.propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                                prob.model.add_constraint('cruise.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                                prob.model.add_constraint('landing.propmodel.motor1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                                prob.model.add_constraint('landing.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                                prob.model.add_objective('mixed_objective') # TODO add this objective
                                
                            
                            elif run_type == 'comp_sizing':
                                print('======Performing Component Sizing Optimization===========')
                                prob.model.add_design_var('ac|weights|MTOW', lower=10, upper=10000, ref0=4000, ref=4100, units='lb')
                                #prob.model.add_design_var('ac|geom|wing|AR', lower = 8.5, upper = 12, ref0=10, ref=12, units = None)
                                #prob.model.add_design_var('ac|geom|vstab|S_ref',lower=35,upper=40, ref0=15, ref=60, units='ft**2')
                                #prob.model.add_design_var('ac|geom|fuselage|length',lower=20,upper=50, ref0=20, ref=60, units='ft')
                                
                                #prob.model.add_design_var('ac|propulsion|propeller|diameter',lower=4.5,upper=13, ref0=4, ref=5, units='ft')
                                #prob.model.add_design_var('ac|propulsion|propeller|proprpm',lower=100, upper=3000, ref0=2000, ref=2100, units='rpm')
                                prob.model.add_design_var('ac|propulsion|propeller|proprpm_takeoff',lower=100,upper=3000, ref0=800, ref=810, units='rpm')
                                prob.model.add_design_var('ac|propulsion|propeller|proprpm_cruise',lower=100, upper=3000, ref0=1200, ref=1210, units='rpm')
                                prob.model.add_design_var('ac|weights|W_battery',lower=10,upper=4000, ref0 = 400, ref = 410, units='lb')
                                #prob.model.add_design_var('ac|propulsion|motor|rating',lower = 10, upper = 500)#, ref0 = 200, ref=500, units='kW')
                                prob.model.add_design_var('ac|propulsion|motor|rating',lower = 5, upper = 100, units='kW')#, ref0 = 30, ref=31, units='kW')
                                #prob.model.add_design_var('climb.Tangle',lower = 0, upper = 45, ref0 = 25, ref=45, units='deg')
                                prob.model.add_design_var('cruise.Tangle',lower = 0, upper = 45, ref0 = 10, ref=45, units='deg')                        
                                #prob.model.add_design_var('descent.Tangle',lower = 0, upper = 45, ref0 = 25, ref=45, units='deg')
                                #----
                                #prob.model.add_constraint('takeoff.propmodel.prop1.eta_prop', equals = 0.6)
                                #prob.model.add_constraint('cruise.propmodel.prop1.eta_prop', equals= 0.8)
                                #----
                                prob.model.add_constraint('margins.MTOW_margin',equals=0,units='lb') # TODO implement
                                prob.model.add_constraint('landing.propmodel.batt1.SOC_final',equals=0.3)
                                prob.model.add_constraint('takeoff.throttle', lower = 0.0, upper= 1.2) 
                                #prob.model.add_constraint('takeoff.throttle', equals=1.2) 
                                prob.model.add_constraint('landing.throttle', lower = 0.0, upper= 1.2)
                                prob.model.add_objective('lowestMTOW') # Fixed payload

                            elif run_type == 'extreme_study':
                                print('======Performing Component Sizing Extreme Optimization===========')

                                # under 8 passangers
                                prob.model.add_design_var('ac|weights|MTOW', lower=100, ref0 = 1000, ref = 15000, units='lb')
                                prob.model.add_design_var('ac|geom|wing|AR',lower = 7, ref0 = 7, ref = 12, units = None)
                                prob.model.add_design_var('ac|geom|wing|S_ref',lower = 16, ref0 = 16, ref = 30, units = 'm**2')
                                prob.model.add_design_var('ac|geom|vstab|S_ref',lower=35, ref0 = 35, ref = 40,units='ft**2')
                                prob.model.add_design_var('ac|geom|fuselage|length',lower=20, ref0 = 20, ref = 25 ,units='ft')
                                prob.model.add_design_var('ac|propulsion|propeller|diameter',lower=9.5, ref0 = 9.5, ref = 12, units='ft')
                                prob.model.add_design_var('ac|propulsion|propeller|proprpm',lower=100, ref0 = 800, ref = 2000,units='rpm')
                                prob.model.add_design_var('ac|weights|W_battery',lower = 10, ref0 = 400, ref = 5000, units = 'lb')
                                prob.model.add_design_var('ac|propulsion|motor|rating',lower = 10, ref0 = 25, ref = 200, units='kW')

                                # 8 to 14 passangers: data: cessna caravan

                                prob.model.add_constraint('cruise.propmodel.prop1.eta_prop', upper = 0.8, ref0 = 0.79, ref = 0.8)
                                #prob.model.add_constraint('cruise.propmodel.prop1.J', lower = 0.5, upper = 4)#, ref0 = 0.0, ref = 4)
                                prob.model.add_constraint('margins.MTOW_margin', equals=0, units='lb')
                                prob.model.add_constraint('takeoff.proprpm_residual', lower=0, units='rpm')

                                prob.model.add_constraint('landing.propmodel.batt1.SOC_final', equals=0.2, ref0 = 0, ref = 1)
                                prob.model.add_constraint('takeoff.throttle', lower = 0.0, upper= 1.0, ref0 = 0, ref = 1) 
                                prob.model.add_constraint('landing.throttle', lower = 0.0, upper= 1.0, ref0 = 0, ref = 1)
                                prob.model.add_objective('lowestMTOW') 
                                
                            else:
                                print('======Analyzing Fuel Burn for Given Mision============')
                                #prob.model.add_design_var('cruise.hybridization', lower=0.01, upper=0.5)
                                #prob.model.add_constraint('descent.propmodel.batt1.SOC_final',lower=0.0)
                                prob.model.add_constraint('landing.propmodel.batt1.SOC_final',lower=0.0)
                                prob.model.add_objective('elec_load_objective')
                                #prob.model.add_objective('descent.fuel_used_final')
                            
                            """
                            prob.driver = om.ScipyOptimizeDriver()
                            prob.driver.options['debug_print'] = ['objs','desvars','nl_cons']#,'totals']#,'ln_cons']
                            prob.driver.options['maxiter'] = 200
                            """
                            prob.driver = om.pyOptSparseDriver()
                            # SNOPT
                            #outfile_to_save = 'case_'+'Heli_'+'SNOPT_'+'range'+str(design_range)+'_'+'battary'+str(spec_energy)+'_'+'payload'+str(payload)+'_'+'cruise'+str(cruise_speed)+'_'+'Rotor'+str(num_rotors)+'.out'
                            prob.driver.options['optimizer'] = 'SNOPT'
                            #prob.driver.opt_settings['Summaryfile'] = outfile_to_save
                            #prob.driver.opt_settings['IFILE'] = outfile_to_save
                            
                            # SLSQP
                            #outfile_to_save = 'case_'+'SLSQP_'+'range'+str(design_range)+'_'+'battary'+str(spec_energy)+'_'+'payload'+str(payload)+'_'+'cruise'+str(cruise_speed)+'_'+'Rotor'+str(num_rotors)+'.out'
                            #prob.driver.options['optimizer'] = 'SLSQP'
                            #prob.driver.opt_settings['IFILE'] = outfile_to_save = 'case_'+'SLSQP_'+'range'+str(design_range)+'_'+'battary'+str(spec_energy)+'_'+'payload'+str(payload)+'_'+'cruise'+str(cruise_speed)+'_'+'Rotor'+str(num_rotors)+'.out'
                            
                            prob.driver.options['debug_print'] = ['objs']#,'desvars','nl_cons']
                            #prob.driver.options['disp'] = True
                            #prob.driver.options['maxiter'] = 20
                            
                            if write_logs:
                                filename_to_save = 'case_'+'JobyS4_'+'SNOPT_'+'range'+str(design_range)+'_'+'battary'+str(spec_energy)+'_'+'payload'+str(payload)+'_'+'cruise'+str(cruise_speed)+'_'+'Rotor'+str(num_rotors)+'.sql'
                                filename_failed_to_save = 'case_'+'JobyS4_'+'SNOPT_'+'range'+str(design_range)+'_'+'battary'+str(spec_energy)+'_'+'payload'+str(payload)+'_'+'cruise'+str(cruise_speed)+'_'+'Rotor'+str(num_rotors)+'_failed.sql'
                                if os.path.isfile(filename_to_save) or os.path.isfile(filename_failed_to_save):
                                    print('Skipping '+filename_to_save)
                                    continue
                                recorder = SqliteRecorder(filename_to_save)
                                prob.add_recorder(recorder)
                                prob.driver.add_recorder(recorder)
                                prob.driver.recording_options['includes'] = []
                                prob.driver.recording_options['record_objectives'] = True
                                prob.driver.recording_options['record_constraints'] = True
                                prob.driver.recording_options['record_desvars'] = True
                                prob.driver.recording_options['record_outputs'] = True
                            
                            prob.setup(check=False)
                            set_values(prob, num_nodes, num_rotors, spec_energy, payload, cruise_speed, design_range)
                            run_flag = prob.run_driver()
                            prob.record('final_state')

                            if run_flag:
                                raise ValueError('Opt failed')

                        except BaseException as e:
                            if write_logs:
                                logging.error('Optimization '+filename_to_save+' failed because '+repr(e))
                            prob.cleanup()
                            try:
                                os.rename(filename_to_save, filename_to_save.split('.sql')[0]+'_failed.sql')
                            #except WindowsError as we:
                            except OSError as we:
                                if write_logs:
                                    logging.error('Error renaming file: '+repr(we))
                                os.remove(filename_to_save)
        om.n2(prob)
        show_outputs(prob)

                        


