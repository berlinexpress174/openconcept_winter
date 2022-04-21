from cmath import e
from enum import auto
import os
from pkgutil import get_data
from unittest import skip 
import openmdao.api as om
import matplotlib.pyplot as plt
from examples.JobyS4Like import eVTOLModel
import numpy as np
#plt.rcParams.update({'font.size': 16})

class read_file:
    def __init__(self, payload,spec_energy,cruise_speed,design_range,num_rotors,optimizer,variable,aircraft_type):
        """
        aircraft_type:
        -------------
        Aircraft type either in 'tiltrotor', 'helicopter', 'multirotor'
        """
        self.payload = payload 
        self.spec_energy = spec_energy 
        self.cruise_speed = cruise_speed 
        self.design_range = design_range 
        self.optimizer = optimizer 
        self.num_rotors = num_rotors
        self.variable = variable # kwarg**
        self.aircraft_type = aircraft_type # kwarg** || 1).tiltrotor || 2). Bell222B || 3). CityAirbus || 4). EHang216
    
    def write_variables(self):
        value_temp = []
        # ----- ------ ----- #
        for filename in os.listdir():
            read = 'case_'+str(self.aircraft_type)+'_'+str(self.optimizer)+'_'+'range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+'_'+'cruise'+str(self.cruise_speed)+'_'+'Rotor['+str(self.num_rotors)+'].sql'
            read_failed = 'case_'+str(self.aircraft_type)+'_'+str(self.optimizer)+'_'+'range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+"_cruise"+str(self.cruise_speed)+'_Rotor['+str(self.num_rotors)+']_failed.sql'
            if filename.endswith(read):
                cr_temp = om.CaseReader(read)
                return read_file.get_values(self,cr_temp)
            elif filename.endswith(read_failed):
                return np.nan
            else:
                pass
        # ----- ------ ----- #
        
    def get_values(self,case_reader):
        cr_temp = case_reader
        driver_casescr_temp = cr_temp.list_cases('driver', recurse = False, out_stream= None)
        last_case_temp = cr_temp.get_case(driver_casescr_temp[-1])
        design_vars_temp = last_case_temp.get_design_vars(scaled = False)
        constraint_temp = last_case_temp.get_constraints(scaled = False)
        ###
        #problem_cases = cr_temp.list_cases('problem')
        #problem_vars = cr_temp.list_source_vars('problem')
        other_output_case_temp = cr_temp.get_case('final_state')
        #constraint_temp = cr_temp.get_responses(scaled = False)
        
        #problem_case = cr_temp.list_outputs(prom_name = 'ac|payload')
        ###get_output_value = other_output_case_temp.get_val(['cruise.drag'][0])
        #print(get_output_value[-1])

        if self.variable == 'MTOW':
            return (design_vars_temp['ac|weights|MTOW'][0])
        elif self.variable == 'AR':
            return (design_vars_temp['ac|geom|wing|AR'][0])
        elif self.variable == 'Wing_S_ref':
            return (design_vars_temp['ac|geom|wing|S_ref'][0])
        elif self.variable == 'rpm':
            return (design_vars_temp['ac|propulsion|propeller|proprpm'][0])
        elif self.variable == 'rotor_diameter':
            return (design_vars_temp['ac|propulsion|propeller|diameter'][0])
        elif self.variable == 'battery_weight':
            return (design_vars_temp['ac|weights|W_battery'][0])
        elif self.variable == 'motor_rating':
            return (design_vars_temp['ac|propulsion|motor|rating'][0])   
        elif self.variable == 'OEW':
            return (design_vars_temp['ac|weights|MTOW'][0]) - (design_vars_temp['ac|weights|W_battery'][0]) - self.payload
        elif self.variable == 'OEW_MTOW_ratio':
            return ((design_vars_temp['ac|weights|MTOW'][0]) - (design_vars_temp['ac|weights|W_battery'][0]) - self.payload) / (design_vars_temp['ac|weights|MTOW'][0])
        elif self.variable == 'battery_weight_MTOW_ratio':
            return (design_vars_temp['ac|weights|W_battery'][0]) / (design_vars_temp['ac|weights|MTOW'][0])
        elif self.variable == 'Payload_MTOW_ratio':
            return self.payload / (design_vars_temp['ac|weights|MTOW'][0])
        elif self.variable == 'takeoff.OEW.W_Motor':
            return other_output_case_temp.get_val(['takeoff.OEW.W_Motor'][0])
        elif self.variable == 'takeoff.OEW.W_prop_fudged':
            return other_output_case_temp.get_val(['takeoff.OEW.W_prop_fudged'][0])
        elif self.variable == 'takeoff.OEW.W_Motor_Controller':
            return other_output_case_temp.get_val(['takeoff.OEW.W_Motor_Controller'][0])
        #elif self.variable == 'takeoff.OEW.Body_S_wet':
            #return other_output_case_temp.get_val(['takeoff.OEW.Body_S_wet'][0])        
        elif self.variable == 'cruise.drag':
            last_drag = other_output_case_temp.get_val(['cruise.drag'][0])
            return last_drag[-1]
        elif self.variable == 'takeoff_proprpm_residual':
            return (constraint_temp['takeoff.proprpm_residual'][0])
        elif self.variable =='W_prop_fudged_MTOW_Ratio':
            return other_output_case_temp.get_val(['takeoff.OEW.W_prop_fudged'][0])/design_vars_temp['ac|weights|MTOW'][0]
        elif self.variable =='Motor_MTOW_Ratio':
            return other_output_case_temp.get_val(['takeoff.OEW.W_Motor'][0])/design_vars_temp['ac|weights|MTOW'][0]
        else:
            return print('input variable not supported')

        
    def reshape_data(self,value_temp):
        values = []
        len_cruise_speeds = len(cruise_speeds)
        len_payloads = len(payloads)
        listsplits = np.array_split(value_temp, len_cruise_speeds)
        for array in listsplits:
            values.append(list(array))
        values_reshape = np.array(values).reshape((len_cruise_speeds, len_payloads)) 
        return values_reshape

    """
    Comparing SLSQP and SNOPT
    def read_sql_inClass(self):
        variables_list = []
        # ----- ----- Todo: Run all cases for SNOPT and SLSQP ----- ----- #
        if self.optimizer == 'SNOPT' or self.optimizer == 'SLSQP':
            for filename in os.listdir():
                read = 'case_'+str(self.optimizer)+'_range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+'_'+'cruise'+str(self.cruise_speed)+'_'+'Rotor'+str(self.num_rotors)+'.sql'
                read_failed = 'case_'+str(self.optimizer)+'_range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+"_cruise"+str(self.cruise_speed)+'_Rotor[6]_failed.sql'    
                #print(read)
                if filename.endswith(read):
                    variables_list = read_file.write_variables(self,read)
                    print(variables_list[-1])
                elif filename.endswith(read_failed):
                    pass #MTOW_list_temp.append(np.nan)
            return variables_list
        # ----- ----- ----- ----- #
        else: 
            raise('current optimizer or case  is unavailable')
    # options: '2D contour', '2D line', '3D surface plot'
    """

class plot_types:
    def __init__(self, plot_list, plot_list_name, plot_list_unit , plot_sweep_type, save, save_type ):
        """
        aircraft_type:
        -------------
        Aircraft type either in 'tiltrotor', 'helicopter', 'multirotor'
        """
        self.plot_sweep_type = plot_sweep_type 
        self.save = save 
        self.save_type = save_type 
        self.plot_list = plot_list
        self.plot_list_name = plot_list_name
        self.plot_list_unit = plot_list_unit
    
    def plot(self):
        if self.plot_sweep_type == 'payload':
            for list in self.plot_list:
                print(list)
                fig1, ax1 = plt.subplots(figsize=(8,6))
                for this_design_ranges in design_ranges:
                    for this_spec_energy in specific_energies:
                        for this_cruise_speed in cruise_speeds: # k
                            arr_EHang216= []
                            arr_CityAirbus= []
                            arr_Bell222B= []
                            arr_JobyS4= []
                            for this_payload in payloads:# x
                                payload = this_payload
                                spec_energy = this_spec_energy
                                cruise_speed = this_cruise_speed
                                design_range = this_design_ranges
                                            
                                color = next(ax1._get_lines.prop_cycler)['color']
                                EHang216_data_read = read_file(payload,spec_energy,cruise_speed,design_range,16,'SNOPT',str(list), aircraft_type = 'EHang216')
                                arr_EHang216.append(EHang216_data_read.write_variables())
                                
                                CityAirbus_data_read = read_file(payload,spec_energy,cruise_speed,design_range,8,'SNOPT',str(list), aircraft_type = 'CityAirbus')
                                arr_CityAirbus.append(CityAirbus_data_read.write_variables())
                                
                                #Bell222B_data_read = read_file(this_sweep,spec_energy,cruise_speed,design_range,1,'SNOPT',str(list), aircraft_type = 'Bell222B')
                                #arr_Bell222B.append(Bell222B_data_read.write_variables())
                                
                                JobyS4_data_read = read_file(payload,spec_energy,cruise_speed,design_range,6,'SNOPT',str(list), aircraft_type = 'JobyS4')
                                arr_JobyS4.append(JobyS4_data_read.write_variables())

                            ax1.plot(payloads,arr_EHang216,label = 'Ehang'+' ' + r'$V_{cruise}$' +' '+ '%d'%cruise_speed + '(mph)', marker = '*',color = color)
                            ax1.plot(payloads,arr_CityAirbus,label = 'CityAirbus'+' ' + r'$V_{cruise}$' +' '+ '%d'%cruise_speed + '(mph)', marker = 'o', color = color)
                            #ax1.plot(payloads,arr_Bell222B,label = 'Bell' + 'Cruise Speed' + '%d'%cruise_speed + '(mph)', marker = '^')#, color = color)
                            ax1.plot(payloads,arr_JobyS4,label = 'Joby'+' '+ r'$V_{cruise}$' +' '+ '%d'%cruise_speed + '(mph)', marker = 's', color = color)
                            
                #ax1.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, ncol = 1)
                ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),ncol = 3)
                
                ax1.set_xlabel('Payload (lb)') # Change this according to the target variables.
                ax1.set_ylabel( str(self.plot_list_name[self.plot_list.index(list)]) + ' (' + str(self.plot_list_unit[self.plot_list.index(list)]) +')')
                
                if self.save == True and self.save_type == 'Windows':
                    # Windows 11
                    plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
                    plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange30Battary100SpeedVsPayload.png", dpi = 300)
                    plt.show()
                    plt.tight_layout()
                elif self.save == True and self.save_type == 'MacOS':
                    # MacOS
                    plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
                    plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/FixRange60Battary300SpeedVsPayload_"+str(list)+".pdf",bbox_inches="tight")
                    plt.tight_layout()
                    #plt.show()
                else:
                    plt.tight_layout()
                    plt.show()
                    continue
        elif self.plot_sweep_type == 'range':        
            for list in self.plot_list:
                print(list)
                fig1, ax1 = plt.subplots(figsize=(8,6))
                for this_payload in payloads:# x
                    for this_spec_energy in specific_energies:
                        for this_cruise_speed in cruise_speeds: # k
                            arr_EHang216= []
                            arr_CityAirbus= []
                            #arr_Bell222B= []
                            arr_JobyS4= []
                            for this_design_ranges in design_ranges:
                                payload = this_payload
                                spec_energy = this_spec_energy
                                cruise_speed = this_cruise_speed
                                design_range = this_design_ranges
                                            
                                color = next(ax1._get_lines.prop_cycler)['color']
                                EHang216_data_read = read_file(payload,spec_energy,cruise_speed,design_range,16,'SNOPT',str(list), aircraft_type = 'EHang216')
                                arr_EHang216.append(EHang216_data_read.write_variables())
                                
                                CityAirbus_data_read = read_file(payload,spec_energy,cruise_speed,design_range,8,'SNOPT',str(list), aircraft_type = 'CityAirbus')
                                arr_CityAirbus.append(CityAirbus_data_read.write_variables())
                                
                                #Bell222B_data_read = read_file(this_sweep,spec_energy,cruise_speed,design_range,1,'SNOPT',str(list), aircraft_type = 'Bell222B')
                                #arr_Bell222B.append(Bell222B_data_read.write_variables())
                                
                                JobyS4_data_read = read_file(payload,spec_energy,cruise_speed,design_range,6,'SNOPT',str(list), aircraft_type = 'JobyS4')
                                arr_JobyS4.append(JobyS4_data_read.write_variables())

                            ax1.plot(design_ranges,arr_EHang216,label = 'Ehang' + r'$V_{cruise}$' + '%d'%cruise_speed + '(mph)', marker = '*',color = color)
                            ax1.plot(design_ranges,arr_CityAirbus,label = 'CityAirbus' + r'$V_{cruise}$' + '%d'%cruise_speed + '(mph)', marker = 'o', color = color)
                            ax1.plot(design_ranges,arr_JobyS4,label = 'Joby' + r'$V_{cruise}$' + '%d'%cruise_speed + '(mph)', marker = 's', color = color)
                            
                ax1.set_xlim([0,design_ranges[-1]])
                ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),ncol = 3)
                ax1.set_xlabel('Range (miles)') # Change this according to the target variables.
                ax1.set_ylabel( str(self.plot_list_name[self.plot_list.index(list)]) + ' (' + str(self.plot_list_unit[self.plot_list.index(list)]) +')')
                
                if self.save == True and self.save_type == 'Windows':
                    # Windows 11
                    plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
                    plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange30Battary100SpeedVsPayload.png", dpi = 300)
                    plt.tight_layout()
                    plt.show()
                elif self.save == True and self.save_type == 'MacOS':
                    # MacOS
                    plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/FixPayload960Battary300Speed200VsRange_"+str(list)+".pdf",bbox_inches="tight")
                    plt.tight_layout()
                    #plt.show()
                else:
                    plt.tight_layout()
                    plt.show()
                    continue
        else:
            return print('else case')

#%%
design_ranges = [60]
specific_energies = [300]
payloads = [240,480,720,960,1200,1440]
cruise_speeds = [150, 200, 300]
num_rotors = [6]
#%%

"""
# Plots some weight (with wing)
plot_list = ['MTOW','AR','Wing_S_ref','rpm','rotor_diameter','battery_weight','motor_rating']
plot_list_name = ['MTOW','AR','Wing Reference Area','Rotor rpm','Rotor Diameter','Battery Weight','Motor Rating']
plot_list_unit = ['lb', None,'m**2','rpm','m','lb','kW']
"""


"""
# Plots some weight (without wing)
plot_list = ['MTOW','rpm','rotor_diameter','battery_weight','motor_rating']
plot_list_name = ['MTOW','Rotor rpm','Rotor Diameter','Battery Weight','Motor Rating']
plot_list_unit = ['lb','rpm','m','lb','kW']
"""


"""
# Plots some MTOW ratio
plot_list = ['OEW','OEW_MTOW_ratio','battery_weight_MTOW_ratio','Payload_MTOW_ratio','W_prop_fudged_MTOW_Ratio','Motor_MTOW_Ratio']
plot_list_name = ['OEW','OEW/MTOW','W_battery/MTOW','Payload/MTOW','Propeller(rotor) weight/MTOW','Motor weight/MTOW']
plot_list_unit = ['lb', None, None, None, None,None]
"""

# Plots some OEW ratio


# plots some component weight
plot_list = ['takeoff.OEW.W_Motor','takeoff.OEW.W_prop_fudged','takeoff.OEW.W_Motor_Controller']#,'cruise.drag']
plot_list_name = ['Motor Weight','Propeller/rotor weight', 'Motor controller weight']#, 'Cruise drag']
plot_list_unit = ['lb', 'lb', 'lb', 'N']
"""
"""


"""
# Test plot
plot_list = ['takeoff_proprpm_residual']
plot_list_name = ['rpm_residual']
plot_list_unit = ['rpm']
"""


"""
type_sweep = plot_types(plot_list, plot_list_name, plot_list_unit, plot_sweep_type ='payload', save = True, save_type = 'MacOS')
plot_payload = type_sweep.plot()
"""

"""
"""
design_ranges = [30,60,120,240,480]#,960,1920]
specific_energies = [300]
payloads = [960]
cruise_speeds = [150,200]
num_rotors = [6]

type_sweep = plot_types(plot_list, plot_list_name, plot_list_unit, plot_sweep_type ='range', save = True, save_type = 'MacOS')
plot_payload = type_sweep.plot()



"""
for list in plot_list:
    print(list)
    fig1, ax1 = plt.subplots(figsize=(10,8))
    for this_design_ranges in design_ranges:
        for this_spec_energy in specific_energies:
            for this_cruise_speed in cruise_speeds: # k
                arr_EHang216= []
                arr_CityAirbus= []
                arr_Bell222B= []
                arr_JobyS4= []
                for this_payload in payloads:# x
                    payload = this_payload
                    spec_energy = this_spec_energy
                    cruise_speed = this_cruise_speed
                    design_range = this_design_ranges
                    
                    color = next(ax1._get_lines.prop_cycler)['color']
                    EHang216_data_read = read_file(payload,spec_energy,cruise_speed,design_range,16,'SNOPT',str(list), aircraft_type = 'EHang216')
                    arr_EHang216.append(EHang216_data_read.write_variables())
                    
                    CityAirbus_data_read = read_file(payload,spec_energy,cruise_speed,design_range,8,'SNOPT',str(list), aircraft_type = 'CityAirbus')
                    arr_CityAirbus.append(CityAirbus_data_read.write_variables())
                    
                    #Bell222B_data_read = read_file(payload,spec_energy,cruise_speed,design_range,1,'SNOPT',str(list), aircraft_type = 'Bell222B')
                    #arr_Bell222B.append(Bell222B_data_read.write_variables())
                    
                    JobyS4_data_read = read_file(payload,spec_energy,cruise_speed,design_range,6,'SNOPT',str(list), aircraft_type = 'JobyS4')
                    arr_JobyS4.append(JobyS4_data_read.write_variables())

                ax1.plot(payloads,arr_EHang216,label = 'Ehang' + 'Cruise Speed' + '%d'%cruise_speed + '(mph)', marker = '*',color = color)
                ax1.plot(payloads,arr_CityAirbus,label = 'CityAirbus' + 'Cruise Speed' + '%d'%cruise_speed + '(mph)', marker = 'o', color = color)
                #ax1.plot(payloads,arr_Bell222B,label = 'Bell' + 'Cruise Speed' + '%d'%cruise_speed + '(mph)', marker = '^')#, color = color)
                ax1.plot(payloads,arr_JobyS4,label = 'Joby' + 'Cruise Speed' + '%d'%cruise_speed + '(mph)', marker = 's', color = color)
                
    ax1.legend(ncol = 3)
    ax1.set_xlabel('Payload (lb)')
    ax1.set_ylabel( str(plot_list_name[plot_list.index(list)]) + ' (' + str(plot_list_unit[plot_list.index(list)]) +')')
    # Windows 11
    #plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
    #plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange30Battary100SpeedVsPayload.png", dpi = 300)
    # MacOS
    #plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/FixRange60Battary300SpeedVsPayload_"+str(list)+".pdf", dpi = 300)
    #plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/FixRange60Battary300SpeedVsPayload_"+str(list)+".pdf", dpi = 300)
    plt.show()
"""
# %%
