#%%
import os 
import openmdao.api as om
import matplotlib.pyplot as plt
from examples.JobyS4Like import eVTOLModel
import numpy as np

#%%
class read_file_test:
    def __init__(self, payload,spec_energy,cruise_speed,design_range,num_rotors,optimizer,variable):
        self.payload = payload 
        self.spec_energy = spec_energy
        self.cruise_speed = cruise_speed 
        self.design_range = design_range
        self.optimizer = optimizer
        self.num_rotors = num_rotors
        self.variable = variable

    def read_sql_test(self):
        if self.optimizer == 'SNOPT':
            print('The current is (' + str(self.optimizer) +') case_'+'range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+'_'+'cruise'+str(self.cruise_speed)+'_'+'Rotor'+str(self.num_rotors)+'.sql')
            #print('You are looking at the varible ' + str(self.variable))
        elif self.optimizer == 'SLSQP':
            print('The current is (' + str(self.optimizer) +') case_'+'range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+'_'+'cruise'+str(self.cruise_speed)+'_'+'Rotor'+str(self.num_rotors)+'.sql')
        else: 
            pass

    def write_variables(self,read):
        MTOW_temp = []
        AR_temp = []
        vstab_Sref_temp = []
        fuselage_temp = []
        propeller_temp = []
        W_battery_temp = []
        motor_rating_temp = []

        cr_temp = om.CaseReader(read)
        driver_temp = cr_temp.get_cases('driver', recurse = False)
        
        for case in driver_temp:
            MTOW_temp.append(case['ac|weights|MTOW'][0])
            AR_temp.append(case['ac|geom|wing|AR'][0])
            vstab_Sref_temp.append(case['ac|geom|vstab|S_ref'][0])
            fuselage_temp.append(case['ac|geom|fuselage|length'][0])
            propeller_temp.append(case['ac|propulsion|propeller|diameter'][0])
            W_battery_temp.append(case['ac|weights|W_battery'][0])
            motor_rating_temp.append(case['ac|propulsion|motor|rating'][0])

        total_temp = [MTOW_temp,AR_temp,vstab_Sref_temp,fuselage_temp,propeller_temp,W_battery_temp,motor_rating_temp]
        #print(MTOW_temp)
        if self.variable == 'MTOW':
            #print(type(total_temp[0]))
            return total_temp[0]
        elif self.variable == 'AR':
            return total_temp[1]
        elif self.variable == 'S_ref':
            return total_temp[2]
        elif self.variable == 'fuselage_length':
            return total_temp[3]
        elif self.variable == 'rotor_diameter':
            return total_temp[4]
        elif self.variable == 'battery_weight':
            return total_temp[5]
        elif self.variable == 'motor_rating':
            return total_temp[6]
        elif self.variable == 'all':
            return total_temp
        else:
            raise('Assigned design variable is not available')

    def read_sql_inClass(self):
        variables_list = []
        if self.optimizer == 'SNOPT' or 'SLSQP':
            for filename in os.listdir():
                read = 'case_'+str(self.optimizer)+'_range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+'_'+'cruise'+str(self.cruise_speed)+'_'+'Rotor'+str(self.num_rotors)+'.sql'
                read_failed = 'case_'+'SNOPT_'+'range'+str(self.design_range)+'_'+'battary'+str(self.spec_energy)+'_'+'payload'+str(self.payload)+"_cruise"+str(self.cruise_speed)+'_Rotor[6]_failed.sql'
                #print(read)
                if filename.endswith(read):
                    variables_list = read_file_test.write_variables(self,read)
                    print(variables_list[-1])
                elif filename.endswith(read_failed):
                    pass #MTOW_list_temp.append(np.nan)
            return variables_list
        else: 
            raise('current optimizer is unavailable')
        #print(variables_list)

#%%
design_ranges = [60]
payloads = [240, 480, 720, 960, 1200]#, 1440]
cruise_speeds = [100]
specific_energies = [300]
num_rotors = [6]

# %%
fig, ax1 = plt.subplots(figsize=(10,8))
fig.subplots_adjust(left = None, bottom = None, right=None, top=None, wspace = 0.5, hspace = None)
for this_design_ranges in design_ranges:
    for this_spec_energy in specific_energies:
        for this_cruise_speed in cruise_speeds:
            for this_payload in payloads:
                payload = this_payload
                spec_energy = this_spec_energy
                cruise_speed = this_cruise_speed
                design_range = this_design_ranges
                #print('payload',payload)
                #print('cruise_speed',cruise_speed)
                #print('design_range',design_range)
                
                color = next(ax1._get_lines.prop_cycler)['color']
                current_data_read = read_file_test(payload,spec_energy,cruise_speed,design_range,num_rotors,'SNOPT','MTOW')
                current_data = current_data_read.read_sql_inClass()
                ax1.plot(np.arange(len(current_data)), current_data, '-', label = 'SNOPT'+' ' + 'payload ' + '%d'%payload + '(lb)', color = color)
                #ax1.legend()

                current_data_read = read_file_test(payload,spec_energy,cruise_speed,design_range,num_rotors,'SLSQP','MTOW')
                current_data = current_data_read.read_sql_inClass()
                ax1.plot(np.arange(len(current_data)), current_data, '--', label = 'SLSQP'+' '+ 'payload ' + '%d'%payload + '(lb)', color = color)
                #ax1.legend()
ax1.legend(ncol = 2)
ax1.set(xlabel = "Function calls", ylabel = 'MTOW (lb)', title = "Optimization History")
ax1.set_yscale('log')
ax1.grid()



#ax2.plot(np.arange(len(SLSQP_MTOW_list)), SLSQP_MTOW_list)
#ax2.set(xlabel = "Iterations", ylabel = 'MTOW_SLSQp', title = "Optimization History")
#ax2.grid()

# Windows
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Prop_eta)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Prop_eta)FixRange30Battary300SpeedVsPayload.png", dpi = 300)
# MacOS
#plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/MTOWHistoryFixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
#plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/MTOWHistoryFixRange60Battary300SpeedVsPayload.png", dpi = 300)
plt.show()