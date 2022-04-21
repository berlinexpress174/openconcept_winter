#%%
from cmath import e
from enum import auto
import os 
import openmdao.api as om
import matplotlib.pyplot as plt
from examples.JobyS4Like import eVTOLModel
import numpy as np
#%%
"""
cr400_6 = om.CaseReader("case_battary400_payload1440_cruise150_Rotor[6].sql")
driver_casescr400_6 = cr400_6.list_cases('driver', recurse = False)
last_case = cr400_6.get_case(driver_casescr400_6[-1])
design_vars = last_case.get_design_vars()
print(design_vars['ac|weights|MTOW'])
"""
#%%
design_ranges = [60]
#payloads = [240,480,720,960,1200,1440]
#cruise_speeds = [100,150,200,250]
payloads = [240,360,480,600,720,840,960,1080,1200,1320,1440]
cruise_speeds = [100,125,150,175,200,225,250]
#specific_energies = [50,75,100,200,300]
specific_energies = [300]
num_rotors = [6]

total_count = 0
c = 0
MTOW_values=[];MTOW_values_temp_list = []
Motor_Rating_values=[];Motor_Rating_values_temp_list = []
Rotor_Diameter_values=[];Rotor_Diameter_values_temp_list = []
RPM_values=[];RPM_values_temp_list = []
AR_values=[];AR_values_temp_list = []
Wing_area_values=[];Wing_area_temp_list = []
Prop_eta_values=[];Prop_eta_values_temp_list = []
Takeoff_Throttle_values=[];Takeoff_Throttle_values_temp_list=[]
Battery_weight_values=[];Battery_weight_values_temp_list=[]
OEW_weight_values = [];OEW_weight_values_temp_list=[]
for this_design_ranges in design_ranges:
    for this_spec_energy in specific_energies:
        for this_cruise_speed in cruise_speeds:
            for this_payload in payloads:
                payload = this_payload
                spec_energy = this_spec_energy
                cruise_speed = this_cruise_speed
                design_range = this_design_ranges
                #print('payload=',payload)
                #print('spec_energy=',spec_energy)
                #print('cruise_speed=',cruise_speed)
                for filename in os.listdir():
                    read = 'case_'+'range'+str(design_range)+'_'+'battary'+str(spec_energy)+'_'+'payload'+str(payload)+'_'+'cruise'+str(cruise_speed)+'_'+'Rotor'+str(num_rotors)+'.sql'
                    # read = "case_battary"+str(spec_energy)+"_payload"+str(payload)+"_cruise"+str(cruise_speed)+"_Rotor[6].sql"
                    read_failed = 'case_'+'range'+str(design_range)+'_'+'battary'+str(spec_energy)+'_'+'payload'+str(payload)+"_cruise"+str(cruise_speed)+'_Rotor[6]_failed.sql'
                    # read_failed = "case_battary"+str(spec_energy)+"_payload"+str(payload)+"_cruise"+str(cruise_speed)+"_Rotor[6]_failed.sql"
                    
                    if filename.endswith(read):
                        cr_temp = om.CaseReader(read)
                        driver_casescr_temp = cr_temp.list_cases('driver', recurse = False, out_stream= None)
                        last_case_temp = cr_temp.get_case(driver_casescr_temp[-1])
                        design_vars_temp = last_case_temp.get_design_vars(scaled = False)
                        constraint_temp = last_case_temp.get_constraints(scaled = False)

                        MTOW_values_temp_list.append(design_vars_temp['ac|weights|MTOW'][0]); #print('MTOW=',design_vars_temp['ac|weights|MTOW'][0])
                        Motor_Rating_values_temp_list.append(design_vars_temp['ac|propulsion|motor|rating'][0])
                        Rotor_Diameter_values_temp_list.append(design_vars_temp['ac|propulsion|propeller|diameter'][0])
                        RPM_values_temp_list.append(design_vars_temp['ac|propulsion|propeller|proprpm'][0])
                        AR_values_temp_list.append(design_vars_temp['ac|geom|wing|AR'][0])
                        Wing_area_temp_list.append(design_vars_temp['ac|geom|wing|S_ref'][0])
                        Prop_eta_values_temp_list.append(constraint_temp['cruise.propmodel.prop1.eta_prop'][0])
                        Takeoff_Throttle_values_temp_list.append(constraint_temp['takeoff.throttle'][0])
                        Battery_weight_values_temp_list.append(design_vars_temp['ac|weights|W_battery'][0])
                        OEW_weight_values_temp_list.append((design_vars_temp['ac|weights|MTOW'][0]-design_vars_temp['ac|weights|W_battery'][0]-payload))
                    elif filename.endswith(read_failed):
                        MTOW_values_temp_list.append(np.nan)
                        Motor_Rating_values_temp_list.append(np.nan)
                        Rotor_Diameter_values_temp_list.append(np.nan)
                        RPM_values_temp_list.append(np.nan)
                        AR_values_temp_list.append(np.nan)
                        Wing_area_temp_list.append(np.nan)
                        Prop_eta_values_temp_list.append(np.nan)
                        Takeoff_Throttle_values_temp_list.append(np.nan)
                        Battery_weight_values_temp_list.append(np.nan)
                        OEW_weight_values_temp_list.append(np.nan)
                        c = c+1
                        #print(c)
                    else:
                        continue
                    #fixed_spec_energy[0].append
                total_count += 1
                #print('payload=',payload)
            #MTOW_values.append(MTOW_values_temp_list)

len_cruise_speeds = len(cruise_speeds)
len_payloads = len(payloads)

MTOWsplits = np.array_split(MTOW_values_temp_list, len_cruise_speeds); print(MTOWsplits)
MotorRatingsplits = np.array_split(Motor_Rating_values_temp_list, len_cruise_speeds)
RotorDiametersplits = np.array_split(Rotor_Diameter_values_temp_list, len_cruise_speeds)
RPMsplits = np.array_split(RPM_values_temp_list, len_cruise_speeds)
ARsplits = np.array_split(AR_values_temp_list, len_cruise_speeds)
prop_eta_splits = np.array_split(Prop_eta_values_temp_list, len_cruise_speeds)
Takeoff_throttle_splits = np.array_split(Takeoff_Throttle_values_temp_list, len_cruise_speeds)
Wing_area_splits = np.array_split(Wing_area_temp_list, len_cruise_speeds)
Battery_weight_splits = np.array_split(Battery_weight_values_temp_list, len_cruise_speeds)
OEW_weight_splits = np.array_split(OEW_weight_values_temp_list, len_cruise_speeds)
for array in MTOWsplits:
    print(list(array))
    MTOW_values.append(list(array))
print(np.shape(MTOW_values))
print('--^MTOW--')
for array in MotorRatingsplits:
    print(list(array))
    Motor_Rating_values.append(list(array))
print('--^Motor Rating--')
for array in RotorDiametersplits:
    print(list(array))
    Rotor_Diameter_values.append(list(array))
print('--^Rotor diameter--')
for array in RPMsplits:
    print(list(array))
    RPM_values.append(list(array))
print('--^RPM--')
for array in ARsplits:
    print(list(array))
    AR_values.append(list(array))
print('--^AR--')
for array in Wing_area_splits:
    print(list(array))
    Wing_area_values.append(list(array))
print('--^Wing Area--')
for array in prop_eta_splits:
    print(list(array))
    Prop_eta_values.append(list(array))
print('--^Prop eta--')
for array in Takeoff_throttle_splits:
    print(list(array))
    Takeoff_Throttle_values.append(list(array))
print('--^Takeoff throttle-')
for array in Battery_weight_splits:
    print(list(array))
    Battery_weight_values.append(list(array))
print('--^Battery Weight-')
for array in OEW_weight_splits:
    print(list(array))
    OEW_weight_values.append(list(array))
print('--^Battery Weight-')
MTOW_values_reshape = np.array(MTOW_values).reshape((len_cruise_speeds, len_payloads))
Motor_Rating_values_reshape = np.array(Motor_Rating_values).reshape((len_cruise_speeds, len_payloads))
Rotor_Diameter_values_reshape = np.array(Rotor_Diameter_values).reshape((len_cruise_speeds, len_payloads))
RPM_values_reshape = np.array(RPM_values).reshape((len_cruise_speeds, len_payloads))
AR_values_reshape = np.array(AR_values).reshape((len_cruise_speeds, len_payloads))
Prop_eta_values_reshape = np.array(Prop_eta_values).reshape((len_cruise_speeds, len_payloads))
Takeoff_Throttle_values_reshape = np.array(Takeoff_Throttle_values).reshape((len_cruise_speeds, len_payloads))
Wing_area_values_reshape = np.array(Wing_area_values).reshape((len_cruise_speeds, len_payloads))
Battery_weight_values_reshape = np.array(Battery_weight_values).reshape((len_cruise_speeds, len_payloads))
OEW_weight_values_reshape = np.array(OEW_weight_values).reshape((len_cruise_speeds, len_payloads))
#print((MTOW_values_reshape.shape))
Battery_MTOW_ratio = Battery_weight_values_reshape/MTOW_values_reshape
#Battery_OEW_ratio = Battery_weight_values_reshape/OEW_weight_values_reshape
print(Battery_MTOW_ratio)
Span = (AR_values_reshape*Wing_area_values_reshape)**0.5
print('--^Span-',Span) #
Chord = Span/AR_values_reshape
print('--^Chord-',Chord) #
#print('--^Battery/OEW-',Battery_OEW_ratio) #
#%%
x = np.array(cruise_speeds);
y = np.array(payloads); #print(y)

X1, Y1 = np.meshgrid(x, y)

fig1, ax1 = plt.subplots()
MTOW_values_reshape = np.flipud(MTOW_values_reshape)
plt.imshow(MTOW_values_reshape, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])

ax1.set_title('MTOW (lb)')
ax1.set_xlabel('Payload (lb)')
ax1.set_ylabel('Cruise speed (mph)')
plt.clim(MTOW_values_reshape.min(),MTOW_values_reshape.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(MTOW)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(MTOW)FixRange60Battary300SpeedVsPayload.png", dpi = 300)
plt.show()
#%%
fig2, ax2 = plt.subplots()
Motor_Rating_values_reshape = np.flipud(Motor_Rating_values_reshape)
plt.imshow(Motor_Rating_values_reshape, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])
ax2.set_title('Single Motor Power (kw)')
ax2.set_xlabel('Payload (lb)')
ax2.set_ylabel('Cruise speed (mph)')
plt.clim(Motor_Rating_values_reshape.min(),Motor_Rating_values_reshape.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Motor)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Motor)FixRange60Battary300SpeedVsPayload.png", dpi = 300)
plt.show()

fig3, ax3 = plt.subplots()
Rotor_Diameter_values_reshape = np.flipud(Rotor_Diameter_values_reshape)
plt.imshow(Rotor_Diameter_values_reshape, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])
ax3.set_title('Rotor Diameter (m)')
ax3.set_xlabel('Payload (lb)')
ax3.set_ylabel('Cruise speed (mph)')
plt.clim(Rotor_Diameter_values_reshape.min(),Rotor_Diameter_values_reshape.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Diameter)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Diameter)FixRange60Battary300SpeedVsPayload.png", dpi = 300)
plt.show()
"""
fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax4.plot_surface(X1, Y1, RPM_values_reshape, cmap = 'viridis_r',antialiased=True)#, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax4.set_title('Propeller RPM')
ax4.set_xlabel('Payload (lb)')
ax4.set_ylabel('Cruise speed (mph)')
ax4.set_zlabel('Rotor RPM in cruise')
ax4.set_xlim(ax4.get_xlim()[::-1])
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(RPM)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(RPM)FixRange30Battary100SpeedVsPayload.png", dpi = 300)

plt.show()


fig5, ax5 = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax5.plot_surface(X1, Y1, AR_values_reshape, cmap = 'viridis_r',antialiased=True)#, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax5.set_title('AR')
ax5.set_xlabel('Payload (lb)')
ax5.set_ylabel('Cruise speed (mph)')
ax5.set_zlabel('AR')
ax5.set_xlim(ax5.get_xlim()[::-1])
ax5.set_ylim(ax5.get_ylim()[::-1])
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(AR)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(AR)FixRange30Battary100SpeedVsPayload.png", dpi = 300)
plt.show()
"""
fig6, ax6 = plt.subplots()
Prop_eta_values_reshape = np.flipud(Prop_eta_values_reshape)
plt.imshow(Prop_eta_values_reshape, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])
ax6.set_title('Prop_eta')
ax6.set_xlabel('Payload (lb)')
ax6.set_ylabel('Cruise speed (mph)')
plt.clim(Prop_eta_values_reshape.min(),Prop_eta_values_reshape.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Prop_eta)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(Prop_eta)FixRange30Battary300SpeedVsPayload.png", dpi = 300)
plt.show()
"""
fig7, ax7 = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax7.plot_surface(X1, Y1, Takeoff_Throttle_values_reshape, cmap = 'viridis_r',antialiased=True)#, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax7.set_title('Takeoff_throttle, Battery density of 100 Wh/kg')
ax7.set_xlabel('Payload (lb)')
ax7.set_ylabel('Cruise speed (mph)')
ax7.set_zlabel('Cruise throttle')
ax7.set_xlim(ax7.get_xlim()[::-1])

#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(TakeoffThrottle)FixRange30Battary100SpeedVsPayload.png", dpi = 300)
plt.show()
"""
"""
#x = x[::-1]
X1, Y1 = np.meshgrid(y, x)
fig8, ax8 = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax8.plot_surface(X1, Y1, Wing_area_values_reshape, cmap = 'viridis_r',antialiased=True)#, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax8.set_title('Wing area')
ax8.set_xlabel('Payload (lb)')
ax8.set_ylabel('Cruise speed (mph)')
ax8.set_zlabel('Wing area (M**2)')
ax8.set_xlim(ax8.get_xlim()[::-1])
ax8.set_ylim(ax8.get_ylim()[::-1])
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(WingArea)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(WingArea)FixRange30Battary100SpeedVsPayload.png", dpi = 300)
plt.show()
"""

fig9, ax9 = plt.subplots()
Battery_weight_values_reshape = np.flipud(Battery_weight_values_reshape)
plt.imshow(Battery_weight_values_reshape, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])
ax9.set_title('Battery weight (lb)')
ax9.set_xlabel('Payload (lb)')
ax9.set_ylabel('Cruise speed (mph)')
plt.clim(Battery_weight_values_reshape.min(),Battery_weight_values_reshape.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(BatteryWeight)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(BatteryWeight)FixRange60Battary300SpeedVsPayload.png", dpi = 300)
plt.show()

fig10, ax10 = plt.subplots()
Battery_MTOW_ratio = np.flipud(Battery_MTOW_ratio)
plt.imshow(Battery_MTOW_ratio, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])
ax10.set_title('Battery/MTOW')
ax10.set_xlabel('Payload (lb)')
ax10.set_ylabel('Cruise speed (mph)')
plt.clim(Battery_MTOW_ratio.min(),Battery_MTOW_ratio.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(BatteryMTOW)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(BatteryMTOW)FixRange60Battary300SpeedVsPayload.png", dpi = 300)
plt.show()

fig11, ax11 = plt.subplots()
OEW_weight_values_reshape = np.flipud(OEW_weight_values_reshape)
plt.imshow(OEW_weight_values_reshape, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])
ax11.set_title('OEW')
ax11.set_xlabel('Payload (lb)')
ax11.set_ylabel('Cruise speed (mph)')
plt.clim(OEW_weight_values_reshape.min(),OEW_weight_values_reshape.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(OEW)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(OEW)FixRange30Battary300SpeedVsPayload.png", dpi = 300)
plt.show()


X1, Y1 = np.meshgrid(y, x)
fig12, ax12 = plt.subplots()

OEW_MTOW = np.flipud(OEW_weight_values_reshape/MTOW_values_reshape)
plt.imshow(OEW_MTOW, aspect='auto', extent =[y.min(),y.max(),x.min(),x.max()])
ax12.set_title('OEW/MTOW')
ax12.set_xlabel('Payload (lb)')
ax12.set_ylabel('Cruise speed (mph)')
plt.clim(OEW_MTOW.min(),OEW_MTOW.max()) 
plt.colorbar()
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(OEWMTOW)FixRange60Battary300SpeedVsPayload.pdf", dpi = 300)
plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(OEWMTOW)FixRange60Battary300SpeedVsPayload.png", dpi = 300)
plt.show()

# MacOS
#plt.savefig("/Users/pcchou/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/FixRange30Battary50SpeedVsPayload.pdf", dpi = 300)
# Window 11
#plt.savefig("C:/Users/Administrator/Dropbox (University of Michigan)/Umich/2021 Fall/AE 590/Report/2022_Winter/(MTOW)FixRange30Battary100SpeedVsPayload.pdf", dpi = 300)
# %%
