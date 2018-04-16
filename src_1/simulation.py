#!/usr/bin/env python

import os
import sys
import requests
import numpy as np
from itertools import groupby
import os.path
from datetime import date
import bisect
import math
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from random import randint
import timeit

a=np.mat('1,2,3;4,5,6')
b=np.array([[1,2,3],[4,5,6]])
null=[]
RouteInx=[	"	200	",	"	201	",	"	202	",	"	203	",	"	204	",	"	205	",	"	206	",	"	207	",
	"	208	",	"	209	",	"	300	",	"	301	",	"	302	",	"	303	",	"	304	",	"	305	",	"	400	",
	"	401	",	"	402	",	"	500	",	"	501	",	"	502	",	"	503	",	"	504	",	"	505	",	"	506	",
	"	507	",	"	508	",	"	600	",	"	601	",	"	602	",	"	603	",	"	604	",	"	700	",	"	V94	",
	"	701	",	"	702	",	"	703	",	"	704	",	"	705	",	"	706	",	"	707	",	"	800	",	"	801	",
	"	803	",	"	804	",	"	805	",	"	806	",	"	900	",	"	901	",	"	902	",	"	903	",	"	904	",
	"	905	",	"	906	",	"	907	",	"	103	",	"	104	",	"	1	",	"	18	",	"	22	",	"	1M	",
	"	3M	",	"	4M	",	"	5M	",	"	7M	",	"	8M	",	"	9M	",	"	10M	",	"	11M	",	"	12M	"]
    
#remove the whitespace in string fromt front and back
RouteInx=[el.strip() for el in RouteInx]


''''

'''

bound_left = -8.661915
bound_right = -8.559543
bound_top = 41.185110
bound_bot = 41.136044


#define bus info extraction function


try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), "..","..","..","..","sumo-0.28-all-src","tools"))  # tutorial in tests
    
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..","tools"))))  # tutorial in docs
    
    print(__file__)
    import sumolib
    from sumolib import checkBinary
    
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci



def euclidean(vector1, vector2):
    '''calculate the euclidean distance, no numpy
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist





#'passenger_87325': 
#[['800', 'BLAM1', 'PDR', 51312], ['305', 'PDR', 'ASP4', 52209]], 'passenger_152955': [['600', 'AAL2', 'AML1', 70756], ['205', 'AML3', 'VNC2', 71750]]



def estimation_in_day(recordInDay, station_xy_dict, OnLineStruct):
    distance = 500#400#640#1000#640
    recordCopy = recordInDay[:]
    ini_flag = 0
    record_w_destination=[]
    
    record_without_destination=[]
    origin ='NONE'
    
    while recordCopy != []:
        Ele_record = recordCopy.pop(0)
        route = Ele_record[0]
        if Ele_record not in recordCopy:# Not a duplicate record
            
            if Ele_record[1] == 'AL1':
                Ele_record[1] = OnLineStruct[Ele_record[0]][0]
                
            if Ele_record[2] == 'AL1':
                Ele_record[2] = OnLineStruct[Ele_record[0]][-1]
                
            
            if Ele_record[1] == '1APS3':
                Ele_record[1] = 'FEL1'
                
            if Ele_record[2] == '1APS3':
                Ele_record[2] = 'FEL1'
            
            
            if Ele_record[1] == 'ARS5' and route =='603_rev':
                Ele_record[1] = 'HSJ12'
            
            if Ele_record[2] == 'ARS5' and route =='603_rev':
                Ele_record[2] = 'HSJ12'
              
            if Ele_record[2] == 'ENX2' and route =='603_rev':
                Ele_record[2] = 'FEL2'
                
            if Ele_record[2] == 'ASP4' and route =='603_rev':
                Ele_record[2] = 'IPO5'
                
            
            if not (recordCopy ==[] and ini_flag ==0):# Not single daily trip
                # until now, this trip is a valid trip
                route = Ele_record[0]
                origin = Ele_record[1]
                if ini_flag ==0:
                    ini_flag = 1
                    daily_origin = Ele_record[1]
                
                heading =[]
                if recordCopy ==[]:# last trip stage of a day?
                    if len(recordInDay)<3:#3:
                        heading = daily_origin
                else:
                    if recordCopy[0][1] != '1APS3':
                        heading = recordCopy[0][1] # set heading as next trip's origin
                    else:
                        heading ='FEL1'
                
                
                #print(origin,route,OnLineStruct[route])
                if origin in OnLineStruct[route]:
                
                    inx = OnLineStruct[route].index(origin)
                    candidate_set = OnLineStruct[route][inx:]
                    if len(candidate_set)>1 and heading!=[]: # remove boarding at terminal illogical samples
                        candidate_set=candidate_set[1:] # down stream stations
                        if not (heading == origin or '' in Ele_record):
                            #remove illogical data
                            destination = 'NONE'
                            if heading in candidate_set:
                                destination = heading
                            else:
                                pre_dist = float("inf")
                                #print(heading,origin,route,OnLineStruct[route])
                                for candidate_ele in candidate_set:
                                    dist = euclidean(station_xy_dict[heading], station_xy_dict[candidate_ele])
                                    if dist <=distance and dist <pre_dist:
                                        pre_dist = dist
                                        destination = candidate_ele
                        
                        
                            if destination != 'NONE':
                                Ele_record.append(destination)
                            
                                #print(Ele_record)
                                inx_on = OnLineStruct[route].index(Ele_record[1])
                                inx_off = OnLineStruct[route].index(Ele_record[2])
                                inf_off = OnLineStruct[route].index(destination)
                            
                                inx_on_name = Ele_record[1]
                                inx_off_name = Ele_record[2]
                                inf_off_name = destination
                            
                            
                                modified_record = [Ele_record[3],Ele_record[0],inx_on_name, inf_off_name, inx_on, inf_off, inx_off_name, inx_off]#[on_time, route_ID, on-stop_ID, off-stop_ID, inx_on, inf_off, true-off-ID,true-off-inx]
                                record_w_destination.append(modified_record)
                            else:
                                if '' not in Ele_record and Ele_record[1] in OnLineStruct[route] and Ele_record[2] in OnLineStruct[route]:
                                    #x1,y1 = station_xy_dict[Ele_record[1]]
                                    #modified_record = [Ele_record[5],x1,y1]
                                    inx_on = OnLineStruct[route].index(Ele_record[1])
                                    inx_off = OnLineStruct[route].index(Ele_record[2])
                                
                                    inx_on_name = Ele_record[1]
                                    inx_off_name = Ele_record[2]
                                
                                    modified_record = [Ele_record[3],Ele_record[0],inx_on_name,inx_on,inx_off_name, inx_off]#[on_time, route_ID, on-stop_ID, inx_on, true-off-ID, true-off-inx]
                                    record_without_destination.append(modified_record)
                            
                    else:
                        if '' not in Ele_record and Ele_record[1] in OnLineStruct[route] and Ele_record[2] in OnLineStruct[route]:
                            #x1,y1 = station_xy_dict[Ele_record[1]]
                            #modified_record = [Ele_record[5],x1,y1]
                            inx_on = OnLineStruct[route].index(Ele_record[1])
                            inx_off = OnLineStruct[route].index(Ele_record[2])
                        
                            inx_on_name = Ele_record[1]
                            inx_off_name = Ele_record[2]
                        
                            modified_record = [Ele_record[3],Ele_record[0],inx_on_name,inx_on,inx_off_name, inx_off]#[on_time, route_ID, on-stop_ID, inx_on, true-off-ID, true-off-inx]
                            record_without_destination.append(modified_record)
                
            
            else:
                if '' not in Ele_record and Ele_record[1] in OnLineStruct[route] and Ele_record[2] in OnLineStruct[route]:
                    #x1,y1 = station_xy_dict[Ele_record[1]]
                    #modified_record = [Ele_record[5],x1,y1]
                    inx_on = OnLineStruct[route].index(Ele_record[1])
                    inx_off = OnLineStruct[route].index(Ele_record[2])
                    
                    inx_on_name = Ele_record[1]
                    inx_off_name = Ele_record[2]
                    
                    modified_record = [Ele_record[3],Ele_record[0],inx_on_name,inx_on,inx_off_name, inx_off]#[on_time, route_ID, on-stop_ID, inx_on, true-off-ID, true-off-inx]
                    record_without_destination.append(modified_record)
                
                            
    
    
    
    return record_w_destination,record_without_destination




#'passenger_87325': 
#[['800', 'BLAM1', 'PDR', 51312], ['305', 'PDR', 'ASP4', 52209]], 'passenger_152955': [['600', 'AAL2', 'AML1', 70756], ['205', 'AML3', 'VNC2', 71750]]





def DestinationEstimate():
    
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']
    
    OnLineSturct = {}
    for ele in Bus_Info_All:
        OnLineSturct.update({ele['name']:ele['sequence']})
    print(OnLineSturct['502'])
    
    ############################### map station geolocation to SUMO x, y #########################################################
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']
    stationDir=busInfo['stationDir'][0]
    
    
    net =sumolib.net.readNet("../osm.net.xml")
    station_xy_dict={}
    for tempStation in stationDir:
        geo= eval(stationDir[tempStation]['geomdesc'])
        geo =geo['coordinates']
        
        if True:#bound_left < geo[0] < bound_right and bound_bot < geo[1] < bound_top:
            x = geo[0]
            y = geo[1]
            [x, y] = net.convertLonLat2XY(x, y)
            station_xy_dict.update({tempStation:[x, y]})

    print(len(station_xy_dict))
    
    
    
    
    ############################### main function as follows:  #########################################################
    simulationRecord = np.load('simulatedData_Wed.npz')
    simulatedData_Wed = simulationRecord['simulatedData_Wed'][0]
    print(len(simulatedData_Wed))
    
    
    
    estimation_completedData = []#{}
    estimation_IncompletedData = []#{}
    count = 0
    
    
    for day in simulatedData_Wed:
        
        dateOfRecord = simulatedData_Wed[day]
        
        
        for person in dateOfRecord:
            
            
            record_w_destination,record_without_destination = estimation_in_day(dateOfRecord[person], station_xy_dict, OnLineSturct)
            
            
            if count ==0 and len(record_w_destination)>0:
                #count =1
                print(person)
                print(record_w_destination)
                
            if count ==0 and len(record_w_destination)>0:
                count =1
                print(person)
                print(record_without_destination)
                
            
            if record_w_destination!=[]:
                estimation_completedData+=record_w_destination
                
                    
                 
            if record_without_destination!=[]:
                estimation_IncompletedData+=record_without_destination
                
    
    count = 0
    error =0
    for item in estimation_completedData:
        if item[5]==item[7]:
            count+=1
        
        error+=(item[5]-item[7])**2
    
    
    print(count*1.0/len(estimation_completedData),error*1.0/len(estimation_completedData))
            
    print(len(estimation_completedData),len(estimation_IncompletedData))
    np.savez('completedData_Incomplete_code_simulated.npz', estimation_completedData=[estimation_completedData],estimation_IncompletedData = [estimation_IncompletedData])
    
    
    


    
    
    
    

def validate():
    estimation_completedData = np.load('completedData_Incomplete_code_simulated.npz')
    estimation_completedData =estimation_completedData['estimation_completedData'][0]
    
    count = 0
    error =0
    for item in estimation_completedData:
        if item[5]==item[7]:
            count+=1
        
        error+=(int(item[5])-int(item[7]))**2
    
    print(count,len(estimation_completedData))
    print("First order inference accuracy:",count*1.0/len(estimation_completedData),"Variance:",error*1.0/len(estimation_completedData))
    







def sample_index(distr_list):
    epsilon = random.uniform(0, 1)
    point = epsilon * np.sum(distr_list)
    
    if sum(distr_list)>0:
        cumulative = 0
        index =0
        for i in range(len(distr_list)):
            cumulative += distr_list[i]
            if point <= cumulative:
                index = i
                break
        return index
    else:
        return []
    




def fill_destination():
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']
    
    OnLineStruct = {}
    for ele in Bus_Info_All:
        OnLineStruct.update({ele['name']:ele['sequence']})
    print(OnLineStruct['502'])
    
    
    #DataOnWed = np.load('../DataOnWed_coded.npz')
    #DataOnWed_test_5D = DataOnWed['DataOnWed_test_5D'][0]
    
    estimation_IncompletedData = np.load('completedData_Incomplete_code_simulated.npz')
    estimation_IncompletedData = estimation_IncompletedData['estimation_IncompletedData'][0]
    
    
    Off_distr_dict = np.load('Off_distr_dict_simulate.npz')
    Off_distr_dict = Off_distr_dict['Off_distr_dict'][0]
    
    first_row = estimation_IncompletedData[0] #[on_time, route_ID, on-stop_ID, inx_on, true-off-ID, true-off-inx]
    d = len(first_row)
    num = len(estimation_IncompletedData)
    print(d,num)
    print(first_row)
    

    timeSeg = range(0,25)#[0,4,8,12,16,20,24]
    timeTag = [str(i-1)+'-'+str(i) for i in timeSeg[1:]]
    timeSeg = [i * 3600 for i in timeSeg]
    timeSeg[0]=-1
    timeSeg[-1]+=2*3600
    print(timeTag)
    print(timeSeg)
    
    fail_count = 0
    DataOnWed_test_inf_for_recon = []
    error_inf =0
    error_rand = 0
    
    
    for i in range(num):
        ele_record = estimation_IncompletedData[i]
        record_time = int(ele_record[0])
        route_ID = ele_record[1]
        inx_on = int(ele_record[3])
        
        true_off = int(ele_record[5]) #ground-true off stop index
        
        seg=bisect.bisect_left(timeSeg, record_time)
        tag=timeTag[seg-1]
        
        if route_ID not in ['505','604','705','700_rev']:
            key_name = route_ID+'_'+tag
            temp_distr = Off_distr_dict[key_name][inx_on,:]
        
        
            if True:#random.uniform(0, 1)>=1:
                inf_off = sample_index(temp_distr)
            else:
                temp_distr =temp_distr.tolist()
                if max(temp_distr)>0:
                    inf_off = temp_distr.index(max(temp_distr))
                else:
                    inf_off = []
        
        
        
            epsilon = random.uniform(0, 1)
            rand_off = int(epsilon * len(temp_distr))
        
            if inf_off != []:
                dest_name = OnLineStruct[route_ID][inf_off]
                temp_filled = [record_time, route_ID, ele_record[2],dest_name, inx_on, inf_off]
                DataOnWed_test_inf_for_recon.append(temp_filled)
            
                error_inf += (inf_off-true_off)**2
                error_rand += (rand_off-true_off)**2
            
            
            
            else:
                fail_count += 1
        else:
            fail_count += 1
        
        
    
    num_inf = len(DataOnWed_test_inf_for_recon)
    print('error_inf:',(1.0*error_inf)/num_inf)
    print('error_rand:',(1.0*error_rand)/num_inf)
    
    
    print(fail_count)
    #np.savez('DataOnWed_train_filled_3Dto5D.npz', DataOnWed_train_filled_3Dto5D=[DataOnWed_train_filled_3Dto5D])

    


#[['57492' '300' 'DC1' '10' 'MAV2' '12']
#['59112' '203' 'MPL3' '0' 'FG1' '1']
# ['57821' '400' 'SBNT2' '2' 'FRX3' '13']

def Off_distr_dict_6min():
    """
    The off_distr_dict is a dictionary in the format of {routeID_hour1:off_distr_matrix, routeID_hour2:off_distr_matrix,...}
    The off_distr_matrix count the off bus record at a certain hour of a route, the row is the boarding bus stop index, and the column is the off bus stop index
    the hour1 hour2.... are denoted as '0-1', '1-2', ..., '23-24'.
    we have 112*24 = 2688 keys in total
    
    5D data example
    ['61148' '302' 'C24A3' 'CMO' '0' '6']
    """
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']
    OnLineStruct = {}
    for ele in Bus_Info_All:
        OnLineStruct.update({ele['name']:ele['sequence']})
    
    
    
    
    estimation_completedData = np.load('completedData_Incomplete_code_simulated.npz')
    estimation_completedData = estimation_completedData['estimation_completedData'][0]
    
    
    
    first_row = estimation_completedData[0]
    d = len(first_row)
    num = len(estimation_completedData)
    
    
    #===================================initial the Off_distr_dict with all zaro matrix
    routeNameList_train = []
    zero_matrix_for_route = {}
    for i in range(num):
        ele_record = estimation_completedData[i]
        if ele_record[1] not in routeNameList_train:
            routeNameList_train.append(ele_record[1])
            route_len = len(OnLineStruct[ele_record[1]])
            temp_matrix = np.zeros((route_len,route_len))
            temp_matrix = temp_matrix
            zero_matrix_for_route.update({ele_record[1]:temp_matrix})
            
    #print(zero_matrix_for_route['604'].shape)
    print(len(routeNameList_train))
    
    Off_distr_dict = {}
    
    timeSeg = range(0,261)#[0,4,8,12,16,20,24]
    timeTag = [str(i-1)+'-'+str(i) for i in timeSeg[1:]]
    timeSeg = [i * 360 for i in timeSeg]
    timeSeg[0]=-1
    #timeSeg[-1]+=2*3600
    
    
    for ele_route in routeNameList_train:
        temp_dict = {}
        for ele_hour in timeTag:
            ppt = zero_matrix_for_route[ele_route]
            temp_dict.update({ele_hour:ppt})
            
            key_name = ele_route+'_'+ele_hour
            
            Off_distr_dict.update({key_name:ppt})
    
    #print(Off_distr_dict['200'])
    print(len(Off_distr_dict))
    
    #============================count the records in the matrix framework for sampling purpose
    
    for i in range(num):
        #print(i)
        route_ID = []
        seg = []
        
        ele_record = estimation_completedData[i]
        record_time = int(ele_record[0])
        route_ID = ele_record[1]
        inx_on = int(ele_record[4])
        inx_off = int(ele_record[5])
        
        
        seg=bisect.bisect_left(timeSeg, record_time)
        tag=timeTag[seg-1]
        
        temp_array=[]
        key_name = route_ID+'_'+tag
        temp_array = Off_distr_dict[key_name]
        temp_array2 = np.copy(temp_array)
        
        temp_array2[inx_on][inx_off] +=1
        Off_distr_dict.update({key_name:temp_array2})
        
        
    np.savez('Off_distr_dict_6min.npz', Off_distr_dict=[Off_distr_dict])
    
    
#============================================================================Iteration===============================================================



def in_step_Off_distr_dict(in_step_distr_dict,in_step_inference,timeSeg,timeTag):
    
    first_row = in_step_inference[0]
    d = len(first_row)
    num = len(in_step_inference)
    
    Off_distr_dict = in_step_distr_dict
    
    for i in range(num):
        
        route_ID = []
        seg = []
        
        ele_record = in_step_inference[i]##5D data example:    ['45865' '207' 'CMP1' 'JM1' '0' '16', true-off-name, true-off-inx]
        record_time = int(ele_record[0])
        route_ID = ele_record[1]
        inx_on = int(ele_record[4])
        inx_off = int(ele_record[5])
        
        
        seg=bisect.bisect_left(timeSeg, record_time)
        tag=timeTag[seg-1]
        
        temp_array=[]
        key_name = route_ID+'_'+tag
        temp_array = Off_distr_dict[key_name]
        temp_array2 = np.copy(temp_array)
        
        temp_array2[inx_on][inx_off] +=1
        Off_distr_dict.update({key_name:temp_array2})
        
        
    
    return Off_distr_dict







def in_step_fill_destination(in_step_distr_dict,incomplete_data_fold,OnLineStruct,selected_volume):
    
    
    Off_distr_dict = in_step_distr_dict
    
    first_row = incomplete_data_fold[0] #5D data example:  ['57492' '300' 'DC1' '10' 'MAV2' '12'] ====> !!!!not:  ['45865' '207' 'CMP1' 'JM1' '0' '16']
    d = len(first_row)
    num = len(incomplete_data_fold)
    print(d,num)
    print(first_row)
    
    timeSeg = range(0,261)#[0,4,8,12,16,20,24]
    timeTag = [str(i-1)+'-'+str(i) for i in timeSeg[1:]]
    timeSeg = [i * 360 for i in timeSeg]
    timeSeg[0]=-1
    
    fail_count = 0
    DataOnWed_train_filled_3Dto5D = []
    still_unknow_data = []
    filleSet_w_prob = []
    
    half_bw = 10*360#2000*360#10*360
    
    for i in range(num):
        ele_record = incomplete_data_fold[i]
        record_time = int(ele_record[0])
        route_ID = ele_record[1]
        
        inx_on = int(ele_record[3])
        
        if route_ID not in ['505','604','705','700_rev']:
            
            
            if record_time - half_bw<0:
                seg_left = 1
                tag_left = timeTag[seg_left-1]
                #seg_right = bisect.bisect_left(timeSeg, record_time + half_bw)
                #tag_right = timeTag[seg_right-1]
            else:
                seg_left = bisect.bisect_left(timeSeg, record_time - half_bw)
                tag_left = timeTag[seg_left-1]
                
            
            if record_time + half_bw>93600:
                #seg_left = bisect.bisect_left(timeSeg, record_time - half_bw)
                #tag_left = timeTag[seg_left-1]
                seg_right = 260
                tag_right = timeTag[seg_right-1]
            else:
                seg_right = bisect.bisect_left(timeSeg, record_time + half_bw)
                tag_right = timeTag[seg_right-1]
                
            
        
        
        
            route_len=len(OnLineStruct[route_ID])
            accumulate_distr_list = np.zeros((1,route_len))
            accumulate_distr_list = accumulate_distr_list[0]
            for piece in range(seg_left-1,seg_right):
                tag=timeTag[piece]
                key_name = route_ID+'_'+tag
                temp_distr = Off_distr_dict[key_name][inx_on,:]
                accumulate_distr_list = np.add(accumulate_distr_list, temp_distr)
        
        
            accumulate_distr_list = accumulate_distr_list.tolist()
            inf_off = accumulate_distr_list.index(max(accumulate_distr_list))
            
            
            if accumulate_distr_list[inf_off] != 0:
                dest_name = OnLineStruct[route_ID][inf_off]
                temp_filled = [record_time, route_ID, ele_record[2],dest_name, inx_on, inf_off,ele_record[4],ele_record[5]] #true destination name and index are the last two
                probability = accumulate_distr_list[inf_off]*1.0/sum(accumulate_distr_list)
                filleSet_w_prob.append([temp_filled,probability])
                
                #DataOnWed_train_filled_3Dto5D.append(temp_filled)
            
            else:
                fail_count += 1
                #still_unknow_data.append(ele_record.tolist())
                
                still_unknow_data.append([ele_record[0],ele_record[1],ele_record[2],ele_record[3],ele_record[4],ele_record[5]])
            
            """    
            if sum(accumulate_distr_list) ==0:
                inf_off = randint(inx_on+1, len(OnLineStruct[route_ID])-1)
                dest_name = OnLineStruct[route_ID][inf_off]
                temp_filled = [record_time, route_ID, ele_record[2],dest_name, inx_on, inf_off,ele_record[4],ele_record[5]] #true destination name and index are the last two
                probability = 0.0
                filleSet_w_prob.append([temp_filled,probability])
            """
            
                
        else:
            fail_count += 1
            #still_unknow_data.append(ele_record.tolist())
            still_unknow_data.append([ele_record[0],ele_record[1],ele_record[2],ele_record[3],ele_record[4],ele_record[5]])
        
            
    filleSet_w_prob.sort(key=lambda x: x[1], reverse=True)
    
    temp_probability=0
    
    for j in range(len(filleSet_w_prob)):
        temp_record = filleSet_w_prob[j]
        if j<selected_volume-1:
            DataOnWed_train_filled_3Dto5D.append(temp_record[0])
            temp_probability=temp_record[1]
        else:
            ele_temp =temp_record[0]
            still_unknow_data.append([ele_temp[0],ele_temp[1],ele_temp[2],ele_temp[4],ele_temp[3],ele_temp[5]])        
            
        
    print("tail probability: ",temp_probability)
    
    inferred_data = DataOnWed_train_filled_3Dto5D
    remain_point = still_unknow_data
    
    return inferred_data, still_unknow_data, fail_count, remain_point
    
    
  
    


    
def inference_validate():
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']
    
    OnLineStruct = {}
    for ele in Bus_Info_All:
        OnLineStruct.update({ele['name']:ele['sequence']})
    
    
    
    estimation_IncompletedData = np.load('completedData_Incomplete_code_simulated.npz')
    estimation_IncompletedData = estimation_IncompletedData['estimation_IncompletedData'][0]
    
    
    Off_distr_dict = np.load('Off_distr_dict_6min.npz')
    Off_distr_dict = Off_distr_dict['Off_distr_dict'][0]
    
    first_row = estimation_IncompletedData[0] #[on_time, route_ID, on-stop_ID, inx_on, true-off-ID, true-off-inx]
    d = len(first_row)
    num = len(estimation_IncompletedData)
    
    
    
    
    
    timeSeg = range(0,261)#[0,4,8,12,16,20,24]
    timeTag = [str(i-1)+'-'+str(i) for i in timeSeg[1:]]
    timeSeg = [i * 360 for i in timeSeg]
    timeSeg[0]=-1
    
    
    
    fail_count = 0
    DataOnWed_test_inf_for_recon = []
    error_inf =0
    error_rand = 0
    true_count =0
    true_random_count = 0
    
    #==========================initialize recon_data_set and incomplete_dataset===========================================
    
    
    in_step_distr_dict = Off_distr_dict # initialize the distribution disctionary
    #in_step_recon_data = DataOnWed['DataOnWed_train_5D'][0]#5D data example:    ['45865' '207' 'CMP1' 'JM1' '0' '16']
    in_step_incomplete_data = estimation_IncompletedData
    
    in_step_inference = []
    inferenced_chunk = []
    num = len(in_step_incomplete_data)
    
    
    iter_num = 200#100#300
    fold_size = int(num/iter_num)
    
    folds = []
    for i in range(iter_num):
        if (i+1) != iter_num:
            temp_fold = in_step_incomplete_data[i*fold_size:(i+1)*fold_size,:]
        
        if (i+1) == iter_num:
            temp_fold = in_step_incomplete_data[i*fold_size:,:]
        
        folds.append(temp_fold)
    
   
    
    
    
    selected_point = folds[0]#initial
    remain_point = in_step_incomplete_data
    selected_volume = fold_size
    iter = 0
    
    pre_chunk_num = 0
    
    
    #===========================iteration start================================
    #for iter in range(iter_num):
    start = timeit.default_timer()
    
    while len(selected_point) >= selected_volume-1:
            
        #==============update distribution dictionary w. newly inferred data fold
        if iter>0:
            in_step_distr_dict = in_step_Off_distr_dict(in_step_distr_dict,in_step_inference,timeSeg,timeTag)
            
        #==============with the updated distribution dictinary, infer next fold
        
        #in_step_inference, still_unknow_data, fail_count = in_step_fill_destination(in_step_distr_dict,folds[iter],OnLineStruct)
        in_step_inference, still_unknow_data, fail_count, remain_point = in_step_fill_destination(in_step_distr_dict,remain_point,OnLineStruct,selected_volume)
        inferenced_chunk+=in_step_inference
        
        """
        if iter<iter_num-1:#not the final iteration
            temp_fold = folds[iter+1]
            #print(temp_fold,still_unknow_data)
            
            folds[iter+1] = np.asarray(temp_fold.tolist()+still_unknow_data)
        """
        remain_point = np.asarray(remain_point+[])
        
        print('Iteration:',iter,len(inferenced_chunk),len(inferenced_chunk)*1.0/num)
        iter+=1
        
        error_inf=0
        true_count =0
        for reconstruct in inferenced_chunk:#5D data example:    ['45865' '207' 'CMP1' 'JM1' '0' '16', true-off-name, true-off-inx]
            inf_off = int(reconstruct[5])
            true_off= int(reconstruct[7])
            
            error_inf += (inf_off-true_off)**2
            if inf_off-true_off==0:
                true_count += 1
        
        
        
            
        print('error_inf:',(1.0*error_inf)/len(inferenced_chunk),'accuracy_inf (in unlabeled set):',(1.0*true_count)/len(inferenced_chunk))
        #print('error_inf:',(1.0*error_inf)/num_inf,'accuracy_inf:',(1.0*true_count)/num_inf)
        
        if pre_chunk_num ==len(inferenced_chunk):
            print("***")
            print('error_inf:',(1.0*error_inf)/len(inferenced_chunk),'total_accuracy_inf(in both labeled and unlabeled set):',(1.0*true_count)/num)
            print("***")
            break
        
        
        
        pre_chunk_num = len(inferenced_chunk)
            
    stop = timeit.default_timer()
    print ("running time:",stop - start) 
    #np.savez('simulated_iteration_inferchunk2.npz', DataOnWed_train_filled_3Dto5D=[inferenced_chunk])
    
    
    

    
def make_standard_dataset():
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']

    DataOnWed_train_filled_3Dto5D = np.load('simulated_iteration_inferchunk.npz')
    DataOnWed_train_filled_3Dto5D = DataOnWed_train_filled_3Dto5D['DataOnWed_train_filled_3Dto5D'][0]#data format: [record_time, route_ID, origin_name, dest_name, inx_on, inf_off]

    stationDir = busInfo['stationDir'][0]
    net = sumolib.net.readNet("../osm.net.xml")
    station_xy_dict = {}
    station_of_interest = []
    for tempStation in stationDir:
        geo = eval(stationDir[tempStation]['geomdesc'])
        geo = geo['coordinates']
        
        if True:#bound_left < geo[0] < bound_right and bound_bot < geo[1] < bound_top:
            x = geo[0]
            y = geo[1]
            [x, y] = net.convertLonLat2XY(x, y)
            station_xy_dict.update({tempStation:[x, y]})
            
        if bound_left < geo[0] < bound_right and bound_bot < geo[1] < bound_top:
            station_of_interest.append(tempStation)

    print(len(station_xy_dict))
    print(len(station_of_interest))
    print(DataOnWed_train_filled_3Dto5D[3])
    
    
    DataOnWed_xy = []
    for ele_record in DataOnWed_train_filled_3Dto5D:
        O_x, O_y = station_xy_dict[ele_record[2]]
        D_x, D_y = station_xy_dict[ele_record[3]]
        temp_point = [int(ele_record[0]),O_x, O_y, D_x, D_y]
        DataOnWed_xy.append(temp_point)
        
    print(len(DataOnWed_xy))
    print(DataOnWed_xy[9])
    np.savez('simulated_iteration_inferchunk.npz', DataOnWed_xy=[DataOnWed_xy])

    DataOnWed_xy = np.load('simulated_iteration_inferchunk.npz')
    DataOnWed_xy = DataOnWed_xy['DataOnWed_xy'][0]
    print(DataOnWed_xy[9])
    









if __name__ == "__main__":
    Bus_Info_All=[]
    
    DestinationEstimate()
    print('saved completedData_Incomplete_code_simulated.npz')
    print('This is the first order inference based on heuristic method: the current destination should be the stop which is closest to the next origin')
    print('=====================step one complete==================')
    
    print('')
    print('')
    print('')
    
    validate()
    print('validate the accuracy of first order inference')
    print('=====================step two complete==================')
    
    print('')
    print('')
    print('')
    
    
    #=========================
    Off_distr_dict_6min()
    print('saved Off_distr_dict_6min.npz')
    print('Alighting distribution in every 6 minutes time window')
    print('=====================step Three complete==================')
    
    print('')
    print('')
    print('')
    inference_validate()
    print('Validate the second order inference accuracy')
    print('=====================step Four complete==================')
    
    #make_standard_dataset()
    
    

