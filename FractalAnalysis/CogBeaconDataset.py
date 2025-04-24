import os
import numpy as np # type: ignore
import re
import pandas as pd

from itertools import chain
from collections import defaultdict

# from FractalAnalysis.CogBeacon_PlotUtils import *

class CogBeaconDataset:
    def __init__(self, root_path):
        self.root_path = root_path
        self.eeg_path = os.path.join(root_path, 'eeg')
        # self.eeg_path = '/Users/athenasaghi/VSProjects/CognitiveFatigueDetection/CogFatigueData/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue-master/eeg'
        self.label_path = os.path.join(root_path,'fatigue_self_report')
        # self.label_path = '/Users/athenasaghi/VSProjects/CognitiveFatigueDetection/CogFatigueData/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue-master/fatigue_self_report'
# 

    def parse_folder_name(self, folder_name): # this is for reading the session folder that contains eeg files
        parts = folder_name.split('_')
        user_id = parts[1]
        stimuli_type = parts[2]
        game_mode = parts[3]
        
        if "b" in user_id:
            session_day = "Second"
            wcst_version = "V2" if game_mode == "m" else "o"
            user_id = user_id.replace("b", "")
        else:
            session_day = "First"
            wcst_version = "V1" if game_mode == "m" else "o"
        
        return {
            'user_id': user_id,
            'stimuli_type': stimuli_type,
            'game_mode': game_mode,
            'session_day': session_day,
            'wcst_version': wcst_version
        }

    def parse_file_name(self, file_name): # this for each eeg data in easch session
        parts = file_name.split('_')
        round_id_same_rule = parts[0] # the same ids of the i=first part is the when the rounds in the game have the same rule to follow
        round_id = parts[1].split('.')[0] 
        
        return {
            'round_id_same_rule': round_id_same_rule,
            'round_id': round_id
        }
    
    def read_eeg_file(self, file_path):
        eeg_data = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('eeg'):
                        data = list(map(float, line.split()[1:]))
                        eeg_data.append(data)
            # print("the  size of the eeg data if {} is {}".format(file_path,eeg_data))            

            return eeg_data
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def process_session(self, session_folder):
        session_path = os.path.join(self.eeg_path, session_folder)
        # print("checkpoint A - session Path ", session_path)
        session_info = self.parse_folder_name(session_folder)
        
        eeg_data = []
        
        file_names = os.listdir(session_path)

        def sort_key(file_name): # sorting the filename to make the reading of the data in the order of the expriment time
            match = re.match(r"(\d+)_(\d+)", file_name)
            if match:
                return (int(match.group(1)), int(match.group(2)))  
            return (float('inf'), float('inf')) 
        file_names.sort(key=sort_key)

        for file_name in file_names:
            # print("process session", file_name)
            file_info = self.parse_file_name(file_name)
            file_path = os.path.join(session_path, file_name)
            data = self.read_eeg_file(file_path)
            eeg_data.append(data) 
        # print("there are {} eeg recoding in the session {}".format(len(eeg_data), file_name))   
        return eeg_data
                # eeg_data.append({
                #     'file_info': file_info,
                #     'data': data
                # })
        
        # return {
        #     'session_info': session_info,
        #     'eeg_data': eeg_data
        # }
        # return eeg_data

    def load_all_sessions(self):
        all_sessions = []
        for session_folder in os.listdir(self.eeg_path):
            if os.path.isdir(os.path.join(self.eeg_path, session_folder)):
                task_data = self.process_session(session_folder)
                all_sessions.append(task_data)
        return all_sessions
    
    def load_session_by_participant(self, user_id, session_day,game_mode,channel):
        # target_sessions = []
        for session_folder in os.listdir(self.eeg_path):
            if os.path.isdir(os.path.join(self.eeg_path, session_folder)):
                session_info = self.parse_folder_name(session_folder)
                if session_info['user_id'] == str(user_id) and session_info['session_day'] == session_day and session_info['game_mode']  == game_mode:
                    task_data = self.process_session(session_folder)
                    session_fatiguereport= self.read_fatigue_self_report(session_folder)
                    print("session loaded successfully",session_info,"size of the session data", len(task_data))
        if not task_data:
                print("file not found!!!!",session_info['user_id'] , str(user_id),session_info['session_day'] , session_day)
        if channel:
            task_data = self.separet_channels_per_turn(task_data)[channel]
        return task_data , session_fatiguereport
    
    
    def read_fatigue_self_report(self,file_path):
        path = os.path.join(self.label_path, file_path+".csv")
        fatigue_reports = pd.read_csv(path,header=None)
        fatigue_list = fatigue_reports.values.tolist()
        fatigue_list = [label[0] if isinstance(label, (list, np.ndarray)) else label for label in fatigue_list]
        return fatigue_list
                       

    def separet_channels_per_turn(self,signal):

        eeg_per_channel_per_channel = {
            "TP9":[],
            "AF7":[],
            "AF8":[],
            "TP10":[]
            
        }
        for turn in signal:
            TP9, AF7,AF8,TP10 = [],[],[],[]
            for rnd in turn:
                TP9.append(rnd[0])
                AF7.append(rnd[1])
                AF8.append(rnd[2])
                TP10.append(rnd[3])
            eeg_per_channel_per_channel["TP9"].append(TP9)
            eeg_per_channel_per_channel["AF7"].append(AF7)
            eeg_per_channel_per_channel["AF8"].append(AF8)
            eeg_per_channel_per_channel["TP10"].append(TP10)
        
        return eeg_per_channel_per_channel
    

    def flattern_turns(self,task_data,task_label):
        """
        This function puts together the whole task turn in a one temporal sequence (all turn of a task in a 1d signal)

        """
        # def flatten_chain(matrix):
        #     return list(chain.from_iterable(matrix))
        flattened_task_data = []
        extended_labels =[]
        for sublist,label in zip(task_data,task_label):
            flattened_task_data.extend(sublist)
            extended_labels.extend([label] * len(sublist))

        # print(type(flattened_task_data), flattened_task_data)
        return flattened_task_data,extended_labels
            

