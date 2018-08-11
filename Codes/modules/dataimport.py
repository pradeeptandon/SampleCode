# -*- coding: utf-8 -*-
"""
Spyder Editor

author: bm30785

Used to import diffrent types of files into panda dataframes.
"""

import pandas as pd
import zipfile
import glob
import os
from time import time

class DataImport:
    def __init__(self, folder_path=(), ext_list=()):
        self.folder_path = folder_path
        self.ext_list = ext_list
        self.t0 = time()
        
    def _get_files (self, folder_path = "Default", ext_list = ["txt","csv","xlsx","zip"]):
        if folder_path =="Default":
            folder_path = os.getcwd()
        doc_list = []
        for ext in ext_list:
            doc_list += glob.glob(folder_path +"/*."+ext)
        return doc_list
    
    def _read_xlsx(self, file, file_format = "xlsx"):
        df = pd.read_excel(file)
        return df
    
    def _read_csv(self, file, file_format = "csv"):
        df = pd.read_csv(file)
        return df

    def _read_zip(self, file, file_format="zip"):
        with zipfile.ZipFile(file) as zip:
            with zip.open(file.replace(self.folder_path + "/","").replace(".zip",".csv")) as myZip:
                df = pd.read_csv(myZip, header=None)
        return df

    def _add_ID(self, input_list, var1_name = "ID", var2_name = "Var_Name"):
        return pd.DataFrame({var1_name: range(len(input_list)), var2_name: input_list})
    
    def read_file(self):
        df = self._add_ID(self._get_files(self.folder_path, self.ext_list), var1_name = "DocID", var2_name = "DocName")
        all_data = pd.DataFrame()
        doc = pd.DataFrame(columns=["DocID","DocName","content"])
        print ("Importing " + str(len(df)) + str(self.ext_list) + " files") 
        for index, row in df.iterrows():
            if ("zip" in self.ext_list):
                doc = self._read_zip(row["DocName"])
                print ("... " + row["DocName"])
            elif ("xlsx" in self.ext_list):
                if row["DocName"][:2] != "~$": #Exclude hidden files which starts with ~$
                    doc = self._read_xlsx(row["DocName"])
            elif ("csv" in self.ext_list):
                if row["DocName"][:2] != "~$": #Exclude hidden files which starts with ~$
                    doc = self._read_csv(row["DocName"])
            else:
                print("File Format not allowed")
            doc["DocID"] = row["DocID"]
            doc["DocName"] = row["DocName"]
            all_data = all_data.append(doc, ignore_index = True)
        print("Done in %0.3fs." % (time() - self.t0))
        return all_data
  

