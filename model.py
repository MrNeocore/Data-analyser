import tkinter as tk
from tkinter import ttk
import pandas as pd
import utils
import psutil
import os
import threading
import time
from math import ceil
from tqdm import tqdm
import random
import numpy as np
import user_agents as ua
from graph import *
from subprocess import run

class Model:
    """ Model class handing data loading, modifications and extraction """ 
    def __init__(self, iface):
        self.iface = iface
        self.file_loaded = False
        self.tk_gui= hasattr(self.iface, 'root') and isinstance(self.iface.root, tk.Tk)
        
        self.filename = None
        self.data = pd.DataFrame()
        
        try:
            self.country_codes = pd.read_csv(utils.COUNTRY_CODES_FILE, na_filter = False)
        except FileNotFoundError:
            raise utils.MissingFileError("Country file {0} not found".format(utils.COUNTRY_CODES_FILE))
    
    def ram_usage(self):
        """ Return the current process RAM usage in MB"""
        proc = psutil.Process(os.getpid())
        return "Ram : {0} MB |".format(round(proc.memory_info()[0]/(1024*1024), 1))
        
    
    def check_file_validity(self, filename):
        """ Description : Check if a given file is a valid Issuu JSON file
            Parameters  : Filename to check"""
        try:
            data = next(pd.read_json(filename, lines=True, chunksize=1))
        except ValueError:
            return False
        if set(['visitor_uuid','visitor_country', 'visitor_useragent', 'subject_doc_id']) <= set(data.keys()):
            self.filename = filename
            return True
        else:
            self.filename = None
            return False
            
    def check_user_validity(self, user):
        """ Description : Check if a given user exists in the currently loading file
            Parameters  : User to check"""
        return user in self.data['visitor_uuid'].tolist()
    
    def check_doc_validity(self, doc):
        """ Description : Check if a given user document in the currently loading file
            Parameters  : Document to check"""
        return doc in self.data['subject_doc_id'].tolist()
        
    def get_rnd_doc(self):
        """ Description : Returns a random document id from the currently loaded file
            Parameters  : Random document id from the currently loaded file"""
           
        rnd = random.randrange(0,len(self.data))
        return self.data.iloc[rnd]['subject_doc_id']
        
    def get_rnd_user(self): 
        """ Description : Returns a random user id from the currently loaded file
            Parameters  : Random user id from the currently loaded file"""
        rnd = random.randrange(0,len(self.data))
        return self.data.iloc[rnd]['visitor_uuid']
    
    def load_main_file_async(self, file_dict, callback=None, pg_val=None): 
        """ Description : Asynchronous wrapper around the load_main_file method
            Parameters  : A dictionnary containing the filename and number of lines in this file [file_dict]. The tkinter progress bar StringVar variable which will be modified to represent progress [pg_val]"""
        thread = threading.Thread(target=self.load_main_file, args=(file_dict, callback, pg_val))
        thread.daemon = True
        thread.start()
    
    def load_main_file(self, file_dict, callback = None, pg_val=None):
        """ Description : Main file loading method for issuu data file. Loads the file in chunk into a instance variable
            Parameters  : A dictionnary containing the filename and number of lines in this file [file_dict]. The tkinter progress bar StringVar variable which will be modified to represent progress [pg_val]"""
           
        utils.logger.info("Started loading file")
        start_time = time.time()
        
        tmp = []
        
        pd_reader = pd.read_json(file_dict['filename'], lines=True, chunksize=utils.CHUNK_SIZE)
        loop_count = ceil(file_dict['linecount']/utils.CHUNK_SIZE)
        
        for i, df in utils.cli_pg_bar(enumerate(pd_reader), total=loop_count): 
            df = df.loc[df['event_type'].isin(['read'])]#, 'pageread'])] # Since we are working with a random sample, some read documents don't have the "read" record but have "pageread" records
            
            tmp.append(df[utils.ISSUU_FIELDS])  
            if self.tk_gui and pg_val is not None:
                pg_val.set(i+1)
        
        self.data = pd.concat(tmp, axis=0)
        self.data.drop_duplicates(['visitor_uuid', 'subject_doc_id'], inplace=True)
        
        if callback is not None:
            callback()
            
        self.file_loaded = True
        self.file_size = utils.get_file_size(file_dict['linecount'])
        utils.logger.info("Loading done")
        df_mem_usage = self.data.memory_usage().sum()/(1024*1024)
        utils.logger.debug("Loaded in {0} seconds - dataframe using {1} MB".format(round(time.time() - start_time, 2), round(df_mem_usage, 2)))
        
    def country_to_continent(self, country):
        """ Description : Return the corresponding continent a given country belongs to
            Parameters  : The country in ISO 3166 Country Code format (string) [country]
            Returns     : The continent on which the country is in. If multiple, returns first, if none, returns 'None'"""
        continent_list = self.country_codes.loc[self.country_codes['a-2'] == country]['CC'].values
        
        if len(continent_list):
            return continent_list[0]
        else:
            return 'None'

        
    def get_plot_data(self, var, doc_id = None, callback=None): # From gui -> Running in seperate thread, get data back with callback. From Cli, just run without callback
        """ Description : Return the necessary data to plot a given variable for a given document (or all) 
            Parameters  : The variable name (column) to plot [doc_id]. The document id to restrict the plot to [doc_id]. The callback to .. call back when the method is done [callback] (asynchronous mode)
            Returns     : The plot data in case of a synchronous call
        """
        if var not in self.data.columns:
            self.preprocess(var)
            
        data = self.data.copy()
        if doc_id is not None: # If we are making a plot for a given document or a on the whole dataset
            data = data.loc[data['subject_doc_id'] == doc_id]#data.iloc[doc_id]['subject_doc_id']] #'130705172251-3a2a725b2bbd5aa3f2af810acf0aeabb'] '130705172251-3a2a725b2bbd5aa3f2af810acf0aeabb']
        
        data.drop_duplicates(['visitor_uuid'], inplace=True)
        
        # Get sum on the given column and sort in descending order
        data = data.groupby([var], as_index=False).size().reset_index(name='counts')
        
        data = data.sort_values('counts', ascending=False)
        
        if callback is not None:
            callback(data)
        else:
            return data
            
    def preprocess(self, var):
        """ Description : Preprocesses (add/modify) the required variable (column) 
            Parameters  : The variable name (column) to preprocess [var]
        """
        utils.logger.info("Starting preprocessing for {0}".format(var))
        start_time = time.time() 
        
        if self.tk_gui and hasattr(self.iface, 'progressbar'):
            self.iface.progressbar["maximum"] = 100
            
        split = np.array_split(self.data, 100) # Data is not copied, no ram increase !
           
        self.data[var] = np.nan
        for i, df in utils.cli_pg_bar(enumerate(split), total=100):
            if var == 'visitor_browser':
                self._preprocess_browser(df)
            elif var == 'visitor_platform':
                self._preprocess_platform(df)
            elif var == 'visitor_continent':
                self._preprocess_continent(df)
                
            if self.tk_gui:
                self.iface.pg_val.set(i+1)
        
        self.data = pd.concat(split)
        utils.logger.info("Done preprocessing - {0} sec".format(time.time()- start_time))
                
    def _preprocess_continent(self, df):
        """ Description : Preprocesses dataframe in place in order to add a new column containing the continent of the readers
            Parameters  : The dataframe to in-place modify [df]
        """
        df['visitor_continent'] = df['visitor_country'].apply(lambda x: self.country_to_continent(x))
        
    def _preprocess_browser(self, df):
        """ Description : Preprocesses dataframe in place in order to add a new column containing the browser of the readers
            Parameters  : The dataframe to in-place modify [df]
        """
        df['visitor_browser'] = df['visitor_useragent'].apply(lambda x: ua.parse(x).browser.family)      
        
    def _preprocess_platform(self, df):
        """ Description : Preprocesses dataframe in place in order to add a new column containing the platforme type of readers
            Parameters  : The dataframe to in-place modify [df]
        """
        df['visitor_platform'] = df['visitor_useragent'].apply(lambda x: 'Mobile' if ua.parse(x).is_mobile else 'Desktop')        
            
    def get_document_readers(self, doc_id):
        """ Description : Returns the list of visitor having read the given document
            Parameters  : The document_id we want to know the readers of [doc_id]
            Returns     : List of visitor having read the input document
        """
        data = self.data.loc[self.data['subject_doc_id'] == doc_id]['visitor_uuid'].unique()
        return data.tolist()
    
    def get_visitor_read(self, vis_id):
        """ Description : Returns the list of documents read by a given visitor
            Parameters  : The visitor we want to know the read documents of [vis_id]
            Returns     : List of documents a visitor has read
        """
        data = self.data.loc[self.data['visitor_uuid'] == vis_id]['subject_doc_id'].unique() # /!\ np.array not pandas.Series
        data = data[~pd.isnull(data)] # Remove nan values, data[True/False with '~' = pandas series inverse]
        return data
     
    def also_likes(self, doc_id, sort, ori_reader = None):
        """ Description : Generates a list of also_like documents as well as the corresponding dot graph
            Parameters  : Document id [doc], sorting algorithm to sort the list of also_likes documents [sort] and an optionnal user
            Returns     : A dictionnary containing the dot graph and the list of also likes sorted"""
            
        utils.logger.info("Starting 'also_likes' procedure")
        start = time.time()
        
        dic = {}
        utils.logger.debug("Original reader : {0}, original doc : {1}".format(ori_reader, doc_id))
        also_readers = self.get_document_readers(doc_id)
        utils.logger.debug("Readers of the same doc : {0}".format(also_readers))
       
        if ori_reader is not None:
            also_readers.remove(ori_reader)  
        
        dic[doc_id] = []
        
        if ori_reader is not None:
            dic[doc_id].append(ori_reader)

        # Find other readers read documents - do not add users that only read the input document
        for reader in also_readers:
            good = False
            for doc in self.get_visitor_read(reader):
                if doc not in dic:
                    dic[doc] = []
                if doc != doc_id:
                    good = True
                    dic[doc].append(reader)
            
            if good:
                dic[doc_id].append(reader)
        
        # Find the most frequently read related documents
        tmp = [(doc_id, len(readers)) for doc_id, readers in dic.items()]
        tmp = sort(tmp) 
        
        # Get the top 10 also_likes document (+1 for input document)
        tmp = tmp[0:11]
        tmp = [x[0] for x in tmp]
        
        new_dic = {doc:readers for doc, readers in dic.items() if doc in tmp}
    
        # Graph header 
        graph_header = ["digraph also_likes {",
                        "ranksep=3; ratio=compress; size = \"30,22\"; orientation=landscape; rotate=180;",
                        "{",
                        "node [shape=plaintext, fontsize=16];",
                        "Readers -> Documents ",
                        "[label=\"Size: {0}\"];".format(self.file_size)]
                        
        # Generate the associated dot graph
        dot_graph = GraphBuilder(graph_header, "style=filled,color=\".3 .9 .7\"").build_tree(doc_id, new_dic, ori_reader)
        
        self.save_graph(dot_graph)
        
        utils.logger.debug("'also-likes' done - Time : {0} sec".format(time.time() - start))
        
        return {'docs':dic.keys(), 'graph':dot_graph}

    def save_graph(self, graph):
        with open("my_diag.dot", "w") as f:
            f.write(graph)

    def get_graph_img(self):
        run(["dot", "-Tpng", "-Gdpi=300", "-o", "my_diag.png", "my_diag.dot"])
        import matplotlib.image as mpimg
        return mpimg.imread('my_diag.png')
