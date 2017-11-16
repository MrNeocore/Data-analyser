from matplotlib import style, use
use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.ticker as ticker

import json
import numpy as np
import pandas as pd
import user_agents as ua

import tkinter as tk
from tkinter import filedialog, StringVar
from tkinter import ttk

from itertools import takewhile, repeat
import threading, time
import psutil, os, sys, random
from subprocess import run, PIPE
from tqdm import tqdm, tqdm_pandas
import contextlib

style.use('ggplot')
plt.ion()
CHUNK_SIZE = 5000


plots = {"None":"None",
         "Visitor country":"visitor_country",
         "Visitor continent":"visitor_continent",
         "Visitor browser":"visitor_browser",
         "Visitor platform":"visitor_platform"}

if sys.platform == 'linux':
    from subprocess import call
    def line_count(filename):
        print("Counting lines in file (*nix)...")
        lines = int(run(["wc", "-l", filename], stdout=PIPE).stdout.decode('utf-8').split(" ")[0])
        print("Done")
        
        return lines
else:   
    # Fastest pure python implementation found
    # Inspired by "Quentin Pradet"'s answer at https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python (7th post)
    def line_count(filename):
        print("Counting lines in file...")
        fd = open(filename, 'rb')
        bufgen = takewhile(lambda x: x, (fd.raw.read(1024*1024) for _ in repeat(None)))
        lines = sum(buf.count(b'\n') for buf in bufgen)
        fd.close()
        print("Done")
        
        return lines

class Gui:
    def __init__(self, model, root):
        self.model = model
        self.root = root
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=2, row=4, sticky=("nswe"))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        button = ttk.Button(self.root, text="Load data file", command=self.load_main_file)
        button.grid(column=1, row=1)
        
        self.pg_val = StringVar(self.root)
        self.progressbar = ttk.Progressbar(self.root, mode='determinate', variable=self.pg_val)
        self.progressbar.grid(column=2, row=1, columnspan=2, sticky='nsew')

        self.ready_lb = tk.Label(self.root, text="Ready", bg="red")
        self.ready_lb.grid(column=4, row =1)
 
        variable = StringVar(self.root)
        variable.set(list(plots.keys())[0]) # default value

        self.dropdown = tk.OptionMenu(self.root, variable, *list(plots.keys()), command=lambda x :self.plot(plots[x]))
        self.dropdown.grid(column=3, row=3)
        self.dropdown.config(state="disabled")
   
        self.mpl_ax = None
        self.mpl_fig = None
        self.show_graph()
        
        self.pg2_val = StringVar(self.root)
        self.progressbar2 = ttk.Progressbar(self.root, mode='determinate', variable=self.pg2_val)
        self.progressbar2.grid(column=2, row=5, columnspan=2, sticky='nsew')

    def file_loaded_cb(self):
        self.ready_lb.config(bg="green")
        self.dropdown.config(state="normal")
        
    def load_main_file(self):
        filename = filedialog.askopenfile().name
        self.progressbar["maximum"] = line_count(filename) / CHUNK_SIZE 
        self.model.load_main_file_async(filename, self.pg_val)
    
    def show_graph(self):
        self.mpl_fig = Figure(figsize=(5,5), dpi=100)
        self.mpl_ax = self.mpl_fig.add_subplot(111)
        
        c = tk.Frame(self.root)
        c.grid(column=1,row=99, columnspan=5)
        
        self.canvas = FigureCanvasTkAgg(self.mpl_fig, c)
        self.canvas.get_tk_widget().grid(column=1, row=100)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2TkAgg(self.canvas, c)
        toolbar.update()
      
    def plot(self, var):
        threshold = 0
        
        self.mpl_fig.clf()
        self.mpl_ax = self.mpl_fig.add_subplot(111)
        
        if var != "None" and self.model.add_column(var):
            print(f"ploting {var}")
            
            df = self.model.data.loc[self.model.data['subject_doc_id'] == random.choice(self.model.data['subject_doc_id'])]#'130705172251-3a2a725b2bbd5aa3f2af810acf0aeabb']
           
            # Get sum on the given column and sort in descending order
            df = df.groupby([var], as_index=False).size().reset_index(name='counts')
            df = df.sort_values('counts', ascending=False)
               
            # Bar plot chart       
            self.mpl_ax = df.plot.bar(var,'counts', ax=self.mpl_ax)
            
            # Change chart y ticks
            #if num_y_ticks:
            self.mpl_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))#num_y_ticks))
                
            # Rotate chart x ticks
            #self.mpl_fig.xticks(rotation=0)
            
        self.canvas.draw()
 
 
class Model:
    def __init__(self, root):
        self.root = root
        self.data = pd.DataFrame()
        self.country_codes = pd.read_csv("country_codes_simple_fixed.csv", na_filter = False)
        self.done_loading = StringVar(self.root)
        self.done_loading.trace_add("write", self.loading_done)
    
    def set_gui(self, gui):
        self.gui = gui
        
    def loading_done(self, *args):
        self.gui.file_loaded_cb()
        print("Loading finished !")
        
    def load_main_file_async(self, filename, pg_val): 
        thread = threading.Thread(target=lambda :self._load_file(filename, pg_val)) # Could use args parameter
        thread.daemon = True
        thread.start()
    
    def _load_file(self, filepath, pg_val):
        print("Started loading file")
        self.data = pd.DataFrame()
        start_time = time.time()
        i = 1
        tmp = []
        for df in pd.read_json(filepath, lines=True, chunksize=CHUNK_SIZE): 
            tmp.append(df[['visitor_uuid','visitor_country', 'visitor_useragent', 'subject_doc_id']])  
            pg_val.set(i)
            i +=1
        
        self.data = pd.concat(tmp, axis=0)

        self.done_loading.set(True)
        print("Loading done")
        df_mem_usage = self.data.memory_usage().sum()/(1024*1024)
        print(f"{time.time() - start_time} seconds, {df_mem_usage} MB")

    def country_to_continent(self, country):
        continent_list = self.country_codes.loc[self.country_codes['a-2'] == country]['CC'].values
        
        if len(continent_list):
            return continent_list[0]
        else:
            return 'None'
            
    def add_column(self, var):
        if var not in list(self.data.columns):
            thread = threading.Thread(target=self.preprocess, args=(var,)) 
            thread.daemon = True
            thread.start()
            return False
        
        else:
            return True
        
    def preprocess(self, var):
        start_time = time.time() 
        self.gui.progressbar2["maximum"] = 100
        df_len = len(self.data.index)
        split = np.array_split(self.data, 100) # Data is not copied, no ram increase !
           
        self.data[var] = np.nan
        for df, i in zip(split, range(1,101)):
            if var == 'visitor_browser':
                self._preprocess_browser(df)
            elif var == 'visitor_platform':
                self._preprocess_platform(df)
            
            self.gui.pg2_val.set(i)  
        self.data = pd.concat(split)
         
        print(f"Done preprocessing - {time.time()- start_time} sec")
        
        
        
    def _preprocess_browser(self, df):
        df['visitor_browser'] = df['visitor_useragent'].apply(lambda x: ua.parse(x).browser.family)      
        
    def _preprocess_platform(self, df):
        df['visitor_platform'] = df['visitor_useragent'].apply(lambda x: 'Mobile' if ua.parse(x).is_mobile else 'Desktop')        
            
            
            
    """def preprocess_browser(self):
        start_time = time.time() 
        self.gui.progressbar2["maximum"] = 100
        df_len = len(self.data.index)
        split = np.array_split(self.data, 100) # Data is not copied, no ram increase !

        self.data['visitor_browser'] = np.nan
        for df, i in zip(split, range(1,101)):
            df['visitor_browser'] = df['visitor_useragent'].apply(lambda x: ua.parse(x).browser.family)
            self.gui.pg2_val.set(i)
        
        self.data = pd.concat(split)
         
        print(f"Done preprocessing - {time.time()- start_time} sec")
        
    
    def preprocess_platform(self):
        start_time = time.time() 
        self.gui.progressbar2["maximum"] = 100
        df_len = len(self.data.index)
        split = np.array_split(self.data, 100) # Data is not copied, no ram increase !

        self.data['visitor_platform'] = np.nan
        for df, i in zip(split, range(1,101)):
            df['visitor_platform'] = df['visitor_useragent'].apply(lambda x: 'Mobile' if ua.parse(x).is_mobile else 'Desktop')        
            self.gui.pg2_val.set(i)
        
        self.data = pd.concat(split)
         
        print(f"Done preprocessing - {time.time()- start_time} sec")"""
        
        
class DataAnalyser:
    def __init__(self):
        root = tk.Tk()
        self.model = Model(root)
        self.gui = Gui(self.model, root)
        self.model.set_gui(self.gui)
        root.mainloop()
    

da = DataAnalyser()
