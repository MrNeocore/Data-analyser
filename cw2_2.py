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

from itertools import takewhile, repeat, chain
import threading, time
import psutil, os, sys, random
from subprocess import run, PIPE
from tqdm import tqdm, tqdm_pandas
import contextlib
import collections
import heapq
import operator

style.use('ggplot')
plt.ion()
CHUNK_SIZE = 5000


######### USE READ EVENTS


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
        
        button2 = ttk.Button(self.root, text="Get document readers", command=lambda : self._also_likes_rnd())#('232eeca785873d35')) 
        button2.grid(column=3, row=3)

    def freq_ascending_sort(self, data): # Data as : [('<DOC_ID>', <count>), ('<DOC_ID>', <count>)...], returns ['<DOC_ID>', '<DOC_ID>'...]
        return data.reverse() # Data sorted (max -> min) by Counter
    
    def _also_likes_rnd(self):
        rnd = random.randrange(0,len(self.model.data))
        self.model.also_likes(self.model.data.iloc[rnd]['subject_doc_id'], self.freq_descending_sort, self.model.data.iloc[rnd]['visitor_uuid'])
        
    def freq_descending_sort(self, data):
        return data # Data already sorted (max -> min) by Counter
    
    def biased_sort(self, data): # Data : JSON encoded strings {"doc":"<DOC_ID>", "weight":"<weight>"} (1 per line)
        doc_weights = {}
        with open("doc_weights.csv") as f:
            for line in f.readlines():
                (doc_id, weight) = line.split(',')
                doc_weights[doc_id] = int(weight)
                            
        sorted_docs = []
        
        for doc_id, freq in data:
            sorted_docs.append((doc_id, doc_weights.get(doc_id, freq))) # If we have no bias toward this document, simply use its frequency
        
        sorted_docs =  sorted(sorted_docs, key=lambda x:x[1], reverse=True)
        
        return sorted_docs
            
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
            
            df = self.model.data 
            dfmi.loc[:,('one','second')]
            df.loc[df['subject_doc_id'] == df.iloc[random.randrange(0,len(df))]['subject_doc_id']] #'130705172251-3a2a725b2bbd5aa3f2af810acf0aeabb'] '130705172251-3a2a725b2bbd5aa3f2af810acf0aeabb']
            df.drop_duplicates(['visitor_uuid'], inplace=True)
            
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
            df = df.loc[df['event_type'] == 'read']
            tmp.append(df[['visitor_uuid','visitor_country', 'visitor_useragent', 'subject_doc_id']])  
            pg_val.set(i)
            i +=1
        
        self.data = pd.concat(tmp, axis=0)
        self.data.drop_duplicates(['visitor_uuid', 'subject_doc_id'], inplace=True) # Also do it on each chunk to reduce each chunk size and speed up concatenation ? Mmm..
        
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
            
    def get_document_readers(self, doc_id):
        data = self.data.loc[self.data['subject_doc_id'] == doc_id]['visitor_uuid'].unique()
        return data.tolist()
    
    def visitor_views(self, vis_id):
        data = self.data.loc[self.data['visitor_uuid'] == vis_id]['subject_doc_id'].unique() # /!\ np.array not pandas.Series
        data = data[~pd.isnull(data)] # Remove nan values, data[True/False with '~' = pandas series inverse]
        return data
     
    def also_likes(self, doc_id, sort, ori_reader = None):
        start = time.time()
        docs = []
        dic = {}
        #import pdb; pdb.set_trace()
        print(f"Original reader : {ori_reader}, original doc : {doc_id}")
        also_readers = self.get_document_readers(doc_id)
        print(f"readers of the same doc : {also_readers}")
       
        if ori_reader:
            also_readers.remove(ori_reader)  
                    
        for reader in self.get_document_readers(doc_id):
            #if reader != ori_reader:
            for doc in self.visitor_views(reader):
                if doc not in dic:
                    dic[doc] = []
                dic[doc].append(reader)
           #docs.extend((reader, self.visitor_views(reader)))
        
        #import pdb; pdb.set_trace()
        #docs = collections.Counter(docs).most_common(10) # Optimized (tested vs dictionnary + sort + indexing and dictionnary + heapq.nlargest) 
        #docs = sort(docs)
        #docs = [x[0] for x in docs]
        dot_graph = GraphBuilder().build_tree(doc_id, dic, ori_reader)
        #print(dot_graph)
        with open("my_diag.dot", "w") as f:
            f.write(dot_graph)
        
        print(f"Time : {time.time() - start} sec")
        
        print (dic)
        #import pdb; pdb.set_trace()
        return dic

class GraphBuilder:
    default_graph_header = ["digraph also_likes {",
                            "ranksep=.75; ratio=compress; size = \"15,22\"; orientation=landscape; rotate=180;",
                            "{",
                            "node [shape=plaintext, fontsize=16];",
                            "Readers -> Documents "]
                        
    def __init__(self, graph_header = None):
        if graph_header == None:
            graph_header = GraphBuilder.default_graph_header
        self.graph = Graph(graph_header, ori_design="style=filled,color=\".3 .9 .7\"")
       
    def build_tree(self, ori_doc, dic, ori_reader=None):
 
        ori_doc = ori_doc[-4:]
        if ori_reader:
            ori_reader = ori_reader[-4:]
    
        n = {}
        for d, readers in dic.items():
            n[d[-4:]] = []
            for reader in readers:
                n[d[-4:]].append(reader[-4:])
                
        #dic = {d[-4:]: v[-4:] for d, v in dic.items()}
        
        self.add_ori(ori_doc, ori_reader)
        self.add_other_docs(set(n.keys()))
        self.add_other_readers(set(chain.from_iterable(n.values())))
        
        if ori_reader:
            self.add_link(ori_reader, ori_doc)
        
        #import pdb; pdb.set_trace()
        for doc, readers in n.items():
            for reader in readers:
                self.add_link(reader, doc)
                
        for reader in chain.from_iterable(n.values()):
            self.add_link(reader, ori_doc)

        return self.graph.get_graph()
        
    def add_other_docs(self, doc_lst):
        for doc in doc_lst:
            self.graph.add_document(doc)
     
    def add_other_readers(self, readers_lst):
        for reader in readers_lst:
            self.graph.add_reader(reader)
            
    def add_ori(self, ori_doc, ori_reader):
          self.graph.add_document(ori_doc, ori=True)
          if ori_reader:
            self.graph.add_reader(ori_reader, ori=True)

    def add_link(self, reader, doc):
        self.graph.add_link(reader, doc)
        
# TODO : Make generic sections
class Graph:
    def __init__(self, header, ori_design):
        self.header = header
        self.ori_design = ori_design
        self.labels = ["[label=\"Size: 1m\"];"]
        self.readers = ["{ rank = same; \"Readers\";"]
        self.docs = ["{ rank = same; \"Documents\";"]
        self.links = []
    
    def add_document(self, doc, align_key=-1, ori=False):
        lb = f"\"{doc}\" [label=\"{doc}\", shape=\"circle\""
        if ori:
            lb += self.ori_design
        
        lb += "];"
        if f"\"{doc}\";" not in self.docs:
            self.docs.append(f"\"{doc}\";")
            self.labels.append(lb)
        
    def add_reader(self, reader, align_key=-1, ori=False):
        lb = f"\"{reader}\" [label=\"{reader}\", shape=\"box\""
        if ori:
            lb += self.ori_design
        
        lb += "];"
        if f"\"{reader}\";" not in self.readers:
            self.readers.append(f"\"{reader}\";")
            self.labels.append(lb)
        
    def add_link(self, reader, doc):
        if f"\"{reader}\" -> \"{doc}\";" not in self.links:
            self.links.append(f"\"{reader}\" -> \"{doc}\";")
    
    """def rearrange_order(self):
        # Overly complicated way to find the original reader and document :)
        tmp = list(filter(lambda x : "shape=\"box\",style=filled" in x, self.labels))
        if len(tmp):
            ori_reader = tmp[0]
            
        tmp = list(filter(lambda x : "shape=\"circle\",style=filled" in x, self.labels))
        if len(tmp):
            ori_doc = tmp[0]
            
        ori_reader_pos = int(len(self.readers) / 2)
        ori_doc_pos = int(len(self.docs) / 2)
        
        new_docs = []
        self.docs.remove(ori_doc)
        
        for idx, doc in enumerate(self.docs):
            if idx == ori_doc_pos:
                new_docs.append(ori_doc)
            new_docs.append(doc)
        
        new_readers = []
        self.readers.remove(ori_reader)
        
        for idx, reader in enumerate(self.readers):
            if idx == ori_reader_pos:
                new_readers.append(ori_reader)
            new_docs.append(reader)
    
        self.docs = new_docs
        self.readers = new_readers"""
        
    def get_graph(self):
        #self.rearrange_order()
        self.readers.append("};")
        self.docs.append("};")
        self.links.append("};")
        graph = self.header + self.labels + self.readers + self.docs + self.links
        graph.append("}")
        graph = '\n'.join(graph)
        
        return graph
    
class DataAnalyser:
    def __init__(self):
        root = tk.Tk()
        self.model = Model(root)
        self.gui = Gui(self.model, root)
        self.model.set_gui(self.gui)
        root.mainloop()
    

da = DataAnalyser()
