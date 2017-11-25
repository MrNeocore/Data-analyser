import numpy as np
import pandas as pd
import user_agents as ua

import tkinter as tk
from tkinter import filedialog, StringVar, LEFT, BOTH, END
from tkinter import ttk

from itertools import takewhile, repeat, chain
import threading, time
import sys, random
from tqdm import tqdm, tqdm_pandas

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.ticker as ticker
from matplotlib import style, use
use("TkAgg")
import matplotlib.pyplot as plt
import argparse
import psutil
import os
import logging
from logging import getLogger, StreamHandler
from math import ceil


style.use('ggplot')
plt.ion()
CHUNK_SIZE = 5000

logger = None

plots = {"None":"None",
         "Visitor country":"visitor_country",
         "Visitor continent":"visitor_continent",
         "Visitor browser":"visitor_browser",
         "Visitor platform":"visitor_platform",
         "Also likes diagram":"also_likes"}

if sys.platform == 'linux':
    from subprocess import run, PIPE
    def line_count(filename):
        logger.debug("Counting lines in file (*nix)...")
        lines = int(run(["wc", "-l", filename], stdout=PIPE).stdout.decode('utf-8').split(" ")[0])
        logger.debug("Done")
        
        return lines
else:
    logger.debug("Platform is not unix-like, file line counting will be (much) slower on large datasets")
    # Fastest pure python implementation found
    # Inspired by Quentin Pradet's answer at https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python (7th post)
    def line_count(filename):
        logger.debug("Counting lines in file (crossplatform)...")
        fd = open(filename, 'rb')
        bufgen = takewhile(lambda x: x, (fd.raw.read(1024*1024) for _ in repeat(None)))
        lines = sum(buf.count(b'\n') for buf in bufgen)
        fd.close()
        logger.debug("Done")
        
        return lines

class Sorting:
    @staticmethod
    def freq_ascending_sort(data): # Data as : [('<DOC_ID>', <count>), ('<DOC_ID>', <count>)...], returns ['<DOC_ID>', '<DOC_ID>'...]
        logger.debug("Sorting documents in ascending order")
        return data.reverse() # Data sorted (max -> min) by Counter
        
    @staticmethod 
    def freq_descending_sort(data):
        logger.debug("Sorting documents in descending order")
        return data # Data already sorted (max -> min) by Counter
    
    @staticmethod
    def biased_sort(data): # Data : JSON encoded strings {"doc":"<DOC_ID>", "weight":"<weight>"} (1 per line)
        doc_weights = {}
        logger.debug("Sorting documents in descending order")
        try: 
            with open("doc_weights_boooom.csv") as f:
                for line in f.readlines():
                    (doc_id, weight) = line.split(',')
                    doc_weights[doc_id] = int(weight)
                    
        except IOError:
            logger.warning("Biased document weights file not found - using default sort (desc_sort)")
            return self.freq_descending_sort(data)
            
        sorted_docs = []
        
        for doc_id, freq in data:
            sorted_docs.append((doc_id, doc_weights.get(doc_id, freq))) # If we have no bias toward this document, simply use its frequency
        
        sorted_docs =  sorted(sorted_docs, key=lambda x:x[1], reverse=True)
        
        return sorted_docs

class Gui:
    def __init__(self, root):
        logger.debug("Tk GUI initialization")
        self.root = root
        self.root.resizable(0,0)
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=2, row=4, sticky=("nswe"))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)
        
        self.create_input_widgets()
        self.create_status_bar()
        self.init_plot()
        
        
   
    def create_status_bar(self):
        status_bar = tk.Frame(self.root)
        status_bar.grid(column=1, row=100, columnspan=5,sticky='nswe')
        self.res_usage = StringVar(self.root)
        self.status_bar = tk.Label(status_bar, text="Ram usage : ", textvariable=self.res_usage)
        self.status_bar.pack(side=LEFT, padx=10)

        self.status_var = StringVar(self.root)
        self.status_var.set("Waiting")
        self.status_lb = tk.Label(status_bar, textvariable=self.status_var)
        self.status_lb.pack(side=LEFT, fill=BOTH, padx=10, ipadx=15)
        
        self.pg_val = StringVar(self.root)
        self.progressbar = ttk.Progressbar(status_bar, mode='determinate', variable=self.pg_val)#, length=300)
        self.progressbar.pack(side=LEFT, padx=10, fill=BOTH, expand=1)
        
        self.ready_lb = tk.Label(status_bar, text="Ready", bg="red")
        self.ready_lb.pack(side=LEFT, padx=10)
        
        #status_bar.pack_propagate(0)
     
    def create_input_widgets(self):
        button = ttk.Button(self.root, text="Load data file", command=self.load_main_file)
        button.grid(column=1, row=1)
        
        doc_sel = ['Select one', 'All documents']
        # Document selection mode
        self.dp_doc_var = StringVar(self.root)
        self.dp_doc_var.set(doc_sel[0])# default value
        self.dp_doc_wg = tk.OptionMenu(self.root, self.dp_doc_var, *doc_sel, command=lambda x: self.dp_doc_sel_update(x))
        self.dp_doc_wg.config(width=max(map(len,doc_sel)) - 5)
        self.dp_doc_wg.grid(column=1, row=2, sticky='ew')
        
        # Plot variable selection
        self.dp_plt_var = StringVar(self.root)
        self.dp_plt_var.set(list(plots.keys())[0]) # default value
        self.dp_plt_wg = tk.OptionMenu(self.root, self.dp_plt_var, *list(plots.keys()), command=lambda x :self.show_plot_graph(plots[x], self.get_doc_id()))
        self.dp_plt_wg.grid(column=2, row=3)
        self.dp_plt_wg.config(state="disabled")
        
        # Document id input
        self.text_entry_default = "Enter a valid document id"
        vcmd = (self.mainframe.register(self.doc_input_update),'%P', '%d')
        self.txt_entry_doc = tk.Entry(self.root, width=43, validate="key", validatecommand=vcmd)
        self.txt_entry_doc.insert(0, self.text_entry_default)
        self.txt_entry_doc.bind('<FocusIn>', (lambda _: self.txt_entry_doc.delete(0, 'end') if self.txt_entry_doc.get() == self.text_entry_default else False))
        self.txt_entry_doc.grid(column=2, row=2)
        
        self.rnd_doc_btn = ttk.Button(self.root, text="Random document", command=self.set_rnd_doc)
        self.rnd_doc_btn.grid(column=5, row=2)
        self.rnd_doc_btn.config(state="disabled")
     
    def show_plot_graph(self, var, doc=None):
        if var != 'None' and var != "also_likes":
            self.embed_plot(var, doc)
        elif var == 'also_likes':
            self.also_likes(doc, Sorting.freq_descending_sort, user=None)
        else:
            Plotting.reset_plot(self.mpl_ax)
            self.canvas.draw()
            
    def dp_doc_sel_update(self, dp_val):
        if dp_val == 'Select one':
            self.txt_entry_doc.config(state="normal")  
            self.rnd_doc_btn.config(state="normal")
            self.doc_input_update(self.get_doc_id(), 1)
            self.dp_plt_wg['menu'].add_command(label='Also likes diagram', command=tk._setit(self.dp_plt_var, 'Also likes diagram'))

        else :
            self.txt_entry_doc.config(state="disabled")
            self.rnd_doc_btn.config(state="disabled")
            self.dp_plt_wg.config(state="normal") 
            if self.dp_plt_var.get() == 'Also likes diagram':
                self.dp_plt_var.set('None')
                self.show_plot_graph('None')
                
            self.dp_plt_wg['menu'].delete('Also likes diagram')
            self.show_plot_graph(plots[self.dp_plt_var.get()]) # All documents
            
    def doc_input_update(self, txt, action): 
        if check_doc_id(txt) and 'subject_doc_id' in self.model.data.columns and txt in self.model.data['subject_doc_id'].tolist(): # test cases order important - better performance and expression evaluated before model is set on startup, but check_doc_id will be false on start 
            self.dp_plt_wg.config(state="normal")
            self.txt_entry_doc.config(bg="pale green")
            if action != 0: # Not delete
                self.show_plot_graph(plots[self.dp_plt_var.get()], txt)
            
        elif txt != self.text_entry_default:
            self.dp_plt_wg.config(state="disabled")
            self.txt_entry_doc.config(bg="tomato")
        
        return True
    
    def set_rnd_doc(self):
        rnd = random.randrange(0,len(self.model.data))
        self.txt_entry_doc.delete(0, END)
        self.txt_entry_doc.insert(0, self.model.data.iloc[rnd]['subject_doc_id'])
            
    def get_doc_id(self):
        if self.dp_doc_var.get() == 'All documents':
            return None
            
        else:
            return self.txt_entry_doc.get()
        
    def also_likes(self, doc, sort, user=None):
        self.mpl_fig.clf()
        self.mpl_ax = self.mpl_fig.add_subplot(111)
        graph = self.model.also_likes(doc, sort, user)
        img = self.model.get_graph_img()
        Plotting.show_graph_img(self.mpl_ax, img)
        self.canvas.draw()
        
    def refresh_res_label(self):
        self.res_usage.set(self.model.res_usage())
        self.root.after(1000, self.refresh_res_label)
         
    def set_model(self, model):
        self.model = model
        self.refresh_res_label()

    def _also_likes_rnd(self):
        rnd = random.randrange(0,len(self.model.data))
        self.model.also_likes(self.model.data.iloc[rnd]['subject_doc_id'], Sorting.freq_descending_sort, self.model.data.iloc[rnd]['visitor_uuid'])
    
    def file_loaded_cb(self):
        self.ready_lb.config(bg="green")
        self.status_var.set("Data file loaded")
        self.rnd_doc_btn.config(state="normal")
        if self.dp_doc_var.get() == 'Select one':
            self.doc_input_update(self.txt_entry_doc.get(), 1) # Forced doc check
    
    def load_main_file(self):
        filename = filedialog.askopenfile()
        if filename is None:
            return
       
        self.status_var.set("Loading data file")
        lines = line_count(filename.name)
        self.progressbar["maximum"] = ceil(lines / CHUNK_SIZE) # int(X) + 1 doesn't work for whole values
        self.model.load_main_file_async({'filename':filename.name, 'linecount':lines}, self.pg_val)
    
    def init_plot(self):
        self.mpl_fig = Figure(figsize=(7,7), dpi=100)
        self.mpl_ax = self.mpl_fig.add_subplot(111)
        
        c = tk.Frame(self.root)
        c.grid(column=1,row=99, columnspan=5)
        
        self.canvas = FigureCanvasTkAgg(self.mpl_fig, c)
        self.canvas.get_tk_widget().grid(column=1, row=99)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2TkAgg(self.canvas, c)
        toolbar.update()
      
    def embed_plot(self, var, doc_id):
        threshold = 0 # TODO 
        
        Plotting.reset_plot(self.mpl_ax)
        if var != 'None':
             self.ready_lb.config(bg="red")
             self.dp_plt_wg.config(state="disabled")
             self.status_var.set("Preprocessing...")
             thread = threading.Thread(target=self.model.get_plot_data, args=(var, doc_id, self.plot_data_ready)) # Thread creation is probably not worth it when preprocessing is already done
             thread.start() 
        else:
            self.canvas.draw()
          
    def plot_data_ready(self, data):
        self.status_var.set("Preprocessing done")
        logger.debug("Plot data ready")
        self.ready_lb.config(bg="green")
        self.dp_plt_wg.config(state="normal")
        Plotting.plot(self.mpl_ax, data)
        self.canvas.draw()

def check_doc_id(txt):
    if len(txt) == 45 and txt[12] == '-' and all(letter in ['a','b','c','d','e','f','0','1','2','3','4','5','6','7','8','9','-'] for letter in txt):
        return True
    else:
        return False
 
class Plotting:
    @staticmethod
    def plot(axes, data):
        logger.debug(f"Plotting {data.columns[0]}")
        
        # Bar plot chart       
        axes = data.plot.bar(data.columns[0],'counts', ax=axes)
        
        # Change chart y ticks
        #if num_y_ticks:
        axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))#num_y_ticks))
            
        # Rotate chart x ticks
        #self.mpl_fig.xticks(rotation=0)
    
    def show_graph_img(axes, img):
        axes.axis('off')
        axes.imshow(img)
        axes.grid(False)
        axes.yaxis.set_visible(False)
        axes.xaxis.set_visible(False)
     
    def reset_plot(axes):
        axes.axis('on')
        axes.grid(True)
        axes.yaxis.set_visible(True)
        axes.xaxis.set_visible(True)
        axes.cla()
        
class Model:
    def __init__(self, iface):
        self.iface = iface
        
        self.tk_gui= hasattr(self.iface, 'root') and isinstance(self.iface.root, tk.Tk)
        
        if self.tk_gui:
            self.root = self.iface.root
            self.done_loading = StringVar(self.root)
            self.done_loading.trace_add("write", self.loading_done)
            
        self.data = pd.DataFrame()
        self.country_codes = pd.read_csv("country_codes_simple_fixed.csv", na_filter = False)
    
    def res_usage(self):
        proc = psutil.Process(os.getpid())
        return f"Ram : {round(proc.memory_info()[0]/(1024*1024), 1)} MB |"
        
    def loading_done(self, *args):
        self.iface.file_loaded_cb()
        
    def load_main_file_async(self, file_dict, pg_val=None): 
        thread = threading.Thread(target=self._load_file, args=(file_dict, pg_val))
        thread.daemon = True
        thread.start()
    
    def _load_file(self, file_dict, pg_val=None):
        logger.info("Started loading file")
        start_time = time.time()
        
        tmp = []
        
        pd_reader = pd.read_json(file_dict['filename'], lines=True, chunksize=CHUNK_SIZE)
        loop_count = ceil(file_dict['linecount']/CHUNK_SIZE)
        
        for i, df in tqdm(enumerate(pd_reader), total=loop_count): 
            df = df.loc[df['event_type'].isin(['read', 'pageread'])] # Since we are working with a random sample, some read documents don't have the "read" record but have "pageread" records
            
            tmp.append(df[['visitor_uuid','visitor_country', 'visitor_useragent', 'subject_doc_id']])  
            if self.tk_gui and pg_val is not None:
                pg_val.set(i+1)
        
        self.data = pd.concat(tmp, axis=0)
        self.data.drop_duplicates(['visitor_uuid', 'subject_doc_id'], inplace=True)
        
        if self.tk_gui and hasattr(self, 'done_loading') and isinstance(self.done_loading, StringVar):
            self.done_loading.set(True)
            
        logger.info("Loading done")
        df_mem_usage = self.data.memory_usage().sum()/(1024*1024)
        logger.debug(f"Loaded in {round(time.time() - start_time, 2)} seconds - dataframe using {round(df_mem_usage, 2)} MB")
        
    def country_to_continent(self, country):
        continent_list = self.country_codes.loc[self.country_codes['a-2'] == country]['CC'].values
        
        if len(continent_list):
            return continent_list[0]
        else:
            return 'None'

        
    def get_plot_data(self, var, doc_id = None, callback=None): # From gui -> Running in seperate thread, get data back with callback. From Cli, just run without callback

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
        logger.info(f"Starting preprocessing for {var}")
        start_time = time.time() 
        
        if self.tk_gui and hasattr(self.iface, 'progressbar'):
            self.iface.progressbar["maximum"] = 100
            
        split = np.array_split(self.data, 100) # Data is not copied, no ram increase !
           
        self.data[var] = np.nan
        for i, df in tqdm(enumerate(split), total=100):
            if var == 'visitor_browser':
                self._preprocess_browser(df)
            elif var == 'visitor_platform':
                self._preprocess_platform(df)
            elif var == 'visitor_continent':
                self._preprocess_continent(df)
                
            if self.tk_gui:
                self.iface.pg_val.set(i+1)
        
        self.data = pd.concat(split)
        logger.info(f"Done preprocessing - {time.time()- start_time} sec")

                
    def _preprocess_continent(self, df):
        df['visitor_continent'] = df['visitor_country'].apply(lambda x: self.country_to_continent(x))
        
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
        logger.info("Starting 'also_likes' procedure")
        start = time.time()
        
        dic = {}
        logger.debug(f"Original reader : {ori_reader}, original doc : {doc_id}")
        also_readers = self.get_document_readers(doc_id)
        logger.debug(f"Readers of the same doc : {also_readers}")
       
        if ori_reader is not None:
            also_readers.remove(ori_reader)  
        
        dic[doc_id] = []
        
        if ori_reader is not None:
            dic[doc_id].append(ori_reader)
                   
        for reader in also_readers:
            good = False
            for doc in self.visitor_views(reader):
                if doc not in dic:
                    dic[doc] = []
                if doc != doc_id:
                    good = True
                    dic[doc].append(reader)
            
            if good:
                dic[doc_id].append(reader)
        
        tmp = [(doc_id, len(readers)) for doc_id, readers in dic.items()]
        sorted(tmp, key=lambda x:x[1])
        tmp = tmp[0:10]
        tmp = [x[0] for x in tmp]
        
        new_dic = {doc:readers for doc, readers in dic.items() if doc in tmp}

        dot_graph = GraphBuilder().build_tree(doc_id, new_dic, ori_reader)
        
        self.save_graph(dot_graph)
        
        logger.debug(f"'also-likes' done - Time : {time.time() - start} sec")
        
        return {'docs':dic.keys(), 'graph':dot_graph}

    def save_graph(self, graph):
        with open("my_diag.dot", "w") as f:
            f.write(graph)

    
    def get_graph_img(self):
        run(["dot", "-Tpng", "-o", "my_diag.png", "my_diag.dot"])
        import matplotlib.image as mpimg
        return mpimg.imread('my_diag.png')
            
class GraphBuilder:
    default_graph_header = ["digraph also_likes {",
                            "ranksep=.75; ratio=compress; size = \"15,22\"; orientation=landscape; rotate=180;",
                            "{",
                            "node [shape=plaintext, fontsize=16];",
                            "Readers -> Documents ",
                            "[label=\"Size: 1m\"];"] # Not really part of the header, but this is done so that the label structure in the Graph class only contains real labels (docs/readers) 
                        
    def __init__(self, graph_header = None):
        if graph_header == None:
            graph_header = GraphBuilder.default_graph_header
        self.graph = Graph(graph_header, ori_design="style=filled,color=\".3 .9 .7\"")
       
    def build_tree(self, ori_doc, dic, ori_reader=None):
        
        ori_doc = ori_doc[-4:]
        if ori_reader is not None:
            ori_reader = ori_reader[-4:]
    
        n = {}
        for d, readers in dic.items():
            n[d[-4:]] = []
            for reader in readers:
                n[d[-4:]].append(reader[-4:])

        self.add_ori(ori_doc, ori_reader)
        self.add_other_docs(set(n.keys()))
        self.add_other_readers(set(chain.from_iterable(n.values())))
        
        if ori_reader:
            self.add_link(ori_reader, ori_doc)

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
          self.graph.add_document(ori_doc, pos="center", align_key="ori", ori=True)
          if ori_reader:
            self.graph.add_reader(ori_reader, pos="center", align_key="ori", ori=True)

    def add_link(self, reader, doc):
        self.graph.add_link(reader, doc)
        
# TODO : Make generic sections
class Graph:
    def __init__(self, header, ori_design):
        self.header = header
        self.ori_design = ori_design
        self.docs_labels = []
        self.readers_labels = []
        self.readers = ["{ rank = same; \"Readers\";"]
        self.docs = ["{ rank = same; \"Documents\";"]
        self.links = []
    
    def add_document(self, doc, pos=None, align_key=None, ori=False):
        lb = f"\"{doc}\" [label=\"{doc}\", shape=\"circle\""
        if ori:
            lb += self.ori_design
        
        lb += "];"
        if f"\"{doc}\";" not in self.docs:
            self.docs.append(f"\"{doc}\";")
            self.docs_labels.append({'label':lb, 'align':align_key, 'pos':pos})
        
    def add_reader(self, reader, pos=None, align_key=None, ori=False): # TODO : Use kwargs for align & ori
        lb = f"\"{reader}\" [label=\"{reader}\", shape=\"box\""
        if ori:
            lb += self.ori_design
        
        lb += "];"
        if f"\"{reader}\";" not in self.readers:
            self.readers.append(f"\"{reader}\";")
            self.readers_labels.append({'label':lb, 'align':align_key, 'pos':pos})
        
    def add_link(self, reader, doc):
        if f"\"{reader}\" -> \"{doc}\";" not in self.links:
            self.links.append(f"\"{reader}\" -> \"{doc}\";")
        
    def get_labels(self):
    
        labels = []
        right_readers   = [x['label'] for x in filter(lambda x : x['pos'] == 'right', self.readers_labels)]
        center_readers  = [x['label'] for x in filter(lambda x : x['pos'] == 'center', self.readers_labels)]
        left_readers    = [x['label'] for x in filter(lambda x : x['pos'] == 'left', self.readers_labels)]
        other_readers   = [x['label'] for x in filter(lambda x : x['pos'] not in ('right', 'left', 'center'), self.readers_labels)]
        
        other_readers_left  = other_readers[:int(len(other_readers)/2)]
        other_readers_right = other_readers[int(len(other_readers)/2):]
        
        right_docs  = [x['label'] for x in filter(lambda x : x['pos'] == 'right', self.docs_labels)]
        center_docs = [x['label'] for x in filter(lambda x : x['pos'] == 'center', self.docs_labels)]
        left_docs   = [x['label'] for x in filter(lambda x : x['pos'] == 'left', self.docs_labels)]
        other_docs  = [x['label'] for x in filter(lambda x : x['pos'] not in ('right', 'left', 'center'), self.docs_labels)]
        
        other_docs_left  = other_docs[:int(len(other_docs)/2)]
        other_docs_right = other_docs[int(len(other_docs)/2):]
        
        readers = left_readers  +   other_readers_left  +   center_readers  +   other_readers_right +   right_readers
        docs    = left_docs     +   other_docs_left     +   center_docs     +   other_docs_right    +   right_docs
        
        labels = readers + docs       
        return labels
        
    def get_graph(self):
        # self.rearrange_order()
        labels = self.get_labels()
        self.readers.append("};")
        self.docs.append("};")
        self.links.append("};")
        graph = self.header + labels + self.readers + self.docs + self.links
        graph.append("}")
        graph = '\n'.join(graph)
        
        return graph
    
class DataAnalyser:
    def __init__(self, iface):
        self.iface = iface
        self.model = Model(self.iface)
        self.iface.set_model(self.model)

def arg_parser():
    parser = argparse.ArgumentParser(description='Issuu data analytics software')

    gui_group = parser.add_mutually_exclusive_group()
    verbosity_group = parser.add_mutually_exclusive_group()


    gui_group.add_argument('-g', '--gui', action='store_true',\
                        help='Use the graphical user interface')

    verbosity_group.add_argument("-v", "--verbose", action="count",\
                        help="Increases verbosity, add 'v's for even more verbosity")

    verbosity_group.add_argument("-q", "--quiet", action="store_true",\
                        help="Do not output to stdout/stderr")

    parser.add_argument("-t", "--task_id", action='store', #required=True, # Cannot since it is not mandatory when using gui -> Manual checking
                        help="Task to execute")

    parser.add_argument("-d", "--doc_uuid", action="store", #required=True,
                        help="Document 'doc_uuid' to analyse")
     
    parser.add_argument("-f", "--input_file", action="store",# required=True,
                        help='Issuu compliant input data file')

    parser.add_argument("-o", "--output_file", action="store",\
                        help='Output file to write to')

    parser.add_argument("-u", "--user_uuid", action="store",\
                        help='Issuu compliant input data file')

    parser.add_argument("-s", "--sort", action="store",\
                        choices=['freq_asc','freq_desc','biased'],
                        help='Also likes document sort algorithm')
                        
    gui_group.add_argument("--plt", action="store_true",\
                        help='Show plots (without gui)')

    return parser

class Cli:    
    def __init__(self):
         self.dispatch = {'2a' : self.task2a,
                          '2b' : self.task2b,
                          '3a' : self.task3a,
                          '3b' : self.task3b,
                          '4d' : self.task4d,
                          '5'  : self.task5,
                          'platform': self.task_platform}
                    
    def load_main_file(self, filename):
        self.model._load_file(filename)
    
    def set_model(self, model):
        self.model = model
            
    def get_plot_data(self, var, doc_id=None):
        return self.model.get_plot_data(var, doc_id)

    def get_doc(self):
        doc_id = args.doc_uuid
        if doc_id == 'random':
            return self.model.data.iloc[random.randrange(0,len(self.model.data))]['subject_doc_id']
        
        elif doc_id is not None and doc_id[12] != '-' and len(doc) != 45:
            return "Invalid document id !"
            
        else:
            return doc_id        
    
    def get_user(self):
        user_id = args.user_uuid
        
        if user_id == 'random':
            return self.model.data.iloc[random.randrange(0, len(self.model.data))]['visitor_uuid']
            
        elif user_id is not None and len(user_id) != 16:
            return "Invalid user id !"
            
        else:
            return user_id
     
    def get_sort(self):
        sorting_algo = {'freq_desc': Sorting.freq_descending_sort,
                'freq_asc' : Sorting.freq_ascending_sort,
                'biased'   : Sorting.biased_sort,
                None       : Sorting.freq_ascending_sort}
                
        sort = args.sort
        algo = sorting_algo.get(args.sort)
        
        if algo == 'None' and not args.quiet:
            logger.warning("Sorting algorithm not provided, using default 'freq_desc'")
            algo = sorting_algo.get('freq_desc') 
        
        return algo
        
    def task2a(self):
        doc = self.get_doc()
        data = self.get_plot_data('visitor_country', doc)
        self.output(data)
        
    def task2b(self):
        doc = self.get_doc() 
        data = self.get_plot_data('visitor_continent', doc)
        self.output(data)
        
    def task3a(self):
        data = self.get_plot_data('visitor_useragent')
        self.output(data)
    
    def task3b(self):
        data = self.get_plot_data('visitor_browser')
        self.output(data)
    
    def task4d(self):
        doc = self.get_doc()
        user = self.get_user()
        sort = self.get_sort()
        data = list(self.model.also_likes(doc, sort, user)['docs'])
        self.output(data)

    def task5(self, **kwargs):
        doc = self.get_doc()
        user = self.get_user()
        sort = self.get_sort()
        data = self.model.also_likes(doc, sort, user)['graph']
        self.output(data)
    
    def task_platform(self):
        doc = self.get_doc()
        data = self.get_plot_data('visitor_platform', doc)
        self.output(data)
   
    def to_string(self, data):
        if isinstance(data, pd.DataFrame):
            return data.to_string(index=False)
        else:
            return data
            
    def output(self, data):
        if not args.quiet:
            logger.results(f"\n\n========= TASK {args.task_id} =========\n {self.to_string(data)}")
            
        if args.plt:
            self.plot(data)
            
    def plot(self, data):
        if args.task_id not in ('4d', '5'):
            ax = plt.gca()
            Plotting.plot(ax, data)
            plt.text(200, 200, 'Image already saved')
            plt.show()
            input("Press a key to exit...")
        
        elif args.task_id == '5':
            img = self.model.get_graph_img()
            fig = plt.figure(figsize=(6,6), dpi=100)
            ax = fig.add_subplot(111)
            Plotting.show_graph_img(self, ax, img)
            fig.show()
            fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
            #plt.annotate("Graph is already saved in file 'my_diag.png'", xy=(350,100), xytext=(10,10))
            input("Press a key to exit...")
            
        elif args.task_id == '4d':
            logger.warning("Flag --plt not available for task 4d - ignoring !")
            
    def execute_task(self, args):
        self.args = args
        self.load_main_file({'filename':args.input_file, 'linecount':line_count(args.input_file)})
        task_func = self.dispatch.get(args.task_id)
        task_func()


def startLogging(level, log_file):
        global logger 
        logging.RESULTS = 25
        logging.addLevelName(logging.RESULTS, "RESULTS")
        
        logger = logging.getLogger(__name__)
        setattr(logger, 'results', lambda msg, *args : logger._log(logging.RESULTS, msg, args))
        
        log_level = {0: 0,
                     1: logging.RESULTS,
                     2: logging.INFO,
                     3: logging.DEBUG}
             
        logger.setLevel(log_level.get(level))
        
        stream_h = StreamHandler()        

        if level != 1:
            formatter = logging.Formatter('[%(levelname)s] : %(message)s')    
            stream_h.setFormatter(formatter)
            
        if log_file is not None and os.access(os.path.dirname(log_file), os.W_OK):
            file_h = logging.FileHandler(log_file)
            logger.addHandler(file_h)
            if level != 1:
                file_h.setFormatter(formatter)

        logger.addHandler(stream_h)

# Imports slow down cli - do something ?
if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    
    verbosity = min((args.verbose or 0) +1, 3) # Equivalent to 'x if x else 0' (coalescing operator) - right hand side used if x = [0, None, False, ""]

    if verbosity < 2:
        def tqdm(x, total): # Override tqdm progress bar method in order to not show it
            return x
            
    startLogging(verbosity, args.output_file)
    
    if args.gui:
        root = tk.Tk()
        gui = Gui(root)
        da = DataAnalyser(iface=gui)
        root.mainloop()
        
    else:
        required_no_gui = args.task_id is not None and args.input_file is not None 

        if not required_no_gui:
            parser.error("the following arguments are required: -t/--task_id and -f/--input_file when not using the gui (--gui)")

        else:
            if args.task_id != '4d' and args.sort is not None:
                logging.info("Ignoring sort algorithm selection - task is not '4d'")
            
            cli = Cli()
            if args.task_id not in list(cli.dispatch.keys()):
                logging.critical("Invalid task !")
                exit(-1)
                
            da = DataAnalyser(iface=cli)
            cli.execute_task(args)
