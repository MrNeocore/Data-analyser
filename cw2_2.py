import numpy as np
import pandas as pd
import user_agents as ua

import tkinter as tk
from tkinter import filedialog, StringVar
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

style.use('ggplot')
plt.ion()
CHUNK_SIZE = 5000


plots = {"None":"None",
         "Visitor country":"visitor_country",
         "Visitor continent":"visitor_continent",
         "Visitor browser":"visitor_browser",
         "Visitor platform":"visitor_platform"}

if sys.platform == 'linux':
    from subprocess import run, PIPE
    def line_count(filename):
        print("Counting lines in file (*nix)...")
        lines = int(run(["wc", "-l", filename], stdout=PIPE).stdout.decode('utf-8').split(" ")[0])
        print("Done")
        
        return lines
else:   
    # Fastest pure python implementation found
    # Inspired by Quentin Pradet's answer at https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python (7th post)
    def line_count(filename):
        print("Counting lines in file...")
        fd = open(filename, 'rb')
        bufgen = takewhile(lambda x: x, (fd.raw.read(1024*1024) for _ in repeat(None)))
        lines = sum(buf.count(b'\n') for buf in bufgen)
        fd.close()
        print("Done")
        
        return lines

class Sorting:
    @staticmethod
    def freq_ascending_sort(data): # Data as : [('<DOC_ID>', <count>), ('<DOC_ID>', <count>)...], returns ['<DOC_ID>', '<DOC_ID>'...]
        return data.reverse() # Data sorted (max -> min) by Counter
        
    @staticmethod 
    def freq_descending_sort(data):
        return data # Data already sorted (max -> min) by Counter
    
    @staticmethod
    def biased_sort(data): # Data : JSON encoded strings {"doc":"<DOC_ID>", "weight":"<weight>"} (1 per line)
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

class Gui:
    def __init__(self, root):
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
 
        self.drop_down_var = StringVar(self.root)
        self.drop_down_var.set(list(plots.keys())[0]) # default value

        self.dropdown = tk.OptionMenu(self.root, self.drop_down_var, *list(plots.keys()), command=lambda x :self.embed_plot(plots[x]))
        self.dropdown.grid(column=3, row=3)
        self.dropdown.config(state="disabled")
   
        self.mpl_ax = None
        self.mpl_fig = None
        self.show_graph()
        
        self.pg2_val = StringVar(self.root)
        self.progressbar2 = ttk.Progressbar(self.root, mode='determinate', variable=self.pg2_val)
        self.progressbar2.grid(column=2, row=5, columnspan=2, sticky='nsew')
        
        button2 = ttk.Button(self.root, text="Get document readers", command=lambda : self.model.also_likes('140228202800-6ef39a241f35301a9a42cd0ed21e5fb0', Sorting.freq_descending_sort, 'b2a24f14bb5c9ea3'))#('232eeca785873d35')) 
        button2.grid(column=5, row=3)
        
        self.res_usage = StringVar(self.root)
        self.status_bar = tk.Label(self.root, text="Ram usage : ", textvariable=self.res_usage)
        self.status_bar.grid(column=1,row=100, sticky='nswe')
   

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
        self.dropdown.config(state="normal")
        
    def load_main_file(self):
        filename = filedialog.askopenfile().name
        self.progressbar["maximum"] = int(line_count(filename) / CHUNK_SIZE)+1
        self.model.load_main_file_async(filename, self.pg_val)
    
    def show_graph(self):
        self.mpl_fig = Figure(figsize=(5,5), dpi=100)
        self.mpl_ax = self.mpl_fig.add_subplot(111)
        
        c = tk.Frame(self.root)
        c.grid(column=1,row=99, columnspan=5)
        
        self.canvas = FigureCanvasTkAgg(self.mpl_fig, c)
        self.canvas.get_tk_widget().grid(column=1, row=99)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2TkAgg(self.canvas, c)
        toolbar.update()
      
    def embed_plot(self, var):
        threshold = 0
        
        self.mpl_fig.clf()
        self.mpl_ax = self.mpl_fig.add_subplot(111)
        if var != 'None':# self.model.data.columns: # TODO : Call back done -> show graph if still on same page
            #if not self.model.get_plot_data(var, random.randrange(0,len(self.model.data)), callback=self.plot_data_ready): # Async, yield to say processing has to be done ?
             self.ready_lb.config(bg="red")
             self.dropdown.config(state="disabled")
             thread = threading.Thread(target=self.model.get_plot_data, args=(var,None, self.plot_data_ready)) # Thread creation is probably not worth it when preprocessing is already done
             thread.start() 
        else:
            self.canvas.draw()
          
    def plot_data_ready(self, data):
        print("Plot data ready")
        self.ready_lb.config(bg="green")
        self.dropdown.config(state="normal")
        
        if plots[self.drop_down_var.get()] == data.columns[0]:
            Plotting.plot(self.mpl_ax, data)
            self.canvas.draw()

class Plotting:
    @staticmethod
    def plot(axes, data):
        print(f"ploting {data.columns[0]}")
        
        # Bar plot chart       
        axes = data.plot.bar(data.columns[0],'counts', ax=axes)
        
        # Change chart y ticks
        #if num_y_ticks:
        axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))#num_y_ticks))
            
        # Rotate chart x ticks
        #self.mpl_fig.xticks(rotation=0)
        
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
        
        print("Loading finished !")
        
    def load_main_file_async(self, filename, pg_val=None): 
        thread = threading.Thread(target=self._load_file, args=(filename, pg_val))
        thread.daemon = True
        thread.start()
    
    def _load_file(self, filepath, pg_val=None):
        print("Started loading file")
        self.data = pd.DataFrame()
        start_time = time.time()
        i = 1
        tmp = []
        
        for df in tqdm(pd.read_json(filepath, lines=True, chunksize=CHUNK_SIZE), total=int(line_count(filepath)/CHUNK_SIZE)+1): 
            df = df.loc[df['event_type'].isin(['read', 'pageread'])] # Since we are working with a random sample, some read documents don't have the "read" record but have "pageread" records
            
            tmp.append(df[['visitor_uuid','visitor_country', 'visitor_useragent', 'subject_doc_id']])  
            if self.tk_gui and pg_val is not None:
                pg_val.set(i)
            i +=1
        
        self.data = pd.concat(tmp, axis=0)
        self.data.drop_duplicates(['visitor_uuid', 'subject_doc_id'], inplace=True)
        
        if self.tk_gui and hasattr(self, 'done_loading') and isinstance(self.done_loading, StringVar):
            self.done_loading.set(True)
            
        print("\nLoading done")
        df_mem_usage = self.data.memory_usage().sum()/(1024*1024)
        print(f"{time.time() - start_time} seconds, {df_mem_usage} MB")
        
    def country_to_continent(self, country):
        continent_list = self.country_codes.loc[self.country_codes['a-2'] == country]['CC'].values
        
        if len(continent_list):
            return continent_list[0]
        else:
            return 'None'

        
    def get_plot_data(self, var, doc_id = None, callback=None): # From gui -> Running in seperate thread, get data back with callback. From Cli, just run without callback

        if var not in self.data.columns:
            self.preprocess(var)
            
        data = self.data.copy()  ### TODO : Find a way to not use it
        
        if doc_id is not None: # If we are making a plot for a given document or a on the whole dataset
            data = data.loc[data['subject_doc_id'] == doc_id]#data.iloc[doc_id]['subject_doc_id']] #'130705172251-3a2a725b2bbd5aa3f2af810acf0aeabb'] '130705172251-3a2a725b2bbd5aa3f2af810acf0aeabb']
        
        data.drop_duplicates(['visitor_uuid'], inplace=True)
        assert(data is not self.data)
        # Get sum on the given column and sort in descending order
        data = data.groupby([var], as_index=False).size().reset_index(name='counts')
        data = data.sort_values('counts', ascending=False)
        
        if callback is not None:
            callback(data)
        else:
            return data
            
    def preprocess(self, var):
        start_time = time.time() 
        if self.tk_gui and hasattr(self.iface, 'progressbar2'):
            self.iface.progressbar2["maximum"] = 100
            
        split = np.array_split(self.data, 100) # Data is not copied, no ram increase !
           
        # TODO : use tqdm here
        self.data[var] = np.nan
        for df, i in tqdm(zip(split, range(1,101)), total=100):
            if var == 'visitor_browser':
                self._preprocess_browser(df)
            elif var == 'visitor_platform':
                self._preprocess_platform(df)
            elif var == 'visitor_continent':
                self._preprocess_continent(df)
                
            if self.tk_gui:
                self.iface.pg2_val.set(i)
        self.data = pd.concat(split)
        
        print(f"Done preprocessing - {time.time()- start_time} sec")

                
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
        start = time.time()
        dic = {}
        print(f"Original reader : {ori_reader}, original doc : {doc_id}")
        also_readers = self.get_document_readers(doc_id)
        print(f"readers of the same doc : {also_readers}")
       
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
           #docs.extend((reader, self.visitor_views(reader)))
        
        tmp = [(doc_id, len(readers)) for doc_id, readers in dic.items()]
        sorted(tmp, key=lambda x:x[1])
        tmp = tmp[0:10]
        tmp = [x[0] for x in tmp]
        
        new_dic = {doc:readers for doc, readers in dic.items() if doc in tmp}

        dot_graph = GraphBuilder().build_tree(doc_id, new_dic, ori_reader)
        
        with open("my_diag.dot", "w") as f:
            f.write(dot_graph)
        
        print(f"Time : {time.time() - start} sec")
        
        return dic

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
                        choices=['freq_asc','freq_desc','biaised'],
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
            print("Sorting algorithm not provided, using default 'freq_desc'")
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
        data = list(self.model.also_likes(doc, sort, user).keys())
        self.output(data)

    def task5(self, **kwargs):
        doc = self.get_doc()
        user = self.get_user()
        sort = self.get_sort()
        data = self.model.also_likes(doc, sort, user)
        self.output(data)
    
    def task_platform(self):
        doc = self.get_doc()
        data = self.get_plot_data('visitor_platform', doc)
        self.output(data)
    
    def output(self, data):
        if not args.quiet:
            print(data)
        
        if args.plt:
            self.plot(data)
            
    def plot(self, data):
        if args.task_id not in ('4d', '5'):
            ax = plt.gca()
            Plotting.plot(ax, data)
            plt.show()
            input("Press a key to exit...")
        
        elif args.task_id == '5': 
            run(["dot", "-Tpng", "-o my_diag.png", "my_diag.dot"])
            import matplotlib.image as mpimg
            img=mpimg.imread('my_diag.png')
            plt.axis('off')
            plt.imshow(img)
            plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
            input("Press a key to exit...")
            
        elif args.task_id == '4d':
            print("Flag --plt not available for task 4d - ignoring !")
            
    def execute_task(self, args):
        self.args = args
        self.load_main_file(args.input_file)
        task_func = self.dispatch.get(args.task_id)
        task_func()
        
# Imports slow down cli - do something ?
if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    if not args.quiet:
        verbosity = (args.verbose or 0) +1 # Equivalent to 'x if x else 0' (coalescing operator) - right hand side used if x = [0, None, False, ""]
    else:
        verbosity = 0  
    
    if args.gui:
        root = tk.Tk()
        gui = Gui(root)
        da = DataAnalyser(iface=gui)
        root.mainloop()
    else:
        required_no_gui = args.task_id is not None and args.task_id is not None 

        if not required_no_gui:
            parser.error("the following arguments are required: -t/--task_id and -f/--input_file when not using the gui (--gui)")

        else:
            if args.task_id != '4d' and args.sort is not None:
                print("Ignoring sort algorithm selection - task is not '4d'")
            
            cli = Cli()
            if args.task_id not in list(cli.dispatch.keys()):
                print("Invalid task !")
                exit(-1)
                
            da = DataAnalyser(iface=cli)
            cli.execute_task(args)

            #logging.disable(logging.CRITICAL)
            #logging.disable(logging.NOTSET)