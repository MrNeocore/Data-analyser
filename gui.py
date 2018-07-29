import utils
import tkinter as tk
from tkinter import filedialog, StringVar, LEFT, BOTH, END
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.ticker as ticker
from matplotlib import style, use
use("TkAgg")
from math import ceil
import threading

class Gui:
    """ View class with tk GUI interface """
    def __init__(self, root):
        """ 
            Description : Initialize TK GUI elements
            Parameters  : tkinter root instance (tk.TK())
            Returns     : Instance of Gui class
        """
        utils.logger.debug("Tk GUI initialization")
        self.root = root
        self.root.resizable(0,0)
        self.root.title("Data Analyser - MEYER Jonathan")
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=2, row=4, sticky=("nswe"))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)
        
        self.create_input_widgets()
        self.create_status_bar()
        self.init_plot()

    def create_status_bar(self):
        """ Create the GUI status bar """
        status_bar = tk.Frame(self.root)
        status_bar.grid(column=1, row=100, columnspan=5,sticky='nswe')
        self.ram_usage = StringVar(self.root)
        self.status_bar = tk.Label(status_bar, text="Ram usage : ", textvariable=self.ram_usage)
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
        
        #status_bar.pack_propagate(0) # Attempt to prevent the status bar elements from moving when label text is changing - should work according to forums but ... yeah
     
    def create_input_widgets(self):
        """ Creates top section of the GUI with all input fields and buttons """ 
        
        self.add_file_sel(row=1)
        self.add_separator(row=2)
        
        self.add_doc_sel(row=3)
        self.add_separator(row=4)
        
        self.add_user_sel(row=5)
        self.add_separator(row=6)
        
        self.add_plot_sel(row=7)
    
    def add_separator(self, row):
        """ Description : Adds an horizontal ttk seperator at the given row 
            Parameters  : Row to which the separator shall be inserted
        """
        sep = ttk.Separator(self.root, orient="horizontal")
        sep.grid(column=1, row=row, sticky="ew", columnspan=5)
        
    def add_user_sel(self, row=5):
        """ Description : Adds the user selection section at the given row 
            Parameters  : Row to which the section shall be inserted
        """
        # Left label
        lb = tk.Label(self.root, text="User selection")
        lb.grid(column=1, row=row, padx=3) 
        pass # Out of time error
        
    def add_file_sel(self, row=1):
        """ Description : Adds the file selection section at the given row 
            Parameters  : Row to which the section shall be inserted
        """
        # Left label
        lb = tk.Label(self.root, text="File selection")
        lb.grid(column=1, row=row, padx=3) 
        
        # Filename entry field
        self.file_entry = tk.Entry(self.root, width=43)
        self.file_entry.insert(0, "Click here to select a file")
        self.file_entry.bind('<ButtonRelease-1>', (lambda _: self.get_file_name()))
        self.file_entry.config(state="readonly")
        self.file_entry.grid(column=2, row=row, padx=3)        
        
        # Load file button
        self.load_btn = ttk.Button(self.root, width=21, text="Load data file", command=self.load_main_file)
        self.load_btn.grid(column=4, row=row, padx=3)
        self.load_btn.config(state="disabled")
        
        # Right label
        self.file_info_var = StringVar(self.root)
        self.file_label = tk.Label(self.root, text="File not loaded", fg="gray", textvariable=self.file_info_var)
        self.file_label.grid(column=5, row=row, padx=3) 
   
    def add_doc_sel(self, row=3):
        """ Description : Adds the document mode selection section at the given row 
            Parameters  : Row to which the section shall be inserted
        """
        # Left label
        lb = tk.Label(self.root, text="Document selection")
        lb.grid(column=1, row=row, padx=3) 
        
        doc_sel = ['Select one', 'All documents']
        
        # Document selection mode
        self.dp_doc_var = StringVar(self.root)
        self.dp_doc_var.set(doc_sel[0])# default value
        self.dp_doc_wg = tk.OptionMenu(self.root, self.dp_doc_var, *doc_sel, command=lambda x: self.dp_doc_sel_update(x))
        self.dp_doc_wg.config(width=15)#max(map(len,doc_sel)) - 5)
        self.dp_doc_wg.grid(column=4, row=row, sticky='ew')
        
        # Document id input
        self.text_entry_default = "Enter a valid document id"
        vcmd = (self.mainframe.register(self.doc_input_update),'%P', '%d')
        self.txt_entry_doc = tk.Entry(self.root, width=43, validate="key", validatecommand=vcmd)
        self.txt_entry_doc.insert(0, self.text_entry_default)
        self.txt_entry_doc.bind('<FocusIn>', (lambda _: self.txt_entry_doc.delete(0, 'end') if self.txt_entry_doc.get() == self.text_entry_default else False))
        self.txt_entry_doc.grid(column=2, row=row, pady=3)
        
        # Random document button
        self.rnd_doc_btn = ttk.Button(self.root, width=15, text="Random document", command=self.set_rnd_doc)
        self.rnd_doc_btn.grid(column=5, row=row, pady=3)
        self.rnd_doc_btn.config(state="disabled")
        
    def add_plot_sel(self, row):
        """ Description : Adds the plot variable selection section at the given row 
            Parameters  : Row to which the section shall be inserted
        """
        # Left label
        lb = tk.Label(self.root, text="Variable selection")
        lb.grid(column=1, row=row, padx=3) 
        
        # Plot variable selection
        self.dp_plt_var = StringVar(self.root)
        self.dp_plt_var.set(list(utils.PLOTS.keys())[0]) # default value
        self.dp_plt_wg = tk.OptionMenu(self.root, self.dp_plt_var, *list(utils.PLOTS.keys()), command=lambda x : self.show_plot_graph(utils.PLOTS[x], self.get_doc_id()))
        self.dp_plt_wg.grid(column=2, row=row)
        self.dp_plt_wg.config(state="disabled")
        self.dp_plt_row = row
     
    def show_plot_graph(self, var, doc=None):
        """ Description : Top level method to plot a given variable - either plot or also_likes diagram
            Parameters  : Variable to plot (var) and a document id (doc). Necessary for task 4d & 5, optional for others. If not provided, plots are taking all document into account
        """
        if var != 'None' and var != "also_likes":
            self.embed_plot(var, doc)
        elif var == 'also_likes':
            self.also_likes(doc, utils.Sorting.freq_descending_sort, user=None)
        else: # None
            utils.Plotting.reset_plot(self.mpl_ax)
            self.canvas.draw()
            
    def dp_doc_sel_update(self, dp_val):
        """ Description : Method called by tkinter when the document selection mode changes - updates gui elements accordingly
            Parameters  : The new variable for the document selection optionMenu
        """
        if dp_val == 'Select one':
            self.txt_entry_doc.config(state="normal")  
            if self.model.file_loaded:
                self.rnd_doc_btn.config(state="normal")
            self.dp_plt_wg.destroy()
            self.dp_plt_wg = tk.OptionMenu(self.root, self.dp_plt_var, *list(utils.PLOTS.keys()), command=lambda x : self.show_plot_graph(utils.PLOTS[x], self.get_doc_id()))
            self.dp_plt_wg.grid(column=2, row=self.dp_plt_row)
            self.doc_input_update(self.get_doc_id(), 1)

        else :
            self.txt_entry_doc.config(state="disabled")
            self.rnd_doc_btn.config(state="disabled")
            if self.model.file_loaded:
                self.dp_plt_wg.config(state="normal") 
            if self.dp_plt_var.get() == 'Also likes diagram':
                self.dp_plt_var.set('None')
                self.show_plot_graph('None')
                
            no_also_likes = dict(utils.PLOTS)
            del no_also_likes['Also likes diagram']
            
            self.dp_plt_wg.destroy()
            self.dp_plt_wg = tk.OptionMenu(self.root, self.dp_plt_var, *list(no_also_likes.keys()), command=lambda x : self.show_plot_graph(utils.PLOTS[x], self.get_doc_id()))
            self.dp_plt_wg.grid(column=2, row=self.dp_plt_row)
            self.show_plot_graph(utils.PLOTS[self.dp_plt_var.get()]) # All documents
            
    def doc_input_update(self, txt, action): 
        """ Description : Method called by tkinter when the input document_id field changes (every key stroke + copy paste)
            Parameters  : The new variable for the document id
        """
       
        if utils.check_doc_id(txt) and self.model.file_loaded and self.model.check_doc_validity(txt): # test cases order important - better performance and expression evaluated before model is set on startup, but check_doc_id will be false on start 
            self.dp_plt_wg.config(state="normal")
            self.txt_entry_doc.config(bg="pale green")
            if action != 0: # Not delete
                self.show_plot_graph(utils.PLOTS[self.dp_plt_var.get()], txt)
            
        else:
            if hasattr(self, 'dp_plt_wg'):
                self.dp_plt_wg.config(state="disabled")
            self.txt_entry_doc.config(bg="tomato")
        
        return True
    
    def set_rnd_doc(self):
        """ Set a random valid document_id in the input document_id field """
        self.txt_entry_doc.delete(0, END)
        self.txt_entry_doc.insert(0, self.model.get_rnd_doc()) 
        
    def get_doc_id(self):
        """ Description : Returns a document_id value depending on the current document selection mode
            Returns     : The current document_id in the document_id field if document selection mode is "Select one" and returns None otherwise""" 
        if self.dp_doc_var.get() == 'All documents':
            return None
            
        else:
            return self.txt_entry_doc.get()
        
    def also_likes(self, doc, sort, user=None):
        """ Description : Ask model to generate also_likes graph based on the given document, sort algorithm and user_id then display it. Synchronous but processing is quite fast and there is nothing the user could do while it's processing anyway...
            Parameters  : Document [doc] to get the also_likes documents from. utils.Sorting algorithm [sort] used for sorting. Optional [user] to display on the graph.
        """
        utils.Plotting.reset_plot(self.mpl_ax)
        graph = self.model.also_likes(doc, sort, user) # Method generates dot file and return the content, but get_graph_img works on the file itself directly...
        img = self.model.get_graph_img()
        utils.Plotting.show_graph_img(self.mpl_ax, img)
        self.canvas.draw()
        
    def refresh_ram_label(self):
        """ Get current ram usage and display it + schedule a call to this function in 1 second """
        self.ram_usage.set(self.model.ram_usage())
        self.root.after(1000, self.refresh_ram_label)
         
    def set_model(self, model):
        """ Set model and start RAM indicator """
        self.model = model
        self.refresh_ram_label()

    def file_loaded_cb(self):
        """ Callback when file is done loading by the model, update gui elements accordingly """
        self.ready_lb.config(bg="green")
        self.status_var.set("Data file loaded")
        self.model.file_loaded = True
        if self.dp_doc_var.get() == 'Select one':
            self.rnd_doc_btn.config(state="normal")
            self.doc_input_update(self.txt_entry_doc.get(), 1) # Forced doc check
        else:
            self.dp_plt_wg.config(state="normal")
    
    def get_file_name(self):
        """Promp the user for a suitable Issuu file. If the file is valid, we store it and unlock gui element accordingly. This is why method "load_main_file" is safe despite being non-stateless"""
        filename = filedialog.askopenfile(filetypes=[("Issuu JSON data files","*.json")])
        if filename is not None:
            self.file_entry.config(state="normal") # Yes, now you can input text into the textbox... that doesn't cause trouble. "readonly" state doesn't show background colors...
            self.file_entry.delete(0, END)
            self.file_entry.insert(0, '.../' + '/'.join(filename.name.split('/')[4:]))
            
            if self.model.check_file_validity(filename.name):
                self.load_btn.config(state="normal")
                self.file_entry.config(bg="pale green")
                self.file_info_var.set("Valid file")
            else:
                self.load_btn.config(state="disabled")
                self.file_entry.config(bg="tomato")
                self.file_info_var.set("Invalid file")
                        
    def load_main_file(self): 
        """ Start asynchronously loading stored filename. If we land in this method we are assured that the store filename is valid. 
            When loading is done, method file_loaded_cb is called"""
            
        # Lock all inputs, clear the plot
        self.rnd_doc_btn.config(state="disabled")
        self.ready_lb.config(bg="red")
        self.dp_plt_wg.config(state="disabled")
        utils.Plotting.reset_plot(self.mpl_ax)
        self.canvas.draw()
        self.txt_entry_doc.delete(0, END)
        self.txt_entry_doc.insert(0, self.text_entry_default)
        self.dp_plt_var.set('None')
        self.status_var.set("Loading data file")
        
        lines = utils.line_count(self.model.filename)
        self.progressbar["maximum"] = ceil(lines / utils.CHUNK_SIZE) # int(X) + 1 doesn't work for whole values
        self.file_info_var.set("{0} entries".format(lines))
        self.model.load_main_file_async({'filename':self.model.filename, 'linecount':lines}, callback=self.file_loaded_cb, pg_val=self.pg_val)
    
    def init_plot(self):
        """ Creates the canvas area on tkinter main window and add a matplotlib figure inside as well as the navigation bar"""
        self.mpl_fig = Figure(figsize=(8,8), dpi=100)
        self.mpl_ax = self.mpl_fig.add_subplot(111)
        self.mpl_ax.set_adjustable('box-forced')
        
        c = tk.Frame(self.root)
        c.grid(column=1,row=99, columnspan=5)
        
        self.canvas = FigureCanvasTkAgg(self.mpl_fig, c)
        self.canvas.get_tk_widget().grid(column=1, row=99)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2TkAgg(self.canvas, c)
        toolbar.update()
      
    def embed_plot(self, var, doc_id):
        """ Description : Request plotting data from the model. Callback "plot_data_ready" is then called on preprocessing completion (if needed)
            Parameters  : Variable to plot [var] for a given document [doc_id]
            Remark      : View using thread explicitely.. not very compliant with an MV model
        """
        threshold = 0 # TODO 
        
        utils.Plotting.reset_plot(self.mpl_ax)
        if var != 'None':
             self.ready_lb.config(bg="red")
             self.dp_plt_wg.config(state="disabled")
             self.status_var.set("Preprocessing...")
             thread = threading.Thread(target=self.model.get_plot_data, args=(var, doc_id, self.plot_data_ready)) # Thread creation is probably not worth it when preprocessing is already done
             thread.start() 
        else:
            self.canvas.draw()
          
    def plot_data_ready(self, data):
        """ Description : Callback method when plotting data is ready. Called by model's get_plot_data method.
            Parameters  : utils.Plotting data [data]
        """
        self.status_var.set("Preprocessing done")
        utils.logger.debug("Plot data ready")
        self.ready_lb.config(bg="green")
        self.dp_plt_wg.config(state="normal")
        utils.Plotting.plot(self.mpl_ax, data)
        self.canvas.draw()
