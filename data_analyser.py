import numpy as np
import pandas as pd
import user_agents as ua

import tkinter as tk
from tkinter import filedialog, StringVar, LEFT, BOTH, END
from tkinter import ttk

from itertools import takewhile, repeat, chain
import threading, time
import sys, random
from tqdm import tqdm

import argparse
import os   
from cli import *
from gui import *
from model import *
from graph import *
from utils import *

style.use('ggplot')
plt.ion()

logger = None


 
def start_data_analyser(iface_cls):  
    """ Description : Program entry point, initialize an instance of the given view class
        Parameters  : [iface_cls], view class to instanciate and initialize
        Returns     : Initialized instance of [iface_cls]
    """ 
    if iface_cls == Gui:
        root = tk.Tk()
        iface = Gui(root)
        
    elif iface_cls == Cli:
        iface = Cli()
        
    else:
        raise InvalidViewInterfaceError(iface_cls)
        
    model = Model(iface)
    iface.set_model(model)

    return iface


# Imports slow down cli - lazy loading attempted but complexity and suspected unstability not worth it.
if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    
    if not args.quiet:
        verbosity = min((args.verbose or 0) +1, 3) # Equivalent to 'x if x else 0' (coalescing operator) - right hand side used if x = [0, None, False, ""]
    else:
        verbosity = 0 
        
    if verbosity < 2:
        def no_pg_bar(x, total): # Override tqdm progress bar method in order to not show it
            return x
        utils.cli_pg_bar = no_pg_bar
    else:
        utils.cli_pg_bar = tqdm  
        
    start_logging(verbosity, args.output_file)
    
    if args.x or not args.gui:
        utils.graph_good = True
        
    if args.gui:
        da = start_data_analyser(Gui)
        da.root.mainloop()
        
    else:
        required_no_gui = args.task_id is not None and args.input_file is not None 

        if not required_no_gui:
            parser.error("the following arguments are required: -t/--task_id and -f/--input_file when not using the gui (--gui)")

        else:
            if args.task_id not in ['4d', '5'] and args.sort is not None:
                logger.info("Ignoring sort algorithm selection - task is not '4d' nor '5'")
            
            da = start_data_analyser(Cli)
            
            if args.task_id not in list(da.dispatch.keys()):
                raise InvalidTaskError("Invalid task !")
                
            da.execute_task(args)