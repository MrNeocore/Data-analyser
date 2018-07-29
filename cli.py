import model
import utils
import pandas as pd
import matplotlib.pyplot as plt


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
        """ Forwards load file request to the model - synchronous"""
        self.model.load_main_file(filename)
    
    def set_model(self, model):
        """ Set the model """
        self.model = model
            
    def get_plot_data(self, var, doc_id=None):
        """ Description : Return plot data for given variable and document 
            Parameters  : Variable (column) [var] to plot, optional document [doc_id] to restrict the plot to
            Returns     : Plotting data
        """
        return self.model.get_plot_data(var, doc_id)

    def get_doc(self, cli_doc_arg, task):
        """ Return a valid doc_id - either the input document id if valid, a random one if requested or raise an exception otherwise """
       
        if cli_doc_arg == 'random':
            return self.model.get_rnd_doc()
        
        # doc_id lenght is 99.9% of the time 45... 0.1% len == 46 (100k sample). Never trust your assumptions... (All entries in 1M sample are 45 in lenght too)
        elif cli_doc_arg is not None and len(cli_doc_arg) > 12 and cli_doc_arg[12] == '-' and len(cli_doc_arg) in [45,46] and self.model.check_doc_validity(cli_doc_arg):
            return cli_doc_arg 
            
        elif task in ['4d', '5']:
            raise utils.InvalidArgumentError("Input document is necessary but '{0}' is invalid !".format(cli_doc_arg))        
        
        if cli_doc_arg is not None:
            raise utils.InvalidArgumentError("An optional document_id has been provided but is invalid !".format(cli_doc_arg))        
        
        return None
            
    def get_user(self, cli_user_arg):
        """ Return a valid user_id - either the input user_id if valid, None if invalid or not provided or a random one if requested"""
        if cli_user_arg == 'random':
            if self.args.task_id == '5':
                utils.logger.warning("Can't pick a random user for task 5") # Because we would need to pick a user that actually read that document - doable but adding complexity for negligible usefulness.
            else:
                return self.model.get_rnd_user()
        
        elif cli_user_arg is not None :
            if len(cli_user_arg) == 16 and self.model.check_user_validity(cli_user_arg):
                return cli_user_arg
            else:
                utils.logger.warning("Provided user_id '{0}' isn't valid - ignoring".format(cli_user_arg))
                return None
        else:
            return None
            
     
    def get_sort(self, cli_sort_arg):
        """ Return a valid sorting algorith for task '4d' and task '5'. If not provided use default freq_desc algorithm. Invalid not possible, filtered by argparse."""
        sorting_algo = {'freq_desc': utils.Sorting.freq_descending_sort,
                'freq_asc' : utils.Sorting.freq_ascending_sort,
                'biased'   : utils.Sorting.biased_sort,
                None       : utils.Sorting.freq_descending_sort}
                
        if self.args.sort is None :
            utils.logger.warning("Sorting algorithm not provided, using default 'freq_desc'")
            
        algo = sorting_algo.get(cli_sort_arg)
        
        return algo
    
    
    """ Task methods - quite self explanatory """
    def task2a(self):
        data = self.get_plot_data('visitor_country', self.doc)
        self.output(data)
        
    def task2b(self):
        data = self.get_plot_data('visitor_continent', self.doc)
        self.output(data)
        
    def task3a(self):
        data = self.get_plot_data('visitor_useragent', self.doc)
        self.output(data)
    
    def task3b(self):
        data = self.get_plot_data('visitor_browser', self.doc)
        self.output(data)
    
    def task4d(self):
        data = list(self.model.also_likes(self.doc, self.sort, self.user)['docs'])
        self.output(data)

    def task5(self):
        data = self.model.also_likes(self.doc, self.sort, self.user)['graph']
        self.output(data)
    
    def task_platform(self):
        data = self.get_plot_data('visitor_platform', self.doc)
        self.output(data)
   
    def to_string(self, data):
        """ Return stringified data if relevant, otherwise returns data unmodified """
        if isinstance(data, pd.DataFrame):
            return data.to_string(index=False)
        else:
            return data
            
    def output(self, data):
        """ Description : Handles cli output in both textual and visual form depending on cli arguments
            Parameters  : [data] to plot / display or output as string 
        """
        
        if not self.args.quiet:
            utils.logger.results("\n\nDocument '{0}'".format(self.doc))
            if self.user:
                utils.logger.results("User '{0}'".format(self.user))
            utils.logger.results("========= TASK '{0}' =========\n {1}".format(self.args.task_id, self.to_string(data)))
            
        if self.args.plt:
            self.plot(data)
            
    def plot(self, data):
        """ Plots data or display also_likes graph based on input data [data] """
        if self.args.task_id not in ('4d', '5'):
            ax = plt.gca()
            utils.Plotting.plot(ax, data)
            plt.text(200, 200, 'Image already saved')
            plt.show()
            input("Press a key to exit...")
        
        elif self.args.task_id == '5':
            img = self.model.get_graph_img()
            fig = plt.figure(figsize=(6,6), dpi=100)
            ax = fig.add_subplot(111)
            utils.Plotting.show_graph_img(ax, img)
            fig.show()
            input("Press a key to exit...")
            
        elif self.args.task_id == '4d':
            utils.logger.warning("Flag --plt not available for task 4d - ignoring !")
            
    def execute_task(self, args):
        """ Description : Stores sanitized argument variables and dispatches task to the relevant method
            Parameters  : Parsed CLI arguments
        """
        self.args = args
        if self.model.check_file_validity(args.input_file):
            self.load_main_file({'filename':args.input_file, 'linecount':utils.line_count(str(args.input_file))})
            self.doc = self.get_doc(args.doc_uuid, args.task_id)
            self.user = self.get_user(args.user_uuid)
            if args.task_id in ['4d', '5']:
                self.sort = self.get_sort(args.sort)
            task_func = self.dispatch.get(args.task_id)
            task_func()
        else:
            raise utils.InvalidArgumentError("Input file is not valid !")
