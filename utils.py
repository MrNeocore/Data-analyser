import sys
import argparse
import logging
from logging import getLogger, StreamHandler
import matplotlib.ticker as ticker
from itertools import takewhile, repeat, chain

# Global variable - not the more elegant choice, but it is quite heavy to pass a variable through 3 layers of functions / classes too..
cli_pg_bar = None # Iterator wrapper to show (or not) a tqdm progressbar. Can be set to either tqdm.tqdm (not quiet) or f(x) return x (quiet)
logger = None 
graph_good = False

CHUNK_SIZE = 5000
ISSUU_FIELDS = ['visitor_uuid','visitor_country', 'visitor_useragent', 'subject_doc_id']
DOC_WEIGHTS_FILE = "doc_weights.csv"
COUNTRY_CODES_FILE = "country_codes.csv"
PLOTS = {"None":"None",
         "Visitor country":"visitor_country",
         "Visitor continent":"visitor_continent",
         "Visitor user agent": "visitor_useragent",
         "Visitor browser":"visitor_browser",
         "Visitor platform":"visitor_platform",
         "Also likes diagram":"also_likes"}

if sys.platform == 'linux':
    from subprocess import run, PIPE
    def line_count(filename):
        logger.debug("Counting lines in file (*nix)...")
        lines = int(run(["grep", "-c", "^", filename], stdout=PIPE).stdout.decode('utf-8').split(" ")[0])
        logger.debug("Done")
        
        return lines
else:
    # Fastest pure python implementation found
    # Inspired by Quentin Pradet's answer at https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python (7th post)
    def line_count(filename):
        logger.debug("Platform is not unix-like, file line counting will be (much) slower on large datasets")
        logger.debug("Counting lines in file (crossplatform)...")
        fd = open(filename, 'rb')
        bufgen = takewhile(lambda x: x, (fd.raw.read(1024*1024) for _ in repeat(None)))
        lines = sum(buf.count(b'\n') for buf in bufgen)
        fd.close()
        logger.debug("Done")
        
        return lines
        
""" Exceptions """   

class InvalidViewInterfaceError(AttributeError):
    def __init__(self, view):
        logger.critical("View {0} isn't valid !".format(view))
        raise CriticalRunError

class InvalidArgumentError(AttributeError):
    def __init__(self, msg):
        logger.critical(msg)
        raise CriticalRunError

class MissingFileError(Exception):
    def __init__self(self, msg):
        logger.critical(msg)
        raise CriticalRunError
        
class InvalidTaskError(Exception):
    def __init__self(self, msg):
        logger.critical(msg)
        raise CriticalRunError
        
class CriticalRunError(Exception):
    def __init__(self):
        logger.critical("Critical error - exiting")
        sys.exit(1)
        
        
        
class Sorting:
    """ Class for keeping sorting methods for the also_likes functionnality together """
    @staticmethod
    def freq_ascending_sort(data): # Data as : , returns ['<DOC_ID>', '<DOC_ID>'...]
        """ Description : Returns the document list sorted in ascending order of frequency
            Parameters  : Document list in format [(<DOC_ID>, <count>), (<DOC_ID>, <count>)...]
            Returns     : Document list in format [<DOC_ID>, <DOC_ID>...] sorted by ascending order
        """
        logger.debug("Sorting documents in ascending order")
        return sorted(data, key=lambda x:x[1], reverse=False)
        
    @staticmethod 
    def freq_descending_sort(data):
        """ Description : Returns the document list sorted in descending order of frequency
            Parameters  : Document list in format [(<DOC_ID>, <count>), (<DOC_ID>, <count>)...]
            Returns     : Document list in format [<DOC_ID>, <DOC_ID>...] sorted by descending frequency
        """
        logger.debug("Sorting documents in descending order")
        return sorted(data, key=lambda x:x[1], reverse=True)
    
    
    @staticmethod
    def biased_sort(data): # Data : JSON encoded strings {"doc":"<DOC_ID>", "weight":"<weight>"} (1 per line)
        """ Description : Return the document list sorted following values in the file DOC_WEIGHTS_FILE in csv format with values <DOC_ID>,<VALUE> for each line / document
            Parameters  : Document list in format [(<DOC_ID>, <count>), (<DOC_ID>, <count>)...]
            Returns     : Document list in format [<DOC_ID>, <DOC_ID>...] sorted by arbitrary order
        """
        doc_weights = {}
        logger.debug("Sorting documents in descending order")
        try: 
            with open(DOC_WEIGHTS_FILE) as f: # Multiple data loading yes... should probably be stored indeed.
                for line in f.readlines():
                    (doc_id, weight) = line.split(',')
                    doc_weights[doc_id] = int(weight)
                    
        except IOError:
            logger.warning("Biased document weights file not found - using default sort (desc_sort)")
            return Sorting.freq_descending_sort(data)
            
        sorted_docs = []
        
        for doc_id, freq in data:
            sorted_docs.append((doc_id, doc_weights.get(doc_id, freq))) # If we have no bias toward this document, simply use its frequency
        
        sorted_docs =  sorted(sorted_docs, key=lambda x:x[1], reverse=True)
        
        return sorted_docs


def check_doc_id(txt):
    """ Quickly checks if a document_id is likely to be valid. Return True / False depending on result """
    if len(txt) == 45 and txt[12] == '-' and all(letter in ['a','b','c','d','e','f','0','1','2','3','4','5','6','7','8','9','-'] for letter in txt):
        return True
    else:
        return False

def get_file_size(line_count):
    """ Description : Returns a human readable file line count. 38734 lines -> 39K
        Parameters  : Raw line count [line_count]
        Returns     : Human readable file line count (ex : 12M)
    """
    suffixes = [{"lower":1, "upper":999, "suffix":""},
                {"lower":999, "upper":999999, "suffix":"K"},
                {"lower":999999, "upper":999999999, "suffix":"M"}]
    
    for suffix in suffixes:
        if line_count > suffix["lower"] and line_count < suffix["upper"]:
            return "{0} {1}".format(round(line_count/suffix["lower"]), suffix["suffix"])
            
    return "N/A"

class Plotting:
    """ Class to gathers all plotting / image display related methods """
    @staticmethod
    def plot(axes, data):
        """ Description : Plots a given dataset on given axes along the first variable of this dataset
            Parameters  : axes [axes] to plot data [data] on. Data is a Pandas DataFrame and the first column is plotted
        """
        logger.debug("Plotting {0}".format(data.columns[0]))
        
        # Bar plot chart       
        axes = data.plot.bar(data.columns[0],'counts', ax=axes)
        axes.xaxis.set_tick_params(rotation=0)
        
        #plt.setp(axes.xaxis.get_majorticklabels(), ha='right') # Correct alignment of rotated ticks when using angle not %90
    
    def show_graph_img(axes, img): 
        """ Description : Displays an image [img}(also_likes graph) in the given matplotlib axis [img]. 
            Parameters  : Axes [axes] to display the image [img] to.
        """
        axes.axis('off')
        if graph_good: # Display graph without distorsion but break next multi-bar plots
            axes.imshow(img)
        else:
            axes.imshow(img, aspect='auto') # Multiple tries to make the image look great without breaking the plots, can't have both for now... distorting the image is less problematic than half-breaking all plots
        axes.grid(False)
        axes.yaxis.set_visible(False)
        axes.xaxis.set_visible(False)
     
    def reset_plot(axes):
        """ Show the axis again after displaying a also_likes graph"""
        #axes.axis('on')
        axes.grid(True)
        axes.yaxis.set_visible(True)
        axes.xaxis.set_visible(True)
        axes.cla()
        
        
def start_logging(level, log_file=None):
        """ Description : Initializes logging module for both stdout and file if requested. /!\ Handling with quiet flag not really perfect... 
            Parameters  : Verbosity level [level], variable to store logg and optional log_file
        """
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

        if level > 1:
            formatter = logging.Formatter('[%(levelname)s] : %(message)s')    
            stream_h.setFormatter(formatter)
        
        
        if log_file is not None: #os.access(os.path.dirname(log_file), os.W_OK):
            try:
               file_h  = logging.FileHandler(log_file)
            except PermissionError:
                logger.warning("Provided log file name is invalid - ignoring")
                
            logger.addHandler(file_h)
            if level > 1:
                file_h.setFormatter(formatter)
        
        if level > 0:
            logger.addHandler(stream_h)
                
def arg_parser():
    """ Returns an initialized ArgumentParser object """
    parser = argparse.ArgumentParser(description='Issuu data analytics software')

    gui_group = parser.add_mutually_exclusive_group()
    verbosity_group = parser.add_mutually_exclusive_group()


    gui_group.add_argument('--gui', action='store_true',\
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
    
    parser.add_argument("-x", action="store_true",
                        help="Display also_likes graph with no distortion /!\ Breaks future multibar plots... If not set, plots are fine but graph is distorted.")
                        
    gui_group.add_argument("--plt", action="store_true",\
                        help='Show PLOTS (without gui)')

    return parser
