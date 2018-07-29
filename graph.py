from itertools import takewhile, repeat, chain

class GraphBuilder:
    """ Top-level class for building a dot graph """
    
    """ Description : Creates a Graph instance for future use in the class
        Parameters  : [Graph_header] is the top section of the dot file, [ori_design] is the design prefix to add to dot labels in case the document or user is the original one 
    """
    def __init__(self, graph_header, ori_design):
        self.graph = Graph(graph_header, ori_design)
       
    def build_tree(self, ori_doc, dic, ori_reader=None):
        """ Description : Top-level class called by also_likes method of class Model. Returns an also_likes dot tree
            Parameters  : initial document [ori_doc], dictionnary {<DOC>:[readers], <DOC>:[readers]...} of also_likes documents with their readers. Optional reader [ori_reader] 
            Returns : Also likes dot tree string
        """
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
        """ Adds all documents in fiven list to the graph """
        for doc in doc_lst:
            self.graph.add_document(doc)
     
    def add_other_readers(self, readers_lst):
        """ Adds all readers in fiven list to the graph """
        for reader in readers_lst:
            self.graph.add_reader(reader)
            
    def add_ori(self, ori_doc, ori_reader):
        """ Description : Add original reader & document to the graph
            Parameters  : Initial document [ori_doc] & initial reader [ori_reader]"""
        self.graph.add_document(ori_doc, pos="center", ori=True)
        if ori_reader:
            self.graph.add_reader(ori_reader, pos="center", ori=True)

    def add_link(self, reader, doc):
        """ Add a link (graphviz arrow) between a given reader and document """ 
        self.graph.add_link(reader, doc)
        
# Make generic sections ?
class Graph:
    """ Low-level class for dot tree creation """
    def __init__(self, header, ori_design):
        self.header = header
        self.ori_design = ori_design
        self.docs_labels = []
        self.readers_labels = []
        self.readers = ["{ rank = same; \"Readers\";"]
        self.docs = ["{ rank = same; \"Documents\";"]
        self.links = []
    
    def add_document(self, doc, pos=None, ori=False):
        """ Description : Add a document to the tree. Node with be colored if document is the original from the also_likes functionnality
            Parameters  : [doc] document to add to the tree, [ori]=True if document is the reference/initial document
        """
        lb = "\"{0}\" [label=\"{0}\", shape=\"circle\"".format(doc)
        if ori:
            lb += self.ori_design
        
        lb += "];"
        if "\"{0}\";".format(doc) not in self.docs:
            self.docs.append("\"{0}\";".format(doc))
            self.docs_labels.append({'label':lb, 'pos':pos})
        
    def add_reader(self, reader, pos=None, ori=False):
        """ Description : Add a reader to the tree. Node with be colored if reader is the original from the also_likes functionnality
            Parameters  : [doc] reader to add to the tree, [ori]=True if reader is the reference/initial reader
        """
        lb = "\"{0}\" [label=\"{0}\", shape=\"box\"".format(reader)
        if ori:
            lb += self.ori_design
        
        lb += "];"
        if "\"{0}\";".format(reader) not in self.readers:
            self.readers.append("\"{0}\";".format(reader))
            self.readers_labels.append({'label':lb, 'pos':pos})
        
    def add_link(self, reader, doc):
        """ Description : Add a graphviz arrow (link) between a reader and a document 
            Parameters  : [doc] & [reader] to draw the arrow to and from
        """
        if "\"{0}\" -> \"{1}\";".format(reader, doc) not in self.links:
            self.links.append("\"{0}\" -> \"{1}\";".format(reader,doc))
        
    def get_labels(self):
        """ Returns all labels - ideally the original reader and document should be in the middle of the graph... but not really """
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
        """ Return the dot graph as a string """
        labels = self.get_labels()
        self.readers.append("};")
        self.docs.append("};")
        self.links.append("};")
        graph = self.header + labels + self.readers + self.docs + self.links
        graph.append("}")
        graph = '\n'.join(graph)
        
        return graph
