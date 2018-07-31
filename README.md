# Data-analyser
Data analyser for Issuu data

## Context :
- Original developement date : 11/2017
- Developped as part of an Industrial Programming coursework at HWU.
- Not yet familiar with several of the used libraries at that time.

## Features 
- GUI and CLI support
- Various plots regarding the data (by countries, by user agent etc), integrated within GUI
- (Sort of) recommender system (i.e. If X read document Y, get the list of documents read by others who have too read Y). 
- Asynchronous data loading 
- MVC architecture

## Libraries
- pandas
- tkinter
- matplotlib
- threading
- logging
- argparse
- json
- graphviz
- user_agents
- psutils

## Feedback from lecturer :
- "v.good coding"
- "v.good report"
- A few minor bugs fixed after demo (some due to Python incompatibilities)
- Grade : 79/100

## Overall : 
- Stability    : OK
- Readability  : Good
- Code quality : Good
- Performance  : MOK / OK -> Loading style optimized for GUI usage (i.e. all computations done once), slow for CLI (as a single query would trigger uncessary computation). Some optimization done as seen in the plot folder.
