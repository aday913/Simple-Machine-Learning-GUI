# GUI.py
# Original authoer: Alexander Day

# GOAL: main script to run a simple GUI through which data can be uploaded and
# machine learning algorithms can be trained

# Import relevant libraries:
import tkinter as tk    # Main GUI library
from tkinter import ttk # Used to create widgets
from tkinter.filedialog import askopenfilename  # User selects single file
import pandas   # Used for data handling

import KNN

class GUI(object):
    '''
    Main class object that will contain all the information of the GUI itself
    '''

    def __init__(self, test=False):
        '''
        Occurs when the class is initially called: GUI()

        If the user wants to test out certain widgets or processes without
        changing the main addWidgets function, set test to "True"
        '''
        self.window = tk.Tk()
        self.window.title('Machine Learning Detection GUI')
        if test:
            self.testAddWidgets()
        else:
            self.addInitialWidgets()
        self.window.mainloop()
    
    def addInitialWidgets(self):
        '''
        Function adds in the default widgets that the GUI will use, such
        as all of the relevant buttons, checkboxes, labels, etc.
        '''
        # Create welcome label
        self.welcomeLabel = ttk.Label(self.window, 
                        text='Welcome to the Machine Learning GUI!')
        self.welcomeLabel.grid(column=0, row=0)

        # Create instructions label
        self.instructionsLabel = ttk.Label(self.window, 
                text='Use the button below to choose your dataset (.csv file)')
        self.instructionsLabel.grid(column=0, row=1)

        # Create data input button
        self.dataInputButton = ttk.Button(self.window, text='Select Data',
                                    command=self.dataInputButtonClick)
        self.dataInputButton.grid(column=0, row=2)
    
    def dataInputButtonClick(self):
        '''
        Allows the user to choose the data file (.csv) that contains the data
        they want to use in the ML analysis
        '''
        filename = askopenfilename()
        output = pandas.read_csv(filename, header=0)
        self.dataFrame = pandas.DataFrame(output)

        self.nextStepLabel = ttk.Label(self.window, 
                        text='Which column name contains the classes?')
        self.nextStepLabel.grid(column=0, row=2)

        self.classColName = tk.StringVar()
        self.classColEntry = ttk.Entry(self.window, width=12, 
            textvariable=self.classColName)
        self.classColEntry.grid(column=0, row=3)

        self.classColumnInput = ttk.Button(self.window, text='Next Step',
                                    command=self.classInputClick)
        self.classColumnInput.grid(column=0, row=4)

        self.classColEntry.focus()
    
    def classInputClick(self):
        '''
        Button for when the user has inputted the name of the column with the
        classes for each observation
        '''
        self.classColName = self.classColName.get()

        self.nextStepLabel.configure(
        text='Input any other columns you want to exclude separated by ", "'
            )
        
        self.trashColNames = tk.StringVar()
        self.trashColNamesEntry = ttk.Entry(self.window, width=12, 
            textvariable=self.trashColNames)
        self.trashColNamesEntry.grid(column=0, row=3)

        self.trashColumnInput = ttk.Button(self.window, text='Next Step',
                                    command=self.trashInputClick)
        self.trashColumnInput.grid(column=0, row=4)

        self.trashColNamesEntry.focus()

    def trashInputClick(self):
        '''
        Button for when the user has inputted the names of columns that
        they do not want to be included in the data
        '''
        droppedCols = [self.classColName]
        temp = list(self.trashColNames.get().split(', '))
        if temp != ['']:
            for i in temp:
                droppedCols.append(i)
        
        self.y = self.dataFrame[self.classColName]
        self.X = self.dataFrame.drop(columns=droppedCols)
        self.X = self.X.values

        KNN.optimizeKNN(self.X, self.y)
        
    
    # ----------------------------------------------
    # Below functions are used for testing purposes!
    # ----------------------------------------------

    def testAddWidgets(self):
        '''
        Function adds in the default widgets that the GUI will use, such
        as all of the relevant buttons, checkboxes, labels, etc.
        '''
        pass

if __name__ == "__main__":
    gui = GUI()