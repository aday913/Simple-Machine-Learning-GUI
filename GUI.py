# GUI.py
# Original authoer: Alexander Day

# GOAL: main script to run a simple GUI through which data can be uploaded and
# machine learning algorithms can be trained

# Import relevant libraries:
import tkinter as tk    # Main GUI library
from tkinter import ttk # Used to create widgets
from tkinter.filedialog import askopenfilename  # User selects single file
import pandas   # Used for data handling

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
            self.addTestWidgets()
        else:
            self.addWidgets()
        self.window.mainloop()
    
    def addWidgets(self):
        '''
        Function adds in the default widgets that the GUI will use, such
        as all of the relevant buttons, checkboxes, labels, etc.
        '''
        # Create video analysis label
        self.videoLabel = ttk.Label(self.window, 
                        text='Click button to show image from video file')
        self.videoLabel.grid(column=0, row=0)

        # Create image analysis label
        self.imageLabel = ttk.Label(self.window, 
                            text='Click button to analyze particle image')
        self.imageLabel.grid(column=0, row=1)

        # Create video analysis button
        # self.testVideoButton = ttk.Button(self.window, text='Select Video',
        #                             command=self.testVideoButtonClick)
        # self.testVideoButton.grid(column=1, row=0)

        # Create image analysis button
        self.testImageButton = ttk.Button(self.window, text='Select Image',
                                    command=self.testImageButtonClick)
        self.testImageButton.grid(column=1, row=1)

        # Create image analysis output label
        self.outputLabel = ttk.Label(self.window,
                            text='Output of image analysis: ')
        self.outputLabel.grid(column=0, row=2)
    
    def testImageButtonClick(self):
        '''
        Test func for when the image button in the test widgets is pressed
        '''
        filename = askopenfilename()
        output = pandas.read_csv(filename, header=0)
        dataFrame = pandas.DataFrame(dataset)
        self.outputLabel.configure(text=dataFrame.columns[0])

if __name__ == "__main__":
    gui = GUI()