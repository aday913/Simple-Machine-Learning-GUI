# Simple-Machine-Learning-GUI
## Original author: Alexander Day

### Overall goal:
Graphical User Interface (GUI) that can be used to quickly apply common machine learning algorithms to user-provided datasets in order to develop a model for future predictions or just to determine whether such a model is feasible.

**Todo's**:
- [x] Develop initial GUI system that reads in data as a pandas dataframe
- [x] Develop initial ML algorithm to test that system works
- [ ] Incorporate ML algorithms that train on the given dataset
- [ ] Add additional ML features such as K-fold CV and confusion matrix creation
- [ ] Ensure proper libraries are installed when user attempts to run the GUI  

### To run:  
Run the following line to clone this repo:
```
git clone https://github.com/aday913/Simple-Machine-Learning-GUI.git
```

Then download all required libraries (assuming that you have Python 3.X and pip/pip3 connected via PATH):
```
pip install -r requirements.txt
```
<em> Note: I'd recommend setting up a virtual environment to store this instance of Python in </em>  

Finally, to run the GUI, simply use the following command:
```
python GUI.py
```
