
# Data
DATASET = 'caltech101'
DATA_DIR = 'data/101_ObjectCategories'
CLASS = 101
NUM_TRAIN = 7766 
BATCH = 64   
SUBSET = 4340 
START = 868 
ADDENDUM  = 434 

# Active learning setting
TRIALS = 3
CYCLES = 7

# Training setting
MARGIN = 1.0  
WEIGHT = 0.1
EPOCH = 50
LR = 0.01    
MOMENTUM = 0.9
WDECAY = 5e-4
MILESTONES = [40]
EPOCHL = 30 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model