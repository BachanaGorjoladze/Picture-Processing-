import os
import sys
from questions import print_qa, qa_list
from summarize import summary

with open('Final.txt', 'w') as file:
    original_stdout = sys.stdout
    sys.stdout = file
    print(summary)
    print('')
    print_qa(qa_list)

    sys.stdout = original_stdout

os.system('start Final.txt')