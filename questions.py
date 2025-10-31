from questiongenerator import QuestionGenerator
from questiongenerator import print_qa
from summarize import summary, result
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qg = QuestionGenerator()

original_string = result
# Define a translation table to remove specified symbols
translation_table = str.maketrans("‘", "“", ",")
translation_table2 = str.maketrans(":", ";", "|")

temp = original_string.translate(translation_table)
modified_string = temp.translate(translation_table2)


print(modified_string)


article = modified_string

qa_list = qg.generate(
    article, 
    num_questions=2, 
    answer_style='all'
)
# print_qa(qa_list)