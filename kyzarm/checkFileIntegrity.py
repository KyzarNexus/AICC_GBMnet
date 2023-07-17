import os 
import numpy as np

main_dir = "D:\\projects\\AI_Campus_Cedars_Group1\\data\\NIfTI-files\\images_structural\\"
# main_dir = "D:/projects/AI_Campus_Cedars_Group1/data/NIfTI-files/images_structural"
keywords = ['T1','T1GD','T2','FLAIR']

subject_list = os.listdir(main_dir)

file_record = dict.fromkeys(subject_list)

for subject in file_record.keys() :
    file_check = [None]*len(keywords)
    subject_folder = os.path.join(os.path.dirname(main_dir),subject)
    for i in range(0,len(keywords)):
        if any(keywords[i] in x for x in os.listdir(subject_folder)):
            # keywords[i] in os.listdir(subject_folder):
            file_check[i] = 1
        else:
            file_check[i] = 0
    file_record[subject] = file_check
    
total_sum = [0]*len(keywords)
for subject in file_record.keys():
    l1 = np.array(total_sum)
    l2 = np.array(file_record[subject])
    total_sum = l1 + l2

print('\nDataset Integrity Check\n')
print('Total Subjects: ' + str(len(file_record)))
print("Keywords: " + str(keywords))
print("Sums per file type: " + str(total_sum))


    
