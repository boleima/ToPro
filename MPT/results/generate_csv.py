import csv
import sys

file_path, name = sys.argv[1:]
# output_path = file_path.split('.')[0] + '.csv'
output_path = '/mounts/work/nie/projects/ProFiT-S/MPT/results/results.csv'

with open(file_path, 'r', encoding='utf-8') as f_read:
    with open(output_path, 'a', encoding='utf-8') as f_write:
        csv_writer = csv.writer(f_write)
        scores = [name]
        for line in f_read.readlines():
            score = line.split()[-1]
            scores.append(round(float(score)*100, 2))
        
        csv_writer.writerow(scores)