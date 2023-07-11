import argparse
import csv

# def to_csv(input_data, name, save_path):
#     with open(save_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         data = [name]
#         for line in input_data.split('\n'):
#             line = line.split('=')
#             if len(line)==2:
#                 data.append(float(line[1])*100)
#         writer.writerow(data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="the path to the result file")
    parser.add_argument('--save_path', type=str, default='results.csv')
    parser.add_argument('--name', type=str, default='baseline', help='name of the input row')

    args = parser.parse_args()
    with open('results.txt','a', newline='') as txtfile:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            string=str(args.name)
            for line in f.readlines():
                if line.split()[0]=="f1":
                    string+=(","+str(float(line.split()[2])*100))
            txtfile.write(string)
"""
    with open(args.input_path, 'r', encoding='utf-8') as f:
        with open(args.save_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            data = [args.name]
            for line in f.readlines():
                if line.split()[0]=="f1":
                    data.append(float(line.split()[2])*100)
            writer.writerow(data)

 the code for classify tasks:
                if line[0].isalpha():
                    line = line.split('=')
                    if len(line)==2:
                        data.append(float(line[1])*100)
            writer.writerow(data)"""

#     input_data = """
# ar=0.35708582834331337
# bg=0.33592814371257484
# de=0.3650698602794411
# el=0.37524950099800397
# en=0.3974051896207585
# es=0.3335329341317365
# fr=0.3724550898203593
# hi=0.36746506986027944
# ru=0.36067864271457084
# sw=0.3590818363273453
# th=0.3500998003992016
# tr=0.3502994011976048
# ur=0.3600798403193613
# vi=0.37005988023952097
# zh=0.38862275449101796
# total=0.362874251497006
#     """

    # to_csv(input_data, args.name, args.save_path)

