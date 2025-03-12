frac = 0.5
import random 

train_jsonl = '/home/wahabu/data/construction/MepDetection/data/YOLO-5/dataset/_annotations.train.jsonl'
valid_jsonl = '/home/wahabu/data/construction/MepDetection/data/YOLO-5/dataset/_annotations.valid.jsonl'
#test_jsonl = main_dir + "_annotations.test.jsonl"

all_files = [train_jsonl, valid_jsonl]  # list of all jsonl files
for file in all_files:
    new_lines = []
    i = 0
    with open(file, "r") as f:
        for line in f:
            i+=1
            if random.random() < frac:
                new_lines.append(line)
    
    
    print(f"Total lines in {file}: {i} {len(new_lines)}")
    #with open(file, "w") as f:
     #   f.writelines(new_lines) 


