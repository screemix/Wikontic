import json
from tqdm import tqdm
import re
import os

IN_DIR = "/data/home/wikidata-dump/"
MAX_ID = 107674455

def extract_ids():
    """
    Merge id info with the label and description info
    :return:
    """

    print("Extracting ids...")
    data = dict()

    with open(os.path.join(IN_DIR, "out.txt")) as f:
        for line in tqdm(f, total=MAX_ID):

            info = line.strip().split(":")
            ln, content = info[0], info[1]
            ln = int(ln.strip())

            content = content.strip()

            try:
                content = re.findall('[A-z][0-9]+', content)[0]
                data[ln] = {
                    "id": content, 
                    "label": '',
                    "description": ''
                }
            except Exception as e:
                print(e)
                print(ln)
                print() 

    json.dump(data, open("preprocessed_entities.json", "w"), indent=3)
    print("Total entries: ", len(data))


def extract_labels():

    print('Processing labels...')
    data = json.load(open("preprocessed_entities.json"))

    with open(os.path.join(IN_DIR, "out-labels.txt")) as f:

        for line in tqdm(f, total=MAX_ID):

            info = line.strip().split(":")
            ln = info[0]

            content = ":".join(info[1:])
            content = content[:len(content)-1]
            content = "{" + content + "}}"
            content = json.loads(content)

            if 'descriptions' in content.keys():
                continue
            else:
                data[ln]["label"] = content['labels']['en']['value']


    print('Saving labels...')

    with open("preprocessed_entities.json", "w") as f:
        json.dump(data, f, indent=3)


def extract_descriptions():
    
    data = json.load(open("preprocessed_entities.json"))

    print('Processing descriptions...')

    with open(os.path.join(IN_DIR, "out-descriptions.txt")) as f:

        for line in tqdm(f, total=MAX_ID):

            info = line.strip().split(":")
            ln =  info[0]

            content = ":".join(info[1:])
            content = content[:len(content)-1]
            content = "{" + content + "}}"

            try:
                content = json.loads(content)
                data[ln]['description'] = content['descriptions']['en']['value']

            except Exception as e:
                print(ln)
                print(e)
                print()

    print('Saving descriptions...')

    with open("preprocessed_entities.json", "w") as f:
        json.dump(data, f, indent=3)


def convert2mapping():

    data = json.load(open("preprocessed_entities.json"))
    id2label_and_desc = dict()

    for key in tqdm(data.keys(), total=len(data)):
        
        wikidata_id = data[key]['id']
        desc = data[key]['description']
        label = data[key]['label']
        
        id2label_and_desc[wikidata_id] = {
            "label": label, 
            "description": desc
        }
    
    with open("id2label_and_desc.json", "w") as f:
        json.dump(id2label_and_desc, f, indent=3)

if __name__=="__main__":
    
    extract_ids()
    extract_labels()
    extract_descriptions()
    convert2mapping()
