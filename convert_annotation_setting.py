
"""
This script converts the jsonlines data format to the csv format for annotation
usage: python convert_from_jsonl.py <inpath>
"""
import plac
import jsonlines
from pathlib import Path
import csv
import jsonlines
from helperutilz import *
from ekphrasis_preprocess import text_processor
import plac
import pandas as pd

# plac.annotations(inpath=("inpath for ", "positional", "i", Path),
#                  outpath=("outpath for jsonlines for prodigy", "positional", "o", Path),
#                  process=("boolean", "option", "p", bool),
#                  label=("string ", "option", "l", str),
#                  )


def convert(inpath, outpath, process=True, label='fullname'):
    print(f"reading in {inpath}")
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    unique_set = set()
    cnt = 0
    kept = 0
    with jsonlines.open(inpath, 'r') as reader:
        with jsonlines.open(outpath, 'w', newline='') as csvfile:
            fieldnames = ['tweetid', 'text', 'source']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for obj in reader:
                cnt += 1
                if cnt % 100 == 0:
                    print(f"processed {cnt} tweets")
                text = obj['text']
                tweetid = obj['metadata']['tweetid']
                urls = obj['metadata']['urls']
                if urls != '':
                    continue

                if text not in unique_set:
                    unique_set.add(text)
                else:
                    continue
                kept += 1
                writer.writerow({'source': inpath.split('/')[-1],
                                 'tweetid': tweetid, 'text': text})

    print(f"read in {cnt} tweets and wrote out {kept} tweets to file {outpath}")
# fid = '/Users/user/Downloads/augmented data-train_aug_05-03-20-22-06-49.csv'
# bn = fid.split('/')[-1]
# bn = bn.replace('.csv', '.jsonl')
# bn = bn.replace(' ', '-')
# print(bn)
train_df = pd.read_csv('data/orig/task4_test_participant.csv')
train_df['class'] = train_df['class'].map(str.strip)

with jsonlines.open(f'data/task4_ekp_test.jsonl', 'w') as writer:
    for i, row in train_df.iterrows():
        text = row['text']
        text = text.replace("_U", "<user>")
        text = " ".join(text_processor.pre_process_doc(text))
        # print(text)
        label = row['class']
        tweetid = row['tweetid']
        accept = class_map.get(label)
        new_obj = {
            'text': text,
            'metadata': {
                tweetid
              # 'tweetid': tweetid
            },
        #   'label': accept
        #  'accept': [accept],
         #  'answer': 'accept'
        }
        writer.write(new_obj)


# if __name__ == '__main__':
#     plac.call(convert)
