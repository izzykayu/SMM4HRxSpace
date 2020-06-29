
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
import en_core_sci_lg

nlp = en_core_sci_lg.load()


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


# import glob
# import os
# folder = 'data/tweets'
# #files = glob.glob(f"{folder}/*.jsonl")
# tweetids = set()
# fout = 'tweets-jsonl.txt'
# cnt = 0
# with open(fout, 'a') as f:
#     for file in files:
#         with jsonlines.open(file, 'r') as reader:
#             for obj in reader:
#                 text = obj['text']
#                 tweetid = obj['metadata']['tweetid']
#
#                 if tweetid in tweetids:
#                     continue
#                 #else:
#                 tweetids.add(tweetid)
#                 cnt += 1
#                 if cnt % 1000 == 0:
#                     print(cnt)
#                 text = " ".join(text_processor.pre_process_doc(text))
#                 f.write(text + os.linesep)
# import glob
# import os
# cnt = 0
# folder = 'data/orig'
# files = glob.glob(f"{folder}/*.csv")
# fout = 'tweets-jsonl.txt'
# with open(fout, 'a') as f:
#     for file in files:
#         df_temp = pd.read_csv(file)
#         texts = df_temp['unprocessed_text'].tolist()
#         for text in texts:
#             cnt += 1
#             if cnt % 1000 == 0:
#                 print(cnt)
#
#             new_text = " ".join(text_processor.pre_process_doc(text))
#             f.write(new_text + os.linesep)

# import glob
import os
import gzip
cnt = 0
unique_sents = set()
fout = 'data/embed/mimic-notes.txt'
with open(fout, 'w') as f:
    with gzip.open('/Users/user/Downloads/NOTEEVENTS.csv.gz', 'r') as file:
        df_temp = pd.read_csv(file, low_memory=False)



    df_temp = df_temp.drop_duplicates('TEXT')

    print(f" {df_temp.shape[0]} notes to process")

    for i, row in df_temp.iterrows():
        text = row['TEXT']
        doc = nlp(text)
        for sent in doc.sents:
            toks = text_processor.pre_process_doc(sent.text)

            new_text = " ".join(toks)
            if len(toks) < 10:
                continue
            if new_text in unique_sents:
                continue
            unique_sents.add(new_text)
            cnt += 1
            if cnt % 100 == 0:
                print(f"{cnt} sents processed")

            f.write(new_text + os.linesep)
print(f"{len(list(unique_sents))} unique sentences added")





# tweetids = set()
# fout = 'tweets-jsonl.txt'
# cnt = 0
# with open(fout, 'a') as f:
#     for file in files:
#         with jsonlines.open(file, 'r') as reader:
#             for obj in reader:
#                 text = obj['text']
#                 tweetid = obj['metadata']['tweetid']
#
#                 if tweetid in tweetids:
#                     continue
#                 #else:
#                 tweetids.add(tweetid)
#                 cnt += 1
#                 if cnt % 1000 == 0:
#                     print(cnt)
#                 text = " ".join(text_processor.pre_process_doc(text))
#                 f.write(text + os.linesep)
# fid = '/Users/user/Downloads/augmented data-train_aug_05-03-20-22-06-49.csv'
# bn = fid.split('/')[-1]
# bn = bn.replace('.csv', '.jsonl')
# bn = bn.replace(' ', '-')
# print(bn)
# train_df = pd.read_csv('/Users/user/Downloads/augmented data-train_aug_05-03-20-22-06-49.csv')
# train_df['class'] = train_df['class'].map(str.strip)
#
# with jsonlines.open(f'data/forprodigy/{bn}', 'w') as writer:
#     for i, row in train_df.iterrows():
#         text = row['text']
#         text = text.replace("_U", "<user>")
#         text = " ".join(text_processor.pre_process_doc(text))
#         # print(text)
#         label = row['class']
#        # tweetid = row['tweetid']
#         accept = class_map.get(label)
#         new_obj = {
#             'text': text,
#             'metadata': {
#               # 'tweetid': tweetid
#             },
#            'label': accept
#         #  'accept': [accept],
#          #  'answer': 'accept'
#         }
#         writer.write(new_obj
#
#
#     )

#
# import glob
# import os
# cnt = 0
# folder = 'data/cadec/CADEC-txt-ann'
# files = glob.glob(f"{folder}/*.txt")
# print(len(files))
# unique_sents = set()
# fout = 'data/embed/cadec.txt' #chqa.txt'
#
# with open(fout, 'w') as f:
#     for file in files:
#         with open(file, 'r') as reader:
#             cnt += 1
#            # print(f"reading in {file}")
#             text = reader.read()
#             toks = text_processor.pre_process_doc(text)
#
#             new_text = " ".join(toks)
#
#             f.write(new_text + os.linesep)
# print(f"wrote out {cnt} docs from cadec corpus")
#
#
#
#
# email_folder = '/Users/user/PycharmProjects/InferDrugTweet/data/CHQA-Corpus-1.0/CHQA-email/20_Practice'
# email_folder2 = '/Users/user/PycharmProjects/InferDrugTweet/data/CHQA-Corpus-1.0/CHQA-email/1720_Unadjudicated/TextFiles'
# web_folder = '/Users/user/PycharmProjects/InferDrugTweet/data/CHQA-Corpus-1.0/CHQA-web/'
# texts = []
# cnt = 0
# with open('data/embed/chqa.txt', 'w') as f:
#     for folder in [email_folder, email_folder2, web_folder, ]:
#
#         files_ = glob.glob(f"{folder}/*.txt")
#         print(f"{len(files_)} in {folder}")
#         for file_ in files_:
#
#
#             with open(file_, 'r') as file_in:
#
#                 text = file_in.read()
#                 new_text = " ".join(text_processor.pre_process_doc(text))
#                 f.write(new_text + os.linesep)
#                 cnt += 1
#
#
# print(f"{cnt} files from chqa corpus")

# if __name__ == '__main__':
#     plac.call(convert)
