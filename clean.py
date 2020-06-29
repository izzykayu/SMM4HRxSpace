import jsonlines
import plac
import jsonlines
from pathlib import Path
import csv

plac.annotations(
    inpath=("inpath for ", "positional", "i", Path),
    outpath=("outpath", "option", "f", str),
                 )

def main(inpath, outpath='example.jsonl'):
    unique_set = set()
    cnt = 0
    kept = 0
    with jsonlines.open(inpath, 'r') as reader:
        with jsonlines.open(outpath, 'w') as writer:
            for obj in reader:
                cnt += 1

                text = obj['text']
                if text not in unique_set:
                    unique_set.add(text)
                else:
                    continue
                kept += 1
                writer.write(obj)
    print(f"read in {cnt} tweets from {inpath} and wrote out {kept} to {outpath} ")

if __name__ == '__main__':
    plac.call(main)