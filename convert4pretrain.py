from __future__ import unicode_literals, print_function
import plac
import random
import ujson
from pathlib import Path
import thinc.extra.datasets
import spacy
import pandas as pd
from spacy.util import minibatch, compounding
from datetime import datetime
today = datetime.today()
print(today)

label_map_binary = {
    'ABUSE': 1,
    'CONSUMPTION': 0,
    'UNRELATED': 0,
    'MENTION': 0,
}


def load_data(reader, limit=0, split=0.8):
    # Partition off part of the train data for evaluation
    train_data = []
    for obj in reader:
        text = obj['text']
        label = obj['label']
        binlabel = label_map_binary.get(label)
        train_data.append((text, binlabel))
        # tweetid = obj['metadata']['tweetid']
    # write_jsonl('data/binary.jsonl')
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{"ABUSE": bool(y), "NONABUSE": not bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])




def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NONABUSE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}




def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
   # cnt = 0
    with Path(file_path).open('r', encoding='utf8') as f:
        for line in f:
            # cnt += 1
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))

# for file in ['data/forprodigy/train-ekp.jsonl', 'data/forprodigy/val-orig-ekp.jsonl']:
#     nn = file.replace('ekp.jsonl', 'ekp-new.jsonl')
#     reader = read_jsonl(file)
#     write_jsonl(file_path=nn, lines=reader)

#
@plac.annotations(
    infile=("infile","option", "i", str),# "positional", "i", Path),
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path),
)
def main(infile='data/forprodigy/train-ekp-new.jsonl', model=None, output_dir=None, n_iter=100, n_texts=2000, init_tok2vec=None):

    infile = Path(infile)
    reader = read_jsonl(file_path=infile)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("ABUSE")
    textcat.add_label("NONABUSE")
    # textcat.add_label("CONSUMPTION")
    # textcat.add_label("MENTION")
    # textcat.add_label("UNRELATED")
    # if n_texts == -1:
    #     max_n = 0
    # else:

   # print(f"utilizing {len(lines)} samples for training")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(reader=reader, limit=0)

    print(
        "Using examples ({} training, {} evaluation)".format(
            len(train_texts), len(dev_texts)
        )
    )

    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
    print(train_data[0])
    random.shuffle(train_data)
    print(train_data[0])
    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # test the trained model
    val_sections = read_jsonl(file_path='data/forprodigy/val-orig-ekp-new.jsonl')
    # val_sections = [
    #     {"text":"<user> diazepam is an addictive benzo . it ' s ok short term . i find sleep & meditation gives me peace & calm . when i can get it .","metadata":{"tweetid":1200394262568943617},"label":"CONSUMPTION"},
    #     {"text":"<user> <user> <user> <user> and the most commonly prescribed non - opiod painkiller ( in my experience ) is tramadol , which can cause <allcaps> severe </allcaps> adverse reactions in people w / even mild mental health issues , pharmacologically treated or not .","metadata":{"tweetid":1202390827181457409},"label":"MENTION"},
    #     {"text":"how did a - <number> working with women artists be about not putting in the work for their marriage ? lyrica are you okay ? ! <repeated> not tf you are not !","metadata":{"tweetid":1201767157232807936},"label":"UNRELATED"},
    #     {"text":"<user> <user> <user> <user> thank god , i have not had a proper sleep in ages ! <repeated> <hashtag> valium </hashtag>","metadata":{"tweetid":1202139241477742594},"label":"CONSUMPTION"},
    #     {"text":"i am on entirely way too much xanax to function during this workout","metadata":{"tweetid":1200499351031894016},"label":"ABUSE"},
    #     {"text":"where ' s my vyvanse when i need it \ud83d\ude02","metadata":{"tweetid":1198476327004856320},"label":"CONSUMPTION"},
    #     {"text":"morphine hits the spot","metadata":{"tweetid":1201337511672524800},"label":"CONSUMPTION"},
    #     {"text":"i \u2019 m still so fucking anxious ! got to pop a pill of valium again ? ! <repeated>","metadata":{"tweetid":1202341837673041920},"label":"CONSUMPTION"},
    #     {"text":"<user> <user> <user> <user> ouch . <repeated> i have had to convert patients to ir because of the ridiculous cost , which is a double edged sword because now there are that many more adderall pills floating around . it ' s a no win for the patient .","metadata":{"tweetid":1198476378729041920},"label":"MENTION"},
    #     {"text":"<user> that was not annie hall or diane keaton that needed the valium but i do now thanks","metadata":{"tweetid":1200980670782300160},"label":"MENTION"},
    #     {"text":"<user> suboxone for opiate dependent individuals does not make them high - so the reason is self medication .","metadata":{"tweetid":1199509721868374022},"label":"MENTION"},
    #     {"text":"small brain : love lil pump med brain : xanax rappers , auto tune are ruining hip hop ! argh ! i miss rap from the clinton administration ! big brain : we can listen to all kinds of hip hop and enjoy them biggest brain : trippie redd is the only relevant artist in the world .","metadata":{"tweetid":1198691681119490050},"label":"MENTION"},
    #     {"text":"<user> do they have a physician ? many will give free samples . if generic some of those meds are cheap even without insurance . xanax generic runs about <money> for a bottle . what medication does your friend need ?","metadata":{"tweetid":1200884551108714497},"label":"MENTION"},
    #     {"text":"<user> <user> <user> the uninformed would think that the story was saying suboxone clinics are equal to pill mills .","metadata":{"tweetid":1198545033261199361},"label":"MENTION"}
    # ]
    comparison_results = []
    vlc = 0
    for val_obs in val_sections:
        vlc += 1
        o = {}
        tweetid = val_obs['metadata']['tweetid']
        test_text = val_obs.get('text')
        true_label = val_obs.get('label')
        doc = nlp(test_text)
        o['Y_TRUE'] = true_label
        o['cats'] = doc.to_json().get('cats')
        o['text'] = doc.to_json().get('text')
        o['tweetid'] = tweetid
        #print(o)
        comparison_results.append(o)
        # if vlc == 10:
        #     break
    df_results = pd.DataFrame(comparison_results)
    print(df_results.head())
    df_results.to_csv(f'data/results/results-{today}.csv', index=False)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)

        doc2 = nlp2('adderall had me doing sit ups at midnight')
        print(doc2.to_json())

if __name__ == "__main__":
    plac.call(main)

