from we import WordEmbedding
from debias import debias
import json


def embedding_dict(words, vecs):
    """
    inputs
    words: list of words in embedding
    vecs: corresponding vector for words

    output: dictionary with keys as the words and values as the word's embeddings in list form.
    """
    emb_dict = {}

    for i in range(len(words)):
        word = words[i]
        vec = vecs[i]
        emb_dict[word] = vec

    return emb_dict


def load_gender_wordlists(words, vecs):
    """
    inputs
    words: list of words in embedding
    vecs: corresponding vector for words

    output: 3 lists of gender-meaningful words, gender_specific_words, defs, and equalize_pairs
    """
    emb_dict = embedding_dict(words, vecs)
    avail_words = emb_dict.keys()

    defs = []
    equalize_pairs = []
    gender_specific_words = []

    with open('./data/definitional_pairs.json', "r") as f:
        temp_defs = json.load(f)
        for pair in temp_defs:
            if pair[0] in avail_words and pair[1] in avail_words:
                defs.append(pair)

    with open('./data/equalize_pairs.json', "r") as f:
        temp_equalize_pairs = json.load(f)
        for pair in temp_equalize_pairs:
            if pair[0] in avail_words and pair[1] in avail_words:
                equalize_pairs.append(pair)

    with open('./data/gender_specific_full.json', "r") as f:
        temp_gender_specific_words = json.load(f)
        for word in temp_gender_specific_words:
            if word in avail_words:
                gender_specific_words.append(word)

    return gender_specific_words, defs, equalize_pairs


def save_debiased_embeddings_to_file(debiased_words, debiased_vecs, target_file):
    """
    inputs
    debiased_emb: dictionary with keys as the words and values as the word's embeddings in list form (without the deleted axis)
    target_file: name of the file that the debiased_emb will be saved to. MUST NOT EXIST YET.

    output: none
    """
    f = open(target_file, "a")
    for i in range(len(debiased_words)):
        word = debiased_words[i]
        emb = debiased_vecs[i]
        emb_str = ""
        for num in emb:
            emb_str += str(num) + " "
        temp = word + " " + emb_str + "\n"
        f.write(temp)
    f.close()
    return


if __name__ == "__main__":
    # ------------------- GUTENBERG DATASET -------------------
    # load original gutenberg embedding
    gutenberg_emb = WordEmbedding('gutenberg_embeddings.txt')
    # load some gender related word lists to help with debiasing
    gender_specific_words, defs, equalize_pairs = load_gender_wordlists(gutenberg_emb.words, gutenberg_emb.vecs)
    # debias
    debiased_emb = debias(gutenberg_emb, gender_specific_words, defs, equalize_pairs)
    # save debiased embeddings to new file
    save_debiased_embeddings_to_file(debiased_emb.words, debiased_emb.vecs, "2016debias_gutenberg_emb.txt")

    # ------------------- WIKIPEDIA DATASET -------------------
    # load original wiki embedding
    # wiki_emb = WordEmbedding('wikipedia_embeddings.txt')
    # load some gender related word lists to help with debiasing
    # gender_specific_words, defs, equalize_pairs = load_gender_wordlists(wiki_emb.words, wiki_emb.vecs)
    # debias
    # debiased_emb = debias(wiki_emb, gender_specific_words, defs, equalize_pairs)
    # save debiased embeddings to new file
    # save_debiased_embeddings_to_file(debiased_emb.words, debiased_emb.vecs, "2016debias_wiki_emb.txt")

