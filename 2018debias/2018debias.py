import numpy as np


def embedding_dict(textfile_of_embeddings):
    """
    input: textfile with the embeddings
    output: dictionary with keys as the words and values as the word's embeddings in list form
    """
    emb_dict = {}

    with open(textfile_of_embeddings) as f:
        lines = f.readlines()

    for line in lines:
        split = line.split()
        word = split[0]
        del split[0]
        split = [float(x) for x in split]
        emb_dict[word] = split

    return emb_dict


def combine_gendered_lists():
    """
    lists of words from: https://github.com/gonenhila/gender_bias_lipstick/tree/master/data/lists
    output: list of gender-pair words with each entry formatted as [male_version, female_version]
    """
    gender_pairs = [["monastery", "convent"], ["spokesman", "spokeswoman"], ["Catholic_priest", "nun"],
                      ["Dad", "Mom"], ["Men", "Women"], ["councilman", "councilwoman"], ["grandpa", "grandma"],
                      ["grandsons", "granddaughters"], ["prostate_cancer", "ovarian_cancer"],
                      ["testosterone", "estrogen"], ["uncle", "aunt"], ["wives", "husbands"], ["Father", "Mother"],
                      ["Grandpa", "Grandma"], ["He", "She"], ["boy", "girl"], ["boys", "girls"], ["brother", "sister"],
                      ["brothers", "sisters"], ["businessman", "businesswoman"], ["chairman", "chairwoman"],
                      ["colt", "filly"], ["congressman", "congresswoman"], ["dad", "mom"], ["dads", "moms"],
                      ["dudes", "gals"], ["ex_girlfriend", "ex_boyfriend"], ["father", "mother"], ["guy", "gal"],
                      ["fatherhood", "motherhood"], ["fathers", "mothers"], ["fella", "granny"],
                      ["fraternity", "sorority"], ["gelding", "mare"], ["gentleman", "lady"], ["gentlemen", "ladies"],
                      ["grandfather", "grandmother"], ["grandson", "granddaughter"], ["he", "she"],
                      ["himself", "herself"], ["his", "her"], ["king", "queen"], ["kings", "queens"],
                      ["male", "female"], ["males", "females"], ["man", "woman"], ["men", "women"], ["nephew", "niece"],
                      ["prince", "princess"], ["schoolboy", "schoolgirl"], ["son", "daughter"], ["sons", "daughters"],
                      ["twin_brother", "twin_sister"]]
    return gender_pairs


def find_gender_axis(emb_dict, gendered_words, size_of_emb):
    """
    This function finds all the existing gender pairs in emb_dict and sutracts the difference between the male and
    female version of the word embeddings. The differences between all the pairs are added together to find which
    axis (index) of the embedding has the greatest difference. This is the axis with the most gender information.

    inputs
    emb_dict: dictionary with keys as the words and values as the word's embeddings in list form
    gendered_words: list of words that are gendered

    output: the index of the most gendered axis
    """
    difference_list = np.zeros(size_of_emb)
    emb_words = emb_dict.keys()

    for pair in gendered_words:
        if (pair[0] in emb_words) and (pair[1] in emb_words):
            word1_emb = np.array(emb_dict.get(pair[0]))
            word2_emb = np.array(emb_dict.get(pair[1]))
            diff = np.subtract(word1_emb, word2_emb)
            abs_diff = np.absolute(diff)
            difference_list = np.add(difference_list, abs_diff)

    gendered_axis = np.where(difference_list == np.amax(difference_list))
    return int(gendered_axis[0][0])


def remove_gender_axis(emb_dict, axis):
    """
    This function removes the axis with the most gender information to attempt to decrease gender bias in word
    embeddings.

    inputs
    emb_dict: dictionary with keys as the words and values as the word's embeddings in list form
    axis: the index that will be deleted from each embedding

    output: dictionary with keys as the words and values as the word's embeddings in list form (without the deleted axis)
    """
    debiased_emb = {}
    emb_words = emb_dict.keys()

    for word in emb_words:
        emb = emb_dict.get(word)
        del emb[axis]
        debiased_emb[word] = emb

    return debiased_emb


def save_debiased_embeddings_to_file(debiased_emb, target_file):
    """
    inputs
    debiased_emb: dictionary with keys as the words and values as the word's embeddings in list form (without the deleted axis)
    target_file: name of the file that the debiased_emb will be saved to. MUST NOT EXIST YET.

    output: none
    """
    words = debiased_emb.keys()
    f = open(target_file, "a")
    for word in words:
        emb = debiased_emb.get(word)
        emb_str = ""
        for num in emb:
            emb_str += str(num) + " "
        temp = word + " " + emb_str+ "\n"
        f.write(temp)
    f.close()
    return


if __name__ == "__main__":
    # get list of gender pairs
    gender_list = combine_gendered_lists()

    # ------------------- GUTENBERG DATASET -------------------
    gutenberg_emb = embedding_dict("../embeddings/gutenberg_embeddings.txt")
    # find gendered axis for gutenberg dataset
    axis = find_gender_axis(gutenberg_emb, gender_list, size_of_emb=300)
    # remove axis (or axes) from embedding
    gutenberg_emb_debias = remove_gender_axis(gutenberg_emb, axis)
    # store debiased embedding in new file
    save_debiased_embeddings_to_file(gutenberg_emb_debias, "2018debias_gutenberg_emb.txt")

    # ------------------- WIKIPEDIA DATASET -------------------
    wiki_emb = embedding_dict("../embeddings/wikipedia_embeddings.txt")
    axis = find_gender_axis(wiki_emb, gender_list, size_of_emb=300)
    wiki_emb_debias = remove_gender_axis(wiki_emb, axis)
    save_debiased_embeddings_to_file(wiki_emb_debias, "2018debias_wiki_emb.txt")


