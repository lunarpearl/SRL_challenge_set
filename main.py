from allennlp_models import pretrained
from nltk.corpus import wordnet
import random, json, sys
import spacy, pyinflect
import pandas as pd
nlp = spacy.load('en_core_web_sm')

def create_base_sentences(vocabulary):
    '''
    This function creates the baseline dataset off the template and saves into a new file.

    :param dict vocabulary: dictionary of lists of terms used to create the base sentences

    :return: None
    '''

    subject_candidates = vocabulary['pronouns']+vocabulary['names']
    verb_candidates = vocabulary['verbs']
    object_candidates = vocabulary['nouns']

    with open('base_sentences.jsonl','w') as outfile:
        sentence_list = []
        while len(sentence_list) <= 50:
            subject = random.choice(subject_candidates)
            verb = random.choice(verb_candidates)
            object_ = random.choice(object_candidates)
            # fix some agreement
            if subject not in ['I','You']:
                vb = nlp(verb)
                verb = vb[0]._.inflect('VBZ')
            sentence = f"{subject} {verb} a {object_}."
            short_sent = f'{subject} {verb}'
            if short_sent in sentence_list:
                continue
            else:
                sentence_list.append(short_sent)
                sentence_dict = {"sentence": sentence,"verb":verb, "allen_gold": ['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'O']}
                outfile.write(f'{json.dumps(sentence_dict)}\n')

def get_base_labels(sentence_dict, model_type):
    '''
    This function extracts baseline sentence's gold labels.

    :param dict sentence_dict: dictionary of the sentence
    :param str model_type: string specifying the type of a model

    :return: subject label, verb label, determiner label, object label, dot label
    '''
    if model_type == 'allen':
        sentence = sentence_dict['allen_gold']
    elif model_type == 'bert':
        sentence = sentence_dict['fine-tuned_gold']
    elif model_type == 'logreg':
        sentence = sentence_dict['logreg_gold']
    subj_l = sentence[0]
    vb_l = sentence[1]
    art_l = sentence[2]
    obj_l = sentence[3]
    dot_l = sentence[-1]
    return subj_l,vb_l,art_l,obj_l,dot_l
    

def passive_voice(base_sentences, vocabulary):
    '''
    This function transforms baseline set into passivisation challenge set and saves to a new file.

    :param list base_sentences: list of dictionaries of sentences
    :param dict vocabulary: dictionary of lists of terms used to create the base sentences

    :return: None
    '''
    with open('passive_sentences.jsonl','w') as outfile:
        for sentence_dict in base_sentences:
            sentence = sentence_dict['sentence']
            subj_l,vb_l,art_l,obj_l,dot_l = get_base_labels(sentence_dict,'allen')
            # change gold labels for allen nlp
            allen_passive_gold = [art_l,obj_l,'O',vb_l,subj_l,'I-ARG0',dot_l]
            sentence_list = sentence.split()
            subject = sentence_list[0]
            verb = sentence_list[1]
            article = sentence_list[2]
            object_ = sentence_list[3]
            object_ = object_[:-1]
            vb = nlp(verb)
            vbd = vb[0]._.inflect('VBN')
            if subject in vocabulary['pronouns']:
                subject = subject.lower()
                if subject == 'he':
                    subject  = 'him'
                elif subject  == 'she':
                    subject  = 'her'
                elif subject  == 'i':
                    subject  = 'me'
                elif subject  == 'we':
                    subject  = 'us'
                elif subject  == 'they':
                    subject  = 'them'
            passive_sentence = f"A {object_} is {vbd} by {subject}."
            sentence_dict = {"sentence": passive_sentence,"verb":vbd, "allen_gold": allen_passive_gold}
            outfile.write(f'{json.dumps(sentence_dict)}\n')

def replace_obj_adjunct(base_sentences, vocabulary):
    '''
    This function transforms baseline set into patient replacement challenge set and saves to a new file.

    :param list base_sentences: list of dictionaries of sentences
    :param dict vocabulary: dictionary of lists of terms used to create the base sentences

    :return: None
    '''
    location_candidates = vocabulary['location']
    with open('replaced_object_sentences.jsonl','w') as outfile:
        for sentence_dict in base_sentences:
            sentence = sentence_dict['sentence']
            subj_l,vb_l,art_l,obj_l,dot_l = get_base_labels(sentence_dict,'allen')
            sentence_cut = sentence.rsplit(' ',2)[0]
            verb = sentence_cut.split()[1]
            vb = nlp(verb)
            vbd = vb[0]._.inflect('VB')
            if vbd in ['break','melt','burn','grow']:
                subj_l = 'B-ARG1'
            allen = [subj_l,vb_l]
            location = random.choice(location_candidates)
            if len(location.split()) == 1:
                allen_loc = ['B-ARGM-LOC']
            elif len(location.split()) == 2:
                allen_loc = ['B-ARGM-LOC','I-ARGM-LOC']
            elif len(location.split()) == 3:
                allen_loc = ['B-ARGM-LOC','I-ARGM-LOC','I-ARGM-LOC']
            else:
                allen_loc = ['B-ARGM-LOC','I-ARGM-LOC','I-ARGM-LOC','I-ARGM-LOC'] 
            allen_gold = allen+allen_loc
            allen_gold.append(dot_l)
            new_sentence = f"{sentence_cut} {location}."
            new_sent_dict = {"sentence": new_sentence,"verb": sentence_dict['verb'],"allen_gold": allen_gold}
            outfile.write(f'{json.dumps(new_sent_dict)}\n')

def flowery_object(base_sentences, vocabulary):
    '''
    This function transforms baseline set into patient expansion challenge set and saves to a new file.

    :param list base_sentences: list of dictionaries of sentences
    :param dict vocabulary: dictionary of lists of terms used to create the base sentences

    :return: None
    '''
    adj_candidates = vocabulary['adjectives']
    det_candidates = vocabulary['determiners']
    adv_candidates = vocabulary['adverbs']
    with open('flowery_object.jsonl','w') as outfile:
        for sentence_dict in base_sentences:
            sentence = sentence_dict['sentence']
            subj_l,vb_l,art_l,obj_l,dot_l = get_base_labels(sentence_dict,'allen')
            adj_allen = [subj_l,vb_l,art_l,'I-ARG1', obj_l, dot_l]
            adv_allen = [subj_l,vb_l,art_l,'I-ARG1', 'I-ARG1', obj_l, dot_l]
            sentence_list = sentence.rsplit(' ', 2)
            before = sentence_list[0]
            obj = sentence_list[2]
            det = random.choice(det_candidates)
            adj = random.choice(adj_candidates)
            adv = random.choice(adv_candidates)
            adj_sentence = f"{before} {det} {adj} {obj}"
            adv_sentence = f"{before} {det} {adv} {adj} {obj}"
            adj_dict = {"sentence": adj_sentence,"verb": sentence_dict['verb'],"allen_gold": adj_allen}
            adv_dict = {"sentence": adv_sentence,"verb": sentence_dict['verb'],"allen_gold": adv_allen}
            outfile.write(f'{json.dumps(adj_dict)}\n')    
            outfile.write(f'{json.dumps(adv_dict)}\n')    

def add_adjunct(base_sentences, vocabulary):
    '''
    This function transforms baseline set into adjunct placement challenge set and saves to a new file.

    :param list base_sentences: list of dictionaries of sentences
    :param dict vocabulary: dictionary of lists of terms used to create the base sentences

    :return: None
    '''
    adjunct_candidates = vocabulary['location']+vocabulary['past time']
    with open('adjunct_sentences.jsonl','w') as outfile:
        for sentence_dict in base_sentences:
            sentence = sentence_dict['sentence']
            subj_l,vb_l,art_l,obj_l,dot_l = get_base_labels(sentence_dict,'allen')
            sentence_list = sentence.rsplit(' ',3)
            subj = sentence_list[0]
            verb = sentence_list[1]
            article = sentence_list[2]
            obj = sentence_list[3]
            vb = nlp(verb)
            vbd = vb[0]._.inflect('VBD')
            adjunct = random.choice(adjunct_candidates)
            if adjunct in vocabulary['location']:
                if len(adjunct.split()) == 1:
                    allen_adj = ['B-ARGM-LOC']
                elif len(adjunct.split()) == 2:
                    allen_adj = ['B-ARGM-LOC','I-ARGM-LOC']
                elif len(adjunct.split()) == 3:
                    allen_adj = ['B-ARGM-LOC','I-ARGM-LOC','I-ARGM-LOC']
                else:
                    allen_adj = ['B-ARGM-LOC','I-ARGM-LOC','I-ARGM-LOC','I-ARGM-LOC'] 
            else:
                if len(adjunct.split()) == 1:
                    allen_adj = ['B-ARGM-TMP']
                elif len(adjunct.split()) == 2:
                    allen_adj = ['B-ARGM-TMP','I-ARGM-TMP']
                elif len(adjunct.split()) == 3:
                    allen_adj = ['B-ARGM-TMP','I-ARGM-TMP','I-ARGM-TMP']
                elif len(adjunct.split()) == 4:
                    allen_adj = ['B-ARGM-TMP','I-ARGM-TMP','I-ARGM-TMP','I-ARGM-TMP']
                else:
                    allen_adj = ['B-ARGM-TMP','I-ARGM-TMP','I-ARGM-TMP','I-ARGM-TMP','I-ARGM-TMP']

            allen_end = [subj_l,vb_l,art_l,obj_l]+allen_adj+[dot_l]
            allen_begin = allen_adj+[subj_l,vb_l,art_l,obj_l,dot_l]

            sent_adj_end = f"{subj} {vbd} {article} {obj[:-1]} {adjunct}."
            pronouns = vocabulary['pronouns']
            if subj in pronouns:
                subj = subj.lower()
            adjunct = adjunct[:1].upper()+adjunct[1:]
            sent_adj_begin = f"{adjunct} {subj} {vbd} {article} {obj}"
            sent_end_dict = {"sentence": sent_adj_end,"verb":vbd,"allen_gold": allen_end}
            sent_begin_dict = {"sentence": sent_adj_begin,"verb":vbd,"allen_gold": allen_begin}
            outfile.write(f'{json.dumps(sent_end_dict)}\n')
            outfile.write(f'{json.dumps(sent_begin_dict)}\n')

def intransitive_sentence(base_sentences):
    '''
    This function transforms baseline set into verb sense change challenge set and saves to a new file.

    :param list base_sentences: list of dictionaries of sentences
    :param dict vocabulary: dictionary of lists of terms used to create the base sentences

    :return: None
    '''
    with open('intransitive_sentences.jsonl','w') as outfile:
        for sentence_dict in base_sentences:
            sentence = sentence_dict['sentence']
            subj_l,vb_l,art_l,obj_l,dot_l = get_base_labels(sentence_dict,'allen')
            sentence_list = sentence.rsplit(' ',2)
            verb = sentence.split()[1]
            vb = nlp(verb)
            vbd = vb[0]._.inflect('VB')
            if vbd in ['break','melt','burn','grow']:
                subj_l = 'B-ARG1'
            allen_gold = [subj_l,vb_l, dot_l]
            new_sent = f"{sentence_list[0]}."
            new_dict = {"sentence": new_sent,"verb": sentence_dict['verb'],"allen_gold": allen_gold}
            outfile.write(f'{json.dumps(new_dict)}\n')

def flowery_subject(base_sentences, vocabulary):
    '''
    This function transforms baseline set into agent expansion challenge set and saves to a new file.

    :param list base_sentences: list of dictionaries of sentences
    :param dict vocabulary: dictionary of lists of terms used to create the base sentences

    :return: None
    '''
    pronouns = vocabulary['pronouns']
    names = vocabulary['names']
    adv_candidates = vocabulary['adverbs']
    adj_candidates = vocabulary['adjectives']
    with open('flowery_subject.jsonl','w') as outfile:
        for sentence_dict in base_sentences:
            sentence = sentence_dict['sentence']
            subj_l,vb_l,art_l,obj_l,dot_l = get_base_labels(sentence_dict,'allen')
            adj_allen = [subj_l,'I-ARG0',vb_l,art_l, obj_l, dot_l]
            adv_allen = [subj_l,'I-ARG0', 'I-ARG0',vb_l,art_l, obj_l, dot_l]
            sentence_list = sentence.rsplit(' ',3)
            subj = sentence_list[0]
            rest = f'{sentence_list[1]} {sentence_list[2]} {sentence_list[3]}'
            if subj in pronouns:
                subj = random.choice(names)
            adv = random.choice(adv_candidates)
            adj = random.choice(adj_candidates)
            adv_sent = f'{adv.capitalize()} {adj} {subj} {rest}'
            adj_sent = f'{adj.capitalize()} {subj} {rest}'
            adj_dict = {"sentence": adj_sent,"verb": sentence_dict['verb'],"allen_gold": adj_allen}
            adv_dict = {"sentence": adv_sent,"verb": sentence_dict['verb'],"allen_gold": adv_allen}
            outfile.write(f'{json.dumps(adj_dict)}\n')
            outfile.write(f'{json.dumps(adv_dict)}\n')

def replace_obj_hyponym(base_sentences):
    '''
    This function transforms baseline set into hyponym challenge set and saves to a new file.

    :param list base_sentences: list of dictionaries of sentences

    :return: None
    '''
    with open('hyponym_sentences.jsonl','w') as outfile:
        for sentence_dict in base_sentences:
            sentence = sentence_dict['sentence']
            sentence_list = sentence.rsplit(' ',2)
            before = sentence_list[0]
            article = sentence_list[1]
            obj = sentence_list[2]
            obj = obj[:-1]
            synsets = wordnet.synsets(obj, pos=wordnet.NOUN)
            synset = synsets[0]
            hyponyms = synset.hyponyms()
            count = 1
            while not hyponyms:
                synset = synsets[count]
                hyponyms = synset.hyponyms()
                count+=1
            hyponym = random.choice(hyponyms)
            hyponym_name = hyponym.name()
            hyponym_synset = wordnet.synset(hyponym_name)
            lemmas = hyponym_synset.lemma_names()
            lemma = lemmas[0]
            if '_' in lemma:
                lemma = lemma.replace('_',' ')
                allen_obj = ['I-ARG1','I-ARG1']
            else:
                allen_obj = ['I-ARG1']
            allen_gold = sentence_dict['allen_gold']
            allen_gold = allen_gold[:3]+allen_obj+['O']
            new_sent = f"{before} the {lemma}."
            new_dict = {"sentence": new_sent,"verb": sentence_dict['verb'],"allen_gold": allen_gold}
            outfile.write(f'{json.dumps(new_dict)}\n')

def load_file(filepath):
    '''
    This function opens a .jsonl formatted file.

    :param str filepath: path to the .jsonl file

    :return: list of dictionaries of sentences
    '''

    list_of_sentences = []
    with open(filepath,'r') as infile:
        for line in infile:
            line.rstrip('\n')
            sentence = json.loads(line)
            list_of_sentences.append(sentence)
    return list_of_sentences

def make_allen_predictions(model, sentences, model_type):
    '''
    This function writes predictions of an SRL model on the challenge set.

    :param model: an allennlp SRL model
    :param list sentences: list of dictionaries of sentences in the challenge set
    :param str model_type: string with a name of the model, either 'bert' or 'lstm'

    :return: list of dictionaries with predictions
    '''
    for sentence in sentences:
        json_preds = model.predict(sentence=sentence['sentence'])
        list_verbs = json_preds['verbs']
        if not list_verbs:
            sentence[f'allen_{model_type}'] = []
        for vb in list_verbs:
            if vb['verb'] == sentence['verb']:
                sentence[f'allen_{model_type}'] = vb['tags']
            else:
                sentence[f'allen_{model_type}'] = []

    return sentences

def save_predictions(list_of_dictionaries, test_name):
    '''
    This function saves a file into .jsonl format.

    :param list list_of_dictionaries: list to save
    :param str test_name: string with a preferred name to save the file under

    :return: None
    '''
    with open(f'{test_name}.jsonl','w') as outfile:
        for line in list_of_dictionaries:
            outfile.write(f'{json.dumps(line)}\n')

def evaluate_allen(sentences, model_type):
    '''
    This function evaluates the predictions of the model and calculates the proportion of incorrect predictions.

    :param list sentences: list of dictionaries of sentences
    :param str model_type: string with a name of the model, either 'bert' or 'lstm'

    :return: a float proportion of incorrect predictions
    '''
    list_of_evals = []
    for sentence in sentences:
        if sentence['allen_gold'] == sentence[f'allen_{model_type}']:
            evaluation = 'Correct'
        else:
            evaluation = 'Incorrect'
        list_of_evals.append(evaluation)
    
    fail_rate = list_of_evals.count('Incorrect')/len(list_of_evals)
    return fail_rate

def main(argv=None):
    '''
    This function runs the entire script to create, test, and evaluate challenge sets on two models - BERT and biLSTM.

    :param bool_1: whether to create the challenge sets
    :param bool_2: whether to make model predictions over the sets
    :param bool_3: whether to calculate the failure rates on the sets
    '''

    if argv == None:
        argv = sys.argv

    create_test_suites = argv[1]
    make_predictions = argv[2]
    evaluate_predictions = argv[3]

    if create_test_suites:
        # base vocabulary 
        vocabulary = {}
        # create a transitive/intransitive verbs list
        vocabulary['verbs'] = ['watch','see','sell','buy','cook','open','eat','drink','smell','hear','move','return','grow','play','run','stop','break','melt','speak','read','win','deliver','paint','pull','watch','clean','attack','cross','change','burn']
        # create determiners list
        vocabulary['determiners'] = ['the','my','their','your','her','his', 'this','that','its']
        # create adjective set
        vocabulary['adjectives'] = ['terrible','beautiful','gorgeous','enormous','dusty','new','orange','violet','violent','wet','cringey','destructive','contaminated','wooden','caffeinated','quixotic','scrumptious', 'elegant','extreme','obnoxious','outlandish','well-fed','peevish','peppy','cranky','smug','normal','boring','snooty','complementary','supportive','clingy','stubborn','familiar']
        # create adverb set
        vocabulary['adverbs'] = ['absolutely','just','precisely','little','nearly','enough','deeply','completely','entirely','quite','rather','vaguely']
        # create names set
        vocabulary['names'] = ['Cecelia','Aina','Sofia','Anastasia','Kiki','Jonny','Nemo','Merlin','Seth','Danny','Maria','Carly','Daria','Dorian','Makeda','Hermann','Alexander','Roxy','Salazar','Nina','Genya','Tamar']
        # create pronouns set
        vocabulary['pronouns'] = ['I','You','He','She','They']
        # create nouns set
        vocabulary['nouns'] = ['bird','cat','book','pavement','table','flower','cactus','cup','notebook','purse','beet','sun','blanket','soup','tomato','hotdog','mint','computer','coffee','toy','tea','cloud','rock']
        # create time and location adjuncts
        vocabulary['location'] = ['outside','indoors','on the market square','in Rotterdam','next to the ditch','on the lake','near the beach','in the national park','on a hike','in the city centre','at the coffee shop','at the carnival', 'at the canal','in the neighbourhood','nearby']
        vocabulary['past time'] = ['yesterday','three weeks ago','about a year ago','last Monday','few days ago','last fall','a while ago', 'more than a decade ago','a few weeks back','previous Friday','a couple months back']
        # creates the core jsonl file with 50 simple transitive sentences
        create_base_sentences(vocabulary)
        # just load it as a list of dictionaries
        list_of_sentences = load_file('base_sentences.jsonl')
        # create passive versions of base sentences
        passive_voice(list_of_sentences, vocabulary)
        # create sentences where objects are replaced by adjuncts
        replace_obj_adjunct(list_of_sentences,vocabulary)
        # create more descriptive object sentences
        flowery_object(list_of_sentences,vocabulary)
        # create sentences with adjuncts attached 
        add_adjunct(list_of_sentences,vocabulary)
        # create intransitive sentences
        intransitive_sentence(list_of_sentences)
        # create more descriptive subject sentences
        flowery_subject(list_of_sentences,vocabulary)
        # create sentences where objects are replaced with hyponyms of them
        replace_obj_hyponym(list_of_sentences)
    if make_predictions:
        # load models
        allen_bert = pretrained.load_predictor('structured-prediction-srl-bert')
        allen_lstm = pretrained.load_predictor('structured-prediction-srl')
        # load tests
        base_test = load_file('base_sentences.jsonl')
        passive_test = load_file('passive_sentences.jsonl')
        intransitive_test = load_file('intransitive_sentences.jsonl')
        adjunct_test = load_file('adjunct_sentences.jsonl')
        flowery_subj_test = load_file('flowery_subject.jsonl')
        flowery_obj_test = load_file('flowery_object.jsonl')
        obj_replacement_test = load_file('replaced_object_sentences.jsonl')
        hyponym_test = load_file('hyponym_sentences.jsonl')
        # make predictions and save them
        bert_base = make_allen_predictions(allen_bert,base_test,'bert')
        lstm_base = make_allen_predictions(allen_lstm,bert_base,'lstm')
        save_predictions(lstm_base,'base_test')
        bert_pass = make_allen_predictions(allen_bert,passive_test,'bert')
        lstm_pass = make_allen_predictions(allen_lstm,bert_pass,'lstm')
        save_predictions(lstm_pass,'passive_test')
        bert_intr = make_allen_predictions(allen_bert,intransitive_test,'bert')
        lstm_intr = make_allen_predictions(allen_lstm,bert_intr,'lstm')
        save_predictions(lstm_intr,'intransitive_test')
        bert_adj = make_allen_predictions(allen_bert,adjunct_test,'bert')
        lstm_adj = make_allen_predictions(allen_lstm,bert_adj,'lstm')
        save_predictions(lstm_adj,'adjunct_test')
        bert_f_subj = make_allen_predictions(allen_bert,flowery_subj_test,'bert')
        lstm_f_subj = make_allen_predictions(allen_lstm,bert_f_subj,'lstm')
        save_predictions(lstm_f_subj,'flowery_subj_test')
        bert_f_obj = make_allen_predictions(allen_bert,flowery_obj_test,'bert')
        lstm_f_obj = make_allen_predictions(allen_lstm,bert_f_obj,'lstm')
        save_predictions(lstm_f_obj,'flowery_obj_test')
        bert_replace = make_allen_predictions(allen_bert,obj_replacement_test,'bert')
        lstm_replace = make_allen_predictions(allen_lstm,bert_replace,'lstm')
        save_predictions(lstm_replace,'obj_replacement_test')
        bert_hypo = make_allen_predictions(allen_bert,hyponym_test,'bert')
        lstm_hypo = make_allen_predictions(allen_lstm,bert_hypo,'lstm')
        save_predictions(lstm_hypo,'hyponym_test')
    if evaluate_predictions:
        # load the datasets
        base_set = load_file('base_test.jsonl')
        passive_set = load_file('passive_test.jsonl')
        intran_set = load_file('intransitive_test.jsonl')
        adjunct_set = load_file('adjunct_test.jsonl')
        f_subj_set = load_file('flowery_subj_test.jsonl')
        f_obj_set = load_file('flowery_obj_test.jsonl')
        replacement_set = load_file('obj_replacement_test.jsonl')
        hyponym_set = load_file('hyponym_test.jsonl')
        # calculate fail rates
        bert_evals = {}
        bert_evals['base test'] = evaluate_allen(base_set,'bert')
        bert_evals['passivisation test'] = evaluate_allen(passive_set,'bert')
        bert_evals['verb sense test'] = evaluate_allen(intran_set,'bert')
        bert_evals['adjunct placement test'] = evaluate_allen(adjunct_set,'bert')
        bert_evals['agent expansion test'] = evaluate_allen(f_subj_set,'bert')
        bert_evals['patient expansion test'] = evaluate_allen(f_obj_set,'bert')
        bert_evals['patient replacement test'] = evaluate_allen(replacement_set,'bert')
        bert_evals['hyponym test'] = evaluate_allen(hyponym_set,'bert')
        # save the fail rates to a dataframe
        fail_df = pd.DataFrame.from_dict(bert_evals,orient='index',columns=['bert'])
        lstm_evals = {}
        lstm_evals['base test'] = evaluate_allen(base_set,'lstm')
        lstm_evals['passivisation test'] = evaluate_allen(passive_set,'lstm')
        lstm_evals['verb sense test'] = evaluate_allen(intran_set,'lstm')
        lstm_evals['adjunct placement test'] = evaluate_allen(adjunct_set,'lstm')
        lstm_evals['agent expansion test'] = evaluate_allen(f_subj_set,'lstm')
        lstm_evals['patient expansion test'] = evaluate_allen(f_obj_set,'lstm')
        lstm_evals['patient replacement test'] = evaluate_allen(replacement_set,'lstm')
        lstm_evals['hyponym test'] = evaluate_allen(hyponym_set,'lstm')
        fail_df['lstm'] = lstm_evals
        # round to percentages
        fail_df['bert'] = fail_df['bert'].astype(float).map("{:.2%}".format)
        fail_df['lstm'] = fail_df['lstm'].astype(float).map("{:.2%}".format)
        # save to .tsv format
        fail_df.to_csv('failure_rates.tsv', sep='\t')
        
        




if __name__ == '__main__':
    my_args = ['main.py', False, False, True]
    main(my_args)


