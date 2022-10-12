# Imports
import pandas as pd
import numpy as np
import math

# for nlp tags & text extraction
import spacy
import regex as re

nlp = spacy.load("en_core_web_trf")

### INTERNAL FUNCTIONS ###
# Here are the functions called within our major extraction and comparison functions...

def which_sh(x,shs,psc_name): # for getting sh names
    """ checked name passed as x against shareholder list
    and returns shareholders it could be via exact matching
    if certain phases are passed we will return the psc_name"""
    
    which = None
    if x in ['director']:
        return psc_name.lower()
    
    if x in ['he','him','she','her','they','them','there','psc','individual']:
        return 'PERSON'
    
    for sh in shs:
        sh_split = sh.split(' ')
        if (x in sh_split):
            which = sh if which == None else which+'/'+sh
    
    return which

def get_pos_tags(txt,nlp,nationalities = None, reduce_ents = True, nat = False):
    """ Returns a dictionary of entities listed in dict_to_return
    This dict can be reffered to when tagging individual part of
    entities in a splitting off text"""
    
    # before adding pos column we need to extract
    doc = nlp(txt)
    doc_ents = doc.ents
    
    if nat != True:
        dict_to_return = {'PERSON': [str(ent) for ent in doc_ents if str(ent.label_) == 'PERSON'],
                         'ORG': [str(ent) for ent in doc_ents if str(ent.label_) == 'ORG'],
                         'DATE':[str(ent) for ent in doc_ents if str(ent.label_) == 'DATE'],
                         'CARDINAL':[str(ent) for ent in doc_ents if str(ent.label_) == 'CARDINAL'],
                         'PERCENT':[str(ent) for ent in doc_ents if str(ent.label_) == 'PERCENT']}
    else:
        dict_to_return = {'PERSON': [str(ent) for ent in doc_ents if str(ent.label_) == 'PERSON'],
                     'ORG': [str(ent) for ent in doc_ents if str(ent.label_) == 'ORG'],
                     'DATE':[str(ent) for ent in doc_ents if str(ent.label_) == 'DATE'],
                     'CARDINAL':[str(ent) for ent in doc_ents if str(ent.label_) == 'CARDINAL'],
                     'PERCENT':[str(ent) for ent in doc_ents if str(ent.label_) == 'PERCENT'],
                     'GPE':[str(ent) for ent in doc_ents if str(ent.label_) == 'GPE'],
                     'NORP':[str(ent) for ent in doc_ents if (str(ent.label_) == 'NORP') or (str(ent) in nationalities)]}
    
    if reduce_ents == True:
        for key in dict_to_return.keys():
            
            values = dict_to_return[key]
            check_len = sum([1 if len(i.split(' ')) > 1 else 0 for i in values])
            
            if check_len > 0:
                new_vals = []
                for val in values:
                    if len(val.split(' ')) > 0:
                        for i in val.split(' '):
                            new_vals.append(i)
                    else:
                        new_vals.append(val)
                        
                dict_to_return[key] = new_vals
            
    return dict_to_return

def assign_pos(pos_dict,item): # assigning found pos tags
    """ Returns the appropriate tag taken from the get_pos_tags
    function and txt_split"""
    
    for key in pos_dict.keys():
        if item in [val.lower() for val in pos_dict[key]]:
            if item == 'psc':
                return np.nan
            return key
    return np.nan

def choose_sh(x,psc_name):
    """ x is list of potential shareholders, these are compared with the psc_name to determine
    our best fit for the problem. Used in Names extraction."""
    if (psc_name == 'psc missing') or (type(x) != str):
        return x
    
    if '/' not in x:
        return x
    
    shs = x.split('/')
    best_count = 0
    best_sh = None
    for sh in shs:
        count = 0
        sh_split = sh.split(' ')# puts into parts-of-name
        for part in sh_split:
                count += 1 if part in psc_name else 0
                
        if count > best_count:
            best_count = count
            best_sh = sh
    
    return best_sh

def consecutive(data, stepsize=1):
    """ Returns consecutive sequences of numbers in an array.
    e.g [1,2,3,5,6,7] returns [1,2,3] & [5,6,7]"""
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def most_frequent(List):
    """ Returns most frequently occuring in list """
    return max(set(List), key = List.count)

def jaccard_similarity(x,y):
    """ jaccard similarity is a vectorised measure of similairty between two sequecnes of numbers x and y
    Is used for comparing matches between strings or sequences, strings must be encoded to a numerical form"""
    # returns the jaccard similarity between two lists 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))

    if union_cardinality == 0:
        return 0

    return intersection_cardinality/float(union_cardinality) # 0 to 1

def squared_sum(x):
  """ Returns 3 rounded square rooted value  """
  return round(math.sqrt(sum([a*a for a in x])),3)

def cos_similarity(x,y):
    """ Similarity measure between two vectors via calculation of the angle between vectors"""
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = squared_sum(x)*squared_sum(y)
    if denominator == 0:
        return 0
    else:
        return round(numerator/float(denominator),3) # 0 to 1

def similarity_scoring(txt_a,txt_b, encode = False, return_encoded = False):
    """ Uses jaccard_similairty, squared_sum, and cos_similairty, and combines these tox get an overall score
    of similarity between two strongs txt_a and txt_b"""
    if (encode == False) & (return_encoded == True):
        return "CANNOT RETURN IF NEVER ENCODED"
    
    if encode == True:
        # For use later on in conversion to numerical representation
        alphabet = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,
               'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26,' ':0, '-':27}
        
        # encoding strings
        encoder = lambda x: [alphabet[char] for char in x if (char in alphabet.keys()) and (char.isdigit() == False)]      
        
        num_a = encoder(txt_a)
        num_b = encoder(txt_b)
        
        
    # scoring    
        num_to_sum = 1/len(txt_a.split(' '))
    
        score_pt1 = np.sum( [num_to_sum if part in txt_a.split(' ') else 0 for part in txt_b.split(' ')] ) # adds sum of matching names with weight 10
        txt_a = num_a
        txt_b = num_b
    else:
        score_pt1 = 5 if txt_a == txt_b else 0
     
    score_pt2 = (jaccard_similarity(txt_a,txt_b) + cos_similarity(txt_a,txt_b))/2 # - euclidean_distance(txt_a,txt_b)/10 
    
    
    score = (score_pt1 + score_pt2)/2
    
    return round(score,3)

def psc_to_sh(shs,psc_name):
    """ Checks best fit of shareholder names against a give psc_name. Very similar to choose_sh but less specific use. """
    titles = ['dr ','mr ','mrs ','mx ','ms ','miss ','master ','sir ','lord ']
    for title in titles:
        if title in psc_name:
            psc_name = psc_name.replace(title,'')
            
    best_count = 0
    best_sh = None
    psc_split = psc_name.split(' ')
    for sh in shs:
        if type(sh) != str:
            continue
        count = 0
        for part in psc_split:
            if type(part) != str:
                continue
            count += 1 if part in sh else 0
        if count > best_count:
            best_count = count
            best_sh = sh
        elif count == best_count:
            sh_fores = [name.split(' ')[0] for name in shs]
            psc_fore = psc_name.split(' ')[0]
            if psc_fore in sh_fores:
                best_sh = shs[sh_fores.index(psc_fore)]
    return best_sh

def get_per_vals(x): # for tagging
    """ When percentage string is passed it will determine if the value
    is a percentage then return the float if it is and nan if not"""
    if ('%' in x) and (x != '%'):
        x = x.replace('%','')
        try:
            return float(x)
        except:
            return np.nan
    if x.isdigit() == True:
        if (float(x) > 10) and (float(x) <= 100):
            return float(x)
    return np.nan

def search_sequence_numpy(arr,seq):
    """ Used for return consecutive sequences within a numpy array
    i.e [1,2,3,7,14,7,8,9] will return 123 and 789"""
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found
    
def get_sh_from_shs(sh,shs): 
    # determines the best sh fit from list of shareholders
    sh_split = sh.split(' ')
    current_best = None
    best_score = 0
    for name in shs:
        current_score = 0
        for part in sh_split:
            if part in name.split(' '):
                current_score += 1
        
        if current_score > best_score:
            current_best = name
            best_score = current_score
            
    return current_best

def is_psc(x,psc_name):
    """ checks if name passed as x is similar to psc_name"""
    # checks names against psc so 'john smith' isnt treated seperately to 'mr john smith'
    psc_split = psc_name.lower().split(' ')
    x_split = x.split(' ') if x != None else ''
    score = 0
    for part in x_split:
        if part in psc_split:
            score+=1

    if score >= 2:
        return psc_name.lower()
    else:
        return x
    
def check_act(cont_id,df_act):
    """ checks action codes for given contact ID"""
    action_codes_avoid = [
    41,4100,41,43,4300,44,4400,50,5000,
    70,7001,7002,71,7101,73,7301,7303,
    7304,7307,7308,74,7401,7402,7403,
    7404,76,7601,7701,90,9000,9100
    ]
    
    act_code = df_act[df_act.CONTACT_ID == cont_id].ACTION_CODE_TYPE_ID.values
    act_code_desc = df_act[df_act.CONTACT_ID == cont_id].ACTION_CODE_DESC.values
    
    if len(act_code) == 0:
        act_code = 0
        act_code_desc = 0
    else:
        act_code = act_code[0]
        act_code_desc = act_code_desc[0]
    
    if act_code in action_codes_avoid:
        return True
    else:
        return False
    
def get_controls(df,multi = False):
    """ get_controls will extract the controls of a given dataframe, here it will be df_noc
     it returns a dictionary of lists with controls inside"""
    
    df.fillna('')
    return_dict = dict.fromkeys(['NOC','VR','SIC'])
    if multi == False:
        for key in return_dict:
            if key == 'SIC':
                return_dict[key] = True if 'SIC' in df.TYPE.values else None
                continue
            if key in df.TYPE.values:
                row = df[df.TYPE == key]
                ub = row.UB.values[0]
                lb = row.LB.values[0]
                ub = 100.0 if lb == 75 else ub # so we have int to compare to

                return_dict[key] = [lb,ub]
            else:
                return_dict[key] = None
    else:
        # HERE WE PUT CODE FOR MULTIPLE CONTROL CASE
        for key in return_dict:
            if key == 'SIC':
                return_dict[key] = True if 'SIC' in df.TYPE.values else None
                continue
            if key in df.TYPE.values:
                rows = df[df.TYPE == key]
                if rows.shape[0] == 1:
                    ub = rows.UB.values[0]
                    lb = rows.LB.values[0]
                    ub = 100.0 if ub == '' else ub # so we have int to compare to
                    ub = 100.0 if lb == 75 else ub
                    return_dict[key] = [lb,ub]
                else:
                    all_bounds = []
                    for i in range(rows.shape[0]):
                        ub = rows.UB.values[i]
                        lb = rows.LB.values[i]
                        ub = 100.0 if ub == '' else ub # so we have int to compare to
                        ub = 100.0 if lb == 75 else ub
                        all_bounds.append([lb,ub])
                    return_dict[key] = all_bounds
                    
    return return_dict

def remove_space_list(arr):
    """ removed whitespace at the start of every item in a list """
    new_arr = []
    for i in range(0,len(arr)):
        item = arr[i]
        if type(item) != str:
            new_arr.append(item)
        else:
            if len(item) == 0:
                continue
            while item[0] == ' ':
                item = item[1:]
            new_arr.append(item)
    return new_arr

def check_old_names(unmatched_list,old_names):
    """ checks list of names with list of other names to try and pair off items. 
    Returns a list of those that have not been matched"""
    old_surnames = [name.split(' ')[-1] for name in old_names]
    matched = []
    for unmatched in unmatched_list:
        if (unmatched in old_names) or (unmatched in old_surnames):
            matched.append(unmatched)
        else:
            unmatched_guess = get_sh_from_shs(unmatched,old_names)
            if unmatched_guess == None:
                continue
            elif unmatched_guess in old_names:
                matched.append(unmatched)
    
    to_return = [x for x in unmatched_list if x not in matched]
    return to_return

def format_chips_op(chips_controls):
    """ Lazy function to format output of controls to match that seen by PSC team"""
    return {'SH' : chips_controls['NOC'],'VR':chips_controls['VR'],'RTA':chips_controls['SIC']}


def merge_names_nat(arr):
    to_combine = []
    for item in arr:
        if type(item) != str:
            continue
        else:
            to_combine.append(item.lower())
            
    return ' '.join(to_combine)

def how_different(x,y):
    """ takes two strings x and y and states how many characters are different"""
    i = 0
    comp = list(x) if len(x) > len(y) else list(y)
    truth = list(x) if comp != list(x) else list(y)
    len_long = len(comp)
    len_short = len(truth)
    diff_count = 0

    while (i < len_short) and (i < len_long):
        if comp[i] != truth[i]:
            diff_count +=1
            del comp[i]
            del truth[i]
            len_long = len(comp)
            len_short = len(truth)
            continue
        
        len_long = len(comp)
        len_short = len(truth)
        i += 1
    
    diff_count += int(np.sqrt((len_long-len_short)**2))
    return diff_count
        

def entity_extract(text):
    
    fields = [
              'Obliged Entity Organisation Name: ',
              'Obliged Entity Contact Name: ',
              'Obliged Entity Email: ',
              'Obliged Entity Telephone Number: ',
              'Obliged Entity Type: ',
              'Company Number: ',
              'Submission Reference: ',
              'PSC Name: ',
              'PSC Date of Birth: ',
              'Discrepancy Detail:',
              'Discrepancy Options:'
            ]
    
    data = {}
    
    if type(text) != str:
        for field in fields:
            key = field.replace(":",'').replace(" ",'_').lower() # defines column names
            key = key if key[-1] != '_' else key[:-1] # removes last empty char
            data[key] = None # assigns none if no data
        return data # returns none data
    
    for field in fields:
        key = field.replace(":",'').replace(" ",'_').lower() # same as above
        key = key if key[-1] != '_' else key[:-1]
        
        if (field in text) and (field != 'Discrepancy Detail:') and (field != 'Discrepancy Options:' ):
            text_split = text.split(field)[1]
          
            data[key] = (text_split.split("\n")[0])

            
        elif (field == 'Discrepancy Detail:') and ('Discrepancy Detail:' in text):
            partial = text.split(field)[1]
            disc_det = partial.split('Discrepancy Options:')

            data[key] = disc_det[0].replace('\r',' ')
        
        elif (field == 'Discrepancy Options:') and ('Discrepancy Options:' in text):
            options = text.split('Discrepancy Options:')[-1].replace("\n ",'')
            options = options.split(',')
            
            #removing first char whitespace
            for i in range(0,len(options)):
                while options[i][0] == " ":
                    options[i] = options[i][1:]
                                    
            data[key] = options
            
        else:
            data[key] =None
    
    return data

def preprocess_df(df):
    df.drop_duplicates(subset = ['PSC_DISCREPANCY_ID', 'PSC_DISCREPANCY_OPTION_DESC'], keep='last', inplace=True)

    
    nulls = df.isnull().sum()
    col_to_remove = []
    for i in range(0,len(nulls)):
        if nulls[i] == df.shape[0]:
            col_to_remove.append(nulls.index[i])

    # Finding where columns contain 1 entry repeated
    columns = df.columns
    cols_added = []
    for col in columns:
        counts = df[col].value_counts()
        leng = len(counts)
        if leng == 1:
            col_to_remove.append(col)
            cols_added.append(col)

    df.drop(col_to_remove, axis = 1, inplace=True)
    
    # removes duplicated columns
    dup_cols_remove = []
    for col in df.columns:
        if (col[-1].isdigit()) and ('SHAREHOLDER_' not in col):
            dup_cols_remove.append(col)
            
    df.drop(dup_cols_remove,axis =1, inplace = True)
    
    # Extracting 'Discrepancy detail' from CONTACT_DETAIL_CLOB
    clob = df.CONTACT_DETAIL_CLOB.values
    result = []
    for row in range(0,len(clob)):
        new_row = entity_extract( clob[row] ) # extraction
        result.append(new_row)


    res = pd.DataFrame(result)

    df['discrepancy_detail'] = res['discrepancy_detail'].values # gets column with raw text for details
    df['COMPANY_NUMBER'] = res['company_number'].values
    df['option_descrip_all'] = res['discrepancy_options'].values
    df['extract_psc_name'] = res['psc_name'].values

    # we want output of multiple discreps to be flagged as multiple
    #df['option_descrip'] = df['option_descrip_all'].apply(lambda x: x[0] if len(x) == 1 else x )
    
    # Getting full names
    df['SHAREHOLDER_FORENAME_1'] = df['SHAREHOLDER_FORENAME_1'].fillna('')
    df['SHAREHOLDER_SURNAME'] = df['SHAREHOLDER_SURNAME'].fillna('')
    
    df['merged_name'] = [merge_names_nat(x) for x in df[['SHAREHOLDER_FORENAME_1','SHAREHOLDER_SURNAME']].values]
    
    return df

def tag_multiple(x):
    for item in x:
        item = item.lower()
        while item[0] == ' ':
            item = item[1:]
        if item in ['name', 'nature of control','company name', 'date of birth', 'nationality','company number', 'correspondence address','other reason']:
            return 'HIGH'
    return None