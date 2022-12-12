from contextlib import redirect_stdout
import os
import re
import asyncio
import sqlite3
import threading
from typing import Tuple, Any, List, Set
from itertools import product
from collections import defaultdict
import tqdm
import random
from parse import get_all_preds_for_execution, remove_distinct
import time
import pickle as pkl
import subprocess
from itertools import chain
from sql_metadata import Parser



threadLock = threading.Lock()
TIMEOUT = 60
EXEC_TMP_DIR = 'tmp/'

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    #alphanumerically sort result1 and result2
    s1 = [unorder_row(row) for row in result1] 
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False #false if count(a) in l1 < count(a) in l2
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints) 

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# check whether two denotations are correct
# Used for final comparison of queries
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)] # set comprehension inside list comprehension

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2): 
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def replace_cur_year(query: str) -> str:
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path: str):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


async def exec_on_db_(sqlite_path: str, query: str) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e

async def exec_on_db(
    sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT
) -> Tuple[str, Any]:
    try:
        return await asyncio.wait_for(exec_on_db_(sqlite_path, query), timeout)
    except asyncio.TimeoutError:
        return ('exception', TimeoutError)
    except Exception as e:
        return ("exception", e)


# postprocess the model predictions to avoid execution errors
# e.g. removing spaces between ">" and "="
def postprocess(query: str) -> str:
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    query = re.sub(' +', ' ',query) # remove consecutive blankspaces
    return query

#get table name 
def get_table_for_col_(num_cols, g_str) -> str:
    if len(num_cols) == 0:
        return None
    g_str = g_str.lower()
    if len(Parser(g_str).tables) == 1:
        return (Parser(g_str).tables)[0]
    columns_dict = (Parser(g_str).columns_dict)
    if columns_dict is None or not 'where' in columns_dict is None: 
        return None
    if len(columns_dict['where']) ==  1:
        return columns_dict['where'][0].lower()
    for col in columns_dict['where']:
        col = col.lower()
        for num_col in num_cols:
            num_col = num_col.lower()
            if col.__contains__(num_col):
                return col.split('.')[0]
    return None

def get_table_for_col(num_cols, g_str) -> str:
    table = get_table_for_col_(num_cols, g_str)
    if isinstance(table, str):
        return table
    elif isinstance(table, list):
      return table[0]
    else:
        return None  




def keep_valid_cols(num_cols, all_cols):
    all_cols = set( [col.split('.')[1].lower() if col.__contains__('.') else col.lower() for col in all_cols ] )
    return( list(num_cols.intersection(all_cols)) )

def get_num_cols(p_str: str) -> set:
    num_cols = set()
    tokens = p_str.split(sep=' ')
    for i in range(5,len(tokens)): #skipping the first 5 tokens
        if tokens[i].isnumeric():
            num_cols.add( tokens[i-2] )
    return {col.split('.')[1] if col.__contains__('.') else col for col in num_cols }

def get_s_values(num_cols, table_name, db_path): #-> list(str)
    if table_name is None:
        return []
    s_values = set()
    for col in num_cols:
        for agg in ['MIN', 'MAX', 'AVG']:
            s_str = 'SELECT ' + agg + '(' + str(col) + ') FROM ' + str(table_name) 
            _, denotation = asyncio.run(exec_on_db(db_path, s_str))
            s_values.add(str(denotation[0][0]))
    return(list(s_values))


    

# approximate whether p_str and g_str are semantically equivalent
# db is the database path
# we are going to evaluate whether they are equivalent in all the databases
# that are in the same directory as db
# 0 if denotationally equivalent
# 1 otherwise
# the meaning of each auxillary argument can be seen in the parser definition in evaluation.py
def eval_exec_match(db: str, p_str: str, g_str: str, plug_value: bool, keep_distinct: bool, progress_bar_for_each_datapoint: bool) -> int:
    # post-process the prediction.
    # e.g. removing spaces between ">" and "="

    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    # we decide whether two denotations are equivalent based on "bag semantics"
    # https://courses.cs.washington.edu/courses/cse444/10sp/lectures/lecture16.pdf
    # if there is order by in query, then we assume order of the rows matter
    # order by might also be used to find the max/min instead of sorting,
    # but in that case the result mostly only contains one row and hence order_matters does not make a difference
    order_matters = 'order by' in g_str.lower()

    # find all databases in the same directory
    db_dir = os.path.dirname(db)
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if '.sqlite' in basename] 

    preds = [p_str]
    # if plug in value (i.e. we do not consider value prediction correctness)
    # enumerate all ways to plug in values in the gold query to the model predictions
    # otherwise, we only evaluate the predicted query with its own value prediction
    if plug_value:

        db_path = db_paths[0]

        #get list of all columns
        all_cols = Parser(g_str).columns
        #get set of numerical columns
        num_cols = get_num_cols(p_str)
        
        #discard non-column tokens
        num_cols = keep_valid_cols(num_cols, all_cols)

        #get table_name for querying
        table_name = get_table_for_col(num_cols, g_str)

        #get statistical values from database
        s_values = get_s_values(num_cols, table_name, db_path)

        #collect different values in the query, and generate all permutations of the values 
        _, preds = get_all_preds_for_execution(g_str, p_str, s_values)

        preds = list(chain([p_str], preds))
        


    for pred in preds:
        pred_passes = 1
        # compare the gold and predicted denotations on each database in the directory
        # wrap with progress bar if required
        if progress_bar_for_each_datapoint:
            ranger = tqdm.tqdm(db_paths)
        else:
            ranger = db_paths

        for db_path in ranger:

            g_flag, g_denotation = asyncio.run(exec_on_db(db_path, g_str))
            p_flag, p_denotation = asyncio.run(exec_on_db(db_path, pred))

            # we should expect the gold to be succesfully executed on the database
            assert g_flag != 'exception', 'gold query %s has error on database file %s' % (g_str, db_path)

            # wrong if execution fails
            if p_flag == 'exception':
                pred_passes = 0

            # if denotations are not equivalent, the prediction must be wrong
            elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = 0
            if pred_passes == 0:
                break

        # the model prediction has the same denotation as the gold for all databases
        if pred_passes == 1:
            return 1

    # none of the predictions passed
    return 0
