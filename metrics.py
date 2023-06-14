import math
import numpy as np

def mAP(q_name, sorted_pool):
    ret_classes = [sorted_pool[i][0].split("_")[0:3] for i in range(len(sorted_pool))]
    q_class = q_name.split("_")[0:3]
    #print(q_class)
    #print(ret_classes)
    initlist = [int(q_class == i) for i in ret_classes]
    #print(initlist)
    den = np.sum(initlist)
    #print(den)
    if den == 0:
        return 0
    x = 0
    preclist = [0]*len(initlist)
    for idx, pts in enumerate(initlist):
        x += pts #rel(n)
        preclist[idx] = x/(idx+1) #rel(n)/k
    #print(preclist)
    num = np.dot(preclist, initlist)
    #print(num)
    #print(num/den)
    return num/den

def nDCG(relevance_list, ideal_relevance_list):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) of a ranked list of documents.

    Args:
        relevance_list: A list of relevance scores for the documents in the ranked list.
        ideal_relevance_list: A list of ideal relevance scores for the documents in the ranked list.

    Returns:
        The NDCG of the ranked list.
    """

    # Calculate the cumulative gain of the ranked list.
    cumulative_gain = 0.0
    for id, relevance in enumerate(relevance_list):
        numerator = 2**relevance - 1
        # add 1 because python 0-index
        denominator =  np.log2(id + 2) 
        cumulative_gain += numerator/denominator
    #print(cumulative_gain)

    # Calculate the ideal cumulative gain.
    ideal_cumulative_gain = 0.0
    for id, ideal_relevance in enumerate(ideal_relevance_list):
        numerator = 2**ideal_relevance - 1
        # add 1 because python 0-index
        denominator =  np.log2(id + 2)
        #print('1', numerator) 
        ideal_cumulative_gain += numerator/denominator
    #print(ideal_cumulative_gain)

    # Normalize the cumulative gain by the ideal cumulative gain.
    if ideal_cumulative_gain==0:
        return 0
    else:
        ndcg = cumulative_gain / ideal_cumulative_gain
        return ndcg


def aCG(relevance_list):
    """
    Calculates the Average Cumulative Gain (ACG) of a ranked list of documents.

    Args:
        relevance_list: A list of relevance scores for the documents in the ranked list.

    Returns:
        The ACG of the ranked list.
    """

    # Calculate the cumulative gain of the ranked list.
    cumulative_gain = 0.0
    for relevance in relevance_list:
        cumulative_gain += relevance

    # Calculate the average cumulative gain.
    return cumulative_gain / len(relevance_list)


def mAPw(relevance_list):
    """
    Calculates the weighted mean Average Precision (mAPw) of a ranked list of documents.

    Args:
        relevance_list: A list of relevance scores for the documents in the ranked list.

    Returns:
        The mAPw of the ranked list.
    """

    # Assign weights based on relevance
    weights = [1 if rel > 0 else 0 for rel in relevance_list]

    # Calculate the average precision of the ranked list
    average_precision = 0.0
    cumulative_relevance = 0.0
    num_relevant_items = sum(weights)
    for i, relevance in enumerate(relevance_list):
        if relevance > 0:
            cumulative_relevance += relevance
            average_precision += cumulative_relevance / (i + 1)

    # Normalize the average precision by dividing by the total number of relevant documents
    if num_relevant_items > 0:
        average_precision /= num_relevant_items

    # Calculate the weighted mean average precision
    weighted_mean_ap = average_precision
    return weighted_mean_ap

def DCG(result):
    dcg = []
    for idx, val in enumerate(result): 
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2) 
        score = numerator/denominator
        dcg.append(score)
    #print(sum(dcg))
    #print('a')
    return sum(dcg)

def ACG(r_list):
    x = 0
    acglist = [0]*len(r_list)
    for idx, pts in enumerate(r_list):
        x += pts #r(i)
        acglist[idx] = x/(idx+1) #r(i)/k'''
    acg_p = np.mean(r_list)
    return acg_p, acglist