import multiprocessing
from multiprocessing import Pool
import tqdm
import time
import os
from heapq import heappush, heappop, heapify
from collections import defaultdict
from bitarray import bitarray
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib
from PIL import Image as im

def _foo(nae):
    print(my_number)
    return 1

def compretionn(name):
    print(33)
    img='photos/'+name
    image = cv2.imread(img)
    #plt.imshow(image)
    #a = im.fromarray(image)
    #print(image)
    #print("====")
    #print(len(image))
    #print(len(image[0]))
    #print(len(image[0][0]))
    #M = image.shape[0]
    #N = image.shape[1]
    #imageR=np.zeros([M,N])
    #imageB=np.zeros([M,N])
    #imageG=np.zeros([M,N])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgshape = image.shape
    #print(imgshape)
    image = np.reshape(image, (1, imgshape[0]*imgshape[1]*imgshape[2]))
    #print(len(image[0]))
    #print(len(image), "a")
    image = image.tolist()[0]
    #print(len(image), "a")
    # image = "HAPPY HAPPY"
    freq_lib = defaultdict(int)    # generate a default library
    for ch in image:                # count each letter and record into the frequency library 
        freq_lib[ch] += 1
    heap = [[fq, [sym, ""]] for sym, fq in freq_lib.items()]  # '' is for entering the huffman code later
    heapify(heap) # transform the list into a heap tree structure
    while len(heap) > 1:
        right = heappop(heap)  # heappop - Pop and return the smallest item from the heap
    #     print('right = ', right)
        left = heappop(heap)
    #     print('left = ', left)

        for pair in right[1:]:  
            pair[1] = '1' + pair[1]   # add one to all the right note
        for pair in left[1:]:  
            pair[1] = '0' + pair[1]   # add zero to all the left note
        heappush(heap, [right[0] + left[0]] + right[1:] + left[1:])  # add values onto the heap. Eg. h = []; heappush(h, (5, 'write code')) --> h = [(5, 'write code')]
    huffman_list = right[1:] + left[1:]
    #print(len(huffman_list),"f")
    huffman_dict = {a[0]:bitarray(str(a[1])) for a in huffman_list}
    #print(len(huffman_dict),"s")
    encoded_image = bitarray()
    encoded_image.encode(huffman_dict, image)
    with open('bin/'+name+'.bin', 'wb') as w:
        encoded_image.tofile(w)
    huffman_dict['imgshape'] = [imgshape[0],imgshape[1],imgshape[2]]
    huffman_dict['encoded_image'] = len(encoded_image)
    #print(huffman_dict)

    a_file = open('dict/'+name+".pkl", "wb")
    pickle.dump(huffman_dict, a_file)
    a_file.close()
    #print(name, "compretion completed...")
    #return a


def multi_compretion(lst,processors):   
    #lst = os.listdir('photos/')
    p = Pool(processors)
    #p = Pool(multiprocessing.cpu_count()-1)
    start_time = time.time()
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    r = list(tqdm.tqdm(p.imap(compretionn, [lst[i] for i in range(len(lst))]), total=len(lst)))
    p.close()
    p.join()
    print("--- %s seconds ---" % (time.time() - start_time))
    return time.time() - start_time