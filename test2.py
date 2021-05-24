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

def decompretion(name):
    decoded_out = bitarray()
    a_file = open('dict/'+name+".pkl", "rb")
    huffman_dict = pickle.load(a_file)
    #print(output)
    
    imgshape = huffman_dict.pop("imgshape")
    e_i = huffman_dict.pop("encoded_image")
    padding = 8 - (e_i % 8)
    
    with open('bin/'+name+'.bin', 'rb') as r:
        decoded_out.fromfile(r)
    
    decoded_out = decoded_out[:-padding] # remove padding
    decoded_out = decoded_out.decode(huffman_dict) 
    # decoded_text = ''.join(decoded_text)

    # print(decoded_out)

    decoded_out = np.array(decoded_out);
    output = np.reshape(decoded_out, (imgshape[0], imgshape[1] , imgshape[2]))
    #plt.imshow(output, cmap='gray')
    #matplotlib.image.imsave('decomp/'+name, output)
    #output.save('decomp/'+name)
    cv2.imwrite('decomp/'+name, output)
    #plt.imshow(output)
    #im.fromarray(output)
    #print(name, "de-compretion completed...")

def multi_decompretion(lst,processors):
    lst = os.listdir('photos/')
    p = Pool(processors)
    #p = Pool(multiprocessing.cpu_count()-1)
    start_time = time.time()
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    r = list(tqdm.tqdm(p.imap(decompretion, [lst[i] for i in range(len(lst))]), total=len(lst)))
    
    p.close()
    p.join()
    print("--- %s seconds ---" % (time.time() - start_time))
    return time.time() - start_time