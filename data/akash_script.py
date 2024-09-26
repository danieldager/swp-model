#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:29:23 2022

Based on dsprites data structure

import numpy as np
data = np.load('data/dsprite_train.npz', allow_pickle= True, encoding='latin1')
print(data.files)

@author: aakash
"""
import os, gc
import numpy as np
from itertools import product
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# from tqdm import tqdm
# import pickle, subprocess, shlex, shutil, io, , , time

def generate(
        savepath=None, text = 'text', index=1, mirror=False,
        invert=False, fontname='Arial', W = 64, H = 64, size=24,
        xshift=0, yshift=0, upper=0, spacing=0, show=None
    ):
    
    if upper: text = text.upper()
    if invert: text = text[::-1]
    
    img = Image.new("L", (W,H), color=10)
    fnt = ImageFont.truetype(fontname+'.ttf', size)
    draw = ImageDraw.Draw(img)

    # Starting word anchor
    w = sum([fnt.getsize(t)[0] for t in text])
    h = sum([fnt.getsize(t)[1] for t in text]) / len(text)
    w = w + spacing * (len(text) - 1)
    h_anchor = (W - w) / 2
    v_anchor = (H - h) / 2

    x, y = (xshift + h_anchor, yshift + v_anchor)
    for t in text:
        draw.text((x,y), t, font=fnt, fill="white")
        letter_w = fnt.getsize(t)[0] # Use the letter width to move the following
        x += letter_w + spacing

    if x > (W + spacing + 2) or (xshift + h_anchor) < -1:
        img.save(f"{savepath}/{text}_{xshift}_{yshift}_{spacing}.jpg")
        raise ValueError(f"Text width is bigger than image. Failed on size:{size}")

    if len(text)==5:
        img.save(f"{savepath}/{text}_{xshift}_{yshift}_{spacing}.jpg")
    
    return img

def generate_ngrams(unigrams, n=1):
    res = unigrams.copy()
    for i in range(2,n+1):
        t = product(unigrams, repeat=i)
        res += ["".join(x) for x in t]

    classes = generate_clases_ngrams(res, unigrams)
    return res, classes, unigrams


def generate_clases_ngrams(res, unigrams):
    max_n = max([len(x) for x in res])
    keys = ["letter"+str(i) for i in range(max_n)]
    classes = {k:np.zeros(len(res)) for k in keys}
    for i_w, w in enumerate(res):
        for i_l, l in enumerate(w):
            k = "letter"+str(i_l)
            letter_code = unigrams.index(l)+1
            classes[k][i_w] = letter_code

    return classes

def add_class(x):
    return zip(range(len(x)), x)

def create_wordset(path_out = './graphemes_data/',
                  ngrams   = 1,
                  words = ["a"]):

    #define words, sizes, fonts
    wordlist, classes, unigrams = generate_ngrams(words, ngrams)
    unigrams = [""]+unigrams
    sizes = [12]#np.arange(12, 21, 3)
    fonts = ['arial']#, 'times']#, 'comic']
    xshifts = np.arange(-2,  4, 1)
    yshifts = np.arange(-10, 11, 2)
    colours = [0]
    uppers  = [1]#[0, 1]
    spacing = range(-2, 3, 1)

    gc.collect()

    imgs, latents_classes,latents_values, latents_values_str = [], [], [], []
    latents_names = ["words", "spacing", "sizes",
                     "xshifts", "yshifts", "fonts", 
                     "uppers"] #+ list(classes.keys())
    latents_size  = [len(wordlist), len(spacing), len(sizes),
                     len(xshifts), len(yshifts), len(fonts), 
                     len(uppers)] #+ [len(set(x)) for x in classes.values()]

    all_stim = product(add_class(spacing), add_class(sizes),
                  add_class(xshifts), add_class(yshifts),
                  add_class(fonts), add_class(uppers))
    all_stim = list(all_stim)

    for w, word in enumerate(wordlist):
        letter_code = [int(x[w]) for x in classes.values()] 
        selected = all_stim
        print(f'Generating images for word: {word}')
        for ((sp, space), (s, size),
             (x, xshift), (y, yshift),
             (f, font), (u, upper)) in selected:

                img = generate(savepath=None, index=None, # no saving
                           text=word, fontname=font, size=size,
                           xshift=xshift, yshift=yshift, upper=upper, spacing=space)

                imgs.append(np.array(img))

                # latents_classes.append([w, sp, s, x, y, f, u] + letter_code)
                # latents_values.append([w, space, size, xshift, yshift, f, upper] + letter_code)
                # latents_values_str.append([word, sp, size, xshift, yshift, font, upper] +
                #                           [unigrams[l] for l in letter_code])
                
                latents_classes.append([w, sp, s, x, y, f, u])
                latents_values.append([w, space, size, xshift, yshift, f, upper])
                latents_values_str.append([word, sp, size, xshift, yshift, font, upper])

    os.makedirs(path_out, exist_ok=True)
    letters = "".join(words).upper()
    f_name = f'/dletters_n{ngrams}_{letters}'
    np.savez(path_out + f_name,
             imgs=imgs, latents_classes=latents_classes, words=wordlist, latents_values=latents_values,
             latents_names=latents_names, latents_size=latents_size, latents_values_str=latents_values_str)


# ngrams = 5
# words = ['a', 'b', 'c']
# words_lists = [["K", "X"], ["M","N"], ["E","F"], ["Q","O"], ["P","R"], ["M","P"],["O","K"],["X","F"]]
# for words in words_lists:
#     create_wordset(f'graphemes_data/test/', ngrams, words)

