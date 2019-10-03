# -*- coding: utf-8 -*-
import argparse
import errno
import glob
import multiprocessing
import os
import random
from collections import namedtuple
import pickle

import yaml
from PIL import ImageFile
from tqdm import tqdm

from data_generator import FakeTextDataGenerator
from string_generator import (create_string_from_dict_with_random_chars,
                              create_strings_from_file,
                              create_strings_from_fonts,
                              create_strings_from_wikipedia,
                              create_strings_randomly, gen_check_font,
                              gen_one_character, generate_char_map_from_font,
                              print_text, random_latin, random_latin_space,
                              random_sequences_sjnk, random_space)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_config():
    """
        Load config from config.yaml
    """
    # 
    try:
        config = yaml.load(open("config.yaml", "r"))
    except FileNotFoundError:
        raise FileNotFoundError("Can't find config.yaml")


    if not config:
        raise ValueError('config.yaml is empty. Aborting ...')

    
    default_config = {'output_dir': './images/', 'input_file': None, 'language': 'en', 'count': 1000, 'format': 32, 
                      'thread_count': 1, 'extension': '.jpg', 'prefix': None, 'random_skew': False, 'skew_angle': 0, 
                      'use_wikipedia': False, 'blur': False, 'background': 0, 'handwritten': False, 'name_format': 0, 
                      'distorsion': 0, 'distorsion_orientation': 0, 'random_crop': False, 'debug': False, 'check_font': False,
                      'random_sequences': False, 'include_letters': False, 'include_numbers': False, 'include_symbols': False, 
                      'random_sequences_from_font': False, 'random_one_character': False, 'random_sequences_sjnk': False, 
                      'random_latin_sjnk': False, 'random_latin_space': False, 'length': 1, 'random': False, 'random_space': False}
    
    invalid_config = [key for key in config if key not in default_config]
    if invalid_config:
        raise ValueError("Found {} invalid config: '{}' in config.yaml".format(
            len(invalid_config), "', ".join(invalid_config)))
    
    for key in config:
        if key in default_config:
            default_config[key] = config[key]
    
    return namedtuple("config", default_config.keys())(*default_config.values())


def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict


def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """
    # if lang == 'cn':
    return glob.glob('./fonts/{0}/*.ttf'.format(lang)) + \
        glob.glob('./fonts/{0}/*.ttc'.format(lang)) + \
        glob.glob('./fonts/{0}/*.otf'.format(lang))
    # else:
    #     return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]


def main():
    """
        Description: Main function
    """

    # Argument parsing
    config = load_config()
    # print(config)
    # Create the directory if it does not exist.
    try:
        os.makedirs(config.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Create font (path) list
    fonts = load_fonts(config.language)

    # Creating synthetic sentences (or word)
    fonts_arr = []
    if not config.check_font:
        fonts_arr = [fonts[random.randrange(0, len(fonts))]
                    for _ in range(0, config.count)]
    else:
        fonts_arr = fonts

    # print('fonts_arr: ',fonts_arr)
    fonts_dict_path = './fonts/' + config.language + '/font_dict.pkl' #only for jp
    try:
        fonts_dict = pickle.load(open(fonts_dict_path, "rb"))
        print("Loaded fonts dict from", fonts_dict_path)
        font_charsets = [fonts_dict[font[2:]] for font in fonts_arr]
    except:
        fonts_dict = {}
        fonts_dict = generate_char_map_from_font(fonts, fonts_dict)
        if not os.path.isdir('./fonts/' + config.language):
            os.system('mkdir ./fonts/' + config.language)
        pickle.dump(fonts_dict, open(fonts_dict_path, "wb"))
        print("Saved fonts dict to", fonts_dict_path)
        font_charsets = [fonts_dict[font[2:]] for font in fonts_arr]

    # print(fonts_dict)
    strings = []
    if config.use_wikipedia:
        strings = create_strings_from_wikipedia(
            config.length, config.count, config.language)
    elif config.input_file != '' and config.input_file is not None:
        strings = create_strings_from_file(
            config.input_file, config.count, config.length)
    elif config.random_sequences:
        strings = create_strings_randomly(config.length, config.random, config.count,
                                          config.include_letters, config.include_numbers, config.include_symbols, config.language)
        # Set a name format compatible with special characters automatically if they are used
        if config.include_symbols or True not in (config.include_letters, config.include_numbers, config.include_symbols):
            config.name_format = 2
    elif config.random_sequences_from_font:
        strings = create_strings_from_fonts(fonts_arr)
    elif config.random_sequences_sjnk:
        strings = random_sequences_sjnk(font_charsets)
    elif config.random_latin_sjnk:
        strings = random_latin(font_charsets)
    elif config.random_space:
        strings = random_space(font_charsets)
    elif config.random_latin_space:
        strings = random_latin_space(font_charsets)
    elif config.check_font:
        strings = gen_check_font(font_charsets)
    elif config.random_one_character:
        fonts_arr, strings = gen_one_character(fonts_dict)
    else:
        # Creating word list
        lang_dict = load_dict(config.language)

        strings = create_string_from_dict_with_random_chars(
            config.length, config.random, config.count, lang_dict)

    strings = [''.join([c for c in text if c in charset])
               for text, charset in zip(strings, font_charsets)]
    # strings = [s.strip() for s in strings if len(s.strip()) > 1]

    if os.path.isdir('./logs/') is False:
        os.system("mkdir logs")

    string_count = len(strings)
    print("Generated", string_count, "string.")
    if string_count < 10:
        [print('\t' + each_string) for each_string in strings]
    else:
        [print('\t' + each_string) for each_string in strings[:5]]
        print('\t...')

    # print(strings)

    print_text("./logs/src-train.txt", ['{}_{}.{}\n'.format(
        config.prefix, str(index), config.extension) for index in range(string_count)])
    print_text("./logs/tgt-train.txt", ['{}\n'.format(x) for x in strings])
    print_text("./logs/tgt-fonts.txt", ['{}\n'.format(x) for x in fonts_arr])

    # exit()
    print('Generating images from the string ...')
    p = multiprocessing.Pool(config.thread_count)
    for _ in tqdm(
            p.imap_unordered(
                FakeTextDataGenerator.generate_from_tuple,
                zip(
                    [i for i in range(0, string_count)],
                    strings,
                    fonts_arr,
                    [config.output_dir] * string_count,
                    # [config.format] * string_count,
                    [random.randint(config.format, config.format + 40) for x in range(string_count)],
                    [config.extension] * string_count,
                    [config.skew_angle] * string_count,
                    [config.random_skew] * string_count,
                    [config.blur] * string_count,
                    [config.background] * string_count,
                    [config.distorsion] * string_count,
                    [config.distorsion_orientation] * string_count,
                    [config.handwritten] * string_count,
                    [config.name_format] * string_count,
                    [-1] * string_count,
                    [config.prefix] * string_count,
                    [config.random_crop] * string_count,
                    [config.debug] * string_count
                )
            ), total=config.count):
        pass
    p.terminate()
    if config.name_format == 2:
        # Create file with filename-to-label connections
        with open(os.path.join(config.output_dir, "labels.txt"), 'w', encoding="utf8") as f:
            for i in range(string_count):
                file_name = str(i) + "." + config.extension
                f.write("{} {}\n".format(file_name, strings[i]))
    print('Generated', config.count, 'images to', config.output_dir)

if __name__ == '__main__':
    main()
    