# coding=utf-8
# Copyright, 2021 Ontocord, LLC, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datasets import load_dataset
import os
import re
import itertools
from re import finditer
import glob
import random
import fsspec
import json
from random import randint, choice
from collections import Counter
import spacy,  itertools
import langid
from nltk.corpus import stopwords
import fsspec, os, gzip
from faker import Faker
from faker.providers import person, company, geo, address
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, AutoTokenizer, pipeline
import torch
import sys
from tqdm import tqdm

model_name = 'Helsinki-NLP/opus-mt-en-hi'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_name = 'Helsinki-NLP/opus-mt-en-ar'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_name = 'Helsinki-NLP/opus-mt-en-zh'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

nlp = spacy.load('en_core_web_lg')

stopwords_en = set(stopwords.words('english'))
