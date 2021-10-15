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

stopwords_en = set(stopwords.words('english'))

junk_dict=dict([(a,1) for a in "' 0123456789¯_§½¼¾×|†—~\"—±′–'°−{}[]·-\'?,./<>!@#^&*()+-‑=:;`→¶'"])

# quick and dirty lexicon management. we could use wikipedia to find some of these words, but I don't have time to do this. 
# A more general version of the code using wikipedia, ontology, etc. is in the pii_processing/pii and onology folder.

country = {'Zimbabwe', 'Panama', 'the Soviet Union', 'Bahamas''Colombia', 'AMERICA', 'Norway', 'The United States of America','Indonesia',  'Switzerland', 'U.S.A.', 'Chile','U.K.', "North Korea's", 'USSR', 'Greece', 'N. Korea', 'Viet Nam', 'america', 'U.S', 'South Korea', 'Sweden', 'New Zealand', 'The United States', 'Pakistan', 'United States', 'Holland', 'Taiwan', 'Ukraine', 'Italy', 'South Africa', 'Turkey', 'Spain', 'the United States of America', 'Britain', 'Saudi Arabia', 'Argentina', 'Vietnam', 'Cuba', 'Syria', 'UK', 'Belgium', 'Iran', 'Latvia', 'India','France', 'Great Britain', 'El Salvador', 'Australia', 'Brazil','Venezuela','England',  'the Roman Empire', 'russia', 'Gaza', 'Costa Rica', 'Yugoslavia', 'the United Kingdom', 'the United States','USA', 'North Korea', 'U.S.', 'Canada', 'US','America', 'China','U.S.','Russia', 'Israel', 'Afghanistan', 'PRC', 'Mexico', 'Iraq', 'Japan', 'Germany', }
public_figures = {'Alec Baldwin', 'Stephen Harper', 'Al Gore', 'tRump', 'Trump', 'Harper', 'Justin','Comey','Obama', 'Gary Johnson', 'Lindsey Graham', 'Michelle Obama', 'Mark Twain', 'Jared Kushner', 'Vladimir Putin', 'Putin', 'Osama bin Laden', 'Fidel Castro', 'Mitch McConnell', 'Elizabeth Warren', 'Hilary Clinton', 'George Bush', 'John Paul II', 'Jimmy Carter', "Hillary Clinton's", 'Ann Coulter', 'Rush Limbaugh', 'JFK', 'Richard Nixon', 'Cameron Sellers', 'Ted Cruz', 'Richard Nixon', 'Jill Stein', 'Brad Wall', 'James Comey', 'Jeff Sessions', 'Loretta Lynch', 'Sean Spicer', 'Nancy Pelosi', 'Ronald Reagan', 'Gray Davis', 'Donald J. Trump', 'Dan Sullivan', 'George W. Bush', 'George Soros', 'Lisa Murkowski', 'Gore', 'Kim Jong Un', "Donald Trump's", 'Steve Bannon', 'Alex Jones', 'Paul Ryan', 'Trumps', 'Clintons', 'John McCain', 'Hillary', 'Obama', 'Trump', 'Clinton', 'Trudeau', 'Donald Trump', 'Bush', 'Jesus', 'Hillary Clinton', 'Justin Trudeau', 'Bernie Sanders', 'Reagan', 'Bill Clinton', 'Pope Francis','Hitler', 'Nixon', 'Sarah Palin', 'Barack Obama', }
NORP = {'Islam', 'Conservative', 'the Conservative Party', 'GOP', 'Christianity', 'Church', 'the Republican Party', 'the Democratic Party', 'the Catholic Church', 'the Liberal Party', 'the Tea Party', 'Republican Party', 'the Roman Catholic Church', 'UNION', 'MAGA', 'the Democrat Party',  }
swap_org = ['Ford', 'Facebook', 'CNN', 'ABC', 'NBC', 'CBS', 'Google', 'Amazon', 'AOL', 'NCR', 'GE', 'Enron']

faker_map = {
      'es': 'es_ES',
      'en': 'en_US',
      'ar': 'ar_AA',
      'pt': 'pt_PT',
      'fr': 'fr_FR',
      'fa': 'fa_IR',
      'hi': 'hi_IN',
      'zh': 'zh_CN',
    }


#from https://github.com/joke2k/faker/blob/master/faker/providers/person/en/__init__.py which is under the MIT License
#not an exausthive list. just to do some filtering for civil_comment. 
first_names = {
        'Aaliyah', 'Abagail', 'Abbey', 'Abbie', 'Abbigail', 'Abby', 'Abigail',
        'Abigale', 'Abigayle', 'Abril', 'Achsah', 'Ada', 'Adah', 'Adaline',
        'Adalyn', 'Adalynn', 'Adamaris', 'Adda', 'Addie', 'Addison', 'Addisyn',
        'Addyson', 'Adel', 'Adela', 'Adelaide', 'Adele', 'Adelia', 'Adelina',
        'Adeline', 'Adell', 'Adella', 'Adelle', 'Adelyn', 'Adelynn', 'Adilene',
        'Adina', 'Adison', 'Adline', 'Adria', 'Adriana', 'Adriane', 'Adrianna',
        'Adrianne', 'Adriene', 'Adrienne', 'Adyson', 'Affie', 'Afton', 'Agatha',
        'Aggie', 'Agnes', 'Agness', 'Agusta', 'Aida', 'Aileen', 'Ailene',
        'Aili', 'Aimee', 'Ainsley', 'Aisha', 'Aiyana', 'Aiyanna', 'Aja',
        'Akeelah', 'Akira', 'Ala', 'Alabama', 'Alaina', 'Alana', 'Alani',
        'Alanna', 'Alannah', 'Alaya', 'Alayna', 'Alba', 'Alberta', 'Albertha',
        'Albertina', 'Albertine', 'Albina', 'Alcie', 'Alda', 'Aldona', 'Aleah',
        'Alease', 'Alecia', 'Aleen', 'Aleena', 'Alejandra', 'Alena', 'Alene',
        'Alesha', 'Alesia', 'Alessandra', 'Aleta', 'Aletha', 'Alethea', 'Alex',
        'Alexa', 'Alexandr', 'Alexandra', 'Alexandrea', 'Alexandria', 'Alexia',
        'Alexina', 'Alexis', 'Alexus', 'Alexys', 'Alfreda', 'Alia', 'Aliana',
        'Alice', 'Alicia', 'Alida', 'Alina', 'Aline', 'Alisa', 'Alisha',
        'Alison', 'Alissa', 'Alisson', 'Alivia', 'Aliya', 'Aliyah', 'Aliza',
        'Alize', 'Alla', 'Allean', 'Alleen', 'Allena', 'Allene', 'Allie',
        'Alline', 'Allison', 'Allisson', 'Ally', 'Allyson', 'Allyssa', 'Alma',
        'Almeda', 'Almedia', 'Almeta', 'Almina', 'Almira', 'Almyra', 'Aloma',
        'Alondra', 'Alpha', 'Alphonsine', 'Alta', 'Altha', 'Althea', 'Altie',
        'Alvena', 'Alvera', 'Alverda', 'Alverta', 'Alvina', 'Alvira', 'Alwilda',
        'Alwina', 'Alwine', 'Alyce', 'Alycia', 'Alys', 'Alysa', 'Alyse',
        'Alysha', 'Alysia', 'Alyson', 'Alyssa', 'Alyssia', 'Alyvia', 'Alzina',
        'Ama', 'Amalia', 'Amalie', 'Amanda', 'Amani', 'Amara', 'Amari',
        'Amaris', 'Amaya', 'Amber', 'Amberly', 'Amelia', 'Amelie', 'America',
        'Amey', 'Ami', 'Amiah', 'Amie', 'Amina', 'Amira', 'Amirah', 'Amiya',
        'Amiyah', 'Amma', 'Ammie', 'Amparo', 'Amy', 'Amya', 'Ana', 'Anabel',
        'Anabella', 'Anabelle', 'Anahi', 'Anais', 'Analia', 'Anastacia',
        'Anastasia', 'Anaya', 'Andra', 'Andrea', 'Andria', 'Angel', 'Angela',
        'Angele', 'Angeles', 'Angelia', 'Angelic', 'Angelica', 'Angelina',
        'Angeline', 'Angelique', 'Angelita', 'Angella', 'Angie', 'Anice',
        'Anie', 'Anika', 'Anissa', 'Anita', 'Anitra', 'Aniya', 'Aniyah',
        'Anjali', 'Anjanette', 'Anjelica', 'Ann', 'Anna', 'Annabel', 'Annabell',
        'Annabella', 'Annabelle', 'Annalise', 'Annamae', 'Annamarie', 'Anne',
        'Anneliese', 'Annemarie', 'Anner', 'Annetta', 'Annette', 'Annice',
        'Annie', 'Annika', 'Annis', 'Annmarie', 'Anona', 'Ansley', 'Antionette',
        'Antoinette', 'Antonetta', 'Antonette', 'Antonia', 'Antonina', 'Anya',
        'April', 'Ara', 'Arabella', 'Araceli', 'Aracely', 'Arah', 'Araminta',
        'Ardath', 'Ardelia', 'Ardell', 'Ardella', 'Ardelle', 'Arden', 'Ardeth',
        'Ardis', 'Ardith', 'Ardyce', 'Areli', 'Arely', 'Aretha', 'Argie',
        'Aria', 'Ariana', 'Ariane', 'Arianna', 'Arie', 'Ariel', 'Ariella',
        'Arielle', 'Arietta', 'Arizona', 'Arkie', 'Arla', 'Arleen', 'Arlena',
        'Arlene', 'Arleth', 'Arletta', 'Arley', 'Arlie', 'Arline', 'Arly',
        'Arlyne', 'Armani', 'Armida', 'Arminda', 'Arminta', 'Arnetta', 'Arra',
        'Arrie', 'Arta', 'Artelia', 'Arvilla', 'Aryana', 'Aryanna', 'Asha',
        'Ashanti', 'Ashely', 'Ashlea', 'Ashlee', 'Ashleigh', 'Ashley', 'Ashli',
        'Ashlie', 'Ashly', 'Ashlyn', 'Ashlynn', 'Ashtyn', 'Asia', 'Ason',
        'Aspen', 'Assunta', 'Astrid', 'Atha', 'Athena', 'Attie', 'Aubree',
        'Aubrey', 'Aubrie', 'Audie', 'Audra', 'Audrey', 'Audriana', 'Audrianna',
        'Audrina', 'Audry', 'Augusta', 'Augustina', 'Aura', 'Aurelia',
        'Aurilla', 'Aurora', 'Aurore', 'Autumn', 'Ava', 'Avah', 'Averi',
        'Averie', 'Avie', 'Avis', 'Ayana', 'Ayanna', 'Ayesha', 'Ayla', 'Ayleen',
        'Aylin', 'Azalee', 'Azaria', 'Azariah', 'Azul', 'Azzie', 'Babette',
        'Baby', 'Bailee', 'Bailey', 'Bama', 'Bambi', 'Barb', 'Barbara',
        'Barbie', 'Barbra', 'Baylee', 'Baylie', 'Bea', 'Beadie', 'Beatrice',
        'Beatrix', 'Beatriz', 'Beaulah', 'Bebe', 'Beckie', 'Becky', 'Beda',
        'Bee', 'Belen', 'Belia', 'Belinda', 'Bell', 'Bella', 'Belle', 'Belva',
        'Bena', 'Benita', 'Bennie', 'Berdie', 'Berenice', 'Bernadette',
        'Bernadine', 'Bernardine', 'Berneice', 'Bernetta', 'Bernice',
        'Berniece', 'Bernita', 'Berta', 'Bertha', 'Bertie', 'Bertina', 'Beryl',
        'Bess', 'Besse', 'Bessie', 'Beth', 'Betha', 'Bethann', 'Bethany',
        'Bethel', 'Bethzy', 'Betsey', 'Betsy', 'Bette', 'Bettie', 'Bettina',
        'Betty', 'Bettye', 'Bettyjane', 'Bettylou', 'Beula', 'Beulah', 'Bev',
        'Beverlee', 'Beverley', 'Beverly', 'Beyonce', 'Bianca', 'Biddie',
        'Billie', 'Billy', 'Billye', 'Bina', 'Bird', 'Birdella', 'Birdie',
        'Birtha', 'Birtie', 'Blair', 'Blake', 'Blanca', 'Blanch', 'Blanche',
        'Blanchie', 'Blossom', 'Bobbi', 'Bobbie', 'Bobby', 'Bobbye', 'Bonita',
        'Bonnie', 'Bonny', 'Braelyn', 'Brande', 'Brandee', 'Brandi', 'Brandie',
        'Brandon', 'Brandy', 'Brea', 'Breana', 'Breann', 'Breanna', 'Breanne',
        'Bree', 'Brenda', 'Brenna', 'Breonna', 'Brett', 'Bria', 'Briana',
        'Brianda', 'Brianna', 'Brianne', 'Bridget', 'Bridgett', 'Bridgette',
        'Brielle', 'Brigette', 'Brigid', 'Brigitte', 'Briley', 'Brinda',
        'Brinley', 'Brionna', 'Brisa', 'Bristol', 'Britany', 'Britney',
        'Britni', 'Britny', 'Britt', 'Britta', 'Brittaney', 'Brittani',
        'Brittanie', 'Brittany', 'Brittnay', 'Brittnee', 'Brittney', 'Brittni',
        'Brittnie', 'Brittny', 'Brook', 'Brooke', 'Brooklyn', 'Brooklynn',
        'Bryana', 'Bryanna', 'Brylee', 'Bryn', 'Brynlee', 'Brynn', 'Buelah',
        'Buena', 'Buffy', 'Bula', 'Bulah', 'Buna', 'Burnice', 'Byrd', 'Byrdie',
        'Caddie', 'Cadence', 'Cailyn', 'Caitlin', 'Caitlyn', 'Caitlynn',
        'Caldonia', 'Caleigh', 'Cali', 'Calista', 'Calla', 'Calleigh', 'Callie',
        'Cambria', 'Cameron', 'Cami', 'Camila', 'Camilla', 'Camille', 'Camisha',
        'Cammie', 'Campbell', 'Camryn', 'Candace', 'Candi', 'Candice',
        'Candida', 'Candis', 'Candy', 'Candyce', 'Cannie', 'Capitola', 'Cappie',
        'Caprice', 'Cara', 'Caren', 'Carey', 'Cari', 'Carie', 'Carin', 'Carina',
        'Carisa', 'Carissa', 'Carla', 'Carlee', 'Carleen', 'Carleigh',
        'Carlene', 'Carley', 'Carli', 'Carlie', 'Carlota', 'Carlotta', 'Carly',
        'Carlyn', 'Carma', 'Carmel', 'Carmela', 'Carmelita', 'Carmella',
        'Carmen', 'Caro', 'Carol', 'Carolann', 'Carole', 'Carolee', 'Carolina',
        'Caroline', 'Carolyn', 'Carolyne', 'Carolynn', 'Caron', 'Carra',
        'Carri', 'Carrie', 'Carrol', 'Carroll', 'Carry', 'Carson', 'Cary',
        'Caryl', 'Caryn', 'Casandra', 'Casey', 'Casie', 'Cassandra', 'Cassidy',
        'Cassie', 'Cassondra', 'Catalina', 'Catharine', 'Catherine', 'Cathern',
        'Cathey', 'Cathi', 'Cathie', 'Cathleen', 'Cathrine', 'Cathryn', 'Cathy',
        'Catina', 'Catrina', 'Caydence', 'Cayla', 'Caylee', 'Cecelia', 'Cecile',
        'Cecilia', 'Cecily', 'Ceil', 'Celena', 'Celesta', 'Celeste', 'Celestia',
        'Celestine', 'Celia', 'Celie', 'Celina', 'Celine', 'Cena', 'Ceola',
        'Chaka', 'Chana', 'Chanda', 'Chandler', 'Chandra', 'Chanel', 'Chanelle',
        'Chaney', 'Chanie', 'Channie', 'Channing', 'Chantal', 'Chante',
        'Chantel', 'Chantelle', 'Charissa', 'Charisse', 'Charity', 'Charla',
        'Charlee', 'Charleen', 'Charlene', 'Charley', 'Charlie', 'Charline',
        'Charlize', 'Charlotta', 'Charlotte', 'Charlottie', 'Charlsie',
        'Charmaine', 'Charolette', 'Chase', 'Chasity', 'Chastity', 'Chaya',
        'Chelsea', 'Chelsey', 'Chelsi', 'Chelsie', 'Chelsy', 'Cher', 'Cherelle',
        'Cheri', 'Cherie', 'Cherilyn', 'Cherise', 'Cherish', 'Cherrelle',
        'Cherri', 'Cherrie', 'Cherry', 'Cherryl', 'Cheryl', 'Cheryle',
        'Cheryll', 'Chessie', 'Chestina', 'Cheyanne', 'Cheyenne', 'Chimere',
        'China', 'Chiquita', 'Chloe', 'Chloie', 'Chris', 'Chrissie', 'Chrissy',
        'Christa', 'Christal', 'Christeen', 'Christel', 'Christen', 'Christena',
        'Christene', 'Christi', 'Christian', 'Christiana', 'Christie',
        'Christin', 'Christina', 'Christine', 'Christy', 'Chrystal', 'Chyna',
        'Chynna', 'Ciara', 'Ciarra', 'Cicely', 'Cielo', 'Ciera', 'Cierra',
        'Ciji', 'Cilla', 'Cinda', 'Cindi', 'Cindy', 'Cinnamon', 'Cinthia',
        'Citlali', 'Citlalli', 'Clair', 'Claire', 'Clara', 'Clarabelle',
        'Clare', 'Claribel', 'Clarice', 'Clarinda', 'Clarine', 'Clarisa',
        'Clarissa', 'Classie', 'Claudette', 'Claudia', 'Claudie', 'Claudine',
        'Cleda', 'Clella', 'Clem', 'Clemence', 'Clementina', 'Clementine',
        'Clemie', 'Clemma', 'Clemmie', 'Cleo', 'Cleola', 'Cleone', 'Cleora',
        'Cleta', 'Cleva', 'Clevie', 'Cliffie', 'Cloe', 'Clora', 'Clotilda',
        'Clotilde', 'Clyda', 'Clydie', 'Clytie', 'Coleen', 'Coletta', 'Colette',
        'Colleen', 'Collette', 'Columbia', 'Concepcion', 'Concetta', 'Concha',
        'Connie', 'Constance', 'Consuela', 'Consuelo', 'Contina', 'Cora',
        'Coraima', 'Coral', 'Coralie', 'Corda', 'Cordelia', 'Cordella',
        'Cordia', 'Cordie', 'Corean', 'Corene', 'Coretta', 'Corey', 'Cori',
        'Corie', 'Corina', 'Corine', 'Corinna', 'Corinne', 'Corliss',
        'Cornelia', 'Cornie', 'Corrie', 'Corrina', 'Corrine', 'Cortney', 'Cory',
        'Courtney', 'Creola', 'Cressie', 'Crete', 'Crissie', 'Crissy', 'Crista',
        'Cristal', 'Cristen', 'Cristi', 'Cristin', 'Cristina', 'Cristine',
        'Cristy', 'Cruz', 'Crysta', 'Crystal', 'Cuba', 'Cydney', 'Cyndi',
        'Cyntha', 'Cynthia', 'Dafne', 'Dagmar', 'Dagny', 'Dahlia', 'Daija',
        'Daijah', 'Daisey', 'Daisha', 'Daisie', 'Daisy', 'Daisye', 'Daja',
        'Dakota', 'Dale', 'Dalia', 'Dallas', 'Damaris', 'Dana', 'Danae',
        'Daneen', 'Danelle', 'Danette', 'Dani', 'Dania', 'Danica', 'Daniela',
        'Daniele', 'Daniella', 'Danielle', 'Danika', 'Danita', 'Danna',
        'Dannie', 'Dannielle', 'Danyel', 'Danyell', 'Danyelle', 'Daphne',
        'Dara', 'Darby', 'Darci', 'Darcie', 'Darcy', 'Daria', 'Darian',
        'Dariana', 'Darla', 'Darleen', 'Darlene', 'Darline', 'Darlyne', 'Dasia',
        'Davina', 'Dawn', 'Dawna', 'Dawne', 'Dayami', 'Dayana', 'Dayanara',
        'Dayle', 'Dayna', 'Dayse', 'Deana', 'Deandra', 'Deann', 'Deanna',
        'Deanne', 'Deasia', 'Deb', 'Debbi', 'Debbie', 'Debbra', 'Debby',
        'Debera', 'Debi', 'Debora', 'Deborah', 'Deborrah', 'Debra', 'Debrah',
        'Debroah', 'Dedra', 'Dee', 'Deeann', 'Deedee', 'Deena', 'Deetta',
        'Deidra', 'Deidre', 'Deirdre', 'Deja', 'Dejah', 'Delaney', 'Delcie',
        'Delfina', 'Delia', 'Deliah', 'Delila', 'Delilah', 'Delina', 'Delinda',
        'Delisa', 'Dell', 'Della', 'Dellar', 'Delle', 'Dellia', 'Dellie',
        'Delma', 'Delois', 'Delora', 'Delores', 'Deloris', 'Delpha', 'Delphia',
        'Delphine', 'Delsie', 'Delta', 'Dema', 'Demetra', 'Demetria', 'Demi',
        'Dena', 'Deneen', 'Denese', 'Denice', 'Denine', 'Denise', 'Denisha',
        'Denisse', 'Denita', 'Dennie', 'Desirae', 'Desiree', 'Dessa', 'Dessie',
        'Destany', 'Destinee', 'Destiney', 'Destini', 'Destiny', 'Devan',
        'Devin', 'Devon', 'Devyn', 'Dewey', 'Deyanira', 'Dezzie', 'Diamond',
        'Dian', 'Diana', 'Diandra', 'Diane', 'Diann', 'Dianna', 'Dianne',
        'Dicie', 'Dicy', 'Dillie', 'Dimple', 'Dina', 'Dinah', 'Dione', 'Dionne',
        'Dixie', 'Diya', 'Djuana', 'Djuna', 'Docia', 'Dola', 'Dollie', 'Dolly',
        'Dollye', 'Dolores', 'Doloris', 'Domenica', 'Dominga', 'Dominique',
        'Dominque', 'Domonique', 'Dona', 'Donia', 'Donie', 'Donita', 'Donna',
        'Donnie', 'Dora', 'Dorathea', 'Dorathy', 'Dorcas', 'Doreen', 'Dorene',
        'Doretha', 'Doretta', 'Dori', 'Dorinda', 'Dorine', 'Doris', 'Dorla',
        'Dorotha', 'Dorothea', 'Dorothy', 'Dorris', 'Dortha', 'Dorthea',
        'Dorthey', 'Dorthy', 'Dosha', 'Doshia', 'Doshie', 'Dosia', 'Dossie',
        'Dot', 'Dottie', 'Dotty', 'Dove', 'Dovie', 'Drema', 'Drew', 'Drucilla',
        'Drusilla', 'Dulce', 'Dulcie', 'Dusty', 'Dwan', 'Dyan', 'Dylan',
        'Earlean', 'Earlene', 'Earlie', 'Earline', 'Earnestine', 'Eartha',
        'Easter', 'Eathel', 'Ebba', 'Eboni', 'Ebony', 'Echo', 'Eda', 'Eddie',
        'Eden', 'Edie', 'Edith', 'Edla', 'Edmonia', 'Edna', 'Ednah', 'Edra',
        'Edrie', 'Edris', 'Edwina', 'Edyth', 'Edythe', 'Effa', 'Effie',
        'Eileen', 'Eithel', 'Ela', 'Elaina', 'Elaine', 'Elana', 'Elayne',
        'Elba', 'Elberta', 'Elda', 'Eldora', 'Eleanor', 'Eleanora', 'Eleanore',
        'Elease', 'Electa', 'Elena', 'Elenor', 'Elenora', 'Elenore', 'Eleonora',
        'Eleonore', 'Elfie', 'Elfreda', 'Elfrieda', 'Elgie', 'Elia', 'Eliana',
        'Elianna', 'Elida', 'Elinor', 'Elinore', 'Elisa', 'Elisabeth', 'Elise',
        'Elisha', 'Elissa', 'Eliza', 'Elizabet', 'Elizabeth', 'Elizbeth',
        'Elizebeth', 'Ella', 'Ellamae', 'Ellar', 'Elle', 'Ellen', 'Eller',
        'Elliana', 'Ellie', 'Ellyn', 'Elma', 'Elmina', 'Elmira', 'Elmire',
        'Elmyra', 'Elna', 'Elnora', 'Elodie', 'Elois', 'Eloisa', 'Eloise',
        'Elouise', 'Elsa', 'Else', 'Elsie', 'Elta', 'Elva', 'Elvera', 'Elvia',
        'Elvie', 'Elvina', 'Elvira', 'Elwanda', 'Elyse', 'Elyssa', 'Elza',
        'Elzada', 'Ema', 'Emaline', 'Ember', 'Emelia', 'Emelie', 'Emeline',
        'Emely', 'Emerald', 'Emerson', 'Emery', 'Emilee', 'Emilia', 'Emilie',
        'Emily', 'Emma', 'Emmalee', 'Emmaline', 'Emmer', 'Emmie', 'Emmy',
        'Emogene', 'Ena', 'Enid', 'Enola', 'Enriqueta', 'Eola', 'Eppie',
        'Epsie', 'Era', 'Erica', 'Ericka', 'Erie', 'Erika', 'Erin', 'Eris',
        'Erla', 'Erlene', 'Erlinda', 'Erline', 'Erma', 'Ermina', 'Ermine',
        'Erna', 'Ernestina', 'Ernestine', 'Erykah', 'Eryn', 'Esmeralda',
        'Esperanza', 'Essa', 'Essence', 'Essie', 'Esta', 'Estefani',
        'Estefania', 'Estefany', 'Estela', 'Estell', 'Estella', 'Estelle',
        'Ester', 'Esther', 'Estie', 'Estrella', 'Etha', 'Ethel', 'Ethelene',
        'Ethelyn', 'Ether', 'Ethie', 'Ethyl', 'Ethyle', 'Etna', 'Etta', 'Etter',
        'Ettie', 'Eudora', 'Eugenia', 'Eugenie', 'Eula', 'Eulah', 'Eulalia',
        'Eulalie', 'Euna', 'Eunice', 'Euphemia', 'Eura', 'Eva', 'Evalena',
        'Evaline', 'Evalyn', 'Evangelina', 'Evangeline', 'Eve', 'Evelena',
        'Evelin', 'Evelina', 'Eveline', 'Evelyn', 'Evelyne', 'Evelynn', 'Ever',
        'Evette', 'Evia', 'Evie', 'Evita', 'Evon', 'Evonne', 'Exa', 'Exie',
        'Fabiola', 'Fae', 'Fairy', 'Faith', 'Fallon', 'Falon', 'Fannie',
        'Fanny', 'Fannye', 'Farah', 'Farrah', 'Fatima', 'Fawn', 'Fay', 'Faye',
        'Felecia', 'Felice', 'Felicia', 'Felicie', 'Felicitas', 'Felicity',
        'Felipa', 'Felisha', 'Fern', 'Fernanda', 'Ferne', 'Fidelia', 'Filomena',
        'Finley', 'Fiona', 'Flavia', 'Fleda', 'Fleeta', 'Fleta', 'Flo',
        'Flonnie', 'Flor', 'Flora', 'Florance', 'Florence', 'Florene',
        'Floretta', 'Florida', 'Florie', 'Florine', 'Florrie', 'Flossie',
        'Floy', 'Fonda', 'Forest', 'Fran', 'Franc', 'Frances', 'Francesca',
        'Francies', 'Francina', 'Francine', 'Francis', 'Francisca',
        'Francisquita', 'Frankie', 'Freda', 'Freddie', 'Frederica',
        'Fredericka', 'Freeda', 'Freida', 'Frida', 'Frieda', 'Frona', 'Fronia',
        'Fronie', 'Fronnie', 'Fumiko', 'Gabriela', 'Gabriella', 'Gabrielle',
        'Gail', 'Gale', 'Galilea', 'Garnet', 'Garnett', 'Gay', 'Gaye', 'Gayla',
        'Gayle', 'Gaylene', 'Gaynell', 'Gearldine', 'Gemma', 'Gena', 'Gene',
        'Genesis', 'Geneva', 'Genevieve', 'Genevra', 'Genie', 'Gennie',
        'Genoveva', 'Georganna', 'Georgeann', 'Georgeanna', 'Georgene',
        'Georgetta', 'Georgette', 'Georgia', 'Georgiana', 'Georgiann',
        'Georgianna', 'Georgie', 'Georgina', 'Georgine', 'Geraldine', 'Geralyn',
        'Gerda', 'Geri', 'Germaine', 'Gerri', 'Gerry', 'Gertha', 'Gertie',
        'Gertrude', 'Gia', 'Giada', 'Giana', 'Gianna', 'Gidget', 'Gigi',
        'Gilda', 'Gillian', 'Gillie', 'Gina', 'Ginger', 'Ginny', 'Giovanna',
        'Girtha', 'Gisele', 'Giselle', 'Gisselle', 'Giuliana', 'Gladis',
        'Gladyce', 'Gladys', 'Glenda', 'Glendora', 'Glenn', 'Glenna', 'Glennie',
        'Glennis', 'Glinda', 'Gloria', 'Glynda', 'Glynis', 'Golda', 'Golden',
        'Goldia', 'Goldie', 'Grace', 'Gracelyn', 'Gracia', 'Gracie', 'Graciela',
        'Grayce', 'Grecia', 'Gregoria', 'Greta', 'Gretchen', 'Gretta', 'Grisel',
        'Griselda', 'Guadalupe', 'Gunda', 'Gussie', 'Gusta', 'Gustie', 'Gwen',
        'Gwenda', 'Gwendolyn', 'Gwyn', 'Gwyneth', 'Hadassah', 'Hadley',
        'Hailee', 'Hailey', 'Hailie', 'Haleigh', 'Haley', 'Hali', 'Halie',
        'Halle', 'Halley', 'Hallie', 'Hana', 'Hanna', 'Hannah', 'Harlene',
        'Harley', 'Harlow', 'Harmony', 'Harper', 'Harriet', 'Harriett',
        'Harriette', 'Haruko', 'Hasel', 'Hassie', 'Hattie', 'Haven', 'Hayden',
        'Haylee', 'Hayleigh', 'Hayley', 'Haylie', 'Hazel', 'Hazelle', 'Hazle',
        'Heather', 'Heaven', 'Hedwig', 'Hedy', 'Heidi', 'Heidy', 'Helaine',
        'Helen', 'Helena', 'Helene', 'Helga', 'Hellen', 'Helma', 'Helyn',
        'Hennie', 'Henretta', 'Henrietta', 'Henriette', 'Herlinda', 'Herma',
        'Hermina', 'Hermine', 'Herminia', 'Hertha', 'Hessie', 'Hester',
        'Hettie', 'Hetty', 'Hilah', 'Hilary', 'Hilda', 'Hildegard',
        'Hildegarde', 'Hildred', 'Hildur', 'Hillary', 'Hilma', 'Holli',
        'Hollie', 'Hollis', 'Holly', 'Honora', 'Hope', 'Hortencia', 'Hortense',
        'Hortensia', 'Hulda', 'Huldah', 'Hunter', 'Ica', 'Icey', 'Icie', 'Icy',
        'Ida', 'Idabelle', 'Idamae', 'Idell', 'Idella', 'Iesha', 'Ieshia',
        'Ila', 'Ilah', 'Ilda', 'Ilene', 'Iliana', 'Illa', 'Ilma', 'Ilo',
        'Ilona', 'Ima', 'Imani', 'Imelda', 'Imo', 'Imogene', 'Ina', 'India',
        'Indiana', 'Inell', 'Ines', 'Inez', 'Infant', 'Inga', 'Ingeborg',
        'Inger', 'Ingrid', 'Iola', 'Iona', 'Ione', 'Ira', 'Ireland', 'Irena',
        'Irene', 'Iridian', 'Irine', 'Iris', 'Irma', 'Irva', 'Isa', 'Isabel',
        'Isabela', 'Isabell', 'Isabella', 'Isabelle', 'Isadora', 'Isamar',
        'Isis', 'Isla', 'Isobel', 'Itzel', 'Iva', 'Ivah', 'Ivana', 'Ivanna',
        'Ivette', 'Ivey', 'Ivie', 'Ivonne', 'Ivory', 'Ivy', 'Iyana', 'Iyanna',
        'Iza', 'Izabella', 'Izabelle', 'Izetta', 'Izola', 'Izora', 'Jacalyn',
        'Jacey', 'Jackeline', 'Jacki', 'Jackie', 'Jacklyn', 'Jaclyn', 'Jacque',
        'Jacquelin', 'Jacqueline', 'Jacquelyn', 'Jacquline', 'Jacqulyn', 'Jada',
        'Jade', 'Jaden', 'Jadyn', 'Jaeda', 'Jaelyn', 'Jaelynn', 'Jaida',
        'Jaiden', 'Jaidyn', 'Jailene', 'Jailyn', 'Jaime', 'Jaimee', 'Jakayla',
        'Jaleesa', 'Jalisa', 'Jalissa', 'Jaliyah', 'Jalyn', 'Jalynn', 'Jamey',
        'Jami', 'Jamie', 'Jamila', 'Jamiya', 'Jammie', 'Jamya', 'Jan', 'Jana',
        'Janae', 'Janay', 'Jane', 'Janeen', 'Janel', 'Janell', 'Janelle',
        'Janene', 'Janessa', 'Janet', 'Janette', 'Janey', 'Janiah', 'Janice',
        'Janie', 'Janine', 'Janis', 'Janiya', 'Janiyah', 'Jann', 'Janna',
        'Jannette', 'Jannie', 'January', 'Janyce', 'Jaquelin', 'Jaqueline',
        'Jaslene', 'Jaslyn', 'Jasmin', 'Jasmine', 'Jasmyn', 'Jasmyne',
        'Jaunita', 'Jaycee', 'Jaycie', 'Jayda', 'Jayde', 'Jayden', 'Jaye',
        'Jayla', 'Jaylah', 'Jaylee', 'Jayleen', 'Jaylen', 'Jaylene', 'Jaylin',
        'Jaylyn', 'Jaylynn', 'Jayme', 'Jayne', 'Jazlene', 'Jazlyn', 'Jazlynn',
        'Jazmin', 'Jazmine', 'Jazmyn', 'Jazmyne', 'Jean', 'Jeana', 'Jeane',
        'Jeanetta', 'Jeanette', 'Jeanie', 'Jeanine', 'Jeanmarie', 'Jeanna',
        'Jeanne', 'Jeannette', 'Jeannie', 'Jeannine', 'Jeffie', 'Jemima',
        'Jena', 'Jenelle', 'Jenifer', 'Jenilee', 'Jenna', 'Jennette', 'Jenni',
        'Jennie', 'Jennifer', 'Jenniffer', 'Jenny', 'Jensen', 'Jeraldine',
        'Jeri', 'Jerica', 'Jerilyn', 'Jerilynn', 'Jerri', 'Jerrica', 'Jerrie',
        'Jerrilyn', 'Jerusha', 'Jeryl', 'Jesenia', 'Jesica', 'Jesse',
        'Jessenia', 'Jessi', 'Jessica', 'Jessie', 'Jessika', 'Jessye', 'Jetta',
        'Jettie', 'Jewel', 'Jewell', 'Jill', 'Jillian', 'Jimena', 'Jinnie',
        'Jo', 'Joan', 'Joana', 'Joanie', 'Joann', 'Joanna', 'Joanne', 'Jocelyn',
        'Jocelyne', 'Jocelynn', 'Jodi', 'Jodie', 'Jody', 'Joell', 'Joella',
        'Joelle', 'Joellen', 'Joetta', 'Joette', 'Johana', 'Johanna',
        'Johannah', 'Johnie', 'Johnna', 'Johnnie', 'Joi', 'Joleen', 'Jolene',
        'Jolette', 'Jolie', 'Joline', 'Jonell', 'Joni', 'Jonna', 'Jonnie',
        'Jordan', 'Jordin', 'Jordyn', 'Joretta', 'Jorja', 'Josefa', 'Josefina',
        'Josefita', 'Joselin', 'Joseline', 'Joselyn', 'Josephine', 'Josette',
        'Josie', 'Josiephine', 'Joslyn', 'Jossie', 'Journey', 'Jovita', 'Joy',
        'Joyce', 'Joycelyn', 'Joye', 'Juana', 'Juanita', 'Judi', 'Judie',
        'Judith', 'Judy', 'Judyth', 'Jule', 'Juli', 'Julia', 'Juliana',
        'Juliann', 'Julianna', 'Julianne', 'Julie', 'Juliet', 'Juliette',
        'Julisa', 'Julissa', 'June', 'Junia', 'Junie', 'Justice', 'Justina',
        'Justine', 'Kaaren', 'Kacey', 'Kaci', 'Kacie', 'Kacy', 'Kadence',
        'Kadijah', 'Kaela', 'Kaelyn', 'Kaelynn', 'Kaia', 'Kaila', 'Kailee',
        'Kailey', 'Kailyn', 'Kaitlin', 'Kaitlyn', 'Kaitlynn', 'Kaiya', 'Kala',
        'Kaleena', 'Kaleigh', 'Kalene', 'Kaley', 'Kali', 'Kalie', 'Kaliyah',
        'Kallie', 'Kalyn', 'Kamari', 'Kameron', 'Kami', 'Kamila', 'Kamilah',
        'Kamora', 'Kamryn', 'Kamya', 'Kandace', 'Kandi', 'Kandice', 'Kandy',
        'Kanesha', 'Kanisha', 'Kara', 'Karan', 'Karel', 'Karen', 'Kari',
        'Karie', 'Karin', 'Karina', 'Karis', 'Karissa', 'Karla', 'Karlee',
        'Karlene', 'Karley', 'Karli', 'Karlie', 'Karly', 'Karma', 'Karol',
        'Karolyn', 'Karon', 'Karren', 'Karri', 'Karrie', 'Karsyn', 'Karyl',
        'Karyme', 'Karyn', 'Kasandra', 'Kasey', 'Kasie', 'Kassandra', 'Kassidy',
        'Kassie', 'Katarina', 'Kate', 'Katelin', 'Katelyn', 'Katelynn',
        'Katerina', 'Kathaleen', 'Katharina', 'Katharine', 'Katharyn',
        'Katherin', 'Katherine', 'Kathern', 'Katheryn', 'Kathey', 'Kathi',
        'Kathie', 'Kathleen', 'Kathlene', 'Kathlyn', 'Kathrine', 'Kathryn',
        'Kathryne', 'Kathy', 'Kathyrn', 'Kati', 'Katia', 'Katie', 'Katina',
        'Katlin', 'Katlyn', 'Katlynn', 'Katrina', 'Kattie', 'Katy', 'Kay',
        'Kaya', 'Kaycee', 'Kayden', 'Kaydence', 'Kaye', 'Kayla', 'Kaylah',
        'Kaylan', 'Kaylee', 'Kayleen', 'Kayleigh', 'Kaylen', 'Kaylene',
        'Kayley', 'Kayli', 'Kaylie', 'Kaylin', 'Kaylyn', 'Kaylynn', 'Kazuko',
        'Keanna', 'Keara', 'Kecia', 'Keeley', 'Keely', 'Keena', 'Keesha',
        'Keila', 'Keira', 'Keisha', 'Kelcie', 'Keli', 'Kelis', 'Kellee',
        'Kelley', 'Kelli', 'Kellie', 'Kelly', 'Kelsea', 'Kelsey', 'Kelsi',
        'Kelsie', 'Kendal', 'Kendall', 'Kendra', 'Kenia', 'Kenisha', 'Kenley',
        'Kenna', 'Kennedi', 'Kennedy', 'Kenya', 'Kenyatta', 'Kenzie', 'Keri',
        'Kerri', 'Kerrie', 'Kerry', 'Kesha', 'Keshia', 'Keyla', 'Khadijah',
        'Khalilah', 'Khloe', 'Kia', 'Kiana', 'Kianna', 'Kiara', 'Kiarra',
        'Kiera', 'Kierra', 'Kiersten', 'Kiley', 'Kim', 'Kimber', 'Kimberely',
        'Kimberlee', 'Kimberley', 'Kimberli', 'Kimberlie', 'Kimberly', 'Kimora',
        'Kindra', 'Kinley', 'Kinsey', 'Kinsley', 'Kira', 'Kirsten', 'Kirstie',
        'Kirstin', 'Kisha', 'Kittie', 'Kitty', 'Kiya', 'Kiyoko', 'Kizzie',
        'Kizzy', 'Kloe', 'Kori', 'Kortney', 'Kourtney', 'Kris', 'Krissy',
        'Krista', 'Kristal', 'Kristan', 'Kristen', 'Kristi', 'Kristian',
        'Kristie', 'Kristin', 'Kristina', 'Kristine', 'Kristy', 'Kristyn',
        'Krysta', 'Krystal', 'Krysten', 'Krystin', 'Krystina', 'Krystle', 'Kya',
        'Kyara', 'Kyla', 'Kylah', 'Kyle', 'Kylee', 'Kyleigh', 'Kylene', 'Kylie',
        'Kyra', 'Kyrie', 'Lacey', 'Laci', 'Lacie', 'Lacy', 'Ladonna', 'Lady',
        'Lahoma', 'Laila', 'Lailah', 'Lainey', 'Laisha', 'Lakeisha', 'Laken',
        'Lakendra', 'Lakesha', 'Lakeshia', 'Lakisha', 'Lala', 'Lalla', 'Lana',
        'Lanette', 'Laney', 'Lani', 'Lanie', 'Lanita', 'Lannie', 'Laquita',
        'Lara', 'Larae', 'Laraine', 'Larissa', 'Larue', 'Lashanda', 'Lashawn',
        'Lashonda', 'Lashunda', 'Lasonya', 'Lassie', 'Latanya', 'Latarsha',
        'Latasha', 'Latesha', 'Latifah', 'Latisha', 'Latonia', 'Latonya',
        'Latoria', 'Latosha', 'Latoya', 'Latoyia', 'Latrice', 'Latricia',
        'Latrina', 'Launa', 'Laura', 'Laureen', 'Laurel', 'Lauren', 'Laurene',
        'Lauretta', 'Laurette', 'Lauri', 'Laurie', 'Laurine', 'Lauryn',
        'Lavada', 'Lavelle', 'Lavenia', 'Lavera', 'Lavern', 'Laverna',
        'Laverne', 'Lavina', 'Lavinia', 'Lavon', 'Lavona', 'Lavonda', 'Lavonia',
        'Lavonne', 'Lawanda', 'Layla', 'Laylah', 'Lea', 'Leafy', 'Leah',
        'Leala', 'Leana', 'Leandra', 'Leaner', 'Leann', 'Leanna', 'Leanne',
        'Leatha', 'Leatrice', 'Leda', 'Lee', 'Leeann', 'Leesa', 'Leia', 'Leigh',
        'Leighton', 'Leila', 'Leilani', 'Leisa', 'Leisha', 'Leitha', 'Lela',
        'Lelah', 'Lelar', 'Lelia', 'Lella', 'Lemma', 'Lempi', 'Lena', 'Lenna',
        'Lennie', 'Lenora', 'Lenore', 'Leola', 'Leoma', 'Leona', 'Leone',
        'Leonia', 'Leonie', 'Leonor', 'Leonora', 'Leonore', 'Leontine', 'Leora',
        'Leota', 'Lera', 'Lesa', 'Lesia', 'Leslee', 'Lesley', 'Lesli', 'Leslie',
        'Lesly', 'Lessie', 'Lesta', 'Leta', 'Letha', 'Lethia', 'Leticia',
        'Letitia', 'Letta', 'Lettie', 'Letty', 'Leva', 'Levina', 'Lexi',
        'Lexie', 'Lexis', 'Lexus', 'Leyla', 'Lia', 'Liana', 'Liane', 'Libbie',
        'Libby', 'Liberty', 'Lida', 'Liddie', 'Lidia', 'Lidie', 'Lila', 'Lilah',
        'Lilia', 'Lilian', 'Liliana', 'Lilianna', 'Lilie', 'Lilla', 'Liller',
        'Lillia', 'Lillian', 'Lilliana', 'Lillianna', 'Lillie', 'Lillis',
        'Lilly', 'Lily', 'Lilyan', 'Lilyana', 'Lilyanna', 'Lina', 'Linda',
        'Lindsay', 'Lindsey', 'Lindy', 'Linette', 'Linna', 'Linnea', 'Linnie',
        'Linsey', 'Lisa', 'Lisbeth', 'Lise', 'Lisette', 'Lisha', 'Lissa',
        'Lissette', 'Lissie', 'Lita', 'Litha', 'Littie', 'Litzy', 'Livia',
        'Liz', 'Liza', 'Lizabeth', 'Lizbeth', 'Lizeth', 'Lizette', 'Lizzie',
        'Lockie', 'Loda', 'Logan', 'Lois', 'Lola', 'Lolita', 'Lolla', 'Lollie',
        'Loma', 'Lona', 'London', 'Londyn', 'Loni', 'Lonie', 'Lonna', 'Lonnie',
        'Lora', 'Loraine', 'Lorayne', 'Lorean', 'Loree', 'Loreen', 'Lorelai',
        'Lorelei', 'Loren', 'Lorena', 'Lorene', 'Lorenza', 'Loretta', 'Loretto',
        'Lori', 'Loria', 'Loriann', 'Lorie', 'Lorinda', 'Lorine', 'Loris',
        'Lorna', 'Lorraine', 'Lorrayne', 'Lorri', 'Lorrie', 'Lossie', 'Lota',
        'Lotta', 'Lottie', 'Lou', 'Louann', 'Louanna', 'Louella', 'Louetta',
        'Louie', 'Louisa', 'Louise', 'Louisiana', 'Loula', 'Lourdes',
        'Louvenia', 'Love', 'Lovey', 'Lovie', 'Lovina', 'Lovisa', 'Loyce', 'Lu',
        'Luana', 'Luann', 'Luanne', 'Luberta', 'Lucero', 'Lucetta', 'Lucia',
        'Luciana', 'Lucie', 'Lucile', 'Lucille', 'Lucina', 'Lucinda', 'Lucindy',
        'Lucretia', 'Lucy', 'Luda', 'Ludie', 'Lue', 'Luella', 'Luetta',
        'Lugenia', 'Luisa', 'Lula', 'Lulah', 'Lular', 'Lulie', 'Lulla', 'Lulu',
        'Luna', 'Lupe', 'Lura', 'Lurana', 'Lurena', 'Lurline', 'Lutie',
        'Luvenia', 'Luverne', 'Luvinia', 'Luz', 'Lyda', 'Lydia', 'Lyla',
        'Lylah', 'Lyn', 'Lynda', 'Lyndia', 'Lyndsay', 'Lyndsey', 'Lynette',
        'Lynn', 'Lynne', 'Lynnette', 'Lynsey', 'Lyric', 'Mabel', 'Mabell',
        'Mabelle', 'Mable', 'Macel', 'Macey', 'Machelle', 'Maci', 'Macie',
        'Mackenzie', 'Macy', 'Madaline', 'Madalyn', 'Madalynn', 'Maddison',
        'Madeleine', 'Madelene', 'Madeline', 'Madelyn', 'Madelynn', 'Madge',
        'Madie', 'Madilyn', 'Madilynn', 'Madisen', 'Madison', 'Madisyn',
        'Madlyn', 'Madonna', 'Madora', 'Madyson', 'Mae', 'Maebell', 'Maebelle',
        'Maegan', 'Maeve', 'Mafalda', 'Magan', 'Magdalen', 'Magdalena',
        'Magdalene', 'Magen', 'Maggie', 'Magnolia', 'Mahala', 'Mahalia',
        'Mahalie', 'Mai', 'Maia', 'Maida', 'Maira', 'Maiya', 'Makaila',
        'Makala', 'Makayla', 'Makena', 'Makenna', 'Makenzie', 'Malaya',
        'Maleah', 'Malia', 'Maliah', 'Malinda', 'Malissa', 'Malissie',
        'Maliyah', 'Mallie', 'Mallorie', 'Mallory', 'Malorie', 'Malvina',
        'Mame', 'Mamie', 'Mammie', 'Manda', 'Mandi', 'Mandie', 'Mandy',
        'Manerva', 'Manervia', 'Manie', 'Manila', 'Manilla', 'Mannie',
        'Manuela', 'Manuelita', 'Mara', 'Maralyn', 'Maranda', 'Marcela',
        'Marcelina', 'Marceline', 'Marcella', 'Marcelle', 'Marci', 'Marcia',
        'Marcie', 'Marcy', 'Mardell', 'Mareli', 'Marely', 'Maren', 'Margaret',
        'Margarete', 'Margaretha', 'Margarett', 'Margaretta', 'Margarette',
        'Margarita', 'Margarite', 'Marge', 'Margene', 'Margeret', 'Margery',
        'Marget', 'Margie', 'Margo', 'Margot', 'Margret', 'Margrett',
        'Margretta', 'Marguerite', 'Margueritte', 'Margurite', 'Margy', 'Mari',
        'Maria', 'Mariah', 'Mariam', 'Marian', 'Mariana', 'Marianita',
        'Mariann', 'Marianna', 'Marianne', 'Maribel', 'Maribeth', 'Maricela',
        'Marie', 'Mariel', 'Mariela', 'Marietta', 'Marilee', 'Marilla',
        'Marilou', 'Marilyn', 'Marilynn', 'Marin', 'Marina', 'Marinda',
        'Marion', 'Marisa', 'Marisela', 'Marisol', 'Marissa', 'Marita',
        'Maritza', 'Mariyah', 'Marjorie', 'Marjory', 'Markita', 'Marla',
        'Marlana', 'Marlee', 'Marleen', 'Marleigh', 'Marlen', 'Marlena',
        'Marlene', 'Marley', 'Marlie', 'Marlo', 'Marlyn', 'Marlys', 'Marni',
        'Marnie', 'Marnita', 'Marolyn', 'Marquita', 'Marry', 'Marsha', 'Marta',
        'Martha', 'Marti', 'Martika', 'Martina', 'Martine', 'Marty', 'Marva',
        'Marvel', 'Mary', 'Maryam', 'Maryann', 'Maryanne', 'Marybelle',
        'Marybeth', 'Maryellen', 'Maryjane', 'Maryjo', 'Marylee', 'Marylin',
        'Marylou', 'Marylouise', 'Marylyn', 'Masako', 'Mathilda', 'Mathilde',
        'Matie', 'Matilda', 'Matilde', 'Mattie', 'Mattye', 'Maud', 'Maude',
        'Maudie', 'Maura', 'Maureen', 'Maurine', 'Mavis', 'Maxie', 'Maxine',
        'May', 'Maya', 'Maybell', 'Maybelle', 'Maye', 'Mayme', 'Maymie',
        'Mayra', 'Mazie', 'Mckayla', 'Mckenna', 'Mckenzie', 'Mckinley',
        'Meadow', 'Meagan', 'Meaghan', 'Mechelle', 'Meda', 'Media', 'Medora',
        'Meg', 'Megan', 'Meggan', 'Meghan', 'Meghann', 'Melanie', 'Melany',
        'Melba', 'Melina', 'Melinda', 'Melisa', 'Melissa', 'Melissia', 'Mell',
        'Mellie', 'Mellisa', 'Mellissa', 'Melodee', 'Melodie', 'Melody',
        'Melonie', 'Melony', 'Melva', 'Melvina', 'Mena', 'Mendy', 'Mercedes',
        'Mercy', 'Meredith', 'Merilyn', 'Merle', 'Merlene', 'Merna', 'Merri',
        'Merrie', 'Merrilee', 'Merrily', 'Merry', 'Mertie', 'Meryl', 'Meta',
        'Metha', 'Metta', 'Mettie', 'Mia', 'Miah', 'Micaela', 'Micah',
        'Micayla', 'Michaela', 'Michaele', 'Michal', 'Michele', 'Michelina',
        'Michell', 'Michelle', 'Mickey', 'Mickie', 'Miesha', 'Migdalia',
        'Mignon', 'Mikaela', 'Mikaila', 'Mikala', 'Mikalah', 'Mikayla', 'Mila',
        'Milagros', 'Milan', 'Milda', 'Mildred', 'Miley', 'Milissa',
        'Millicent', 'Millie', 'Milly', 'Mima', 'Mimi', 'Mina', 'Minda',
        'Mindi', 'Mindy', 'Minerva', 'Minervia', 'Minna', 'Minnie', 'Minta',
        'Mintie', 'Mira', 'Miracle', 'Miranda', 'Mireya', 'Miriah', 'Miriam',
        'Mirna', 'Mirtie', 'Missie', 'Missouri', 'Missy', 'Misti', 'Mistie',
        'Misty', 'Mittie', 'Mitzi', 'Miya', 'Modena', 'Moesha', 'Moira',
        'Mollie', 'Molly', 'Mona', 'Monica', 'Monika', 'Monique', 'Monna',
        'Monnie', 'Monserrat', 'Montana', 'Montie', 'Mora', 'Morgan', 'Moriah',
        'Mossie', 'Mozell', 'Mozella', 'Mozelle', 'Muriel', 'Murl', 'Mya',
        'Myah', 'Myla', 'Mylee', 'Mylie', 'Myra', 'Myranda', 'Myrl', 'Myrle',
        'Myrna', 'Myrta', 'Myrtice', 'Myrtie', 'Myrtis', 'Myrtle', 'Nada',
        'Nadia', 'Nadine', 'Naima', 'Nakia', 'Nakisha', 'Nakita', 'Nallely',
        'Nan', 'Nana', 'Nanci', 'Nancie', 'Nancy', 'Nanette', 'Nanie', 'Nanna',
        'Nannette', 'Nannie', 'Naoma', 'Naomi', 'Narcissus', 'Natalee',
        'Natalia', 'Natalie', 'Nataly', 'Natalya', 'Natasha', 'Nathalia',
        'Nathalie', 'Nathaly', 'Natosha', 'Nautica', 'Nayeli', 'Nayely',
        'Nealie', 'Nealy', 'Nedra', 'Neha', 'Nelda', 'Nelia', 'Nelie', 'Nell',
        'Nella', 'Nelle', 'Nellie', 'Nelly', 'Nena', 'Neola', 'Neoma', 'Neppie',
        'Nereida', 'Neta', 'Netta', 'Nettie', 'Neva', 'Nevada', 'Nevaeh',
        'Neveah', 'Nia', 'Nichelle', 'Nichol', 'Nichole', 'Nicki', 'Nicola',
        'Nicole', 'Nicolette', 'Nicolle', 'Niki', 'Nikia', 'Nikita', 'Nikki',
        'Nikole', 'Nila', 'Nilda', 'Nina', 'Ninnie', 'Nira', 'Nita', 'Nobie',
        'Noel', 'Noelia', 'Noelle', 'Noemi', 'Noemie', 'Nohely', 'Nola',
        'Nolia', 'Nolie', 'Noma', 'Nona', 'Nonie', 'Nora', 'Norah', 'Noreen',
        'Norene', 'Noreta', 'Noretta', 'Norine', 'Norita', 'Norma', 'Nova',
        'Novella', 'Nya', 'Nyah', 'Nyasia', 'Nyla', 'Nylah', 'Nyree', 'Ocie',
        'Octa', 'Octavia', 'Octavie', 'Oda', 'Odalis', 'Odalys', 'Odelia',
        'Odell', 'Odessa', 'Odette', 'Odie', 'Odile', 'Ofelia', 'Ola', 'Olar',
        'Olena', 'Olene', 'Oleta', 'Olevia', 'Olga', 'Olie', 'Olinda', 'Oline',
        'Oliva', 'Olive', 'Olivia', 'Olivine', 'Ollie', 'Olympia', 'Oma',
        'Omie', 'Ona', 'Oneida', 'Oneta', 'Oney', 'Onie', 'Onnie', 'Opal',
        'Opha', 'Ophelia', 'Ora', 'Orah', 'Oral', 'Oralia', 'Orelia', 'Orene',
        'Orilla', 'Orlena', 'Orma', 'Orpha', 'Orra', 'Orrie', 'Osa', 'Osie',
        'Ossie', 'Ota', 'Otelia', 'Otha', 'Ottie', 'Ottilia', 'Ottilie',
        'Ouida', 'Ova', 'Ozell', 'Ozella', 'Ozie', 'Paige', 'Pairlee',
        'Paisley', 'Paityn', 'Pallie', 'Palma', 'Paloma', 'Pam', 'Pamala',
        'Pamela', 'Pamelia', 'Pamella', 'Pandora', 'Pansy', 'Paola', 'Paralee',
        'Paris', 'Parker', 'Parlee', 'Parthenia', 'Pat', 'Patience', 'Patrica',
        'Patrice', 'Patricia', 'Patsy', 'Patti', 'Pattie', 'Patty', 'Paula',
        'Pauletta', 'Paulette', 'Paulina', 'Pauline', 'Payten', 'Payton',
        'Pearl', 'Pearla', 'Pearle', 'Pearlene', 'Pearlie', 'Pearline',
        'Pearly', 'Peggie', 'Peggy', 'Penelope', 'Penni', 'Pennie', 'Penny',
        'Pepper', 'Perla', 'Permelia', 'Perri', 'Petra', 'Peyton', 'Phebe',
        'Pheobe', 'Phillis', 'Philomena', 'Philomene', 'Phoebe', 'Phoenix',
        'Phylicia', 'Phylis', 'Phyliss', 'Phyllis', 'Pink', 'Pinkey', 'Pinkie',
        'Piper', 'Pluma', 'Pollie', 'Polly', 'Porsche', 'Porsha', 'Portia',
        'Precious', 'Presley', 'Pricilla', 'Princess', 'Priscila', 'Priscilla',
        'Prudence', 'Prudie', 'Qiana', 'Queen', 'Queenie', 'Quiana', 'Quinn',
        'Rachael', 'Racheal', 'Rachel', 'Rachelle', 'Racquel', 'Rae', 'Raegan',
        'Raelyn', 'Raelynn', 'Rafaela', 'Ragna', 'Raina', 'Ramona', 'Randi',
        'Raquel', 'Rashida', 'Raven', 'Rayna', 'Rayne', 'Reagan', 'Reanna',
        'Reatha', 'Reba', 'Rebeca', 'Rebecca', 'Rebekah', 'Reece', 'Reese',
        'Regan', 'Regena', 'Regenia', 'Regina', 'Reilly', 'Reina', 'Rella',
        'Rena', 'Renada', 'Renae', 'Renata', 'Rene', 'Renea', 'Renee', 'Renita',
        'Rennie', 'Ressie', 'Reta', 'Retha', 'Retta', 'Rettie', 'Reva', 'Reyna',
        'Rhea', 'Rheta', 'Rhianna', 'Rhiannon', 'Rhoda', 'Rhona', 'Rhonda',
        'Rianna', 'Richelle', 'Ricki', 'Rihanna', 'Rikki', 'Riley', 'Rilla',
        'Rillie', 'Rinda', 'Risa', 'Rita', 'River', 'Riya', 'Robbie', 'Robbin',
        'Roberta', 'Robin', 'Robyn', 'Rochelle', 'Rocio', 'Roena', 'Rolanda',
        'Roma', 'Romaine', 'Romona', 'Rona', 'Ronda', 'Roni', 'Ronna', 'Ronnie',
        'Rory', 'Rosa', 'Rosabelle', 'Rosalee', 'Rosalia', 'Rosalie',
        'Rosalind', 'Rosalinda', 'Rosaline', 'Rosalyn', 'Rosamond', 'Rosann',
        'Rosanna', 'Rosanne', 'Rosaria', 'Rosario', 'Rose', 'Roseann',
        'Roseanna', 'Roseanne', 'Rosella', 'Roselyn', 'Rosemarie', 'Rosemary',
        'Rosena', 'Rosetta', 'Rosey', 'Rosia', 'Rosie', 'Rosina', 'Rosita',
        'Roslyn', 'Rossie', 'Rosy', 'Rowan', 'Rowena', 'Roxana', 'Roxane',
        'Roxann', 'Roxanna', 'Roxanne', 'Roxie', 'Roxy', 'Rozanne', 'Rozella',
        'Rubi', 'Rubie', 'Ruby', 'Rubye', 'Ruie', 'Ruth', 'Rutha', 'Ruthann',
        'Ruthanne', 'Ruthe', 'Ruthie', 'Ryann', 'Rylan', 'Rylee', 'Ryleigh',
        'Rylie', 'Sabina', 'Sable', 'Sabra', 'Sabrina', 'Sada', 'Sade', 'Sadie',
        'Sadye', 'Sage', 'Saige', 'Salena', 'Salina', 'Sallie', 'Sally',
        'Salma', 'Salome', 'Samantha', 'Samara', 'Samatha', 'Samira', 'Samiyah',
        'Sammie', 'Sanaa', 'Sanai', 'Sandi', 'Sandie', 'Sandra', 'Sandy',
        'Saniya', 'Saniyah', 'Sanjuana', 'Sanjuanita', 'Sannie', 'Santa',
        'Santana', 'Santina', 'Santos', 'Sara', 'Sarah', 'Sarahi', 'Sarai',
        'Sariah', 'Sarina', 'Sarita', 'Sarrah', 'Sasha', 'Saundra', 'Savana',
        'Savanah', 'Savanna', 'Savannah', 'Savilla', 'Scarlet', 'Scarlett',
        'Sebrina', 'Selah', 'Selena', 'Selene', 'Selina', 'Selma', 'Sena',
        'Senora', 'Serena', 'Serenity', 'Serina', 'Shae', 'Shaina', 'Shakira',
        'Shalon', 'Shalonda', 'Shameka', 'Shamika', 'Shana', 'Shanae', 'Shanda',
        'Shandra', 'Shane', 'Shaneka', 'Shanell', 'Shanelle', 'Shanequa',
        'Shani', 'Shania', 'Shanice', 'Shaniece', 'Shanika', 'Shaniqua',
        'Shanita', 'Shaniya', 'Shanna', 'Shannan', 'Shannen', 'Shannon',
        'Shanon', 'Shanta', 'Shante', 'Shantel', 'Shantell', 'Shaquana',
        'Shaquita', 'Shara', 'Shardae', 'Sharday', 'Sharde', 'Sharee', 'Sharen',
        'Shari', 'Sharita', 'Sharla', 'Sharleen', 'Sharlene', 'Sharman',
        'Sharon', 'Sharonda', 'Sharron', 'Sharyl', 'Sharyn', 'Shasta',
        'Shatara', 'Shauna', 'Shaunna', 'Shavon', 'Shavonne', 'Shawanda',
        'Shawna', 'Shawnda', 'Shawnee', 'Shawnna', 'Shawnte', 'Shay', 'Shayla',
        'Shaylee', 'Shayna', 'Shea', 'Sheena', 'Sheila', 'Sheilah', 'Shelba',
        'Shelbi', 'Shelbie', 'Shelby', 'Shelia', 'Shelley', 'Shelli', 'Shellie',
        'Shelly', 'Shelva', 'Shelvia', 'Shelvie', 'Shena', 'Shenna', 'Sheree',
        'Sheri', 'Sheridan', 'Sherie', 'Sherilyn', 'Sherita', 'Sherlyn',
        'Sheron', 'Sherree', 'Sherri', 'Sherrie', 'Sherrill', 'Sherron',
        'Sherry', 'Sherryl', 'Sheryl', 'Sheryll', 'Sheyla', 'Shianne', 'Shiela',
        'Shiloh', 'Shira', 'Shirl', 'Shirlee', 'Shirleen', 'Shirlene',
        'Shirley', 'Shirleyann', 'Shirlie', 'Shona', 'Shonda', 'Shonna',
        'Shreya', 'Shyann', 'Shyanne', 'Shyla', 'Sibbie', 'Sibyl', 'Siddie',
        'Sidney', 'Siena', 'Sienna', 'Sierra', 'Signa', 'Signe', 'Sigrid',
        'Silvia', 'Simona', 'Simone', 'Sina', 'Sinda', 'Siobhan', 'Sister',
        'Sky', 'Skye', 'Skyla', 'Skylar', 'Skyler', 'Sloane', 'Socorro',
        'Sofia', 'Soledad', 'Somer', 'Sommer', 'Sondra', 'Sonia', 'Sonja',
        'Sonji', 'Sonya', 'Sophia', 'Sophie', 'Sophronia', 'Spring', 'Stacey',
        'Staci', 'Stacia', 'Stacie', 'Stacy', 'Star', 'Starla', 'Starr',
        'Stasia', 'Stefani', 'Stefanie', 'Stella', 'Stephaine', 'Stephani',
        'Stephania', 'Stephanie', 'Stephany', 'Stephenie', 'Stevie', 'Stormy',
        'Sudie', 'Sue', 'Suellen', 'Sula', 'Summer', 'Sunday', 'Sunny',
        'Sunshine', 'Susan', 'Susana', 'Susann', 'Susanna', 'Susannah',
        'Susanne', 'Susie', 'Sussie', 'Suzan', 'Suzann', 'Suzanna', 'Suzanne',
        'Suzette', 'Suzie', 'Suzy', 'Sybil', 'Sybilla', 'Syble', 'Sydell',
        'Sydnee', 'Sydney', 'Sydni', 'Sydnie', 'Sylva', 'Sylvania', 'Sylvia',
        'Symone', 'Syreeta', 'Tabatha', 'Tabetha', 'Tabitha', 'Tai', 'Taina',
        'Taja', 'Takisha', 'Talia', 'Taliyah', 'Tamala', 'Tamara', 'Tamatha',
        'Tambra', 'Tameka', 'Tamekia', 'Tamela', 'Tamera', 'Tami', 'Tamia',
        'Tamica', 'Tamie', 'Tamika', 'Tamiko', 'Tamisha', 'Tammi', 'Tammie',
        'Tammy', 'Tamra', 'Tamya', 'Tana', 'Tanesha', 'Tangela', 'Tania',
        'Tanika', 'Tanisha', 'Taniya', 'Taniyah', 'Tanja', 'Tanya', 'Tara',
        'Tarah', 'Taraji', 'Tari', 'Tarsha', 'Taryn', 'Tasha', 'Tashina',
        'Tasia', 'Tatia', 'Tatiana', 'Tatianna', 'Tatum', 'Tatyana', 'Tatyanna',
        'Tawana', 'Tawanda', 'Tawanna', 'Tawny', 'Tawnya', 'Taya', 'Tayla',
        'Tayler', 'Taylor', 'Tea', 'Teagan', 'Teela', 'Teena', 'Tella',
        'Tempie', 'Tena', 'Tenika', 'Tenisha', 'Tennessee', 'Tennie',
        'Tennille', 'Tera', 'Teresa', 'Terese', 'Teressa', 'Teri', 'Terra',
        'Terri', 'Terrie', 'Terry', 'Tess', 'Tessa', 'Tessie', 'Texanna',
        'Texas', 'Texie', 'Thalia', 'Thea', 'Theda', 'Thekla', 'Thelma',
        'Theodocia', 'Theodora', 'Theodosia', 'Theola', 'Theresa', 'Therese',
        'Theresia', 'Theta', 'Thomasina', 'Thora', 'Thresa', 'Thursa', 'Thyra',
        'Tia', 'Tiana', 'Tianna', 'Tiara', 'Tiarra', 'Tiera', 'Tierra',
        'Tiesha', 'Tiffani', 'Tiffanie', 'Tiffany', 'Tilda', 'Tilla', 'Tillie',
        'Tina', 'Tiney', 'Tinie', 'Tinnie', 'Tiny', 'Tisa', 'Tisha', 'Tishie',
        'Tobi', 'Toby', 'Toccara', 'Tomasa', 'Tomeka', 'Tomika', 'Tommie',
        'Tonda', 'Toni', 'Tonia', 'Tonja', 'Tonya', 'Tori', 'Torie', 'Torrie',
        'Tory', 'Tosha', 'Toshiko', 'Towanda', 'Toya', 'Tracee', 'Tracey',
        'Traci', 'Tracie', 'Tracy', 'Treasure', 'Treena', 'Trena', 'Tresa',
        'Tressa', 'Tressie', 'Treva', 'Tricia', 'Trilby', 'Trina', 'Trinidad',
        'Trinity', 'Trish', 'Trisha', 'Trista', 'Tristan', 'Tristen', 'Trudi',
        'Trudie', 'Trudy', 'Trula', 'Tula', 'Twila', 'Twyla', 'Tyesha', 'Tyra',
        'Ula', 'Una', 'Unique', 'Unknown', 'Ura', 'Ursula', 'Vada', 'Val',
        'Valarie', 'Valencia', 'Valentina', 'Valentine', 'Valeria', 'Valerie',
        'Valery', 'Valinda', 'Vallie', 'Valorie', 'Vanesa', 'Vanessa', 'Vannie',
        'Vara', 'Vashti', 'Vassie', 'Veda', 'Vela', 'Velda', 'Velia', 'Vella',
        'Velma', 'Velva', 'Velvet', 'Vena', 'Venessa', 'Venice', 'Venie',
        'Venita', 'Vennie', 'Venus', 'Veola', 'Vera', 'Verda', 'Verdell',
        'Verdie', 'Verena', 'Vergie', 'Verla', 'Verlene', 'Verlie', 'Verna',
        'Verne', 'Vernell', 'Vernelle', 'Vernetta', 'Vernia', 'Vernice',
        'Vernie', 'Vernita', 'Verona', 'Veronica', 'Versa', 'Versie', 'Vertie',
        'Vessie', 'Vesta', 'Veta', 'Veva', 'Vicie', 'Vickey', 'Vicki', 'Vickie',
        'Vicky', 'Victoria', 'Victorine', 'Victory', 'Vicy', 'Vida', 'Vikki',
        'Villa', 'Vilma', 'Vina', 'Vincenza', 'Viney', 'Vinie', 'Vinnie',
        'Viola', 'Violet', 'Violeta', 'Violetta', 'Violette', 'Vira', 'Virdie',
        'Virgia', 'Virgie', 'Virginia', 'Viridiana', 'Vita', 'Viva', 'Vivian',
        'Viviana', 'Vivien', 'Vivienne', 'Vlasta', 'Vonda', 'Vonetta', 'Vonnie',
        'Wanda', 'Waneta', 'Wanita', 'Wava', 'Wende', 'Wendi', 'Wendy',
        'Whitley', 'Whitney', 'Wilda', 'Wilhelmina', 'Wilhelmine', 'Willa',
        'Willene', 'Willia', 'Willie', 'Williemae', 'Willodean', 'Willow',
        'Wilma', 'Windy', 'Winifred', 'Winnie', 'Winnifred', 'Winona', 'Winter',
        'Wynona', 'Xena', 'Ximena', 'Xiomara', 'Yadira', 'Yahaira', 'Yajaira',
        'Yamilet', 'Yamilex', 'Yareli', 'Yaretzi', 'Yaritza', 'Yasmeen',
        'Yasmin', 'Yasmine', 'Yazmin', 'Yesenia', 'Yessenia', 'Yetta',
        'Yolanda', 'Yolonda', 'Yoselin', 'Yoshiko', 'Yuliana', 'Yulisa',
        'Yulissa', 'Yuridia', 'Yvette', 'Yvonne', 'Zada', 'Zadie', 'Zaida',
        'Zana', 'Zandra', 'Zaniyah', 'Zara', 'Zaria', 'Zariah', 'Zela', 'Zelda',
        'Zelia', 'Zella', 'Zelma', 'Zelpha', 'Zena', 'Zenobia', 'Zeta', 'Zetta',
        'Zettie', 'Zhane', 'Zillah', 'Zilpah', 'Zilpha', 'Zina', 'Zion', 'Zita',
        'Zoa', 'Zoe', 'Zoey', 'Zoie', 'Zola', 'Zona', 'Zora', 'Zula',
        'Aaden', 'Aarav', 'Aaron', 'Ab', 'Abb', 'Abbott', 'Abdiel', 'Abdul',
        'Abdullah', 'Abe', 'Abel', 'Abelardo', 'Abie', 'Abner', 'Abraham',
        'Abram', 'Ace', 'Acey', 'Acie', 'Acy', 'Adalberto', 'Adam', 'Adams',
        'Adan', 'Add', 'Adelard', 'Adelbert', 'Aden', 'Adin', 'Aditya', 'Adlai',
        'Admiral', 'Adolf', 'Adolfo', 'Adolph', 'Adolphus', 'Adonis', 'Adrain',
        'Adrian', 'Adriel', 'Adrien', 'Adron', 'Aedan', 'Agustin', 'Agustus',
        'Ah', 'Ahmad', 'Ahmed', 'Aidan', 'Aiden', 'Aidyn', 'Aime', 'Akeem',
        'Al', 'Alan', 'Alanzo', 'Albert', 'Alberto', 'Albertus', 'Albin',
        'Albion', 'Alby', 'Alcee', 'Alcide', 'Alden', 'Aldo', 'Alec', 'Aleck',
        'Alejandro', 'Alek', 'Alessandro', 'Alex', 'Alexande', 'Alexander',
        'Alexandre', 'Alexandro', 'Alexis', 'Alexzander', 'Alf', 'Alferd',
        'Alfie', 'Alfonse', 'Alfonso', 'Alfonzo', 'Alford', 'Alfred', 'Alfredo',
        'Alger', 'Algernon', 'Algie', 'Algot', 'Ali', 'Alijah', 'Allan',
        'Allen', 'Allyn', 'Almer', 'Almon', 'Almond', 'Almus', 'Alois',
        'Alonso', 'Alonza', 'Alonzo', 'Aloys', 'Aloysius', 'Alpheus', 'Alphons',
        'Alphonse', 'Alphonso', 'Alphonsus', 'Alston', 'Alto', 'Alton', 'Alva',
        'Alvah', 'Alvan', 'Alvaro', 'Alver', 'Alvia', 'Alvie', 'Alvin', 'Alvis',
        'Alvy', 'Alwin', 'Amado', 'Amare', 'Amari', 'Amarion', 'Amasa',
        'Ambers', 'Ambrose', 'Americo', 'Amerigo', 'Amil', 'Amin', 'Amir',
        'Amit', 'Ammon', 'Amon', 'Amos', 'Ananias', 'Anastacio', 'Anatole',
        'Ancel', 'Ancil', 'Anders', 'Anderson', 'Andon', 'Andra', 'Andrae',
        'Andre', 'Andreas', 'Andres', 'Andrew', 'Andy', 'Anfernee', 'Angel',
        'Angelo', 'Angus', 'Anibal', 'Ansel', 'Anson', 'Anthoney', 'Anthony',
        'Antione', 'Antoine', 'Anton', 'Antone', 'Antonio', 'Antony', 'Antwain',
        'Antwan', 'Antwon', 'Anwar', 'Arba', 'Arbie', 'Arch', 'Archer',
        'Archibald', 'Archie', 'Ardell', 'Arden', 'Ari', 'Aric', 'Arjun',
        'Arlan', 'Arland', 'Arlen', 'Arley', 'Arlie', 'Arlin', 'Arlington',
        'Arlis', 'Arlo', 'Arlyn', 'Arman', 'Armand', 'Armando', 'Armani',
        'Armin', 'Armond', 'Armstead', 'Arnav', 'Arne', 'Arnett', 'Arnie',
        'Arno', 'Arnold', 'Arnoldo', 'Arnulfo', 'Aron', 'Arron', 'Arsenio',
        'Art', 'Arther', 'Arthor', 'Arthur', 'Artie', 'Artis', 'Arturo',
        'Arvel', 'Arvid', 'Arvil', 'Arvin', 'Arvo', 'Aryan', 'Asa', 'Asberry',
        'Asbury', 'Ashby', 'Asher', 'Ashton', 'Atha', 'Atlas', 'Atticus',
        'Attilio', 'Aubra', 'Aubrey', 'Audie', 'Audley', 'Audy', 'August',
        'Auguste', 'Augustin', 'Augustine', 'Augustus', 'Aurelio', 'Aurthur',
        'Austen', 'Austin', 'Auston', 'Austyn', 'Auther', 'Author', 'Authur',
        'Autry', 'Avery', 'Avon', 'Axel', 'Ayaan', 'Aydan', 'Ayden', 'Aydin',
        'Babe', 'Babyboy', 'Bailey', 'Baker', 'Baldwin', 'Ballard', 'Banks',
        'Barnard', 'Barnett', 'Barney', 'Barnie', 'Baron', 'Barrett', 'Barrie',
        'Barron', 'Barry', 'Bart', 'Bartholomew', 'Bartley', 'Barton', 'Bascom',
        'Basil', 'Baxter', 'Bayard', 'Beau', 'Beckett', 'Beckham', 'Bedford',
        'Beecher', 'Bell', 'Belton', 'Ben', 'Benard', 'Benedict', 'Benito',
        'Benjaman', 'Benjamen', 'Benjamin', 'Benjamine', 'Benji', 'Benjiman',
        'Benjman', 'Bennett', 'Bennie', 'Benny', 'Benson', 'Bentley', 'Benton',
        'Berkley', 'Berlin', 'Bernard', 'Bernardo', 'Bernhard', 'Bernie',
        'Berry', 'Bert', 'Bertie', 'Berton', 'Bertram', 'Bertrand', 'Beryl',
        'Bethel', 'Bilal', 'Bill', 'Billie', 'Billy', 'Bird', 'Birt', 'Bishop',
        'Bjorn', 'Blain', 'Blaine', 'Blair', 'Blaise', 'Blake', 'Blanchard',
        'Blane', 'Blas', 'Blaze', 'Bliss', 'Bluford', 'Bo', 'Bob', 'Bobbie',
        'Bobby', 'Bode', 'Bolden', 'Booker', 'Boone', 'Boris', 'Bose', 'Boss',
        'Boston', 'Bowman', 'Boyce', 'Boyd', 'Boysie', 'Brad', 'Braden',
        'Bradford', 'Bradley', 'Bradly', 'Brady', 'Bradyn', 'Braeden',
        'Braedon', 'Braiden', 'Brain', 'Branch', 'Brandan', 'Branden',
        'Brandin', 'Brandon', 'Brandt', 'Brandy', 'Brandyn', 'Brannon',
        'Branson', 'Brant', 'Brantley', 'Braulio', 'Braxton', 'Brayan',
        'Brayden', 'Braydon', 'Braylen', 'Braylon', 'Brendan', 'Brenden',
        'Brendon', 'Brennan', 'Brennen', 'Brennon', 'Brent', 'Brenton', 'Bret',
        'Brett', 'Brian', 'Brice', 'Bridger', 'Brien', 'Brion', 'Britt',
        'Brittany', 'Britton', 'Brock', 'Broderick', 'Brodie', 'Brody',
        'Brogan', 'Bronson', 'Brook', 'Brooks', 'Brown', 'Bruce', 'Bruno',
        'Bryan', 'Bryant', 'Bryce', 'Brycen', 'Bryon', 'Bryson', 'Bryton',
        'Buck', 'Bud', 'Budd', 'Buddie', 'Buddy', 'Buel', 'Buell', 'Buford',
        'Bunk', 'Burdette', 'Buren', 'Burgess', 'Burk', 'Burke', 'Burl',
        'Burleigh', 'Burley', 'Burnell', 'Burnett', 'Burney', 'Burnice',
        'Burnie', 'Burns', 'Burr', 'Burrel', 'Burrell', 'Burt', 'Burton',
        'Bush', 'Buster', 'Butch', 'Butler', 'Bynum', 'Byrd', 'Byron', 'Cade',
        'Caden', 'Cael', 'Caesar', 'Caiden', 'Cain', 'Cal', 'Cale', 'Caleb',
        'Calhoun', 'Callie', 'Callum', 'Calvin', 'Cam', 'Camden', 'Cameron',
        'Camilo', 'Campbell', 'Camren', 'Camron', 'Camryn', 'Candido', 'Cannon',
        'Canyon', 'Cap', 'Captain', 'Carey', 'Carl', 'Carleton', 'Carlie',
        'Carlisle', 'Carlo', 'Carlos', 'Carlton', 'Carlyle', 'Carmel',
        'Carmelo', 'Carmen', 'Carmine', 'Carnell', 'Carrie', 'Carrol',
        'Carroll', 'Carsen', 'Carson', 'Carter', 'Cary', 'Cas', 'Case', 'Casen',
        'Casey', 'Cash', 'Casimer', 'Casimir', 'Casimiro', 'Cason', 'Casper',
        'Cass', 'Cassidy', 'Cassie', 'Cassius', 'Caswell', 'Cato', 'Cayden',
        'Ceasar', 'Cecil', 'Cedric', 'Cedrick', 'Celestino', 'Cephus', 'Cesar',
        'Ceylon', 'Chace', 'Chad', 'Chadd', 'Chadrick', 'Chadwick', 'Chaim',
        'Chalmer', 'Chalmers', 'Champ', 'Chance', 'Chancey', 'Chancy',
        'Chandler', 'Channing', 'Charle', 'Charles', 'Charley', 'Charlie',
        'Charls', 'Charlton', 'Charly', 'Chas', 'Chase', 'Chauncey', 'Chauncy',
        'Chaz', 'Che', 'Chesley', 'Chester', 'Chet', 'Cheyenne', 'Chin', 'Chip',
        'Chris', 'Christ', 'Christian', 'Christina', 'Christion', 'Christop',
        'Christoper', 'Christophe', 'Christopher', 'Chuck', 'Cicero', 'Clabe',
        'Claiborne', 'Clair', 'Clarance', 'Clare', 'Clarence', 'Clark',
        'Clarke', 'Clarnce', 'Claud', 'Claude', 'Claudie', 'Claudio',
        'Claudius', 'Claus', 'Clay', 'Clayton', 'Clearence', 'Cleave', 'Clell',
        'Clem', 'Clemence', 'Clemens', 'Clement', 'Clemente', 'Clemmie',
        'Clemon', 'Cleo', 'Cleon', 'Cletus', 'Cleve', 'Cleveland', 'Clide',
        'Cliff', 'Clifford', 'Clifton', 'Clint', 'Clinton', 'Clive', 'Clovis',
        'Cloyd', 'Clyde', 'Coby', 'Codey', 'Codi', 'Codie', 'Cody', 'Coen',
        'Cohen', 'Colbert', 'Colby', 'Cole', 'Coleman', 'Coleton', 'Coley',
        'Colie', 'Colin', 'Collie', 'Collier', 'Collin', 'Collins', 'Collis',
        'Colon', 'Colonel', 'Colt', 'Colten', 'Colter', 'Colton', 'Columbus',
        'Colvin', 'Commodore', 'Con', 'Conard', 'Conley', 'Conner', 'Connie',
        'Connor', 'Conor', 'Conrad', 'Constantine', 'Conway', 'Coolidge',
        'Cooper', 'Corbett', 'Corbin', 'Cordaro', 'Cordell', 'Cordero', 'Corey',
        'Cornel', 'Cornelious', 'Cornelius', 'Cornell', 'Corry', 'Cortez',
        'Cortney', 'Corwin', 'Cory', 'Cosmo', 'Coty', 'Council', 'Courtland',
        'Courtney', 'Coy', 'Craig', 'Crawford', 'Creed', 'Cris', 'Cristian',
        'Cristobal', 'Cristofer', 'Cristopher', 'Crockett', 'Cruz', 'Cullen',
        'Curley', 'Curt', 'Curtis', 'Curtiss', 'Cyril', 'Cyrus', 'Dabney',
        'Dakoda', 'Dakota', 'Dakotah', 'Dale', 'Dallas', 'Dallin', 'Dalton',
        'Dalvin', 'Damarcus', 'Damari', 'Damarion', 'Dameon', 'Damian',
        'Damien', 'Damion', 'Damon', 'Damond', 'Dan', 'Dana', 'Dandre', 'Dane',
        'Dangelo', 'Danial', 'Daniel', 'Dann', 'Dannie', 'Danniel', 'Danny',
        'Dante', 'Daquan', 'Darby', 'Darcy', 'Darell', 'Daren', 'Darian',
        'Darien', 'Darin', 'Dario', 'Darion', 'Darius', 'Darl', 'Darnell',
        'Darold', 'Daron', 'Darrel', 'Darrell', 'Darren', 'Darrian', 'Darrick',
        'Darrien', 'Darrin', 'Darrion', 'Darrius', 'Darron', 'Darry', 'Darryl',
        'Darryle', 'Darryll', 'Darryn', 'Darvin', 'Darwin', 'Darwyn', 'Daryl',
        'Daryle', 'Daryn', 'Dashawn', 'Daulton', 'Daunte', 'Davante', 'Dave',
        'Davey', 'Davian', 'David', 'Davie', 'Davin', 'Davion', 'Davis',
        'Davon', 'Davonta', 'Davonte', 'Davy', 'Dawson', 'Dax', 'Daxton',
        'Dayne', 'Dayton', 'Deacon', 'Dean', 'Deandre', 'Deane', 'Deangelo',
        'Deante', 'Declan', 'Dedric', 'Dedrick', 'Deegan', 'Deforest', 'Deion',
        'Dejon', 'Dejuan', 'Del', 'Delano', 'Delbert', 'Dell', 'Della', 'Delma',
        'Delmar', 'Delmas', 'Delmer', 'Delmus', 'Delos', 'Delphin', 'Delton',
        'Delvin', 'Delwin', 'Demarco', 'Demarcus', 'Demario', 'Demarion',
        'Demetri', 'Demetric', 'Demetrios', 'Demetrius', 'Demian', 'Demond',
        'Demonte', 'Dempsey', 'Denis', 'Dennie', 'Dennis', 'Denny', 'Denton',
        'Denver', 'Denzel', 'Denzell', 'Denzil', 'Deon', 'Deondre', 'Deonta',
        'Deontae', 'Deonte', 'Dequan', 'Derald', 'Dereck', 'Derek', 'Dereon',
        'Deric', 'Derick', 'Derik', 'Derl', 'Deron', 'Derrek', 'Derrell',
        'Derrick', 'Derwin', 'Deryl', 'Desean', 'Deshaun', 'Deshawn', 'Desi',
        'Desmond', 'Dessie', 'Destin', 'Destry', 'Devan', 'Devante', 'Devaughn',
        'Deven', 'Devin', 'Devon', 'Devonta', 'Devontae', 'Devonte', 'Devyn',
        'Deward', 'Dewayne', 'Dewey', 'Dewitt', 'Dexter', 'Diallo', 'Diamond',
        'Diane', 'Dickie', 'Diego', 'Dijon', 'Dilan', 'Dillan', 'Dillard',
        'Dillion', 'Dillon', 'Dimitri', 'Dimitrios', 'Dink', 'Dino', 'Dion',
        'Dionicio', 'Dionte', 'Dirk', 'Dixon', 'Doc', 'Dock', 'Doctor', 'Doll',
        'Dolph', 'Dolphus', 'Domenic', 'Domenick', 'Domenico', 'Domingo',
        'Dominic', 'Dominick', 'Dominik', 'Don', 'Donaciano', 'Donal', 'Donald',
        'Donat', 'Donato', 'Donavan', 'Donavon', 'Dondre', 'Donell', 'Donn',
        'Donnell', 'Donnie', 'Donny', 'Donovan', 'Donta', 'Dontae', 'Donte',
        'Dora', 'Dorian', 'Dorman', 'Dorr', 'Dorris', 'Dorsey', 'Doss', 'Doug',
        'Douglas', 'Douglass', 'Dow', 'Doyle', 'Dozier', 'Drake', 'Draven',
        'Drew', 'Drury', 'Duane', 'Duard', 'Dudley', 'Duff', 'Duke', 'Duncan',
        'Durell', 'Durrell', 'Durward', 'Durwood', 'Dustan', 'Dustin', 'Dusty',
        'Dustyn', 'Duwayne', 'Dwain', 'Dwaine', 'Dwane', 'Dwayne', 'Dwight',
        'Dwyane', 'Dylan', 'Dyllan', 'Dylon', 'Ean', 'Earl', 'Earle', 'Earley',
        'Earlie', 'Early', 'Earnest', 'Easton', 'Ebb', 'Ebbie', 'Eben',
        'Ebenezer', 'Eber', 'Ebert', 'Ed', 'Edd', 'Eddie', 'Eddy', 'Eden',
        'Edgar', 'Edgardo', 'Edie', 'Edison', 'Edmon', 'Edmond', 'Edmund',
        'Edsel', 'Edson', 'Eduardo', 'Edw', 'Edward', 'Edwardo', 'Edwin',
        'Effie', 'Efrain', 'Efrem', 'Efren', 'Egbert', 'Einar', 'Eino', 'Elam',
        'Elbert', 'Elbridge', 'Elby', 'Elden', 'Elder', 'Eldon', 'Eldred',
        'Eldridge', 'Elex', 'Elgie', 'Elgin', 'Eli', 'Elian', 'Elias', 'Elick',
        'Elie', 'Eliezer', 'Eliga', 'Eligah', 'Elige', 'Elihu', 'Elijah',
        'Eliot', 'Eliseo', 'Elisha', 'Elizah', 'Ell', 'Ellery', 'Elliot',
        'Elliott', 'Ellis', 'Ellison', 'Ellsworth', 'Ellwood', 'Elmer', 'Elmo',
        'Elmore', 'Elon', 'Elonzo', 'Eloy', 'Elroy', 'Elsworth', 'Elton',
        'Elvin', 'Elvis', 'Elwin', 'Elwood', 'Elwyn', 'Ely', 'Elza', 'Elzie',
        'Elzy', 'Emanuel', 'Emerson', 'Emery', 'Emett', 'Emil', 'Emile',
        'Emiliano', 'Emilio', 'Emit', 'Emma', 'Emmanuel', 'Emmet', 'Emmett',
        'Emmit', 'Emmitt', 'Emmons', 'Emory', 'Emry', 'Encarnacion', 'Ennis',
        'Enoch', 'Enos', 'Enrico', 'Enrique', 'Enzo', 'Ephraim', 'Ephram',
        'Ephriam', 'Epifanio', 'Erasmo', 'Erasmus', 'Erastus', 'Erby', 'Eric',
        'Erich', 'Erick', 'Erie', 'Erik', 'Erin', 'Erland', 'Erle', 'Erling',
        'Ernest', 'Ernesto', 'Ernie', 'Ernst', 'Errol', 'Ervin', 'Erving',
        'Erwin', 'Esau', 'Esco', 'Esequiel', 'Esker', 'Esley', 'Essex',
        'Esteban', 'Estel', 'Estes', 'Estevan', 'Estill', 'Eston', 'Ethan',
        'Ethelbert', 'Ethen', 'Eugene', 'Eugenio', 'Eusebio', 'Eustace', 'Evan',
        'Evander', 'Evans', 'Evelyn', 'Everet', 'Everett', 'Everette', 'Evert',
        'Evertt', 'Ewald', 'Ewart', 'Ewell', 'Ewin', 'Ewing', 'Ezekiel',
        'Ezell', 'Ezequiel', 'Ezra', 'Ezzard', 'Fabian', 'Faron', 'Farrell',
        'Farris', 'Fate', 'Faustino', 'Fayette', 'Fed', 'Federico', 'Felipe',
        'Felix', 'Felton', 'Fenton', 'Ferd', 'Ferdinand', 'Ferman', 'Fernand',
        'Fernando', 'Ferrell', 'Ferris', 'Festus', 'Fidel', 'Fidencio',
        'Fielding', 'Finis', 'Finley', 'Finn', 'Finnegan', 'Firman', 'Fisher',
        'Fitzgerald', 'Fitzhugh', 'Fleet', 'Flem', 'Fleming', 'Fletcher',
        'Flint', 'Florencio', 'Florentino', 'Florian', 'Floy', 'Floyd', 'Foch',
        'Ford', 'Forest', 'Forrest', 'Foster', 'Fount', 'Foy', 'Frances',
        'Francesco', 'Francis', 'Francisco', 'Franco', 'Frank', 'Frankie',
        'Franklin', 'Franklyn', 'Franz', 'Frazier', 'Fred', 'Freddie', 'Freddy',
        'Frederic', 'Frederick', 'Fredie', 'Fredric', 'Fredrick', 'Fredy',
        'Freeman', 'Fremont', 'French', 'Friend', 'Fritz', 'Fuller', 'Fulton',
        'Furman', 'Gabe', 'Gabriel', 'Gael', 'Gaetano', 'Gage', 'Gaige', 'Gail',
        'Gaines', 'Gaither', 'Gale', 'Galen', 'Gannon', 'Gardner', 'Garett',
        'Garey', 'Garfield', 'Garland', 'Garner', 'Garnet', 'Garnett', 'Garold',
        'Garret', 'Garrett', 'Garrick', 'Garrison', 'Garry', 'Garth', 'Garvin',
        'Gary', 'Gasper', 'Gaston', 'Gauge', 'Gaven', 'Gavin', 'Gavyn', 'Gay',
        'Gayle', 'Gaylen', 'Gaylon', 'Gaylord', 'Gearld', 'Geary', 'Gee',
        'Genaro', 'Gene', 'General', 'Genie', 'Gennaro', 'Geno', 'Geo', 'Geoff',
        'Geoffrey', 'George', 'Georgie', 'Geovanni', 'Gerald', 'Geraldo',
        'Gerard', 'Gerardo', 'Gerhard', 'Gerhardt', 'Germaine', 'German',
        'Gerold', 'Gerrit', 'Gerry', 'Giancarlo', 'Gianni', 'Gibson', 'Gideon',
        'Gifford', 'Gil', 'Gilbert', 'Gilberto', 'Giles', 'Gilford', 'Gilman',
        'Gilmer', 'Gilmore', 'Gino', 'Giovani', 'Giovanni', 'Giovanny',
        'Giuseppe', 'Gladstone', 'Glen', 'Glendon', 'Glenn', 'Glenwood',
        'Glover', 'Glynn', 'Godfrey', 'Goebel', 'Golden', 'Gonzalo', 'Gorden',
        'Gordon', 'Gorge', 'Gottlieb', 'Governor', 'Grady', 'Grafton', 'Graham',
        'Grant', 'Granville', 'Graves', 'Gray', 'Graydon', 'Grayling',
        'Grayson', 'Green', 'Greene', 'Greg', 'Gregg', 'Greggory', 'Gregorio',
        'Gregory', 'Greyson', 'Griffin', 'Griffith', 'Grove', 'Grover', 'Guido',
        'Guilford', 'Guillermo', 'Gunnar', 'Gunner', 'Gurney', 'Gus', 'Guss',
        'Gussie', 'Gust', 'Gustaf', 'Gustav', 'Gustave', 'Gustavo', 'Gustavus',
        'Guthrie', 'Guy', 'Haden', 'Hadley', 'Haiden', 'Hakeem', 'Hakim', 'Hal',
        'Halbert', 'Hale', 'Hall', 'Halley', 'Hallie', 'Halsey', 'Ham',
        'Hamilton', 'Hamp', 'Hampton', 'Hamza', 'Handy', 'Hank', 'Hans',
        'Hansel', 'Hansford', 'Hanson', 'Harden', 'Hardie', 'Hardin', 'Harding',
        'Hardy', 'Harl', 'Harlan', 'Harland', 'Harlen', 'Harley', 'Harlie',
        'Harlon', 'Harlow', 'Harm', 'Harman', 'Harmon', 'Harold', 'Harper',
        'Harrell', 'Harrie', 'Harris', 'Harrison', 'Harrold', 'Harry', 'Hart',
        'Hartley', 'Hartwell', 'Harve', 'Harvey', 'Harvie', 'Harvy', 'Hasan',
        'Haskell', 'Hassan', 'Hattie', 'Haven', 'Hayden', 'Hayes', 'Hays',
        'Hayward', 'Haywood', 'Hazen', 'Heath', 'Heber', 'Hebert', 'Hector',
        'Helmer', 'Hence', 'Henderson', 'Henery', 'Henri', 'Henry', 'Herb',
        'Herbert', 'Heriberto', 'Herman', 'Hermann', 'Hermon', 'Hernan',
        'Herschel', 'Hershel', 'Hershell', 'Hervey', 'Heyward', 'Hezekiah',
        'Hezzie', 'Hideo', 'Hilario', 'Hilary', 'Hilbert', 'Hill', 'Hillard',
        'Hillary', 'Hillery', 'Hilliard', 'Hilmer', 'Hilton', 'Hiram',
        'Hiroshi', 'Hjalmar', 'Hjalmer', 'Hobart', 'Hobert', 'Hobson', 'Hoke',
        'Holden', 'Holland', 'Hollie', 'Hollis', 'Holmes', 'Homer', 'Hoover',
        'Hope', 'Horace', 'Horacio', 'Horatio', 'Horton', 'Hosea', 'Hosie',
        'Hosteen', 'Houston', 'Howard', 'Howell', 'Hoy', 'Hoyt', 'Hubbard',
        'Hubert', 'Hudson', 'Huey', 'Hugh', 'Hughes', 'Hughey', 'Hughie',
        'Hugo', 'Humberto', 'Humphrey', 'Hung', 'Hunt', 'Hunter', 'Hurbert',
        'Hurley', 'Huston', 'Huy', 'Hyman', 'Hymen', 'Hyrum', 'Ian', 'Ibrahim',
        'Ida', 'Ignacio', 'Ignatius', 'Ignatz', 'Ike', 'Illya', 'Imanol',
        'Immanuel', 'Infant', 'Ingram', 'Ira', 'Irl', 'Irven', 'Irvin',
        'Irvine', 'Irving', 'Irwin', 'Isaac', 'Isaak', 'Isadore', 'Isai',
        'Isaiah', 'Isaias', 'Isam', 'Ishaan', 'Isham', 'Ishmael', 'Isiah',
        'Isidor', 'Isidore', 'Isidro', 'Ismael', 'Isom', 'Israel', 'Isreal',
        'Issac', 'Iva', 'Ivan', 'Iver', 'Iverson', 'Ivey', 'Ivor', 'Ivory',
        'Ivy', 'Izaiah', 'Izayah', 'Jabari', 'Jabbar', 'Jabez', 'Jace', 'Jack',
        'Jackson', 'Jacky', 'Jacob', 'Jacoby', 'Jacques', 'Jacquez', 'Jade',
        'Jaden', 'Jadiel', 'Jadon', 'Jadyn', 'Jaeden', 'Jagger', 'Jaheem',
        'Jaheim', 'Jahiem', 'Jahir', 'Jaiden', 'Jaidyn', 'Jaime', 'Jaimie',
        'Jair', 'Jairo', 'Jajuan', 'Jake', 'Jakob', 'Jakobe', 'Jaleel', 'Jalen',
        'Jalon', 'Jamaal', 'Jamal', 'Jamar', 'Jamarcus', 'Jamari', 'Jamarion',
        'Jame', 'Jameel', 'Jamel', 'James', 'Jameson', 'Jamey', 'Jamie',
        'Jamil', 'Jamin', 'Jamir', 'Jamison', 'Jammie', 'Jan', 'Jaquan',
        'Jaquez', 'Jarad', 'Jared', 'Jaren', 'Jaret', 'Jarett', 'Jarod',
        'Jaron', 'Jarrad', 'Jarred', 'Jarrell', 'Jarret', 'Jarrett', 'Jarrod',
        'Jarvis', 'Jase', 'Jasen', 'Jasiah', 'Jason', 'Jasper', 'Javen',
        'Javier', 'Javion', 'Javon', 'Javonte', 'Jax', 'Jaxen', 'Jaxon',
        'Jaxson', 'Jaxton', 'Jay', 'Jayce', 'Jaycob', 'Jaydan', 'Jayden',
        'Jaydin', 'Jaydon', 'Jaylan', 'Jaylen', 'Jaylin', 'Jaylon', 'Jayme',
        'Jaymes', 'Jayson', 'Jayvion', 'Jayvon', 'Jean', 'Jeb', 'Jed',
        'Jedediah', 'Jedidiah', 'Jeff', 'Jefferey', 'Jefferson', 'Jeffery',
        'Jeffie', 'Jeffrey', 'Jeffry', 'Jelani', 'Jemal', 'Jennings', 'Jens',
        'Jensen', 'Jep', 'Jeptha', 'Jerad', 'Jerald', 'Jeramiah', 'Jeramie',
        'Jeramy', 'Jere', 'Jered', 'Jerel', 'Jereme', 'Jeremey', 'Jeremiah',
        'Jeremie', 'Jeremy', 'Jerimiah', 'Jerimy', 'Jermain', 'Jermaine',
        'Jermey', 'Jerod', 'Jerold', 'Jerome', 'Jeromy', 'Jerrad', 'Jerrel',
        'Jerrell', 'Jerrod', 'Jerrold', 'Jerry', 'Jess', 'Jesse', 'Jessee',
        'Jessie', 'Jessy', 'Jesus', 'Jethro', 'Jett', 'Jettie', 'Jevon',
        'Jewell', 'Jiles', 'Jim', 'Jimmie', 'Jimmy', 'Joaquin', 'Job', 'Jobe',
        'Joe', 'Joel', 'Joeseph', 'Joesph', 'Joey', 'Johan', 'Johathan', 'John',
        'Johnathan', 'Johnathon', 'Johney', 'Johnie', 'Johnnie', 'Johnny',
        'Johnpaul', 'Johnson', 'Johny', 'Jon', 'Jonah', 'Jonas', 'Jonatan',
        'Jonathan', 'Jonathon', 'Jones', 'Jonnie', 'Jordan', 'Jorden', 'Jordi',
        'Jordon', 'Jordy', 'Jordyn', 'Jorge', 'Jory', 'Jose', 'Josef',
        'Joseluis', 'Joseph', 'Josephus', 'Josh', 'Joshua', 'Joshuah', 'Josiah',
        'Josue', 'Jovan', 'Jovani', 'Jovanni', 'Jovanny', 'Jovany', 'Joy',
        'Juan', 'Judah', 'Judd', 'Jude', 'Judge', 'Judson', 'Juelz', 'Jule',
        'Jules', 'Julian', 'Julien', 'Julio', 'Julious', 'Julius', 'Juluis',
        'Junior', 'Junious', 'Junius', 'Justen', 'Justice', 'Justin', 'Juston',
        'Justus', 'Justyn', 'Juwan', 'Kade', 'Kadeem', 'Kaden', 'Kadin',
        'Kadyn', 'Kaeden', 'Kael', 'Kahlil', 'Kai', 'Kaiden', 'Kale', 'Kaleb',
        'Kalel', 'Kalen', 'Kalvin', 'Kamari', 'Kamden', 'Kameron', 'Kamren',
        'Kamron', 'Kamryn', 'Kane', 'Kanye', 'Kareem', 'Kareen', 'Karim',
        'Karl', 'Karson', 'Karter', 'Kasen', 'Kasey', 'Kash', 'Kason', 'Kavon',
        'Kayden', 'Kaye', 'Kayson', 'Kazuo', 'Keagan', 'Keandre', 'Keanu',
        'Keaton', 'Keegan', 'Keenan', 'Keenen', 'Kegan', 'Keifer', 'Keion',
        'Keith', 'Kelan', 'Kelby', 'Kellan', 'Kellen', 'Kelley', 'Kelly',
        'Kelsey', 'Kelton', 'Kelvin', 'Kem', 'Ken', 'Kenan', 'Kendal',
        'Kendall', 'Kendell', 'Kendrick', 'Kenji', 'Kennard', 'Kennedy',
        'Kenneth', 'Kenney', 'Kennith', 'Kennth', 'Kenny', 'Kent', 'Kenton',
        'Kenya', 'Kenyatta', 'Kenyon', 'Keon', 'Kermit', 'Kerry', 'Kerwin',
        'Keshaun', 'Keshawn', 'Kevan', 'Keven', 'Kevin', 'Kevon', 'Keyon',
        'Keyshawn', 'Khalid', 'Khalil', 'Khari', 'Khiry', 'Kian', 'Kiara',
        'Kiefer', 'Kiel', 'Kieran', 'Kieth', 'Kiley', 'Killian', 'Kim',
        'Kimball', 'Kimberly', 'King', 'Kingston', 'Kinte', 'Kip', 'Kipp',
        'Kirby', 'Kirk', 'Kirt', 'Kit', 'Kiyoshi', 'Knox', 'Knute', 'Kobe',
        'Koby', 'Koda', 'Kody', 'Koen', 'Kolby', 'Kole', 'Kolten', 'Kolton',
        'Konner', 'Konnor', 'Korbin', 'Kordell', 'Korey', 'Kory', 'Kraig',
        'Kris', 'Krish', 'Kristen', 'Kristian', 'Kristin', 'Kristofer',
        'Kristoffer', 'Kristopher', 'Kunta', 'Kurt', 'Kurtis', 'Kwame', 'Kyan',
        'Kylan', 'Kyle', 'Kyler', 'Kymani', 'Kyree', 'Kyson', 'Lacey', 'Lacy',
        'Ladarius', 'Laddie', 'Lafayette', 'Lafe', 'Lamar', 'Lamarcus',
        'Lambert', 'Lamont', 'Lamonte', 'Lance', 'Landan', 'Landen', 'Landin',
        'Landon', 'Landyn', 'Lane', 'Lannie', 'Lanny', 'Laquan', 'Lark',
        'Larkin', 'Laron', 'Larry', 'Lars', 'Larue', 'Lary', 'Lashawn',
        'Latrell', 'Laurance', 'Laurel', 'Laurence', 'Lavar', 'Lavern',
        'Laverne', 'Lavon', 'Lawerence', 'Lawrance', 'Lawrence', 'Lawson',
        'Lawton', 'Lawyer', 'Layne', 'Layton', 'Lazaro', 'Le', 'Lea', 'Leamon',
        'Leander', 'Leandro', 'Lee', 'Leeroy', 'Leif', 'Leigh', 'Leighton',
        'Leland', 'Lem', 'Lemmie', 'Lemon', 'Lemuel', 'Len', 'Lena', 'Lenard',
        'Lennie', 'Lennon', 'Lenny', 'Lenon', 'Lenord', 'Lenwood', 'Leo',
        'Leon', 'Leonard', 'Leonardo', 'Leonce', 'Leonel', 'Leonidas',
        'Leopold', 'Leopoldo', 'Leroy', 'Les', 'Lesley', 'Leslie', 'Less',
        'Lessie', 'Lester', 'Levar', 'Levern', 'Levi', 'Levie', 'Levin',
        'Levon', 'Levy', 'Lew', 'Lewis', 'Lex', 'Lexie', 'Liam', 'Lige',
        'Lilburn', 'Lillard', 'Lim', 'Lincoln', 'Lindbergh', 'Lindell',
        'Linden', 'Lindsay', 'Lindsey', 'Lindy', 'Link', 'Linn', 'Linnie',
        'Linton', 'Linus', 'Linwood', 'Linzy', 'Lionel', 'Lisandro', 'Lish',
        'Lisle', 'Liston', 'Little', 'Littleton', 'Llewellyn', 'Lloyd', 'Logan',
        'Lon', 'London', 'Lone', 'Loney', 'Long', 'Lonie', 'Lonnie', 'Lonny',
        'Lonzo', 'Lora', 'Loran', 'Loren', 'Lorenz', 'Lorenza', 'Lorenzo',
        'Lorin', 'Loring', 'Lorne', 'Lott', 'Lou', 'Louie', 'Louis', 'Love',
        'Lovell', 'Lovett', 'Lovie', 'Lowell', 'Loy', 'Loyal', 'Loyd', 'Luc',
        'Luca', 'Lucas', 'Lucian', 'Luciano', 'Lucien', 'Lucio', 'Lucious',
        'Lucius', 'Lucky', 'Ludwig', 'Lue', 'Luigi', 'Luis', 'Luka', 'Lukas',
        'Luke', 'Lula', 'Lum', 'Lupe', 'Luster', 'Lute', 'Luther', 'Luverne',
        'Lydell', 'Lyle', 'Lyman', 'Lyn', 'Lyndon', 'Lynn', 'Lynwood', 'Lyric',
        'Mac', 'Macarthur', 'Mace', 'Maceo', 'Mack', 'Mackenzie', 'Madden',
        'Maddox', 'Maddux', 'Madison', 'Mae', 'Mahlon', 'Major', 'Makai',
        'Makhi', 'Mal', 'Malachi', 'Malakai', 'Malaki', 'Malcolm', 'Malcom',
        'Male', 'Malik', 'Malvin', 'Mamie', 'Manford', 'Manley', 'Manly',
        'Mannie', 'Manning', 'Mansfield', 'Manson', 'Manuel', 'Marc', 'Marcel',
        'Marcelino', 'Marcell', 'Marcello', 'Marcellus', 'Marcelo', 'Marchello',
        'Marco', 'Marcos', 'Marcus', 'Margarito', 'Mariano', 'Mario', 'Marion',
        'Marius', 'Mark', 'Markel', 'Markell', 'Markus', 'Marland', 'Marley',
        'Marlin', 'Marlo', 'Marlon', 'Marlyn', 'Marques', 'Marquez', 'Marquis',
        'Marquise', 'Marrion', 'Marsh', 'Marshal', 'Marshall', 'Mart',
        'Martell', 'Martez', 'Martin', 'Marty', 'Marvin', 'Masao', 'Mason',
        'Mat', 'Mateo', 'Math', 'Mathew', 'Mathews', 'Mathias', 'Matias',
        'Matt', 'Matteo', 'Matthew', 'Matthias', 'Maurice', 'Mauricio', 'Mauro',
        'Maury', 'Maverick', 'Max', 'Maxie', 'Maxim', 'Maximilian',
        'Maximiliano', 'Maximillian', 'Maximo', 'Maximus', 'Maxwell', 'Maxx',
        'May', 'Maynard', 'Mayo', 'Mcarthur', 'Mckinley', 'Mearl', 'Mekhi',
        'Mel', 'Melbourne', 'Mell', 'Melton', 'Melville', 'Melvin', 'Melvyn',
        'Memphis', 'Menachem', 'Mercer', 'Merl', 'Merle', 'Merlin', 'Merlyn',
        'Merrill', 'Merritt', 'Merton', 'Mervin', 'Mervyn', 'Merwin', 'Messiah',
        'Metro', 'Meyer', 'Micah', 'Michael', 'Michal', 'Michale', 'Micheal',
        'Michel', 'Michial', 'Mickey', 'Micky', 'Miguel', 'Miguelangel',
        'Mikal', 'Mike', 'Mikeal', 'Mikel', 'Mikhail', 'Milan', 'Milas',
        'Milburn', 'Miles', 'Milford', 'Millard', 'Miller', 'Mills', 'Milo',
        'Milton', 'Miner', 'Minor', 'Minoru', 'Misael', 'Mitch', 'Mitchel',
        'Mitchell', 'Moe', 'Mohamed', 'Mohammad', 'Mohammed', 'Moises',
        'Monroe', 'Mont', 'Montana', 'Monte', 'Montel', 'Montgomery', 'Montie',
        'Montrell', 'Monty', 'Moody', 'Mordechai', 'Morgan', 'Morris',
        'Mortimer', 'Morton', 'Mose', 'Moses', 'Moshe', 'Muhammad', 'Murdock',
        'Murl', 'Murphy', 'Murray', 'Murry', 'Mustafa', 'Mychal', 'Myer',
        'Mykel', 'Myles', 'Myrl', 'Myron', 'Myrtle', 'Najee', 'Nakia', 'Namon',
        'Napoleon', 'Nash', 'Nasir', 'Nat', 'Nathan', 'Nathanael', 'Nathanial',
        'Nathaniel', 'Nathen', 'Neal', 'Ned', 'Needham', 'Neely', 'Nehemiah',
        'Neil', 'Nello', 'Nels', 'Nelson', 'Nery', 'Nestor', 'Nevin', 'Newell',
        'Newman', 'Newt', 'Newton', 'Nicholas', 'Nicholaus', 'Nick', 'Nicklaus',
        'Nickolas', 'Nicky', 'Nico', 'Nicolas', 'Nigel', 'Nikhil', 'Nikko',
        'Niko', 'Nikolai', 'Nikolas', 'Nile', 'Niles', 'Nils', 'Nim', 'Noah',
        'Noble', 'Noe', 'Noel', 'Nolan', 'Nolen', 'Norbert', 'Norberto',
        'Norman', 'Normand', 'Norris', 'North', 'Norton', 'Norval', 'Norwood',
        'Nunzio', 'Oakley', 'Obe', 'Obed', 'Obie', 'Ocie', 'Octave', 'Octavio',
        'Octavius', 'Oda', 'Oddie', 'Odell', 'Odie', 'Odin', 'Odis', 'Odus',
        'Offie', 'Ogden', 'Okey', 'Ola', 'Olaf', 'Olan', 'Oland', 'Ole', 'Olen',
        'Oley', 'Olie', 'Olin', 'Oliver', 'Ollie', 'Olof', 'Omar', 'Omari',
        'Omarion', 'Omer', 'Oneal', 'Ora', 'Oral', 'Oran', 'Orange', 'Oren',
        'Orie', 'Orin', 'Orion', 'Oris', 'Orla', 'Orland', 'Orlando', 'Orley',
        'Orlin', 'Orlo', 'Orren', 'Orrie', 'Orrin', 'Orris', 'Orson', 'Orval',
        'Orvel', 'Orvil', 'Orville', 'Orvin', 'Orvis', 'Osbaldo', 'Osborn',
        'Osborne', 'Oscar', 'Osie', 'Ossie', 'Osvaldo', 'Oswald', 'Oswaldo',
        'Otha', 'Othel', 'Otho', 'Otis', 'Ott', 'Ottie', 'Ottis', 'Otto', 'Ova',
        'Ovid', 'Ovila', 'Owen', 'Owens', 'Ozell', 'Ozie', 'Ozzie', 'Pablo',
        'Page', 'Palmer', 'Paris', 'Park', 'Parker', 'Parley', 'Parrish',
        'Pascal', 'Pasquale', 'Pat', 'Pate', 'Patric', 'Patrick', 'Paul',
        'Paulo', 'Paxton', 'Payton', 'Pearley', 'Pedro', 'Percival', 'Percy',
        'Perley', 'Pernell', 'Perry', 'Pershing', 'Pete', 'Peter', 'Peyton',
        'Phil', 'Philip', 'Phillip', 'Philo', 'Phoenix', 'Pierce', 'Pierre',
        'Pinkney', 'Pleas', 'Pleasant', 'Ples', 'Plummer', 'Polk', 'Porfirio',
        'Porter', 'Posey', 'Powell', 'Pranav', 'Pratt', 'Prentice', 'Prentiss',
        'Presley', 'Press', 'Preston', 'Price', 'Primus', 'Prince', 'Prosper',
        'Pryor', 'Purl', 'Quentin', 'Quincy', 'Quinn', 'Quint', 'Quinten',
        'Quintin', 'Quinton', 'Rae', 'Raekwon', 'Rafael', 'Rafe', 'Raheem',
        'Rahn', 'Rahsaan', 'Rahul', 'Raiden', 'Rakeem', 'Raleigh', 'Ralph',
        'Ramiro', 'Ramon', 'Ramsey', 'Rance', 'Rand', 'Randal', 'Randall',
        'Randel', 'Randell', 'Randle', 'Randolf', 'Randolph', 'Randy', 'Ransom',
        'Raoul', 'Raphael', 'Raquan', 'Ras', 'Rashaad', 'Rashaan', 'Rashad',
        'Rashawn', 'Rasheed', 'Raul', 'Raven', 'Ray', 'Rayan', 'Rayburn',
        'Rayfield', 'Rayford', 'Raymon', 'Raymond', 'Raymundo', 'Raynard',
        'Rayshawn', 'Reagan', 'Reason', 'Red', 'Redden', 'Redmond', 'Reece',
        'Reed', 'Reese', 'Refugio', 'Regan', 'Reggie', 'Reginal', 'Reginald',
        'Regis', 'Reid', 'Reilly', 'Reinaldo', 'Reinhold', 'Reino', 'Remington',
        'Remy', 'Renaldo', 'Renard', 'Rene', 'Reno', 'Reuben', 'Reubin', 'Rex',
        'Rexford', 'Rey', 'Reyes', 'Reynaldo', 'Reynold', 'Reynolds', 'Rhett',
        'Rhoda', 'Rhys', 'Rian', 'Ricardo', 'Ricci', 'Rice', 'Rich', 'Richard',
        'Richie', 'Richmond', 'Rick', 'Rickey', 'Ricki', 'Rickie', 'Ricky',
        'Rico', 'Ridge', 'Rigoberto', 'Riley', 'Rishi', 'Ritchie', 'River',
        'Rob', 'Robb', 'Robbie', 'Robbin', 'Robby', 'Robert', 'Roberto',
        'Robin', 'Robley', 'Robt', 'Roby', 'Rocco', 'Rock', 'Rocky', 'Rod',
        'Roddy', 'Roderic', 'Roderick', 'Rodger', 'Rodney', 'Rodolfo',
        'Rodrick', 'Rodrigo', 'Roe', 'Roel', 'Rogelio', 'Roger', 'Rogers',
        'Rohan', 'Roland', 'Rolando', 'Rolf', 'Roll', 'Rolla', 'Rolland',
        'Rollie', 'Rollin', 'Rollo', 'Roma', 'Roman', 'Rome', 'Romello',
        'Romeo', 'Romie', 'Ron', 'Ronal', 'Ronald', 'Ronaldo', 'Ronan',
        'Rondal', 'Ronin', 'Ronnie', 'Ronny', 'Roosevelt', 'Rory', 'Rosario',
        'Rosco', 'Roscoe', 'Rosendo', 'Rosevelt', 'Ross', 'Rossie', 'Roswell',
        'Rowan', 'Rowland', 'Roy', 'Royal', 'Royce', 'Rube', 'Ruben', 'Rubin',
        'Ruby', 'Rudolf', 'Rudolfo', 'Rudolph', 'Rudy', 'Rueben', 'Ruel',
        'Ruffin', 'Ruffus', 'Rufus', 'Rupert', 'Rush', 'Russ', 'Russel',
        'Russell', 'Rustin', 'Rusty', 'Rutherford', 'Ryan', 'Ryder', 'Ryker',
        'Rylan', 'Ryland', 'Rylee', 'Ryley', 'Ryne', 'Sabastian', 'Sage',
        'Saint', 'Sal', 'Salomon', 'Salvador', 'Salvatore', 'Sam', 'Samie',
        'Samir', 'Sammie', 'Sammy', 'Sampson', 'Samson', 'Samual', 'Samuel',
        'Sanders', 'Sandy', 'Sanford', 'Santana', 'Santiago', 'Santino',
        'Santo', 'Santos', 'Saul', 'Saverio', 'Savion', 'Savon', 'Sawyer',
        'Schley', 'Schuyler', 'Scot', 'Scott', 'Scottie', 'Scotty', 'Seaborn',
        'Seamus', 'Sean', 'Sebastian', 'Sedrick', 'Seldon', 'Selmer', 'Semaj',
        'Seneca', 'Sergio', 'Seth', 'Severo', 'Severt', 'Seward', 'Seymour',
        'Shad', 'Shade', 'Shafter', 'Shamar', 'Shan', 'Shane', 'Shannon',
        'Shanon', 'Shaquan', 'Shaquille', 'Sharif', 'Sharon', 'Shaun', 'Shawn',
        'Shay', 'Shayne', 'Shea', 'Shedrick', 'Shelby', 'Sheldon', 'Shelley',
        'Shellie', 'Shelly', 'Shelton', 'Shemar', 'Shep', 'Shepherd',
        'Sheridan', 'Sherman', 'Sherrill', 'Sherwin', 'Sherwood', 'Shirley',
        'Shoji', 'Shon', 'Shyheim', 'Sid', 'Sidney', 'Sie', 'Sigmund', 'Sigurd',
        'Silas', 'Silver', 'Silvester', 'Silvio', 'Sim', 'Simeon', 'Simmie',
        'Simon', 'Simpson', 'Sincere', 'Sing', 'Skip', 'Skylar', 'Skyler',
        'Slade', 'Smith', 'Sol', 'Soloman', 'Solomon', 'Solon', 'Son', 'Sonny',
        'Soren', 'Spencer', 'Spenser', 'Spurgeon', 'Squire', 'Stacey', 'Stacy',
        'Stafford', 'Stan', 'Stanford', 'Stanislaus', 'Stanley', 'Stanton',
        'Starling', 'Stefan', 'Stephan', 'Stephanie', 'Stephen', 'Stephon',
        'Sterling', 'Stetson', 'Stevan', 'Steve', 'Steven', 'Stevie', 'Steward',
        'Stewart', 'Stone', 'Stonewall', 'Stoney', 'Storm', 'Stuart',
        'Sullivan', 'Sumner', 'Susie', 'Sydney', 'Syed', 'Sylas', 'Sylvan',
        'Sylvanus', 'Sylvester', 'Tab', 'Tad', 'Taft', 'Tahj', 'Taj', 'Tal',
        'Talan', 'Talen', 'Tallie', 'Talmadge', 'Talmage', 'Talon', 'Tandy',
        'Tanner', 'Tarik', 'Tariq', 'Tate', 'Tatsuo', 'Taurean', 'Taurus',
        'Tavares', 'Tavaris', 'Tavian', 'Tavion', 'Tavon', 'Tayler', 'Taylor',
        'Tayshaun', 'Teagan', 'Ted', 'Teddie', 'Teddy', 'Tegan', 'Telly',
        'Terance', 'Terell', 'Terence', 'Terrance', 'Terrell', 'Terrence',
        'Terrill', 'Terry', 'Tevin', 'Tex', 'Thad', 'Thaddeus', 'Theadore',
        'Thedore', 'Theo', 'Theodis', 'Theodore', 'Theophile', 'Therman',
        'Theron', 'Thomas', 'Thompson', 'Thor', 'Thornton', 'Thorwald', 'Thos',
        'Thurlow', 'Thurman', 'Thurston', 'Tilden', 'Tillman', 'Tilman', 'Tim',
        'Timmie', 'Timmothy', 'Timmy', 'Timothy', 'Tito', 'Titus', 'Tobe',
        'Tobias', 'Tobie', 'Tobin', 'Toby', 'Tod', 'Todd', 'Toivo', 'Tolbert',
        'Tollie', 'Tom', 'Toma', 'Tomas', 'Tomie', 'Tommie', 'Tommy', 'Toney',
        'Tony', 'Torey', 'Toriano', 'Torrance', 'Torrence', 'Torrey', 'Torry',
        'Tory', 'Toshio', 'Toy', 'Trace', 'Tracey', 'Tracy', 'Trae', 'Travis',
        'Travon', 'Trayvon', 'Tre', 'Tremaine', 'Tremayne', 'Trent', 'Trenten',
        'Trenton', 'Trever', 'Trevin', 'Trevion', 'Trevon', 'Trevor', 'Trey',
        'Treyton', 'Treyvon', 'Trinidad', 'Trinity', 'Tripp', 'Tristan',
        'Tristen', 'Tristian', 'Tristin', 'Triston', 'Troy', 'True', 'Trumaine',
        'Truman', 'Trystan', 'Tuan', 'Tucker', 'Turner', 'Ty', 'Tye', 'Tyler',
        'Tylor', 'Tyquan', 'Tyree', 'Tyreek', 'Tyreese', 'Tyrek', 'Tyreke',
        'Tyrel', 'Tyrell', 'Tyrese', 'Tyrik', 'Tyrin', 'Tyriq', 'Tyrique',
        'Tyron', 'Tyrone', 'Tyrus', 'Tyshawn', 'Tyson', 'Ulises', 'Ulysses',
        'Unknown', 'Unnamed', 'Urban', 'Uriah', 'Uriel', 'Urijah', 'Val',
        'Valentin', 'Valentine', 'Valentino', 'Van', 'Vance', 'Vander',
        'Vashon', 'Vaughn', 'Vera', 'Vere', 'Vergil', 'Verl', 'Verle', 'Verlin',
        'Verlon', 'Verlyn', 'Vern', 'Verna', 'Vernal', 'Verne', 'Vernell',
        'Verner', 'Vernie', 'Vernon', 'Vester', 'Vic', 'Vicente', 'Vick',
        'Victor', 'Victoriano', 'Vidal', 'Vince', 'Vincent', 'Vincenzo',
        'Vinson', 'Vinton', 'Virge', 'Virgel', 'Virgie', 'Virgil', 'Virgle',
        'Vito', 'Vollie', 'Volney', 'Von', 'Wade', 'Waino', 'Waldemar', 'Waldo',
        'Walker', 'Wallace', 'Wally', 'Walt', 'Walter', 'Walton', 'Ward',
        'Wardell', 'Warner', 'Warren', 'Wash', 'Washington', 'Watson', 'Watt',
        'Waverly', 'Wayde', 'Wayland', 'Waylon', 'Wayman', 'Waymon', 'Wayne',
        'Weaver', 'Webb', 'Webster', 'Weldon', 'Wellington', 'Wells', 'Welton',
        'Wendel', 'Wendell', 'Wenzel', 'Werner', 'Wes', 'Wesley', 'Wess',
        'West', 'Westin', 'Westley', 'Weston', 'Wheeler', 'Whit', 'Whitney',
        'Wilber', 'Wilbert', 'Wilbur', 'Wilburn', 'Wiley', 'Wilford', 'Wilfred',
        'Wilfredo', 'Wilfrid', 'Wilhelm', 'Wiliam', 'Wilkie', 'Will', 'Willaim',
        'Willam', 'Willard', 'William', 'Williams', 'Willian', 'Williard',
        'Willie', 'Willis', 'Willy', 'Wilmer', 'Wilson', 'Wilton', 'Windell',
        'Winfield', 'Winford', 'Winfred', 'Wing', 'Winifred', 'Winnie',
        'Winston', 'Winthrop', 'Winton', 'Wirt', 'Wm', 'Wong', 'Wood', 'Woodie',
        'Woodroe', 'Woodrow', 'Woodson', 'Woody', 'Worley', 'Worth', 'Wright',
        'Wyatt', 'Wylie', 'Wyman', 'Xander', 'Xavier', 'Xzavier', 'Yaakov',
        'Yadiel', 'Yael', 'Yahir', 'Yair', 'Yancy', 'Yandel', 'Yee', 'Yehuda',
        'Yoel', 'York', 'Yosef', 'Yoshio', 'Young', 'Yurem', 'Yusuf',
        'Zachariah', 'Zachary', 'Zachery', 'Zack', 'Zackary', 'Zackery', 'Zaid',
        'Zaiden', 'Zain', 'Zaire', 'Zakary', 'Zander', 'Zane', 'Zavier',
        'Zavion', 'Zayden', 'Zayne', 'Zeb', 'Zebulon', 'Zechariah', 'Zed',
        'Zeke', 'Zenas', 'Zeno', 'Zigmund', 'Zion', 'Zollie',
    }


def _get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
  _BASE_DATA_URL_FORMAT_STR = ("https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/")
  _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
  base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
            shuffled=shuffled, language=language, deduplicated=deduplicated
        )
  checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(language=language)
  with fsspec.open(checksum_url, encoding="utf-8") as f:
    data_filenames = [line.decode().split("\t")[0] for line in f if line]
    return [base_data_url + data_filename for data_filename in data_filenames]

def _download_urls(urls):
  for url in urls:
    if not os.path.exists(url.split("/")[-1]):
      os.system(f"wget {url}")

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def stem(s):
  s = s.replace(".", " ").replace("!", " ").replace("?", " ").replace(",", " ").replace("-", " ").replace(";", " ").replace("'", " '").replace("\"", " \"")
  sArr = s.lower().split()
  if len(sArr) > 4:
    sArr = sArr[:4]
  s = " ".join([s1[:4] if len(s1) > 4 else s1 for s1 in sArr if s1.strip()])
  return s

def save_enron_line(l2, prev, o):
        """ Process Enron Data """
        l2 = remove_html_tags(l2)
        l2 = l2.split('-----Original Message-----')[0].strip()
        l2 = l2.split('---------------------- Forwarded')[0].strip()
        l2 = l2.split('----- Forwarded')[0].strip()
        l2 = l2.split('---------From:')[0].strip()
        l2 = l2.split('**********************************************************************This')[0].strip()
        l2 = l2.split('**********************************************************************   This')[0].strip()
        l2 = l2.split('******************************************************************This')[0].strip()
        l2 = l2.split('*************************************************This')[0].strip()
        l2 = l2.split('********************************************************************** This')[0].strip()
        l2 = l2.split('--------- Inline attachment follows')[0].strip()
        l2 = l2.split('The information contained in this e-mail message and')[0].strip()
        l2 = l2.split('This message is for the designated recipient')[0].strip()
        l2 = l2.split('***Please be advised')[0].strip()
        l2 = l2.split('*******This message')[0].strip()
        l2 = l2.split('This message (including any attachments) contains')[0].strip()
        l2 = l2.split('*********************************************************')[0].strip()
        l2 = l2.split('_________________________________________________________________Get')[0].strip()
        l2 = l2.split('___________________________________________')[0].strip()
        l2 = l2.split('__________________________________________________ Do')[0].strip()
        l2 = l2.replace("\\\"", " \" ").replace("(", " (").replace(")", ") ").replace("[", " [").replace("]", "] ").replace("?", "? ").replace("!", "! ").replace("? ?", "??").replace("! !", "!!").replace(":", ": ").replace("\t", " ").replace("= ", "").replace("=20","").replace("=90","").replace("=018","").replace("=09","").replace("=3D","")
        l2 = l2.replace(" s ", " 's ").replace(" ve ", " 've ").replace(" re ", " 're ").replace(" ll ", " 'll ").replace(" m ", " 'm ").replace(" t ", " 't ").replace(" d ", " 'd ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace(". . .", "...")
        l2 = l2.replace(". yahoo", ".yahoo").replace("www. ", "www.").replace(". htm", ".htm").replace(". co", ".co").replace(". org", ".org").replace(". edu", ".edu").replace(". net", ".net").replace(". NET", ".NET").replace(". CO", ".CO").replace(". ORG", ".ORG").replace(". EDU", ".EDU").replace(": //", "://")
        l2 = l2.replace(": 0", ":0").replace(": 1", ":1").replace(": 2", ":2").replace(": 3", ":3").replace(": 4", ":4").replace(": 5", ":5").replace(": 6", ":6").replace(": 7", ":7").replace(": 8", ":8").replace(": 9", ":9")
        l2 = l2.replace(". url -", ".url - <<").replace(". doc -", ".doc - <<").replace(". pdf -", ".pdf <<").replace(". xls -", ".xls <<").replace(". url", ".url>>").replace(". doc", ".doc>>").replace(". pdf", ".pdf>>").replace(". xls", ".xls>>").replace("<< ", "<<").replace("> >", " ").replace("  ", " ")
        l2 = l2.replace(". URL -", ".URL - <<").replace(". DOC -", ".DOC - <<").replace(". PDF -", ".PDF <<").replace(". XLS -", ".xls <<").replace(". URL", ".URL>>").replace(". DOC", ".DOC>>").replace(". PDF", ".PDF>>").replace(". XLS", ".XLS>>").replace("<< ", "<<").replace("> >", " ").replace("  ", " ")
        l2 = l2.replace("RE:", "").replace("Re:", "").replace("RE: ", "").replace("Re: ", "").replace("Fw: ", "").replace("FW: ", "").replace("FWD: ", "").replace("Fwd: ", "")
        l2 = l2.replace('Importance: High',':')
        if "Sent:" in l2: return
        l2 = l2.replace("...", "... ").replace("\"\"", " \" ").replace("  ", " ").strip(" -:;[]()\=<>\"").rstrip(".!?")
        l2Arr = l2.split()
        if len(l2Arr) > 3:
          l2 = " ".join(itertools.chain(*[camel_case_split(a) for a in l2Arr]))
          if l2.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!", "").replace("?", "").replace(",", "").replace("-", "").replace(";", "").replace(" ", "").lower() in prev: return
          l2 = l2.replace("==", "--")
          l2 = l2.replace("++", "--")
          l2 = l2.replace("*~", "--")
          l2 = l2.replace("||", "--")
          l2 = l2.replace("**", "--")
          l2 = l2.replace("__", "--")
          l2 = l2.replace("##", "--")
          for l3 in l2.split('--'):
            l3 = l3.strip()
            if l3:
              for l4 in l3.split("Subject: "):
                l4 = l4.strip('=, ')
                if l4: o.write(l4+"\tenron\n")
          prev[l2.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!", "").replace("?", "").replace(",", "").replace("-", "").replace(";", "").replace(" ", "").lower()] = 1

def has_any(s, lst):
  for l in lst:
    if l in s: return True
  return False

     
def create_english_dataset(share_dir='/content/drive/Shareddrives/BigScience/'):   
  """
    Creates a English dataset from different domains useful for doing PII detection. 
    Domains include enron (email), a subset of civil comments (forum message)

    We do some deduplication and cleanup. 
    We add some PII as augumentation (TBD: some <PERSON> and <ORG> tags added. Need to add some more categories).

    See specific licenses for each dataset. 
  """    

  prev={}
  if not os.path.exists("cleaned_english.tsv"):
    with open("english.tsv", "w", encoding="utf8") as o:

      #https://github.com/reglab/casehold court cases are government works and in the public domain. 
      #Annotations and selections are under Apache-2.0 License
      """
      @inproceedings{zhengguha2021,
	title={When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset},
	author={Lucia Zheng and Neel Guha and Brandon R. Anderson and Peter Henderson and Daniel E. Ho},
	year={2021},
	eprint={2104.08671},
	archivePrefix={arXiv},
	primaryClass={cs.CL},
	booktitle={Proceedings of the 18th International Conference on Artificial Intelligence and Law},
	publisher={Association for Computing Machinery}
  }      
      """
      with open(f"{share_dir}/casehold.csv", "rb") as f:
        while True:
          line = f.readline().decode()
          if not line: break
          line = line.split(",\"")
          if len(line) >= 2:
            line = line[1]
            line = line.split("(<HOLDING>)")
            if len(line) == 2:
              s1, s2 = line
              s1 = s1.replace("  ", " ").replace("  ", " ")
              s2 = s2.replace("  ", " ").replace("  ", " ")
              s2 = ' '.join(s2.split(',')[:-6]).strip(';: ')
              if s2:
                o.write(s1+' HOLDING: '+s2+"\tcasehold\n")
              else:
                o.write(s1+"\tcasehold\n")
            else:
              s1 = s1.replace("  ", " ").replace("  ", " ")
              o.write(s1+"\tcasehold\n")

      #from https://www.kaggle.com/wcukierski/enron-email-dataset, originally from https://www.cs.cmu.edu/~enron/ 
      # public data and partially copyrighted works (annotations) used by permission of authors
      """
      Public record data origially published by www.ferc.gov. Subsequent data cleansing by the authors and released
      "as a resource for researchers who are interested in improving current email tools, or understanding how email is currently used". 
      """
      with open(f"{share_dir}/kaggle_enron_emails.csv", "rb") as f:
        in_message = False
        l2 = ""
        while True:
          l = f.readline()
          if not l: break
          l = l.decode().strip()
          if not in_message and l.startswith("Subject:"):
            l = l.replace("Subject:", "").strip()
            if l: l2 = l+ ":"
          if "X-FileName" in l:
            in_message = True
            continue
          elif "Message-ID" in l:
            save_enron_line(l2, prev, o)
            l2 = ""
            in_message = False
          if in_message:
            l2 += " "+l
            
        if l2:
          save_enron_line(l2, prev, o)
  
      #https://huggingface.co/datasets/civil_comments - CC0
      """
      @article{DBLP:journals/corr/abs-1903-04561,
  author    = {Daniel Borkan and
               Lucas Dixon and
               Jeffrey Sorensen and
               Nithum Thain and
               Lucy Vasserman},
  title     = {Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
               Classification},
  journal   = {CoRR},
  volume    = {abs/1903.04561},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.04561},
  archivePrefix = {arXiv},
  eprint    = {1903.04561},
  timestamp = {Sun, 31 Mar 2019 19:01:24 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-04561},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
      """
      # An experiment we could try is tagging with another public figure tag called PUBLIC_FIGURE_OPINION 
      # so that we could train a model to distinguish between public figure mentions in opinion domains vs non-opinion domains
      
      dataset = load_dataset("civil_comments")
      for d in (dataset['train'], ):
         for idx, data in enumerate(d):
          score = sum ([data[feature] for feature in [ 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']])
          l2 = data['text']
          l2 = l2.replace("\n", " ").replace("  ", " ").replace("  ", " ")
          l2Arr = l2.split()
          has_a_name = has_any(first_names, l2Arr)
          l2_lower = l2.lower()
          if random.choice([0,1]) and not has_a_name and "mr." not in l2_lower and "ms." not in l2_lower and  "mrs." not in l2_lower and "president" not in l2_lower and "governor" not in l2_lower and  "mayor" not in l2_lower:
            continue
          if len(l2Arr) > 10 and len(l2Arr) < 50 and (score <= 0.5 or random.randint(0, 10)==0): # having too much toxic content may skew the data
            if has_a_name or "mr." in l2_lower or "ms." in l2_lower or "mrs." in l2_lower or "senator" in l2_lower or "president" in l2_lower or "governor" in l2_lower or "mayor" in l2_lower:
              o.write (l2+"\tcivil_comments\n")

    os.system("sort --parallel=32 english.tsv -o english.tsv")

  with open("english_cleaned.tsv", "w", encoding="utf8") as o:
    with open("english.tsv", "rb") as f:
      prev=""
      while True:
        l = f.readline().decode()
        if not l: break
        l = l.strip()
        l2 = l.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!", "").replace("?", "").replace(",", "").replace("-", "").replace(";", "").replace(" ", "").lower()
        prev2 = prev.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!", "").replace("?", "").replace(",", "").replace("-", "").replace(";", "").replace(" ", "").lower()
        if prev != "" and (l2==prev2 or (len(prev) > 10 and len(l) > 10 and prev2[:10]== l2[:10])):
          if len(l) > len(prev):
            prev = l
          continue
        else:
          if prev:
            if prev[0] < 'וח': 
              o.write (prev.lstrip(':;.+- ')+"\n")
          prev = l
    if prev:
      if prev[0] < 'וח': 
        o.write (prev.lstrip(':;.+- ')+"\n")

  
  os.system("sort --parallel=32 english_cleaned.tsv -o english_cleaned.tsv")
  os.system(f"cp english_cleaned.tsv {share_dir}/english_cleaned.tsv")


def check_good_sentence(s, en_lang_cutoff=0.1, junk_ratio=0.5, stopword_check=True):
  #basic dejunk
  s = s.lower().strip()
  if not s: return False
  jr = len([s2 for s2 in s if s2 in junk_dict])/len(s)
  if jr >= junk_ratio:
    return False
  sArr = [s2.strip("' 0123456789¯_§½¼¾×|†—~\"—±′–'°−{}[]·-\'?,./<>!@#^&*()+-‑=:;`→¶'") for s2 in s.lower().split()]
  if len(sArr) == 0:
    return False
  #stopword check
  if stopword_check and len([s2 for s2 in sArr if s2 in stopwords_en])/len(sArr) < en_lang_cutoff:
    return False
  else:
    #langid check
    try:
      lang =  langid.classify(s)[0]
    except:
      lang = ""
    return lang == "en"


def create_oscar_subset_for_ner():
  with open("pii_oscar.txt", "w", encoding="utf8") as o:
    with open("oscar_sample.txt", "rb") as f:
      while True:
        sent = f.readline().decode()
        if not sent: break
        sent = sent.strip()
        for sent2 in sent.split("<|endoftext|>"):
            sent2 = sent2.strip()
            sentArr = sent2.split()
            if len(sentArr) > 150: 
              #print ("truncating sent", sent)
              sent2 = " ".join(sentArr[:150])
            if "Alzheimer's" in sent2 or "Alzheimer" in sent2 or 'heart disease' in sent2 or ' AIDS ' in sent2 or ' HIV ' in sent2 or ' was born ' in sent2 or 'Social Secu' in sent2 or 'socialist' in sent2 or 'republican' in sent2 or 'democrat' in sent2 or 'lower class' in sent2 or ' union ' in sent2 or 'upper class' in sent2 or 'middle class' in sent2 or ' cancer ' in sent2:
              if 'pussy' not in sent2 and ' cock ' not in sent2:
                o.write (sent2+"\n")
    url = _get_oscar_urls("en")[0]
    _download_urls([url])
    file = url.split("/")[-1]
    for sent2 in gzip.open(file):
      sent2 = sent2.decode()
      sent2 = sent2.strip()
      sentArr = sent2.split()
      if len(sentArr) > 150: 
        sent2 = " ".join(sentArr[:150])
      #let's just look for disease age, for the bigger set, since terms like "republic" and "democract" are over represented
      if "Alzheimer's" in sent2 or "Alzheimer" in sent2 or 'heart disease' in sent2 or ' AIDS ' in sent2 or ' HIV ' in sent2 or ' was born ' in sent2  or ' cancer ' in sent2:
          if not check_good_sentence(sent2):
            continue
          if 'pussy' not in sent2 and ' cock ' not in sent2:
            o.write (sent2+"\n")
  os.system("sort --parallel=32 pii_oscar.txt -o pii_oscar.txt")

def do_ner(do_casehold=False):
  """ Create English based NER/PII dataset """
  faker_target_lang = Faker(faker_map["en"])
  faker_target_lang.add_provider(person)
  nlp = spacy.load('en_core_web_lg')
  row_id = 0
  with open("pii_en.jsonl", "w", encoding="utf8") as o:

    with open("pii_oscar.txt", "rb") as f:
      prev = ""
      for sent2 in tqdm(f):
        #sent2 = f.readline().decode().strip()
        sent2 = sent2.decode().strip()
        if not sent2: break
        domain = "oscar"
        if sent2 == prev:
          continue
        if prev:
          sent3 = prev
          if sent3[0] in "0123456789":
            sent3 = sent3.split(" ",1)[1]
          sentArr = sent3.split()
          if sentArr[0].endswith(":"):
            sentArr= sentArr[:1]
          if len(sentArr) > 100: 
            sentArr = sentArr[:100]
          sent3 = " ".join(sentArr)
          if True:
            doc = nlp(sent3)
            entities = list(doc.ents)
            if [entity for entity in entities if entity.label_ == 'PERSON']:
              ents = [[entity.text, entity.label_] for entity in entities if entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP') and 'http:' not in entity.text]
              if len(ents) > 1 or 'cancer' in sent3 or 'class' in sent3 or 'union' in sent3 or 'democrat' in sent3 or 'republican' in sent3 or 'socialist' in sent3:
                if len(ents) < 5:
		  if '@' in sent3 or 'Social Sec' in sent3 or 'password' in sent3:
		      context = {}
		      ents2 = []
		      for item in ents:
			if item[1] == 'PERSON':
			  if item[0] in public_figures:
			    item[1] = 'PUBLIC_FIGURE'
			  else:
			    context[item[0]] = context.get(item[0], \
						      faker_target_lang.name() if " " in item[0]  else \
						      faker_target_lang.first_name())
			    sent3 = sent3.replace(item[0], context[item[0]])
			    if " " in item[0]:
			      context[item[0].split()[0]] = context[item[0]].split()[0]
			      context[item[0].split()[-1]] = context[item[0]].split()[-1]
			    item[0] = context[item[0]]
			ents2.append(item)
		    else:
		      ents2 = ents
                  o.write (json.dumps({"text": sent3, "ner": ents2, "domain": domain, "target_lang": "en", "id": row_id})+"\n")
                  row_id += 1
        prev = sent2
        
    with open("english_cleaned.tsv", "rb") as f:
      #while True:
      #  l = f.readline().decode()
      #  if not l: break
      for l in tqdm(f):
        #sent2 = f.readline().decode().strip()
        l = l.decode().strip()
        l = l.split("\t")
        sent = l[0]
        domain = l[-1].strip()
        if not check_good_sentence(sent):
          continue
        if not do_casehold and domain == "casehold": continue
        if "Notice No." in sent: continue
        if "TO: ALL COMEX" in sent: continue
        if "TO: All NYMEX" in sent: continue
        if "TO: All New" in sent: continue
        if sent[0] in "0123456789":
          sent = sent.split(" ",1)[1]
        sentArr = sent.split()
        if sentArr[0].endswith(":"):
          sentArr= sentArr[:1]
        if len(sentArr) > 100: 
          sentArr = sentArr[:100]
        sent = " ".join(sentArr)
        doc = nlp(sent)
        entities = list(doc.ents)
        
        if [entity for entity in entities if entity.label_ == 'PERSON']:
          ents = [[entity.text, entity.label_] for entity in entities if entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP') and 'http:' not in entity.text]
          if len(ents) > 1 and len(ents) < 5:
            if random.randint(0,1)==0 or '@' in sent or 'Social Sec' in sent or 'password' in sent:
              context = {}
              ents2 = []
              for item in ents:
                if item[1] == 'PERSON':
                  if item[0] in public_figures:
                    item[1] = 'PUBLIC_FIGURE'
                  else:
                    context[item[0]] = context.get(item[0], \
                                              faker_target_lang.name() if " " in item[0]  else \
                                              faker_target_lang.first_name())
                    sent = sent.replace(item[0], context[item[0]])
                    if " " in item[0]:
                      context[item[0].split()[0]] = context[item[0]].split()[0]
                      context[item[0].split()[-1]] = context[item[0]].split()[-1]
                    item[0] = context[item[0]]
                ents2.append(item)
            else:
              ents2 = ents
            o.write (json.dumps({"text": sent, "ner": ents2, "domain": domain, "target_lang": "en", "id": row_id})+"\n")
            row_id += 1

def do_translation(target_lang='hi', mariam_mt=True, person_swap=True):
  """ 
  Translate the English PII/NER dataset to target_lang. 
  Mariam_mt is faster but less accurate. 
  If not mariam_mt, then do m2m100 which is slower bu more accurate. 
  """
  lbracket = "["
  rbracket = "]"
  if target_lang in ('zh', 'ja', 'ko'):
    lbracket = "[["
    rbracket = "]]"  
  faker_target_lang = Faker(faker_map[target_lang])
  faker_target_lang.add_provider(person)
  faker_target_lang.add_provider(geo)
  if mariam_mt:
    try:
      model_name = 'Helsinki-NLP/opus-mt-en-'+ target_lang
      model = MarianMTModel.from_pretrained(model_name)
      model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      trans = pipeline("translation", model=model, tokenizer=tokenizer)
    except:
      print ("loading m2m100")
      mariam_mt = False
      model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
      model =  torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
      tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
  else:
    print ("loading m2m100")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model =  torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
  row_id = -1
  with open(f"pii_{target_lang}.jsonl", "w", encoding="utf8") as o:
    for s in tqdm(open("pii_en.jsonl", "rb")):
      s = s.decode().strip()
      if not s: continue
      dat = json.loads(s)
      domain = dat['domain']
      ner = dat['ner']
      text = dat['text']
      if 'id' not in dat:
        row_id += 1
      else:
        row_id = int(dat['id'])
      if 'NYMEX' in text: continue
      ner = [n for n in ner if n[0] not in ("FREE", "’m", 'Social Security')]
      ner2 = []
      if ' cancer ' in text:
        ner2.append(['cancer', 'DISEASE'])
      elif ' HIV ' in text:
        ner2.append(['HIV', 'DISEASE'])
      elif ' AIDS ' in text:
        ner2.append(['AIDS', 'DISEASE'])
      elif "Alzheimer's" in text:
        ner2.append(["Alzheimer's", 'DISEASE'])
      elif "Alzheimer" in text:
        ner2.append(['Alzheimer', 'DISEASE'])
      elif 'heart disease' in text:
        ner2.append(['heart disease', 'DISEASE'])
      elif 'democractic' in text:
        ner2.append(['democractic', 'NORP'])
      elif 'democrats' in text:
        ner2.append(['democrats', 'NORP'])
      elif 'Democrats' in text:
        ner2.append(['Democrats', 'NORP'])
      elif 'democrat' in text:
        ner2.append(['democrat', 'NORP'])
      elif 'Democrat' in text:
        ner2.append(['Democrat', 'NORP'])
      elif 'republican' in text:
        ner2.append(['republican', 'NORP'])
      elif 'republicans' in text:
        ner2.append(['republicans', 'NORP'])
      elif 'Republicans' in text:
        ner2.append(['Republicans', 'NORP'])
      elif 'republican' in text:
        ner2.append(['republican', 'NORP'])
      elif 'Republican' in text:
        ner2.append(['Republican', 'NORP'])
      elif 'socialist' in text:
        ner2.append(['socialist', 'NORP'])
      elif 'Socialist' in text:
        ner2.append(['Socialist', 'NORP'])
      for item in ner:
        itemArr =  item[0].split()
        if itemArr[0] in ('Association', 'Society', 'Union') or itemArr[-1] in ('Association', 'Society', 'Party'):
          item[1] = 'NORP'
        if item[0] == 'Enron':
          item[0] = choice(swap_org)
          text = text.replace("Enron", item[0])
          text = text.replace("@enron", '@'+item[0].lower())
          text = text.replace(" enron", ' '+item[0].lower())
          #print (text)
        if item[0] in public_figures:
          item[1] = 'PUBLIC_FIGURE'
        elif item[0] in country:
          item[1] = 'COUNTRY'
        elif item[0] in NORP:
          item[1] = 'NORP'
        elif '@' in item[0]:
          item[1] = 'EMAIL'
        ner2.append(item)
      #col.extend ([d[0] for d in ner2 if d[1] == 'NORP'])
      context = {}
      ner_mapping = {}
      seen = {}
      text = "... " + text + " "
      text = text.replace(lbracket, "(")
      text = text.replace(rbracket, ")",)
      if person_swap:
        _idx = 0
        for item in ner2:
          if item[0] in seen: continue
          seen[item[0]] = 1
          if item[1] in ('COUNTRY', 'GPE', 'PERSON'): #ORG
            text = text.replace(" "+ item[0] + " ", " *"+ str(_idx) +  "* ")
            text = text.replace(" "+ item[0] + ",", " *"+ str(_idx) + "* ,")
            text = text.replace(" "+ item[0] + "'", " *"+ str(_idx) + "*'")
            text = text.replace(item[0], "*"+ str(_idx) + "*")
            context[item[0]] = context.get(item[0], \
                                          faker_target_lang.name() if " " in item[0] and item[1] == 'PERSON' else \
                                          faker_target_lang.first_name() if item[1] == 'PERSON' else \
                                          faker_target_lang.country() if item[1] == 'COUNTRY' else \
                                          faker_target_lang.state() if item[1] == 'GPE' and target_lang != 'zh'  else \
                                          faker_target_lang.province() if item[1] == 'GPE' and target_lang == 'zh' else \
                                          #faker_target_lang.company()  if item[1] == 'ORG' else 
                                          item[0])
            ner_mapping["*"+ str(_idx) + "*"] = [context[item[0]] , item[1] if item[1] != 'COUNTRY' else 'GPE']
            if " " in item[0]:
              context[item[0].split()[0]] = context[item[0]].split()[0]
              context[item[0].split()[-1]] = context[item[0]].split()[-1]
            item[0] = context[item[0]]
          else:
            text = text.replace(item[0], ' ' + str(_idx) +  " "+lbracket+item[0] + rbracket)
            ner_mapping[str(_idx) +  " "+lbracket] = item
          _idx += 1
      
      text_trans = ""
      if not mariam_mt:
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        trans_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
      else:
        #try:
          encoded = tokenizer(text)
          if len(encoded.input_ids) > 512: 
            print ("sentence too long: ", text)
            continue
          trans_text = trans(text)[0]['translation_text']
        #except:

          #continue
      if True:
        ner_found = []
        trans_text = trans_text.lstrip(".")
        trans_text = trans_text.strip()
        trans_text = trans_text.replace("* ", "*").replace(" *", "*").replace('"0*', '*0* ').replace('"1*', '*1* ').replace('"2*', '*2* ').replace('"3*', '*3* ').replace('"4*', '*4* ').\
          replace('*0:', '*0* ').replace('*1:', '*1* ').replace('*2:', '*2* ').replace('*3:', '*3* ').replace('*4:', '*4* ').\
          replace('*0 ', '*0* ').replace('*1 ', '*1* ').replace('*2 ', '*2* ').replace('*3 ', '*3* ').replace('*4 ', '*4* ').\
          replace(' 0*', '*0* ').replace(' 1*', '*1* ').replace('2 *', '*2* ').replace(' 3*', '*3* ').replace(' 4*', '*4* ')
        if trans_text.startswith('0') and not trans_text.startswith('0 ['):
          trans_text = '*0* ' + trans_text[1:]
        trans_text = trans_text.replace(". [", " [").replace(".[", " [").replace("  ", " ")
        orig_trans_text = trans_text
        for key, ner_item in ner_mapping.items():
          found = False
          if key in trans_text:
            found = True
          elif key.replace(" ", "") in trans_text:
            found = True
            key = key.replace(" ", "")
          elif key.lstrip('*') in trans_text:
            found = True
            key = key.lstrip('*')
          elif key.rstrip('*') in trans_text:
            found = True
            key = key.rstrip('*')
          if found:
            if key[0] == '*':
              trans_text = trans_text.replace(key, " " + ner_item[0] + " ")
              ner_found.append(list(ner_item))
            else:
              trans_text2 = ""
              for segment in trans_text.split(key):
                if rbracket in segment:
                  entity, rest = segment.split(rbracket, 1)
                  entity = entity.strip("[]")
                  ner_found.append([entity, ner_item[1]])
                  trans_text2 += " " + entity + " " + rest
                else:
                  trans_text2 += " " + segment
              trans_text = trans_text2.strip()
        trans_text = trans_text.replace("*", " ").replace("[", " ").replace("]", " ").replace(" .", ".").replace(" ,", ",").replace("  ", " ").replace("  ", " ").strip()
        if target_lang in ('zh', 'ja', 'ko'):
          trans_text.replace(" ", "")
          trans_text.strip('#.')
        #print (text, ner2)
        #print ('orig_trans_text', orig_trans_text)
        #print (trans_text, ner_found)
        if ner_found:
          j = {'text': trans_text, 'ner': ner_found, 'domain': domain, 'id': row_id, 'lang': target_lang}
          o.write (json.dumps(j)+"\n")


#create_english_dataset()
#create_oscar_subset_for_ner()
#do_ner()
#do_translation(target_lang="hi")
if __name__ == "__main__":  
  if "-target_lang" in sys.argv:
    target_lang = sys.argv[sys.argv.index("-target_lang")+1]   
    if target_lang == "en":
      do_ner()
    else:
      do_translation(target_lang=target_lang)
