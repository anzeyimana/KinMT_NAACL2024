from typing import List

from token_stats import get_all_token_stats, TokenStats
from misc_functions import read_lines, write_lines
from names_dataset import NameDataset, NameWrapper
import unicodedata as ud

latin_letters= {}

def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def only_roman_chars(unistr):
    return all(is_latin(uchr)
           for uchr in unistr
           if uchr.isalpha()) # isalpha suggested by John Machin
def read_places(input_file = 'raw/world-cities_csv.csv') -> List[str]:
    lines = read_lines(input_file)
    # Skip header
    lines = lines[1:]
    places_list = []
    for line in lines:
        tokens = line.strip().split(',')
        for part in tokens:
            place = part.strip()
            place = place.rstrip()
            place = place.strip("\t;,.’‘\'\"")
            place = place.rstrip("\t;,.’‘\'\"")
            places_list.append(place)
    places_list = list(set(places_list))
    return places_list

def read_all_token_stats(all_tokens_file="data/AllTokens.csv") -> List[TokenStats]:
    global all_tokens_stats
    lines = read_lines(all_tokens_file)
    return [TokenStats(line,id,len(lines)) for id,line in enumerate(lines)]

def export_people_numbers_and_places():
    places_list = read_places()
    print('places_list:', len(places_list))

    rw_tokens_stats = read_all_token_stats()
    rwanda = []
    for tok in rw_tokens_stats:
        if (tok.totalDocuments > 3) and (tok.totalCount > 6):
            if ((tok.firstUpCount*1.0) > (0.7*tok.totalCount)):
                pn = (tok.id[:1].upper())+(tok.id[1:])
                rwanda.append(pn)
            elif tok.isNumeric:
                pn = tok.id
                rwanda.append(pn)
    print('rwanda:', len(rwanda))

    females = read_lines("raw/cmu_names/female.txt")
    print('females:', len(females))

    males = read_lines("raw/cmu_names/male.txt")
    print('males:', len(males))

    nd = NameDataset()
    people = []
    people_data = [nd.get_top_names(n=1000, country_alpha2='IN'),
                   nd.get_top_names(n=1000, country_alpha2='CN'),
                   nd.get_top_names(n=5000, country_alpha2='BI'),
                   nd.get_top_names(n=3000, country_alpha2='US'),
                   nd.get_top_names(n=3000, country_alpha2='GB'),
                   nd.get_top_names(n=2000, country_alpha2='FR'),
                   nd.get_top_names(n=2000, country_alpha2='BE')]
    for data in people_data:
        for country in data:
            for sex in data[country]:
                people.extend(data[country][sex])

    people = list(set(people))
    print('people:', len(people))

    full_list = list(set(places_list + rwanda + males + females + people))
    full_list = [tok.strip().rstrip().strip("\t;,.’‘\'\"").rstrip("\t;,.’‘\'\"") for tok in full_list if only_roman_chars(tok)]
    print('full_list:', len(full_list))

    full_list = sorted(full_list)
    write_lines(full_list,'txt/names_and_numbers_rw.txt')
    write_lines(full_list,'txt/names_and_numbers_en.txt')


if __name__ == '__main__':
    # nd = NameDataset()
    # data = nd.get_top_names(n=5, country_alpha2='CN')
    # people = []
    # for country in data:
    #     for sex in data[country]:
    #         people.extend(data[country][sex])
    # print('\n'.join(sorted(list(set(people)))))

    export_people_numbers_and_places()
