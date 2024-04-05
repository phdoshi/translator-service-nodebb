from vertexai.language_models import ChatModel, InputOutputTextPair
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
credentials = service_account.Credentials.from_service_account_info({
  "type": "service_account",
  "project_id": "skilled-compass-419014",
  "private_key_id": "9a9f9b0978755a7873f66370d46f98c80fc32d78",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDfYOstQfSiPUpE\nh9L3nzgazCymT5sRDQfBnjDkT2BBf9P8pa/IY0r5+lwn6G5jyv1GO9K2oZgur2MQ\n2bmrMPV4WkaAzv10LUbycP23ksk/vyeya1fweUJi0f5S+sZtFc5oOTOWbgDM+tW0\nA3TEnfSA3Gm8W55CIy0uUyei3tN3PgT7xyGhGPbYUUkpk4H0lPlYph2/sbLM6Xz9\nn1gvJQXKm7abL2YewGX7MGWRtznfbJP82HXj3xK6c5CG6Q8igJ4PW8pbyiX3P6Ae\nBhCFJF2YCjHzX7wLabVUeKyo4SQSIfns1GUKSZ7PNeMI2QBSiv+LiVWEoJPQCZox\nXWDwccX5AgMBAAECggEAE4VwHV0S4012khVJ9eNE5Jf3RvQqBGFzzhM5s+95OqrQ\nfIHWTNgwkzaymwzlqMmdPL1kUBrrJm6kD7Lsq5ifEHeDNc/hDSgad9GGuruUjteb\nwSyUW4BIBu7Rylqpa0W+lhPO/anQtBkvrTL9dLlWcXR8YROjTvsCqyEP6Qc5dRrH\n9Lj1L4yKQjTyuTnbZNf1VwZ0+N9lsacy+zoocpfYCrYcCGXR9jU8vKhAqepvruxL\n3yAg3NLVudEvWArKypE9CfPNdATzU7KhbJlLIMOGeRZXJNhvItnRVOxOz/eWZRkl\nnS4NXMS5GNCRyEKYrSub7nPHn4Oe3lHLps7QNvpJiQKBgQDvTK8z1YfLrzBsbWuT\nCCl8f+YBIy6JJ34ByMZtcE4nWAVFoSoP7kJ9XThGlQS1Ynjjz03GLMXHAoiv3GVs\ngb4XfSPuKrLRZARPvlmV1ezOyb+i4wN2AvTxRCuBrctGlTQVrlMci9c8QWj9G/s+\n8l3ucL0fPp8xISVUQgEsKXH4HwKBgQDu98qEIutJykFUuezI/QRRDu70j+u7BDqc\n619iBo01SSkqkdJVzQAOfu82OPXo/2lEkjOV5ZkjGoKeslhcNm/hwua+upgAESmf\nfmTTZHPvJa8eboNexJ8hoMgCeOCZTtJLx5+YDKRdAJHexCyXKVV0TTM49QRJ5LVI\nM9iwXJDe5wKBgQDf/yYZHQ3KCakINbx2mzNKSOZhti7/T6pRvUCQfImLpob40I1w\n8BPpCXN+DkukBhMnG4uvr4VKbgLIq0N9OE1Z61fQvsM34alvg7yT1vWd85egv2Mj\n+kyR8r/O2YHoBg4FJXLCy9ujmY7PLnwWRgTLB1jggFu/P0mT/1zdm8YE/QKBgC41\nqmqs2fTV8JYysdAX5TrLeXe/UOVEJxVrCnHJNIsmT5iUxCilUKp870L79smMgk2p\nO+JzmI5KNRH9CMskF7R/XOFq8bOpnfPinBqopSaMkfV/h1XPDbqR/btEpqHetRnE\nS186qpn1Xz/FjZ3Zd1XbC9mqv56OZGPG6tMHw1fnAoGAaLNwOj10buVk6zR8iu2o\n7SR55F6b59U0Ie1FSdp650sZI5IOIVvo4kIqdaQw5KtIZGKnm8iSOAnd9dGld3Vp\nUzMMq85IUifUOFd4AtvVNTm2Ws5GQIVSBP0jcMYjMd6vcKzzao/SGcfm9MYt7x8S\nap/dAIUocRwFRWjRAwdAwHk=\n-----END PRIVATE KEY-----\n",
  "client_email": "deploy-for-nodebb@skilled-compass-419014.iam.gserviceaccount.com",
  "client_id": "106562052692852693829",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/deploy-for-nodebb%40skilled-compass-419014.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
)
aiplatform.init(credentials=credentials)

def get_translation(post: str) -> str:
    parameters = {
        "temperature": 0.7,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    }
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    context_translate = "This is a post from an online web-discussion server. Translate the post into English if it is not already in English."
    chat = chat_model.start_chat(context=context_translate)
    response = chat.send_message(post, **parameters)
    return response.text


def get_language(post: str) -> str:
    parameters = {
        "temperature": 0.7,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    }
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    context_classify = "This is a post from an online web-discussion server. Classify the language the post is written in."
    chat = chat_model.start_chat(context=context_classify)
    response = chat.send_message(post, **parameters)
    return response.text

# language pairs from https://gist.github.com/alexanderjulo/4073388

language_pairs = [
    ('aa', 'Afar'),
    ('ab', 'Abkhazian'),
    ('af', 'Afrikaans'),
    ('ak', 'Akan'),
    ('sq', 'Albanian'),
    ('am', 'Amharic'),
    ('ar', 'Arabic'),
    ('an', 'Aragonese'),
    ('hy', 'Armenian'),
    ('as', 'Assamese'),
    ('av', 'Avaric'),
    ('ae', 'Avestan'),
    ('ay', 'Aymara'),
    ('az', 'Azerbaijani'),
    ('ba', 'Bashkir'),
    ('bm', 'Bambara'),
    ('eu', 'Basque'),
    ('be', 'Belarusian'),
    ('bn', 'Bengali'),
    ('bh', 'Bihari languages'),
    ('bi', 'Bislama'),
    ('bo', 'Tibetan'),
    ('bs', 'Bosnian'),
    ('br', 'Breton'),
    ('bg', 'Bulgarian'),
    ('my', 'Burmese'),
    ('ca', 'Catalan; Valencian'),
    ('cs', 'Czech'),
    ('ch', 'Chamorro'),
    ('ce', 'Chechen'),
    ('zh', 'Chinese'),
    ('cu', 'Church Slavic; Old Slavonic; Church Slavonic; Old Bulgarian; Old Church Slavonic'),
    ('cv', 'Chuvash'),
    ('kw', 'Cornish'),
    ('co', 'Corsican'),
    ('cr', 'Cree'),
    ('cy', 'Welsh'),
    ('cs', 'Czech'),
    ('da', 'Danish'),
    ('de', 'German'),
    ('dv', 'Divehi; Dhivehi; Maldivian'),
    ('nl', 'Dutch; Flemish'),
    ('dz', 'Dzongkha'),
    ('el', 'Greek, Modern (1453-)'),
    ('en', 'English'),
    ('eo', 'Esperanto'),
    ('et', 'Estonian'),
    ('eu', 'Basque'),
    ('ee', 'Ewe'),
    ('fo', 'Faroese'),
    ('fa', 'Persian'),
    ('fj', 'Fijian'),
    ('fi', 'Finnish'),
    ('fr', 'French'),
    ('fy', 'Western Frisian'),
    ('ff', 'Fulah'),
    ('Ga', 'Georgian'),
    ('de', 'German'),
    ('gd', 'Gaelic; Scottish Gaelic'),
    ('ga', 'Irish'),
    ('gl', 'Galician'),
    ('gv', 'Manx'),
    ('el', 'Greek, Modern (1453-)'),
    ('gn', 'Guarani'),
    ('gu', 'Gujarati'),
    ('ht', 'Haitian; Haitian Creole'),
    ('ha', 'Hausa'),
    ('he', 'Hebrew'),
    ('hz', 'Herero'),
    ('hi', 'Hindi'),
    ('ho', 'Hiri Motu'),
    ('hr', 'Croatian'),
    ('hu', 'Hungarian'),
    ('hy', 'Armenian'),
    ('ig', 'Igbo'),
    ('is', 'Icelandic'),
    ('io', 'Ido'),
    ('ii', 'Sichuan Yi; Nuosu'),
    ('iu', 'Inuktitut'),
    ('ie', 'Interlingue; Occidental'),
    ('ia', 'Interlingua (International Auxiliary Language Association)'),
    ('id', 'Indonesian'),
    ('ik', 'Inupiaq'),
    ('is', 'Icelandic'),
    ('it', 'Italian'),
    ('jv', 'Javanese'),
    ('ja', 'Japanese'),
    ('kl', 'Kalaallisut; Greenlandic'),
    ('kn', 'Kannada'),
    ('ks', 'Kashmiri'),
    ('ka', 'Georgian'),
    ('kr', 'Kanuri'),
    ('kk', 'Kazakh'),
    ('km', 'Central Khmer'),
    ('ki', 'Kikuyu; Gikuyu'),
    ('rw', 'Kinyarwanda'),
    ('ky', 'Kirghiz; Kyrgyz'),
    ('kv', 'Komi'),
    ('kg', 'Kongo'),
    ('ko', 'Korean'),
    ('kj', 'Kuanyama; Kwanyama'),
    ('ku', 'Kurdish'),
    ('lo', 'Lao'),
    ('la', 'Latin'),
    ('lv', 'Latvian'),
    ('li', 'Limburgan; Limburger; Limburgish'),
    ('ln', 'Lingala'),
    ('lt', 'Lithuanian'),
    ('lb', 'Luxembourgish; Letzeburgesch'),
    ('lu', 'Luba-Katanga'),
    ('lg', 'Ganda'),
    ('mk', 'Macedonian'),
    ('mh', 'Marshallese'),
    ('ml', 'Malayalam'),
    ('mi', 'Maori'),
    ('mr', 'Marathi'),
    ('ms', 'Malay'),
    ('Mi', 'Micmac'),
    ('mk', 'Macedonian'),
    ('mg', 'Malagasy'),
    ('mt', 'Maltese'),
    ('mn', 'Mongolian'),
    ('mi', 'Maori'),
    ('ms', 'Malay'),
    ('my', 'Burmese'),
    ('na', 'Nauru'),
    ('nv', 'Navajo; Navaho'),
    ('nr', 'Ndebele, South; South Ndebele'),
    ('nd', 'Ndebele, North; North Ndebele'),
    ('ng', 'Ndonga'),
    ('ne', 'Nepali'),
    ('nl', 'Dutch; Flemish'),
    ('nn', 'Norwegian Nynorsk; Nynorsk, Norwegian'),
    ('nb', 'Bokmål, Norwegian; Norwegian Bokmål'),
    ('no', 'Norwegian'),
    ('oc', 'Occitan (post 1500)'),
    ('oj', 'Ojibwa'),
    ('or', 'Oriya'),
    ('om', 'Oromo'),
    ('os', 'Ossetian; Ossetic'),
    ('pa', 'Panjabi; Punjabi'),
    ('fa', 'Persian'),
    ('pi', 'Pali'),
    ('pl', 'Polish'),
    ('pt', 'Portuguese'),
    ('ps', 'Pushto; Pashto'),
    ('qu', 'Quechua'),
    ('rm', 'Romansh'),
    ('ro', 'Romanian; Moldavian; Moldovan'),
    ('ro', 'Romanian; Moldavian; Moldovan'),
    ('rn', 'Rundi'),
    ('ru', 'Russian'),
    ('sg', 'Sango'),
    ('sa', 'Sanskrit'),
    ('si', 'Sinhala; Sinhalese'),
    ('sk', 'Slovak'),
    ('sk', 'Slovak'),
    ('sl', 'Slovenian'),
    ('se', 'Northern Sami'),
    ('sm', 'Samoan'),
    ('sn', 'Shona'),
    ('sd', 'Sindhi'),
    ('so', 'Somali'),
    ('st', 'Sotho, Southern'),
    ('es', 'Spanish; Castilian'),
    ('sq', 'Albanian'),
    ('sc', 'Sardinian'),
    ('sr', 'Serbian'),
    ('ss', 'Swati'),
    ('su', 'Sundanese'),
    ('sw', 'Swahili'),
    ('sv', 'Swedish'),
    ('ty', 'Tahitian'),
    ('ta', 'Tamil'),
    ('tt', 'Tatar'),
    ('te', 'Telugu'),
    ('tg', 'Tajik'),
    ('tl', 'Tagalog'),
    ('th', 'Thai'),
    ('bo', 'Tibetan'),
    ('ti', 'Tigrinya'),
    ('to', 'Tonga (Tonga Islands)'),
    ('tn', 'Tswana'),
    ('ts', 'Tsonga'),
    ('tk', 'Turkmen'),
    ('tr', 'Turkish'),
    ('tw', 'Twi'),
    ('ug', 'Uighur; Uyghur'),
    ('uk', 'Ukrainian'),
    ('ur', 'Urdu'),
    ('uz', 'Uzbek'),
    ('ve', 'Venda'),
    ('vi', 'Vietnamese'),
    ('vo', 'Volapük'),
    ('cy', 'Welsh'),
    ('wa', 'Walloon'),
    ('wo', 'Wolof'),
    ('xh', 'Xhosa'),
    ('yi', 'Yiddish'),
    ('yo', 'Yoruba'),
    ('za', 'Zhuang; Chuang'),
    ('zh', 'Chinese'),
    ('zu', 'Zulu')
]

languages = set()
for _, l in language_pairs:
  for elem in l.split("; "):
    languages.add(elem.lower())

def translate_content(content: str) -> tuple[bool, str]:
    try:
        translation = ''
        language = ''
        translation = get_translation(content)
        language = get_language(content)
        assert(language.lower() in languages)

    except:
        return (language, translation)
    
    if ("don't understand" in translation) or ("cannot" in translation) or ("can't" in translation):
            return True, content
    
    if language.lower() == 'english': return (True, content)
    return (False, translation)