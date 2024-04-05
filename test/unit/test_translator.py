from src.translator import translate_content
from mock import patch
import vertexai
from sentence_transformers import SentenceTransformer, util

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    expected_translation = "This is a Chinese message"
    assert is_english == False
    model = SentenceTransformer("all-MiniLM-L6-v2")
    expected_encode = model.encode(expected_translation.lower())
    response_encode = model.encode(translated_content.lower())
    sim = util.cos_sim(expected_encode, response_encode)[0,0]
    assert(sim >= 0.75)


def test_llm_normal_response():
    examples =     [
        {
            "post": "What's one more example to write? We'll keep writing this example until we're almost done....We're almost there....I guess it can still be a bit longer. That would be alright. It would be nicer if I had more ideas about tests.",
            "expected_answer": (True, "What's one more example to write? We'll keep writing this example until we're almost done....We're almost there....I guess it can still be a bit longer. That would be alright. It would be nicer if I had more ideas about tests.")
        },
        {
            "post": "Aquí está su primer ejemplo.",
            "expected_answer": (False, "This is your first example.")
        },
        {
            "post": "ma asmuk?",
            "expected_answer" : (False, "What is your name?")
        },
    ]
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for ex in examples:
        eng, translated = translate_content(ex["post"])
        expected_lang, expected_translation = ex["expected_answer"]
        assert(eng == expected_lang)
        expected_encode = model.encode(expected_translation.lower())
        response_encode = model.encode(translated.lower())
        sim = util.cos_sim(expected_encode, response_encode)[0,0]
        assert(sim >= 0.5)

@patch('google.generativeai.GenerativeModel.generate_content')
def test_llm_gibberish_response(mocker):
    mocker.return_value.text = "I don't understand your request"
    examples = [{
        "post": "GIBBERSIHWEFadsflakwefjoawepfansg@@@9afdaslfkj###\\\\",
        "expected_answer": (False, "Sorry, we couldn't currently parse this. Please try again later!")
    }]

    for ex in examples:
        eng, translated = translate_content(ex["post"])
        expected_lang, expected_translation = ex["expected_answer"]
        assert(eng == expected_lang)
        assert(expected_translation == translated)

@patch('google.generativeai.GenerativeModel.generate_content')
def test_unexpected_language(mocker):
    # we mock the model's response to return a random message, and the response should be the same default
    mocker.return_value.text = "I don't understand your request"
    assert translate_content("GIBBERSIHWEFadsflakwefjoawepfansg@@@9afdaslfkj###\\\\") == (False, "Sorry, we couldn't currently parse this. Please try again later!")

    eng, translated = translate_content("Aquí está su primer ejemplo.")
    expected_translation = "Sorry, we couldn't currently parse this. Please try again later!"
    
    assert(eng == False)
    assert(expected_translation == translated)