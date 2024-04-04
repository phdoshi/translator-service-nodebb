from src.translator import translate_content
from mock import patch
import vertexai
from sentence_transformmers import SentenceTransformer, util

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

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

    for ex in examples:
        eng, translated = translate_content(ex["post"])
        expected_lang, expected_translation = ex["expected_answer"]
        assert(eng == expected_lang)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        expected_encode = model.encode(expected_translation.lower())
        response_encode = model.encode(translated.lower())
        sim = util.cos_sim(expected_encode, response_encode)[0,0]
        assert(sim >= 0.9)

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_llm_gibberish_response(mocker):
    mocker.return_value.text = "I don't understand your request"
    examples = [{
        "post": "GIBBERSIHWEFadsflakwefjoawepfansg@@@9afdaslfkj###\\\\",
        "expected_answer": (True, "GIBBERSIHWEFadsflakwefjoawepfansg@@@9afdaslfkj###\\\\")
    }]

    for ex in examples:
        eng, translated = translate_content(ex["post"])
        expected_lang, expected_translation = ex["expected_answer"]
        assert(eng == expected_lang)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        expected_encode = model.encode(expected_translation.lower())
        response_encode = model.encode(translated.lower())
        sim = util.cos_sim(expected_encode, response_encode)[0,0]
        assert(sim >= 0.9)

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_unexpected_language(mocker):
    # we mock the model's response to return a random message
    mocker.return_value.text = "I don't understand your request"
    assert translate_content("GIBBERSIHWEFadsflakwefjoawepfansg@@@9afdaslfkj###\\\\") == (True,"GIBBERSIHWEFadsflakwefjoawepfansg@@@9afdaslfkj###\\\\")
    eng, translated = translate_content("Aquí está su primer ejemplo.")
    expected_translation = "This is your first example."
    assert(eng == False)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    expected_encode = model.encode(expected_translation.lower())
    response_encode = model.encode(translated.lower())
    sim = util.cos_sim(expected_encode, response_encode)[0,0]
    assert(sim >= 0.9)