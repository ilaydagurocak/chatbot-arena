import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from langdetect import detect, LangDetectException
import matplotlib.pyplot as plt

# Data preprocessing and feature engineering
def preprocess_data(df):
    df['prompt_len'] = df['prompt'].apply(len)
    df['response_a_len'] = df['response_a'].apply(len)
    df['response_b_len'] = df['response_b'].apply(len)
    df['prompt_word_count'] = df['prompt'].apply(lambda x: len(x.split()))
    df['response_a_word_count'] = df['response_a'].apply(lambda x: len(x.split()))
    df['response_b_word_count'] = df['response_b'].apply(lambda x: len(x.split()))
    return df

# Function to apply custom rules
def apply_custom_rules(prompt, response_a, response_b, expected_language='en'):
    winner_a, winner_b, winner_tie = 0, 0, 1
    tie = False
    
    ai_phrases = [
        "As an AI", "I don't have feelings","I don't","I do not","I can't","I am just",
        "I am just a machine", "I don't have preferences", "I don't have emotions",
        "I cannot form opinions", "I do not have personal experiences",
        "I don't have a favorite", "I don't have tastes", "I do not have personal views",
        "I don't have personal opinions", "I am not capable of feelings", "I don't have a favorite movie"
    ]
    ai_a = any(phrase in response_a for phrase in ai_phrases)
    ai_b = any(phrase in response_b for phrase in ai_phrases)
    if ai_a and not ai_b:
        winner_a, winner_b, winner_tie = 0, 1, 0
    elif ai_b and not ai_a:
        winner_a, winner_b, winner_tie = 1, 0, 0
    elif ai_a and ai_b:
        tie = True

    question_phrases = ["What is", "How do", "Can you", "Could you", "Explain", "Describe", "Why is", "Tell me", "Show me"]
    if any(phrase in prompt for phrase in question_phrases):
        explanation_phrases = ["Explanation:", "Let me explain", "Here is an explanation", "Allow me to elaborate"]
        explanation_a = any(phrase in response_a for phrase in explanation_phrases)
        explanation_b = any(phrase in response_b for phrase in explanation_phrases)
        if explanation_a and not explanation_b:
            winner_a, winner_b, winner_tie = 0, 1, 0
        elif explanation_b and not explanation_a:
            winner_a, winner_b, winner_tie = 1, 0, 0
        elif explanation_a and explanation_b:
            tie = True

    if " ".join(response_a.split()[:4]) == " ".join(response_b.split()[:4]):
        tie = True

    incorrect_phrases = ["incorrect", "wrong", "not true", "false", "error", "mistake"]
    incorrect_a = any(phrase in response_a for phrase in incorrect_phrases)
    incorrect_b = any(phrase in response_b for phrase in incorrect_phrases)
    if incorrect_a and not incorrect_b:
        winner_a, winner_b, winner_tie = 0, 1, 0
    elif incorrect_b and not incorrect_a:
        winner_a, winner_b, winner_tie = 1, 0, 0
    elif incorrect_a and incorrect_b:
        tie = True

    example_phrases = ["Here is an example:", "For example:", "Example:", "Such as", "Let's say"]
    example_a = any(phrase in response_a for phrase in example_phrases)
    example_b = any(phrase in response_b for phrase in example_phrases)
    if any(phrase in prompt for phrase in ["example", "sample", "show me how", "demonstrate"]):
        if example_a and not example_b:
            winner_a, winner_b, winner_tie = 0, 1, 0
        elif example_b and not example_a:
            winner_a, winner_b, winner_tie = 1, 0, 0
        elif example_a and example_b:
            tie = True

    keywords = prompt.split()
    keyword_a = any(keyword in response_a.split()[:4] for keyword in keywords)
    keyword_b = any(keyword in response_b.split()[:4] for keyword in keywords)
    if keyword_a and not keyword_b:
        winner_a, winner_b, winner_tie = 1, 0, 0
    elif keyword_b and not keyword_a:
        winner_a, winner_b, winner_tie = 0, 1, 0
    elif keyword_a and keyword_b:
        tie = True

    apologize_phrases = ["I apologize", "I don't", "I'm sorry", "I regret", "I cannot", "Unfortunately"]
    apologize_a = any(phrase in response_a for phrase in apologize_phrases)
    apologize_b = any(phrase in response_b for phrase in apologize_phrases)
    if apologize_a and not apologize_b:
        winner_a, winner_b, winner_tie = 0, 1, 0
    elif apologize_b and not apologize_a:
        winner_a, winner_b, winner_tie = 1, 0, 0
    elif apologize_a and apologize_b:
        tie = True

    step_phrases = ["Step 1:", "Step 2:", "Step 3:", "First,", "Second,", "Third,"]
    step_a = any(phrase in response_a for phrase in step_phrases)
    step_b = any(phrase in response_b for phrase in step_phrases)
    if step_a and not step_b:
        winner_a, winner_b, winner_tie = 0, 1, 0
    elif step_b and not step_a:
        winner_a, winner_b, winner_tie = 1, 0, 0
    elif step_a and step_b:
        tie = True

    dialogue_phrases = ["construct", "construct a", "create a dialogue", "write a dialogue", "create a conversation", "write a conversation", "create an argument", "write an argument", "create a story", "write a story", "create a rap battle", "write a rap battle"]
    if any(phrase in prompt for phrase in dialogue_phrases):
        dialogue_a_count = response_a.lower().count("said") + response_a.lower().count("replied") + response_a.lower().count("asked")
        dialogue_b_count = response_b.lower().count("said") + response_b.lower().count("replied") + response_b.lower().count("asked")
        if dialogue_a_count > dialogue_b_count:
            winner_a, winner_b, winner_tie = 1, 0, 0
        elif dialogue_b_count > dialogue_a_count:
            winner_a, winner_b, winner_tie = 0, 1, 0
        else:
            tie = True

    try:
        response_a_lang = detect(response_a)
    except LangDetectException:
        response_a_lang = None

    try:
        response_b_lang = detect(response_b)
    except LangDetectException:
        response_b_lang = None

    if response_a_lang != expected_language and response_b_lang == expected_language:
        winner_a, winner_b, winner_tie = 0, 1, 0
    elif response_b_lang != expected_language and response_a_lang == expected_language:
        winner_a, winner_b, winner_tie = 1, 0, 0
    elif response_a_lang != expected_language and response_b_lang != expected_language:
        tie = True

    if tie:
        return 0, 0, 1  # Tie
    return winner_a, winner_b, winner_tie

# Known languages and their codes
language_codes = {
    'ab': 'Abkhazian',
    'aa': 'Afar',
    'af': 'Afrikaans',
    'ak': 'Akan',
    'sq': 'Albanian',
    'am': 'Amharic',
    'ar': 'Arabic',
    'an': 'Aragonese',
    'hy': 'Armenian',
    'as': 'Assamese',
    'av': 'Avaric',
    'ae': 'Avestan',
    'ay': 'Aymara',
    'az': 'Azerbaijani',
    'bm': 'Bambara',
    'ba': 'Bashkir',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bn': 'Bengali',
    'bh': 'Bihari',
    'bi': 'Bislama',
    'bs': 'Bosnian',
    'br': 'Breton',
    'bg': 'Bulgarian',
    'my': 'Burmese',
    'ca': 'Catalan',
    'ch': 'Chamorro',
    'ce': 'Chechen',
    'ny': 'Chichewa',
    'zh': 'Chinese',
    'zh-cn': 'Simplified Chinese',
    'zh-tw': 'Traditional Chinese',
    'cv': 'Chuvash',
    'kw': 'Cornish',
    'co': 'Corsican',
    'cr': 'Cree',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'dv': 'Divehi',
    'nl': 'Dutch',
    'dz': 'Dzongkha',
    'en': 'English',
    'eo': 'Esperanto',
    'et': 'Estonian',
    'ee': 'Ewe',
    'fo': 'Faroese',
    'fj': 'Fijian',
    'fi': 'Finnish',
    'fr': 'French',
    'ff': 'Fulah',
    'gl': 'Galician',
    'ka': 'Georgian',
    'de': 'German',
    'el': 'Greek',
    'gn': 'Guarani',
    'gu': 'Gujarati',
    'ht': 'Haitian',
    'ha': 'Hausa',
    'he': 'Hebrew',
    'hz': 'Herero',
    'hi': 'Hindi',
    'ho': 'Hiri Motu',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'io': 'Ido',
    'ig': 'Igbo',
    'id': 'Indonesian',
    'ia': 'Interlingua',
    'ie': 'Interlingue',
    'iu': 'Inuktitut',
    'ik': 'Inupiaq',
    'ga': 'Irish',
    'it': 'Italian',
    'ja': 'Japanese',
    'jv': 'Javanese',
    'kl': 'Kalaallisut',
    'kn': 'Kannada',
    'kr': 'Kanuri',
    'ks': 'Kashmiri',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'ki': 'Kikuyu',
    'rw': 'Kinyarwanda',
    'ky': 'Kyrgyz',
    'kv': 'Komi',
    'kg': 'Kongo',
    'ko': 'Korean',
    'ku': 'Kurdish',
    'kj': 'Kwanyama',
    'lo': 'Lao',
    'la': 'Latin',
    'lv': 'Latvian',
    'lb': 'Luxembourgish',
    'li': 'Limburgish',
    'ln': 'Lingala',
    'lt': 'Lithuanian',
    'lu': 'Luba-Katanga',
    'lg': 'Ganda',
    'mk': 'Macedonian',
    'mg': 'Malagasy',
    'ms': 'Malay',
    'ml': 'Malayalam',
    'mt': 'Maltese',
    'gv': 'Manx',
    'mi': 'Maori',
    'mr': 'Marathi',
    'mh': 'Marshallese',
    'mn': 'Mongolian',
    'na': 'Nauru',
    'nv': 'Navajo',
    'nd': 'Northern Ndebele',
    'nr': 'Southern Ndebele',
    'ng': 'Ndonga',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'nb': 'Norwegian Bokmål',
    'nn': 'Norwegian Nynorsk',
    'ii': 'Nuosu',
    'oc': 'Occitan',
    'oj': 'Ojibwe',
    'or': 'Oriya',
    'om': 'Oromo',
    'os': 'Ossetian',
    'pi': 'Pali',
    'pa': 'Punjabi',
    'ps': 'Pashto',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'qu': 'Quechua',
    'ro': 'Romanian',
    'rm': 'Romansh',
    'rn': 'Kirundi',
    'ru': 'Russian',
    'se': 'Northern Sami',
    'sm': 'Samoan',
    'sg': 'Sango',
    'sa': 'Sanskrit',
    'sc': 'Sardinian',
    'sr': 'Serbian',
    'sn': 'Shona',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovene',
    'so': 'Somali',
    'st': 'Southern Sotho',
    'es': 'Spanish',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'ss': 'Swati',
    'sv': 'Swedish',
    'tl': 'Tagalog',
    'ty': 'Tahitian',
    'tg': 'Tajik',
    'ta': 'Tamil',
    'tt': 'Tatar',
    'te': 'Telugu',
    'th': 'Thai',
    'bo': 'Tibetan',
    'ti': 'Tigrinya',
    'to': 'Tonga',
    'ts': 'Tsonga',
    'tn': 'Tswana',
    'tr': 'Turkish',
    'tk': 'Turkmen',
    'tw': 'Twi',
    'ug': 'Uighur',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    've': 'Venda',
    'vi': 'Vietnamese',
    'vo': 'Volapük',
    'wa': 'Walloon',
    'cy': 'Welsh',
    'wo': 'Wolof',
    'fy': 'Western Frisian',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'za': 'Zhuang',
    'zu': 'Zulu',
}

# Load the training dataset
train_df = pd.read_csv('train.csv')

# Preprocess the training data
train_df = preprocess_data(train_df)

# Text vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

X_prompt = tfidf_vectorizer.fit_transform(train_df['prompt'])
X_response_a = tfidf_vectorizer.transform(train_df['response_a'])
X_response_b = tfidf_vectorizer.transform(train_df['response_b'])

# Combine features
X = np.hstack((X_prompt.toarray(), X_response_a.toarray(), X_response_b.toarray(),
               train_df[['prompt_len', 'response_a_len', 'response_b_len', 'prompt_word_count', 'response_a_word_count', 'response_b_word_count']].values))

# Target variables
y_a = train_df['winner_model_a']
y_b = train_df['winner_model_b']
y_tie = train_df['winner_tie']

# Split into training and validation sets
X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(X, y_a, test_size=0.2, random_state=42)
X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X, y_b, test_size=0.2, random_state=42)
X_train_tie, X_val_tie, y_train_tie, y_val_tie = train_test_split(X, y_tie, test_size=0.2, random_state=42)

# Create models
model_a = XGBClassifier(eval_metric="logloss", early_stopping_rounds=10, reg_alpha=0.1, reg_lambda=0.1)
model_b = XGBClassifier(eval_metric="logloss", early_stopping_rounds=10, reg_alpha=0.1, reg_lambda=0.1)
model_tie = XGBClassifier(eval_metric="logloss", early_stopping_rounds=10, reg_alpha=0.1, reg_lambda=0.1)

# Train the models
model_a.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)], verbose=True)
model_b.fit(X_train_b, y_train_b, eval_set=[(X_val_b, y_val_b)], verbose=True)
model_tie.fit(X_train_tie, y_train_tie, eval_set=[(X_val_tie, y_val_tie)], verbose=True)

# Evaluate the models
y_pred_a = model_a.predict(X_val_a)
y_pred_b = model_b.predict(X_val_b)
y_pred_tie = model_tie.predict(X_val_tie)

# Calculate log loss
y_pred_proba_a = model_a.predict_proba(X_val_a)
y_pred_proba_b = model_b.predict_proba(X_val_b)
y_pred_proba_tie = model_tie.predict_proba(X_val_tie)

log_loss_a = log_loss(y_val_a, y_pred_proba_a)
log_loss_b = log_loss(y_val_b, y_pred_proba_b)
log_loss_tie = log_loss(y_val_tie, y_pred_proba_tie)

print(f'Log Loss for model_a: {log_loss_a}')
print(f'Log Loss for model_b: {log_loss_b}')
print(f'Log Loss for model_tie: {log_loss_tie}')

# Calculate general log loss
y_true_combined = np.concatenate((y_val_a, y_val_b, y_val_tie))
y_pred_proba_combined = np.concatenate((y_pred_proba_a, y_pred_proba_b, y_pred_proba_tie), axis=0)

general_log_loss = log_loss(y_true_combined, y_pred_proba_combined)
print(f'General Log Loss: {general_log_loss}')

# Visualize log loss
train_logloss_a = model_a.evals_result()['validation_0']['logloss']
val_logloss_a = model_a.evals_result()['validation_0']['logloss']
train_logloss_b = model_b.evals_result()['validation_0']['logloss']
val_logloss_b = model_b.evals_result()['validation_0']['logloss']
train_logloss_tie = model_tie.evals_result()['validation_0']['logloss']
val_logloss_tie = model_tie.evals_result()['validation_0']['logloss']

plt.figure(figsize=(12, 6))
plt.plot(train_logloss_a, label='Train Log Loss (Model A)')
plt.plot(val_logloss_a, label='Validation Log Loss (Model A)')
plt.plot(train_logloss_b, label='Train Log Loss (Model B)')
plt.plot(val_logloss_b, label='Validation Log Loss (Model B)')
plt.plot(train_logloss_tie, label='Train Log Loss (Model Tie)')
plt.plot(val_logloss_tie, label='Validation Log Loss (Model Tie)')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.title('Training and Validation Log Loss')
plt.show()

# Load the test dataset
test_df = pd.read_csv('test.csv')

# Preprocess the test data
test_df = preprocess_data(test_df)

# Text vectorization for test data
X_test_prompt = tfidf_vectorizer.transform(test_df['prompt'])
X_test_response_a = tfidf_vectorizer.transform(test_df['response_a'])
X_test_response_b = tfidf_vectorizer.transform(test_df['response_b'])

# Combine test features
X_test = np.hstack((X_test_prompt.toarray(), X_test_response_a.toarray(), X_test_response_b.toarray(),
                    test_df[['prompt_len', 'response_a_len', 'response_b_len', 'prompt_word_count', 'response_a_word_count', 'response_b_word_count']].values))

# Making predictions on test data
test_pred_proba_a = model_a.predict_proba(X_test)[:, 1]
test_pred_proba_b = model_b.predict_proba(X_test)[:, 1]
test_pred_proba_tie = model_tie.predict_proba(X_test)[:, 1]

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'winner_model_a': test_pred_proba_a,
    'winner_model_b': test_pred_proba_b,
    'winner_tie': test_pred_proba_tie
})

# Save the submission DataFrame to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")