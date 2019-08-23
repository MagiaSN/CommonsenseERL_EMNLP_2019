# Preprocessing

First, prepare the NYT corpus for pretraining. Run the following scripts sequentially:

- `nyt_untar.sh` Untar the NYT corpus

- `nyt_extract_raw_text.sh` Extract raw text from xml

- `nyt_extract_event.sh` Extract events with ollie

- `nyt_doc_on_line.py` Make events from the same document on the same line

- `nyt_sample_negative.py` Negative sampling

- `nyt_generate_dataset_ep.py` Generate dataset for "event prediction" training objective

- `nyt_generate_dataset_wp.py` Generate dataset for "word prediction" training objective

Second, prepare the ATOMIC dataset for joint training with intent and sentiment:

- `atomic_filter_intent_react.py` Filter instances with intent and react

- `atomic_build_svo.py` Change to svo format

- `atomic_sample_negative.py` Sample negative intents and reacts

- `atomic_generate_dataset.py` Generate dataset

- `atomic_senti.py` Convert react to sentiment using SenticNet
