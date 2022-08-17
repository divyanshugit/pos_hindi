# Parts of Speech tagger for Hindi

postagger.py
------------

`postagger.py` is a part of speech tagger for Hindi which is built using [NLTK](https://www.nltk.org/index.html).
I took the inspiration from [this](https://github.com/JayeshSuryavanshi/POS-Tagger-for-Hindi-Language)
repository. It maps the tags of word according token generated by any frequently used tokenizer and it gives
a list of pos tags.

Example:

```py
from posttagger import pos_tag

list_of_pos_tags = pos_tag(sentence,tokenizer)
```
