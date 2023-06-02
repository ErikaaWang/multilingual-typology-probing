import os
from os import path
language = ["UD_Arabic", 'UD_Basque','UD_Catalan', 'UD_Chinese', "UD_Chinese-CFL", "UD_English", "UD_French", "UD_Hindi", "UD_Marathi", "UD_Portuguese","UD_Spanish","UD_Tamil","UD_Urdu","UD_Vietnamese"]
_DEFAULT_TREEBANKS_ROOT = path.join("./data", "ud/ud-treebanks-v2.1")

for lang in language:
    treebank_path = os.path.join(_DEFAULT_TREEBANKS_ROOT, lang)
    last_layer_path = path.join(treebank_path, 'last-layer')
    os.mkdir(last_layer_path)
    os.system(f"mv {treebank_path}/*-bloom-560m.pkl {last_layer_path}")

print("Done.")