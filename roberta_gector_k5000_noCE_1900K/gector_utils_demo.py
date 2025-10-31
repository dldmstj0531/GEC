
from gector_utils import GectorVocab

# Adjust these paths if your notebook uses a different folder.
vocab_dir = "/content"
verb_file = "/content/verb-form-vocab.txt"

voc = GectorVocab.from_files(vocab_dir, verb_file=verb_file)

print(f"#labels = {len(voc.labels)}")
print("First 10 labels:", voc.labels[:10])
print("KEEP id:", voc.keep_id)
if voc.d_tags:
    print(f"#d_tags = {len(voc.d_tags)} -> {voc.d_tags}")
if voc.non_padded_namespaces:
    print("non_padded_namespaces:", voc.non_padded_namespaces)
print("Verb table loaded:", voc.verb_forms is not None)

# quick toy apply
tokens  = ["he", "go", "to", "school"]
actions = ["$KEEP", "$TRANSFORM_VERB_VB_VBD", "$KEEP", "$KEEP"]
print("Input :", tokens)
print("Acts  :", actions)
print("Output:", voc.apply_actions(tokens, actions))
