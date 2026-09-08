import json, os

# Check key formats
with open(r'E:\Saad_audi deepfakes\_casr\results\step2b_probs\clean.json') as f:
    clean = json.load(f)
with open(r'E:\Saad_audi deepfakes\_casr\results\step2b_probs\h264_crf38.json') as f:
    crf38 = json.load(f)

ck = list(clean.keys())[:3]
pk = list(crf38.keys())[:3]
print('clean keys:', ck)
print('crf38 keys:', pk)

# Check matching
for k in list(crf38.keys())[:5]:
    name = k.split(os.sep)[-1]
    print(f'crf38 key: {k}, name: {name}')
    # Find in clean
    matches = [ck for ck in clean if ck.endswith(name)]
    if matches:
        print(f'  match: {matches[0]}')
    else:
        print(f'  NO MATCH')