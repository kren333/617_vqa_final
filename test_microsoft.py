from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import pdb
import json
import random
from tqdm import tqdm
import torch
from collections import Counter

if __name__ == "__main__":
    data = json.load(open('./v2_mscoco_val2014_annotations.json'))
    annotations = data['annotations']
    data2 = json.load(open('./v2_OpenEnded_mscoco_val2014_questions.json'))
    questions = data2['questions']
    questions = {x['question_id'] : {'image_id': x['image_id'], 'question': x['question']} for x in questions}

    processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")

    indices = [i for i in range(len(annotations))]
    random.shuffle(indices)

    correct = 0

    ## old logic 
    for i in tqdm(range(len(indices))):
        annotation_ind = indices[i]
        annotation = annotations[annotation_ind]
        img_num = str(annotation['image_id']).zfill(12)
        question_num = annotation['question_id']
        image = Image.open("./val2014/COCO_val2014_{}.jpg".format(img_num))
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        question = questions[question_num]['question']
        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        guess = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        question_idx = guess.find("?")
        guess = guess[question_idx+1:].strip() if question_idx != -1 else "idk man"

        possible_answers = Counter(x['answer'] for x in annotation['answers'])

        correct += min(possible_answers[guess]/3, 1)
    
    print("ACCURACY: {}".format(correct/len(annotations)))

    pdb.set_trace()