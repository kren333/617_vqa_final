from PIL import Image
from transformers import pipeline
import pdb
import json
import random
from tqdm import tqdm
from collections import Counter

def out_of_box_accuracy(model, indices):
    pass

if __name__ == "__main__":
    data = json.load(open('./v2_mscoco_val2014_annotations.json'))
    annotations = data['annotations']
    data2 = json.load(open('./v2_OpenEnded_mscoco_val2014_questions.json'))
    questions = data2['questions']
    questions = {x['question_id'] : {'image_id': x['image_id'], 'question': x['question']} for x in questions}
    # vqa_pipeline = pipeline("visual-question-answering", model="microsoft/git-large-vqav2")
    vqa_pipeline = pipeline("visual-question-answering", model="microsoft/git-base-textvqa")
    # vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base") # .927, 88 each, 85 w new metric
    # vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa") # .9 81 w new metric

    correct = 0
    # samples = 1000
    indices = [i for i in range(len(annotations))]
    random.shuffle(indices)

    for i in tqdm(range(len(indices))):
        annotation_ind = indices[i]
        annotation = annotations[annotation_ind]
        img_num = str(annotation['image_id']).zfill(12)
        question_num = annotation['question_id']
        image = Image.open("./val2014/COCO_val2014_{}.jpg".format(img_num))
        question = questions[question_num]['question']
        possible_answers = Counter(x['answer'] for x in annotation['answers'])
        x = vqa_pipeline(image, question, top_k=5)
        guess = x[0]['answer']
        pdb.set_trace()
        # print("GUESS: {}, \nPOSSIBLE ANSWERS: {}".format(guess, possible_answers))
        # if guess in possible_answers: correct += 1
        correct += min(possible_answers[guess]/3, 1)
    
    print("ACCURACY: {}".format(correct/len(annotations)))

    image =  Image.open("./train2014/COCO_train2014_000000000009.jpg")
    question = "Is there an elephant?"

    x = vqa_pipeline(image, question, top_k=3)
    pdb.set_trace()