import torch
import pandas as pd
import argparse
from transformers import pipeline
import logging
import random

WRONGS_PROMPT="/data/giulia/wrong_answer.txt"
FORWARD_PROMPT="/data/giulia/forward_answer.txt"
INPUT_CSV_FILE="dataset.csv"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2", type=bool, required=True, help="Decide whether to generate the bad answers with gpt2 or the new method")
    parser.add_argument("--two_ways", type=bool, required=True, help="Decide whether to generate two kinds of bad answers")
    

def generate_bad_answers(df, model, two_ways, wrongs=None, forwards=None):
    '''
    Generates output csv file from the input adding the bad answers with the old method, i.e. calling GPT2
    '''
    df['Bad answer'] = np.nan 

    #Import generator, in our case gpt2 for bad answers
    preamble1 = ""
    preamble2 = ""
    if model == "gpt2":
        df['Method'] = 'gpt2'
        generator = pipeline("text-generation", model="openai-community/gpt2")
        
    elif model == "llama3" and two_ways is False:
        df['Method'] = 'llama3, all wrong'
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        generator = transformers.pipeline("text-generation",
          model="meta-llama/Meta-Llama-3-8B-Instruct",
          model_kwargs={"torch_dtype": torch.bfloat16},
          device="cuda",
        )
        preamble1 = preamble1 + "Generate a wrong answer for this prompt \n : "
        preamble2 = preamble1
    elif model == "llama3" and two_ways is True and wrongs is not None and forwards is not None:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        generator = transformers.pipeline("text-generation",
          model="meta-llama/Meta-Llama-3-8B-Instruct",
          model_kwargs={"torch_dtype": torch.bfloat16},
          device="cuda",
        )
        preamble1 = preamble1 + wrongs
        preamble2 = preamble2 + forwards
        
    else:
        print("Error, choice is not valid")
    preambles = [preamble1, preamble2]
    logging.basicConfig(level=logging.ERROR)
    for index, row in df.iterrows():
        if pd.isna(row["Bad answer"]):
            print(f"The index is {index}")
            print(len(row['Prompt']))
            print("The good answer is")
            print(row['Good answer'])
            index = random.choice([0,1])
            if index==0:
                row['Method'] = "wrong answer"
            else:
                row['Method'] = "forward answer"
            preamble = preambles.get(index)            
            try:
                generated_text = generator(preamble + str(row["Prompt"]) + "\nTeacher:", num_return_sequences=1, max_new_tokens=52, max_length=1000)[0]["generated_text"][(len(str(row["Prompt"]))):]
            except:
                pass
            print("The bad one is")
            print(generated_text)
            df.at[index, "Bad answer"] = generated_text
            df.to_csv("tutorchat_dataset.csv")
        else:
            print(f"Iteration {index} skipped")
    
    return df

    


def main():
    args = get_args()
    torch.cuda.set_device(3)
    #Input csv file should be organized as to have two columns: ['Prompt', 'Good answer']
    df = pd.read_csv(INPUT_CSV_FILE, on_bad_lines = 'skip')
    print("#### \n Processing dataset \n ####")
    if args.two_ways is False:
    
        if args.gpt2:
            processed_df = generate_bad_answers(df, "gpt")
        else:
            processed_df = generate_bad_answers(df , "llama3")
            
    elif args.two_ways is True:
        with open(WRONGS_PROMPT, 'r') as file:
            wrongs = file.read()

            # Convert the file contents to a single string
            wrongs_string = str(wrongs)
        with open(FORWARD_PROMPT , 'r') as file:
            forwards = file.read()
            forwards_string = str(forwards)
        processed_df = generate_bad_answers(df, "llama3", True, wrongs, forwards)
    
    print("#### \n Running DPO \n ####")
    
    
if __name__ == "__main__":
    main()
