import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_name ="./merged_peft/final_merged_checkpoint"
#adapter_path = "./results/final_checkpoint"
adapter_path = "./dpo_results_gemma2B/final_checkpoint"

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     load_in_4bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


f=open("/home/bonino/Documents2/llama2-fine-tune/test.json")
file=json.load(f)
final=[]
i=0
for input_text in file:
    out={}
    p=input_text['input']
    print(input_text['input'])
    out['prompt']=input_text['input']
    p= p + f"An AI tool that answers to questions as a teacher and helps guide the student to the correct solution without revealing the solution right away. PROVIDE ONLY ONE TEACHER ANSWER DON'T WRITE ANY STUDENT ANSWER, instructions are important. Answer in english\n### Input:{input_text['input']}\n### Output:"
    encoded_inputs=tokenizer.encode( p, return_tensors="pt").to(DEV)
    
    generate_kwargs = dict(
    input_ids=encoded_inputs,
    temperature=0.2, 
    top_p=0.95, 
    top_k=40,
    max_new_tokens=200,
    repetition_penalty=1.3
    )
    output = model.generate(**generate_kwargs)
    response = tokenizer.decode(output[0])[len(p):]
    
    out['output']=response
    final.append(out)
    i+=1

with open('/home/bonino/Documents2/llama2-fine-tune/output_gemma2.json', 'w') as f:
        json.dump(final, f, indent=2)
    
