import os
os.system('python3 -m pip install vllm  --quiet')

from vllm import LLM, SamplingParams

llm = LLM(model="LoneStriker/Smaug-72B-v0.1-GPTQ", dtype="float16",max_model_len=2048,gpu_memory_utilization=0.9)

prompts = [
	    "Hello, my name is",
	    "The president of the United States is",
	    "The capital of France is",
            "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\n Prompt: {prompt!r}, \n Answer: {generated_text!r}")



