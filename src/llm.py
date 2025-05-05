import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from retriever import retrieve_context

# 3. Choose Mistral 7B
GEN_MODEL = "mistralai/Mistral-7B-v0.1"

# 4. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)

if torch.cuda.is_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
    model.to("cpu")

model.eval()

# 5. Initialize LLM
llm = HuggingFaceLLM(
    model=model, tokenizer=tokenizer,
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] </s>")
)

context = retrieve_context(combined, k=3)  # you can bump k up to 5 safely

# 7. Create Query Engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm
)

def generate_question(prompt_type: str, user_answer: str = "") -> str:
    if prompt_type == "initial":
        query = "Generate an initial interview question:"
    else:
         query = f"Generate a follow-up question based on this answer: {user_answer}"

    response = query_engine.query(query)
    return response.response

"""import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig
from retriever import retrieve_context
import os


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── 1) MODEL SELECTION ───────────────────────────────────────
GEN_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# ── 2) LOAD TOKENIZER & MODEL ───────────────────────────────
tokenizer = LlamaTokenizer.from_pretrained(GEN_MODEL, use_auth_token="YOUR_TOKEN")
if torch.cuda.is_available():
    # GPU path: 8-bit quantize + device_map
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    model = LlamaForCausalLM.from_pretrained(
        GEN_MODEL,
        use_auth_token="YOUR_TOKEN",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
else:
    # CPU path: no quantization
    model = LlamaForCausalLM.from_pretrained(
        GEN_MODEL,
        use_auth_token="YOUR_TOKEN",
        device_map=None,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.to("cpu")

model.eval()

# Confirm the new window:
print(f"Context window size: {model.config.max_position_embeddings} tokens")

def generate_question(prompt_type: str, combined: str, user_answer: str = "") -> str:
    # ── 3) Retrieve RAG context ────────────────────────────────
    context = retrieve_context(combined, k=5)  # you can bump k up to 5 safely

    # ── 4) Build prompt text ──────────────────────────────────
    if prompt_type == "initial":
        prompt = (
            "### System:\nYou are a technical interviewer.\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Question:\n"
        )
    else:
        prompt = (
            "### System:\nYou are a technical interviewer.\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Candidate Answer:\n"
            f"{user_answer}\n\n"
            "### Next Question:\n"
        )

    # ── 5) Tokenize & truncate to fit the new window ───────────
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ── 6) Generate new tokens ─────────────────────────────────
    out = model.generate(
        **inputs,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
    )

    # ── 7) Decode & strip prompt echo ──────────────────────────
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Split on our marker
    if "### Question:" in text:
        return text.split("### Question:")[-1].strip()
    if "### Next Question:" in text:
        return text.split("### Next Question:")[-1].strip()
    return text.strip()
"""

"""import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import retrieve_context

# have to look up an audio based model later
GEN_MODEL = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)

# Use GPU if available, otherwise fall back to CPU
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_question(prompt_type: str, combined: str, user_answer: str = "") -> str:
    context = retrieve_context(combined, k=3)

    if prompt_type == "initial":
        prompt = (
            "You are a technical interviewer. Use the context below to generate "
            "the first interview question:\n\n"
            f"{context}\n\nQuestion:"
        )
    else:
        prompt = (
            "You are a technical interviewer reviewing the candidate's answer. "
            "Use the context and the answer provided by the candidate to produce either a follow-up question "
            "for clarity or move on to the next interview question.\n\n"
            f"Context:\n{context}\n\n"
            f"Candidate answer: {user_answer}\n\nNext question:"
        )
        # 3) Tokenize (this *should* truncate, but we'll double-check below)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length  # usually 2048 for GPT-Neo
    )

    # 4) Move tensors to GPU/CPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5) **Manually clamp** to the last 2048 tokens if still too long
    max_len = tokenizer.model_max_length
    seq_len = inputs["input_ids"].size(1)
    if seq_len > max_len:
        # keep only the last max_len tokens of both input_ids & attention_mask
        inputs["input_ids"] = inputs["input_ids"][:, -max_len:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_len:]

    # 6) Generate only new tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 7) Decode & strip off the prompt echo
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Question:" in full:
        return full.split("Question:")[-1].strip()
    if "Next question:" in full:
        return full.split("Next question:")[-1].strip()
    return full.strip()"""