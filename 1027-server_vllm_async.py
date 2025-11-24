from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import json, re, asyncio
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_logger")

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PROMPT_PATH = "./prompts/1027_prompt(en).txt"

# -------------------------------
# vLLM 엔진 초기화
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=4096
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# 동시 요청을 순차 처리하기 위한 Lock
engine_lock = asyncio.Lock()

app = FastAPI(title="Qwen Fast Server (vLLM)")

# -------------------------------
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_json_blocks(text: str):
    stack, start = [], None
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack: start = i
            stack.append(ch)
        elif ch in "]}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    json_str = text[start:i + 1]
                    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
                    try: return json.loads(json_str)
                    except: return None
    return None

# -------------------------------
class InputData(BaseModel):
    sentences: Union[str, List[str]]

@app.post("/extract")
async def extract_info(data: InputData):
    sentences = data.sentences if isinstance(data.sentences, list) else [data.sentences]
    prompt_template = load_prompt(DEFAULT_PROMPT_PATH)
    
    results = []
    for idx, s in enumerate(sentences):
        logger.info(f"[INPUT] {s}")

        prompt = prompt_template.replace("{sentence}", s)

        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=400
        )

        async with engine_lock:
            logger.info("Waiting for vLLM engine lock...")
            generator = engine.generate(prompt, sampling_params, request_id=f"req_{idx}")

            final_output = None
            async for output in generator:
                final_output = output

        generated_text = final_output.outputs[0].text
        parsed_result = extract_json_blocks(generated_text)

        missing = []
        if not parsed_result or not isinstance(parsed_result, dict):
            missing = ["location", "activity"]
        else:
            if not parsed_result.get("location"): missing.append("location")
            if not parsed_result.get("activity"): missing.append("activity")

        res_dict = parsed_result if parsed_result else {"error": "추출 실패"}
        if missing:
            res_dict["status"] = "missing_info"
            res_dict["message"] = f"부족한 정보: {', '.join(missing)}"
        else:
            res_dict["status"] = "complete"

        results.append(res_dict)

    return {"results": results}
