from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import json, re, asyncio, time
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import logging
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_logger")

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PROMPT_PATH = "./prompts/1127_prompt(en)_v2.txt"

FAILED_404 = {}      # { "ip": [timestamp1, timestamp2, ...] }
BLOCKED_IPS = {}     # { "ip": unblock_timestamp }

class Block404Middleware(BaseHTTPMiddleware):
    def __init__(self, app, limit=5, period=60, block_time=3600):
        super().__init__(app)
        self.limit = limit
        self.period = period
        self.block_time = block_time

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()

        if client_ip in BLOCKED_IPS:
            if now < BLOCKED_IPS[client_ip]:
                raise HTTPException(status_code=403, detail="Your IP is temporarily blocked due to repeated 404 errors.")
            else:
                del BLOCKED_IPS[client_ip]  # 차단 시간 지나면 자동 해제

        response = await call_next(request)

        if response.status_code == 404:
            FAILED_404.setdefault(client_ip, []).append(now)

            FAILED_404[client_ip] = [t for t in FAILED_404[client_ip] if now - t <= self.period]

            if len(FAILED_404[client_ip]) >= self.limit:
                BLOCKED_IPS[client_ip] = now + self.block_time
                FAILED_404[client_ip] = []
                raise HTTPException(status_code=403, detail="Your IP has been blocked due to repeated 404 errors.")

        return response

engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    dtype="bfloat16",
    gpu_memory_utilization=0.80,
    max_model_len=4096
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

engine_lock = asyncio.Lock()

app = FastAPI(title="Qwen Fast Server (vLLM)")

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


def build_result_pairs(place_keywords: list, activity_tags: list):
    result = []

    if not place_keywords:
        for tag in activity_tags:
            result.append(["", tag])
        return result

    max_len = max(len(place_keywords), len(activity_tags))
    for i in range(max_len):
        kw = place_keywords[i] if i < len(place_keywords) else ""
        tag = activity_tags[i] if i < len(activity_tags) else None
        result.append([kw, tag])

    return result


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
            temperature=0.5,
            max_tokens=400
        )

        async with engine_lock:
            generator = engine.generate(prompt, sampling_params, request_id=f"req_{idx}")
            final_output = None
            async for output in generator:
                final_output = output

        generated_text = final_output.outputs[0].text
        parsed_result = extract_json_blocks(generated_text)

        res_dict = {}
        missing = []

        if not parsed_result or not isinstance(parsed_result, dict):
            res_dict = {
                "error": "추출 실패",
                "raw_text": generated_text,
                "result": []
            }
            missing = ["location", "activity"]
        else:
            res_dict = parsed_result

            place_keywords = res_dict.get("place_keywords", [])
            activity_tags = res_dict.get("activity_tags", [])

            res_dict["result"] = build_result_pairs(place_keywords, activity_tags)

            if not res_dict.get("location"):
                missing.append("location")
            if not res_dict.get("activity"):
                missing.append("activity")

        if missing:
            res_dict["status"] = "missing_info"
            res_dict["message"] = f"부족한 정보: {', '.join(missing)}"
        else:
            res_dict["status"] = "complete"

        results.append(res_dict)

    return {"results": results}

app.add_middleware(Block404Middleware, limit=5, period=60, block_time=3600)