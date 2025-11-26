from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import json, re, asyncio
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_logger")

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# [중요] 프롬프트 파일 내용이 v2로 업데이트되어야 합니다.
DEFAULT_PROMPT_PATH = "./prompts/1126_prompt(en).txt" 

# -------------------------------
# vLLM 엔진 초기화
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=4096
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

engine_lock = asyncio.Lock()

app = FastAPI(title="Qwen Fast Server (vLLM)")

# -------------------------------
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_json_blocks(text: str):
    """
    LLM 응답 텍스트에서 JSON 블록을 찾아 파싱합니다.
    """
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
                    # Trailing comma 제거 (JSON 문법 오류 방지)
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

        # Sampling Params: 정형화된 출력을 위해 temperature를 낮게 유지
        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=400
        )

        async with engine_lock:
            # logger.info("Waiting for vLLM engine lock...")
            generator = engine.generate(prompt, sampling_params, request_id=f"req_{idx}")

            final_output = None
            async for output in generator:
                final_output = output

        generated_text = final_output.outputs[0].text
        parsed_result = extract_json_blocks(generated_text)

        res_dict = {}
        missing = []

        if not parsed_result or not isinstance(parsed_result, dict):
            # 파싱 실패 시
            res_dict = {"error": "추출 실패", "raw_text": generated_text}
            missing = ["location", "activity"]
        else:
            res_dict = parsed_result
            
            # [수정됨] 1. place_keywords 키 보장 (Schema Enforcement)
            # LLM이 실수로 키를 누락해도 빈 리스트([])로 초기화하여 에러 방지
            if "place_keywords" not in res_dict:
                res_dict["place_keywords"] = []

            # [수정됨] 2. 필수 정보 검증 (Validation)
            # location은 '지도 중심점'이므로 필수 (null이면 missing 처리)
            if not res_dict.get("location"): 
                missing.append("location")
            
            # activity는 필수
            if not res_dict.get("activity"): 
                missing.append("activity")
            
            # 주의: place_keywords는 빈 리스트([])일 수 있으므로 missing 체크에서 제외함
            # 예: "공원 산책할래" -> location:공원, place_keywords:[] (정상)

        if missing:
            res_dict["status"] = "missing_info"
            res_dict["message"] = f"부족한 정보: {', '.join(missing)}"
        else:
            res_dict["status"] = "complete"

        results.append(res_dict)

    return {"results": results}