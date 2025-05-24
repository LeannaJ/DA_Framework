import os
import datetime
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agent import run_agent, create_notebook_from_codes
from typing import List, Optional
from pydantic import BaseModel

load_dotenv()

UPLOAD_FOLDER = './tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunPipelineRequest(BaseModel):
    filepath: str
    target_variable: str
    independent_variables: List[str]
    modelling_context: str
    dataset_description: Optional[str] = ""
    # llm_type: Optional[str] = None 

def get_new_session_id():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

@app.post("/analyze")
async def analyze_data(
    modelling_context: str = Form(...),
    dataset: UploadFile = File(...),
    dataset_description: str = Form(...),
    target_variable: str = Form(...),
    independent_variables: str = Form(...),
):
    try:
        session_id = get_new_session_id()
        filename = dataset.filename
        filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
        with open(filepath, "wb") as buffer:
            buffer.write(await dataset.read())

        user_inputs = {
            'input_file': filepath,
            'target_variable': target_variable,
            'independent_variables': independent_variables.split(','),
            'modelling_context': modelling_context,
            'dataset_description': dataset_description,
            'session_id': session_id,
            'uploaded_file_name': filename
        }

        code_dict = run_agent(user_inputs, filepath, llm_type='openai')

        # Example: Extract notebook paths for each step
        ic_nb = code_dict.get("ic_notebook")
        eda_nb = code_dict.get("eda_notebook")
        mod_nb = code_dict.get("mod_notebook")

        # Example: Return notebook paths for each step
        return {
            "ic_notebook": ic_nb,
            "eda_notebook": eda_nb,
            "mod_notebook": mod_nb
        }
        # Or return specific steps as FileResponse:
        # return FileResponse(ic_nb, media_type='application/x-ipynb+json', filename=f'ic_{session_id}.ipynb')

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post('/run')
async def run_pipeline(body: RunPipelineRequest):
    session_id = get_new_session_id()
    uploaded_file_name = os.path.basename(body.filepath)
    user_inputs = {
        'input_file': body.filepath,
        'target_variable': body.target_variable,
        'independent_variables': body.independent_variables,
        'modelling_context': body.modelling_context,
        'dataset_description': body.dataset_description,
        'session_id': session_id,
        'uploaded_file_name': uploaded_file_name
    }
    try:
        llm_type = 'openai'  # Always use openai
        result = run_agent(user_inputs, body.filepath, llm_type='openai')
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})