import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI, HuggingFaceHub
from langchain.chains import LLMChain
import pandas as pd
from dotenv import load_dotenv
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import subprocess
from fpdf import FPDF

load_dotenv()

# --- Session & Path Management ---
def get_session_output_dir(session_id, step):
    """Return the output directory for a given session and step (e.g., 'eda', 'modeling', etc.)"""
    base_dir = f"{step}_output/{session_id}"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def get_cleaned_file_path(session_id, uploaded_file_name):
    """Return the path to the cleaned file after inspection/cleaning step."""
    return f"ic_full_output/{session_id}/cleaned_final_{uploaded_file_name}.csv"

# --- Template Loading & Placeholder Replacement ---
def load_code_template(template_path, user_inputs):
    """Load a code template and replace placeholders with user_inputs values."""
    with open(template_path, 'r', encoding='utf-8') as f:
        code = f.read()
    # Replace placeholders like {target_variable}, {session_id}, etc.
    for key, value in user_inputs.items():
        code = code.replace(f'{{{key}}}', str(value))
    return code

# --- Example user_inputs structure ---
# user_inputs = {
#     'input_file': '/uploads/NYC-BikeShare-2015-2017-combined.csv',
#     'target_variable': 'Gender',
#     'independent_variables': ['Start Time', 'Trip_duration_in_min', 'User Type'],
#     'modelling_context': "Your task is to predict 'Gender'...",
#     'dataset_description': 'The dataset contains anonymized trip information...',
#     'session_id': 'abc123',
#     'uploaded_file_name': 'NYC-BikeShare-2015-2017-combined.csv'
# }

# --- Gemini LLM Wrapper ---
try:
    from langchain_community.llms import Gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def get_llm(llm_type='openai'):
    if llm_type == 'gemini':
        # Gemini 지원이 되면 이 부분 활성화
        # ...
        raise ImportError("Gemini LLM is not available.")
    elif llm_type == 'openai' or llm_type is None:
        return OpenAI(temperature=0)
    else:
        return HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0})

# --- System prompt for LLM ---
SYSTEM_PROMPT = (
    "You are an expert Python data scientist. "
    "Your task is to read the provided code template and, based on the uploaded dataset and user modeling context, "
    "dynamically modify all dataset-dependent parts to generate a final, executable code file. "
    "Ensure the code is robust, reproducible, and tailored to the user's dataset and objectives. "
    "Only output the final, complete Python code file."
)

def llm_generate_code(template_path, user_inputs, llm_type='openai'):
    llm = get_llm(llm_type)
    template_code = load_code_template(template_path, user_inputs)
    # Compose the prompt: system instruction + template code + context
    prompt_text = SYSTEM_PROMPT + "\n\n" + template_code + "\n\n" + f"# Context: {user_inputs}"
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{prompt}"))
    code = chain.run({"prompt": prompt_text})
    return code

def inspection_and_cleaning_step(user_inputs, llm_type='openai'):
    template_path = os.path.join('code_template', 'ic.py')
    code = llm_generate_code(template_path, user_inputs, llm_type)
    # Optionally, save or execute the code here
    return code

def eda_step(user_inputs, llm_type='openai'):
    template_path = os.path.join('code_template', 'eda.py')
    code = llm_generate_code(template_path, user_inputs, llm_type)
    return code

def modeling_step(user_inputs, llm_type='openai'):
    template_path = os.path.join('code_template', 'mod.py')
    code = llm_generate_code(template_path, user_inputs, llm_type)
    return code

def report_step(user_inputs, llm_type='openai'):
    template_path = os.path.join('code_template', 'report.py')
    code = llm_generate_code(template_path, user_inputs, llm_type)
    return code

def run_agent(user_inputs, data_path, llm_type='openai'):
    ic_code = inspection_and_cleaning_step(user_inputs, llm_type)
    eda_code = eda_step(user_inputs, llm_type)
    mod_code = modeling_step(user_inputs, llm_type)
    report_code = report_step(user_inputs, llm_type)

    session_id = user_inputs['session_id']
    ic_nb = create_step_notebook(ic_code, session_id, 'ic')
    eda_nb = create_step_notebook(eda_code, session_id, 'eda')
    mod_nb = create_step_notebook(mod_code, session_id, 'mod')
    # (report 단계도 필요하면 추가)

    return {
        "inspection_cleaning_code": ic_code,
        "eda_code": eda_code,
        "modeling_code": mod_code,
        "report_code": report_code,
        "ic_notebook": ic_nb,
        "eda_notebook": eda_nb,
        "mod_notebook": mod_nb
    }

def create_notebook_from_codes(code_dict, session_id):
    """
    Combine generated code for all steps into a single Jupyter notebook file.
    Each step's code is a separate code cell, with markdown headers.
    Returns the notebook file path.
    """
    nb = new_notebook()
    cells = []
    # Add a title
    cells.append(new_markdown_cell(f"# Automated Prediction Modeling Pipeline\nSession: {session_id}"))
    # Add each step as a code cell with a markdown header
    step_titles = [
        ("inspection_cleaning_code", "## 1. Inspection & Cleaning"),
        ("eda_code", "## 2. EDA"),
        ("modeling_code", "## 3. Modeling"),
        ("report_code", "## 4. Report Generation")
    ]
    for key, title in step_titles:
        if key in code_dict and code_dict[key]:
            cells.append(new_markdown_cell(title))
            cells.append(new_code_cell(code_dict[key]))
    nb.cells = cells
    # Save notebook to session output dir
    output_dir = get_session_output_dir(session_id, 'notebook')
    notebook_path = os.path.join(output_dir, f'final_pipeline_{session_id}.ipynb')
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    return notebook_path

def create_step_notebook(code, session_id, step_name):
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(f"# {step_name.capitalize()} Step\nSession: {session_id}"),
        new_code_cell(code)
    ]
    output_dir = get_session_output_dir(session_id, step_name)
    notebook_path = os.path.join(output_dir, f"{step_name}_{session_id}.ipynb")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    return notebook_path

def save_code_to_file(code, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)

def execute_python_file(file_path):
    """Return the execution result (standard output, error)"""
    result = subprocess.run(
        ['python', file_path],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr

def generate_pdf_report(session_id, eda_summary, modeling_summary, eda_images=None, modeling_images=None):
    output_dir = get_session_output_dir(session_id, 'report')
    pdf_path = os.path.join(output_dir, f'result_report_{session_id}.pdf')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="AI Prediction Modeling Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt="1. EDA Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, eda_summary)
    if eda_images:
        for img in eda_images:
            pdf.image(img, w=150)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="2. Modeling Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, modeling_summary)
    if modeling_images:
        for img in modeling_images:
            pdf.image(img, w=150)
    pdf.output(pdf_path)
    return pdf_path

def run_and_report(code_dict, session_id):
    # 1. Save each step's code to a file and execute
    output_dir = get_session_output_dir(session_id, 'executed_code')
    os.makedirs(output_dir, exist_ok=True)
    step_files = {}
    step_outputs = {}
    for step, code in code_dict.items():
        if code:
            file_path = os.path.join(output_dir, f'{step}_{session_id}.py')
            save_code_to_file(code, file_path)
            stdout, stderr = execute_python_file(file_path)
            step_files[step] = file_path
            step_outputs[step] = {'stdout': stdout, 'stderr': stderr}
    # 2. Extract EDA/Modeling summaries (e.g., extract summary text from stdout, or read from file)
    eda_summary = step_outputs.get('eda_code', {}).get('stdout', '')
    modeling_summary = step_outputs.get('modeling_code', {}).get('stdout', '')
    # 3. (Optional) Collect paths to key image files
    eda_images = []  # e.g., os.path.join(output_dir, 'eda_output', session_id, 'some_plot.png')
    modeling_images = []
    # 4. Create PDF report
    pdf_path = generate_pdf_report(session_id, eda_summary, modeling_summary, eda_images, modeling_images)
    return {
        'executed_code_files': step_files,
        'step_outputs': step_outputs,
        'pdf_report_path': pdf_path
    } 