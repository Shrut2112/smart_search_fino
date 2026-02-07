import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import cpu_count
from graph.graph import main_graph  
from agents.universal_parser_agent import init_worker
import fitz  # PyMuPDF
from datetime import datetime   

MAX_PAGES = 200

def is_pdf_too_large(pdf_path):
    with fitz.open(pdf_path) as doc:
        return doc.page_count > MAX_PAGES
    
def process_pdf_safe(pdf_path):
    """Safe wrapper with error handling"""
    pdf_path = Path(pdf_path)
    try:
        if is_pdf_too_large(pdf_path):
            with open("skipped_files.log", "a") as log_file:
                log_file.write(f"{pdf_path.name}\n")
            return {
                "success": False,
                "file": pdf_path.name,
                "error": f"Skipped: >{MAX_PAGES} pages"
            }
        graph = main_graph()
        result = graph.invoke({"original_filename": str(pdf_path)})
        with open("processed_files.log", "a") as log_file:
            log_file.write(f"{pdf_path.name}:\t\t {datetime.now()}\n\n")
        return {"success": True, "file": pdf_path.name, "result": result}
    except Exception as e:
        with open("error_files.log", "a") as log_file:
            log_file.write(f"{pdf_path.name}:\t\t {str(e)}\n\n")
        return {"success": False, "file": pdf_path.name, "error": str(e)}
    
from multiprocessing import Pool, cpu_count

def batch_ingest_pdfs(pdf_dir=r"Website_Backup", max_workers=2):
    pdf_files = [str(p) for p in Path(pdf_dir).glob("*.pdf")]
    
    if not pdf_files:
        return []

    with Pool(processes=max_workers, initializer=init_worker, maxtasksperchild=50) as pool:
        results = []
        try:
            
            for res in tqdm(pool.imap_unordered(process_pdf_safe, pdf_files), 
                           total=len(pdf_files), desc="Processing"):
                results.append(res)
        except Exception as e:
            print(f"Pool crashed: {e}")
            
    return results

if __name__ == "__main__":
    # Ensure any shared resources (like DB pools) are initialized here 
    # OR inside the worker process itself.
    final_results = batch_ingest_pdfs()