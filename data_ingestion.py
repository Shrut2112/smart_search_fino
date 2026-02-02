import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import cpu_count
from graph.graph import main_graph  
from agents.universal_parser_agent import init_worker

def process_pdf_safe(pdf_path):
    """Safe wrapper with error handling"""
    pdf_path = Path(pdf_path)
    try:
        graph = main_graph()
        result = graph.invoke({"original_filename": str(pdf_path)})
        return {"success": True, "file": pdf_path.name, "result": result}
    except Exception as e:
        return {"success": False, "file": pdf_path.name, "error": str(e)}
    
def batch_ingest_pdfs(pdf_dir=r"Website_Backup", max_workers=2):
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs found!")
        return []

    workers = min(max_workers, cpu_count())
    print(f"🚀 Processing {len(pdf_files)} PDFs with {workers} PROCESSES")

    results = []
    
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as executor:
        # Pass strings instead of Path objects to ensure easy serialization
        futures = {executor.submit(process_pdf_safe, str(pdf)): pdf for pdf in pdf_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                res = future.result()
                if res is None:
                    print("Warning: A worker returned None. Check internal graph errors.")
                results.append(res)
            except Exception as e:
                print(f"\n❌ Process crashed for a file: {e}")
                results.append({"success": False, "error": str(e)})

    return results

# THE CRITICAL GUARD
if __name__ == "__main__":
    # Ensure any shared resources (like DB pools) are initialized here 
    # OR inside the worker process itself.
    final_results = batch_ingest_pdfs()