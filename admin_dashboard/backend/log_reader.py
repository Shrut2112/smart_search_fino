from pathlib import Path

def read_logs(log_file_path: str, lines=100, level_filter=None, search_text=None):
    path = Path(log_file_path)
    if not path.exists():
        return [f"Log file not found: {log_file_path}"]
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            
        filtered = []
        for line in all_lines:
            if level_filter and level_filter != "ALL":
                if f"[{level_filter}]" not in line:
                    continue
            if search_text and search_text.lower() not in line.lower():
                continue
            filtered.append(line)
            
        # Return last N lines
        return filtered[-lines:] if lines > 0 else filtered
    except Exception as e:
        return [f"Error reading logs: {e}"]
