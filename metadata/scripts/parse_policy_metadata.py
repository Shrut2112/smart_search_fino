import json
import re
from datetime import datetime, timedelta
from urllib.parse import unquote
from pathlib import Path
from collections import defaultdict

def to_unix_timestamp(date_obj):
    #Convert date to Unix timestamp for Pinecone filtering
    try:
        return int(datetime(
            date_obj.year,
            date_obj.month,
            date_obj.day
        ).timestamp())
    except OSError:
        return int(datetime(2100, 1, 1).timestamp())

def parse_policy_metadata(jsonl_path):
    """
    Parse policy_details.jsonl → generate Pinecone-ready metadata with:
    - Stable policy_id from policy_name
    - Numeric timestamps for temporal filtering (_ts fields)
    - Non-overlapping expiry dates (next_eff - 1 day)
    - Clean version numbers
    """

    policies = []
    today = datetime.now().date()
    
    # Load raw jsonl
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 1. Stable policy_id: deterministic, lowercase, normalized
            policy_name = data['policy_name'].lower().strip()
            # Remove common policy words, normalize spacing
            policy_id = re.sub(r'\s+(policy|comprehensive|revised|w\.e\.f|kyc/aml/cft)\s+', '_', policy_name)
            policy_id = re.sub(r'[^a-z0-9_]', '', policy_id).strip('_')
            if not policy_id:  # Fallback
                policy_id = f"policy_{data['sr']}"
            
            # 2. Clean version: V7 → "7", V6. → "6"
            version_match = re.match(r'^V?(\d+(?:\.\d+)?)', data['version'].upper())
            version_num = version_match.group(1) if version_match else "1"
            
            # 3. Parse DD-MM-YYYY → date object
            try:
                eff_date = datetime.strptime(data['effective_date'], '%d-%m-%Y').date()
            except ValueError:
                print(f"Invalid date format: {data['effective_date']} for {data['policy_name']}")
                continue
            
            # 4. Initial expiry (will adjust for versions)
            MAX_EXPIRY_DATE = datetime(2100, 1, 1).date()
            expiry_date = MAX_EXPIRY_DATE
            
            # 5. Initial status
            status = "Active" if eff_date <= today else "Scheduled"
            
            # 6. Clean filename
            file_name = unquote(data['file_path'])
            
            policy_data = {
                'policy_id': policy_id,
                'version_number': version_num,
                'status': status,
                
                # Pinecone filtering fields (numeric)
                'effective_date_ts': to_unix_timestamp(eff_date),
                'expiry_date_ts': to_unix_timestamp(expiry_date),
                
                # Human-readable fields
                'effective_date': eff_date.isoformat(),
                'expiry_date': expiry_date.isoformat(),
                
                'language': 'en',  # Detect per PDF later
                'file_name': file_name,
                'last_updated_at': datetime.now().isoformat(),
                'download_url': data['download_url'],
                'sr': data['sr']
            }
            
            policies.append(policy_data)
    
    # 7. Versioning logic: expire older versions per policy_id
    policy_groups = defaultdict(list)
    for policy in policies:
        policy_groups[policy['policy_id']].append(policy)
    
    for policy_id, versions in policy_groups.items():
        # Sort by effective_date DESC (newest first)
        versions.sort(
        key=lambda x: (
        float(x['version_number']), 
        x['effective_date_ts']
        ),
        reverse=True
        )
        
        for i, version in enumerate(versions):
            if i == 0:  # Latest version
                version['status'] = "Active" if version['effective_date_ts'] <= to_unix_timestamp(today) else "Scheduled"
            else:  # Older versions → Expired
                # Expiry = next_version.effective_date - 1 day (non-overlapping)
                next_eff_date = datetime.fromisoformat(
                versions[i-1]['effective_date']
                ).date()
                
                calculated_expiry = next_eff_date - timedelta(days=1)
                
                # Guardrail: expiry can NEVER be before its own effective date
                safe_expiry = max(
                    calculated_expiry,
                    datetime.fromisoformat(version['effective_date']).date()
                )
                
                version['status'] = 'Expired'
                version['expiry_date'] = safe_expiry.isoformat()
                version['expiry_date_ts'] = to_unix_timestamp(safe_expiry)

    
    # Flatten back to list, sorted by policy_id + version desc
    final_policies = []
    for policy_id, versions in policy_groups.items():
        versions.sort(key=lambda x: x['effective_date_ts'], reverse=True)
        final_policies.extend(versions)
    
    return dict(policy_groups)  # {policy_id: [versions]}

def save_metadata_for_ingestion(policy_groups, output_path='metadata/canonical/fino_policies_metadata.json'):
    """Save processed metadata for ingestion pipeline."""
    flat_list = []
    for policy_id, versions in policy_groups.items():
        for version in versions:
            flat_list.append(version)
    
    with open(output_path, 'w') as f:
        json.dump(flat_list, f, indent=2, default=str)
    print(f"✅ Saved {len(flat_list)} policy versions to {output_path}")
    
    # Summary
    active_count = sum(1 for v in flat_list if v['status'] == 'Active')
    print(f"Active policies: {active_count}/{len(flat_list)}")

if __name__ == "__main__":
    policies = parse_policy_metadata('metadata/raw/policy_details.jsonl')
    save_metadata_for_ingestion(policies)
    
    # Example: Check Comprehensive Deposit versioning
    if 'comprehensive_deposit' in policies:
        print("\n=== Comprehensive Deposit Policy Versions ===")
        for v in policies['comprehensive_deposit']:
            print(json.dumps({
                'policy_id': v['policy_id'],
                'version': v['version_number'],
                'status': v['status'],
                'effective': v['effective_date'],
                'expiry': v['expiry_date'],
                'effective_ts': v['effective_date_ts']
            }, indent=2))