import requests

# SmartDoc API configuration
BASE_URL = "http://172.24.250.17:8001"

# =============================================
# 1. UPLOAD DOCUMENTS
# =============================================
print("Uploading documents...")

# Upload document
files = {'file': open('/Users/delfi/Downloads/ManualRenta2024Tomo2_es_es.pdf', 'rb')}
response = requests.post(f'{BASE_URL}/upload', files=files)
doc_id = response.json()['doc_id']
print(f"Document uploaded. ID: {doc_id}")
files['file'].close()

# =============================================
# 2. LIST DOCUMENTS
# =============================================
documents = requests.get(f'{BASE_URL}/documents').json()
print(f"Total documents: {len(documents)}")

# =============================================
# 3. GET SUMMARY
# =============================================
summary = requests.get(f'{BASE_URL}/document/{doc_id}/summary').json()
print(f"Summary: {summary['summary']}")

# =============================================
# 4. ASK QUESTION ABOUT SPECIFIC DOCUMENT
# =============================================
query = {'query': 'What is this document about?'}
answer = requests.post(f'{BASE_URL}/document/{doc_id}/query', json=query).json()
print(f"Answer: {answer['answer']}")

# =============================================
# 5. SEARCH ALL DOCUMENTS (NEW!)
# =============================================
print("\n--- GLOBAL SEARCH EXAMPLES ---")

# Example 1: Topic search across all documents
query_all = {'query': 'What information do you have about artificial intelligence?'}
global_answer = requests.post(f'{BASE_URL}/query_all', json=query_all).json()

print(f"Global Answer: {global_answer['answer']}")
print(f"Documents searched: {global_answer['total_documents_searched']}")
print(f"Sources: {[s['filename'] for s in global_answer['sources']]}")

# Example 2: Comparative analysis
query_compare = {'query': 'Compare the main conclusions from all documents'}
comparison = requests.post(f'{BASE_URL}/query_all', json=query_compare).json()
print(f"Comparison: {comparison['answer']}")

# Example 3: Topic synthesis
query_synthesis = {'query': 'Summarize all information about machine learning from all documents'}
synthesis = requests.post(f'{BASE_URL}/query_all', json=query_synthesis).json()
print(f"Synthesis: {synthesis['answer']}")

# =============================================
# 6. SYSTEM STATUS
# =============================================
health = requests.get(f'{BASE_URL}/health').json()
print(f"System status: {health['status']}")
print(f"vLLM status: {health['vllm_status']}")
print(f"Global index: {health['global_index']['chunks']} chunks, {health['global_index']['documents']} documents")

config = requests.get(f'{BASE_URL}/config').json()
print(f"vLLM model: {config['vllm_config']['model']}")
print(f"Global index enabled: {config['global_index']['enabled']}")