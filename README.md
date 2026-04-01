# RUNNING MODELS LOCALLY:
### Ollama:

### HF (reranker/embedding models)
1. Create a separate directory to save huggingface models (preferebly in users/)
2. Define an env. variable called "HF_MODELS" which stores the absolute path of the directory created in step 1.
3. Copy the HF models (directories) located in the .cache/huggingface/hub/ in the directory created in step 1. (with the same name)



keys for combined_data.json: "values", "raw_contexts", "raw_contexts_ids", "raw_qa"

raw jsons must be saved as: azienda_name : {}
combined raw json is saved in data/jsons/<TRAIN || TEST>/combined_data.json
output files are saved in run/<TRAIN || TEST>/run_time/ with name: pred.json
match_scores files are saved in run/<TRAIN || TEST>/run_time/ with name: match_scores.json

# How to run TEST:
1. After inserting raw data in jsons, find raw_context_ids by running find_raw_chunks_ids.py function: insert_chunk_id_to_combined_raw_json
2. Run extraction pipeline by running run_extraction.sh
3. Create match_scores.json by running the script tests/utils/match_pred_output_llm.py
4. Verify manually and correct if needed match_scores.json by consulting the decision_logs.txt
5. Create match_scores_qa.json by running the script tests/utils/match_pred_rag_QA.py
6. Verify manually and correct if needed match_scores_qa.json by consulting the decision_logs_qa.txt
7. Finally run eval_overall.py

# Notes:
- qwen3.5:4b is not good at extracting structured json format
- To select the nodes of rag_pipeline, change it in scripts/rag_pipeline.py
- when loading huggingface models from local paths, we need to adjust a bit for e.g._
    - Pruner: copy path directly from huggingface for first download, set local_files_only=False in scripts/rag_pipeline.py for first run


# TODOs: 
1. Include explanation of the keys to be extracted in the prompt passed to extractor llm (explain what each key to be extracted means)
2. Add structures of raw and pred jsons in README
3. When running experiments, save a block/flow diagram of rag_pipeline used in that run to extract fields. (need to adjust scripts/rag_pipeline.py and/or scripts/extract_info_json.py)
4. Implement the extract_and_save_all_info in scripts/extract_info.py in asynchronous manner
5. Integrate the extraction process with streamlit frontend (a button which extracts the pre-defined fields for 1 statuto)
6. Check if "keyBERT" model can be implemented for analyze_query node.
7. Check OLLAMA implementation for Qwen3.5
8. When testing classify the type of errors: half correct, correct but extra contradictory stuff, extract values for fields for which there is no ref values (false positives), missing values for which there is ref values (false negatives)


# Possibile Improvements:
1. In the anlyze_query node for the rag_pipeline, see if there are better alternatives than using LLM for optimizing the query such as spaCy or KeyBERT?? If not, at least use a differnt, smaller model for this step with zero-shot prompting. (TEST KeyBERT in analyze query thoroughly and if works better/good will need to modify the section 3.3.2 Node 1: Query Analysis and Pre-Retrieval Optimization in the thesis to reflect this)
2. In the extractor llm, try to use a smaller llm (1b/2b) and compare if performance is sufficient/not too low. If works, edit the section 3.4.1 The Two-Step Extraction System (paragraph 3) in the thesis.