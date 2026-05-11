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

# How to run Extraction:
1. Check and adjust config.yaml file if needed
2. Run the script run_apis.sh to run microservices of embedder, re-ranker and pruner
3. Run the script run_extraction.sh

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

# Structures of raw and pred jsons:
**raw_data.json:**
{
    Azienda_name: {
        "BILANCI_E_UTILI": {
            "values": {
                "CapitaleSociale": {
                    "capitale_sociale_euro": ""
                },
                "DataChiusuraEsercizio": {
                    "data_chiusura_esercizio": ""
                },
                "PercentualeRiservaLegale": {
                    "percentuale_utili": ""
                },
                "TermineApprovazioneBilancio": {
                    "termine_ordinario_giorni": "",
                    "termine_prorogato_giorni": ""
                },
                "UtiliResidui": {
                    "utili_residui": ""
                }
            },
            "raw_contexts": {
                "CapitaleSociale": "",
                ...
            },
            "raw_contexts_ids": {
                "CapitaleSociale": {
                    "parents": [],
                    "children": []
                },
                ...
            },
            "raw_qa": {
                "CapitaleSociale": {
                    "Q": "",
                    "A": ""
                },
                ...
            }
        },
        "COMPENSO_DEGLI_AMMINISTRATORI": {
            ...
        },
        "INFO_GENERALI": {
            ...
        }
    },
    ...
}

**pred.json:**
{
    "Azienda_name": {
        "BILANCI_E_UTILI": {
            "output": {
                "CapitaleSociale": {
                    "capitale_sociale_euro": ""
                },
                "DataChiusuraEsercizio": {
                    "data_chiusura_esercizio": ""
                },
                "PercentualeRiservaLegale": {
                    "percentuale_utili": ""
                },
                "TermineApprovazioneBilancio": {
                    "termine_ordinario_giorni": "",
                    "termine_prorogato_giorni": ""
                },
                "UtiliResidui": {
                    "utili_residui": ""
                }
            },
            "retrieved_docs": {
                "CapitaleSociale": {
                    "parents": [],
                    "children": []
                },
                ...
            },
            "re_ranked_docs": {
                "CapitaleSociale": {
                    "parents": [],
                    "children": []
                },
                ...
            },
            "retrieved_docs_texts": {
                "CapitaleSociale": {
                    "parents": [],
                    "children": []
                },
                ...
            },
            "re_ranked_docs_texts": {
                "CapitaleSociale": {
                    "parents": [],
                    "children": []
                },
                ...
            },
            "rag_qa": {
                "CapitaleSociale": {
                    "Q": "",
                    "A": ""
                },
                ...
            },
            "run_times": {
                "CapitaleSociale": {
                    "analyze_query": "",
                    "retrieve": "",
                    "pruning": "",
                    "generate": "",
                    "re_ranking": "",
                    "faster_retrieve_and_rerank": "",
                    "overall": "",
                    "extract_sub_module": ""
                },
                ...
            },
            "optimized_query": {
                "CapitaleSociale": {
                    "query": "",
                    "azienda": ""
                },
                ...
            }
        },
        "COMPENSO_DEGLI_AMMINISTRATORI": {
            ...
        },
        "INFO_GENERALI": {
            ...
        }
    },
    ...
}


# Cloud vs. Local Deployment Trade-offs
During development, I observed that while local hosting ensures 100% data privacy and zero deprecation risk, CPU-only inference can lead to high latency. I integrated the Google GenAI API to offer a 'High Performance' mode, while documenting the inherent risk of model retirement (as seen with the transition from Gemma 3 to Gemma 4).


# TODOs: 
4. Implement the extract_and_save_all_info in scripts/extract_info.py in asynchronous manner
5. Integrate the extraction process with streamlit frontend (a button which extracts the pre-defined fields for 1 statuto)
6. Check if "keyBERT" model can be implemented for analyze_query node.
7. Check OLLAMA implementation for Qwen3.5
8. Implement tests for faithfulness, answer relevance, Conciseness.
9. Read about GemmaEmbeddings and use it for embedding instead of e5 (check also prompt for embedders at https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers?hl=it&authuser=1)
10. Apply isort and black to all scripts for uniform code formatting 
11. Write all the tests in a separate folder for pytest testings


# Possibile Improvements:
1. In the anlyze_query node for the rag_pipeline, see if there are better alternatives than using LLM for optimizing the query such as spaCy or KeyBERT?? If not, at least use a differnt, smaller model for this step with zero-shot prompting. (TEST KeyBERT in analyze query thoroughly and if works better/good will need to modify the section 3.3.2 Node 1: Query Analysis and Pre-Retrieval Optimization in the thesis to reflect this)
2. In the extractor llm, try to use a smaller llm (1b/2b) and compare if performance is sufficient/not too low. If works, edit the section 3.4.1 The Two-Step Extraction System (paragraph 3) in the thesis.
3. For better query optimization, use the HyDE + re-ranker technique
4. In rag_info_extractor, for ingestion with custom_chunking, we can use semantic_chunking as fallback instead of fixed-size since it showed better performance


# Suggestion to switch from local to Gemini API:
A. Use a "Dual-Mode" Architecture
Instead of just switching to the API, implement a toggle. This proves you can optimize for different environments.

**Example**
def get_llm(mode="api"):
    if mode == "local":
        return LocalGemma(model_path="path/to/gemma-4-e4b.gguf") # High Latency / Privacy
    else:
        return GoogleGenAI(model_id="gemma-4-26b-a4b-it") # Low Latency / Cloud