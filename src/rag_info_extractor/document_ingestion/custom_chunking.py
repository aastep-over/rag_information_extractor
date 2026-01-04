from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer


# Python native
import asyncio
import re
from typing import TypedDict, List, Dict, Pattern, Optional, Tuple, Any
from collections import defaultdict
import os


# from other modules
from rag_info_extractor.llm_connector import OllamaLLM

# Logging
import logging
logger = logging.getLogger(__name__)


# Pattern 1: "Art./Articolo/Articoli + numero" (OCR-tolerant)
_ART_WITH_NUM = re.compile(
    r"""(?imx)                           # i: ignore case, m: multi-line, x: verbose
    ^\s*
    (?P<articolo>
        (?:                          # "Art." or "Articolo/Articoli" with random inner spaces
            A\s*[rR]\s*[tT]\s*\.? |
            A\s*[rR]\s*[tT]\s*[iI1]\s*[cCÇ]\s*[oO0]\s*[lL1I]\s*[oO0](?:\s*[iI1])?
        )
        \s*
        (?:                          # optional "n./num./n°/nº" with OCR noise
            (?:[Nn]\s*(?:[\.\º°]|o)) |
            (?:[Nn]\s*[uU]\s*[mM]\s*[\.\º°]?)
        )?
        \s*
        (?:                          # number: arabic or roman
            [0-9]{1,3} |
            [IVXLCDMivxlcdm]{1,8}
        )
        (?:\s*
            (?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)
        )?
        (?:\s*[-–]\s*
            (?:[0-9]{1,3}|[IVXLCDMivxlcdm]{1,8})
        )?
    )
    \s*[\-–:\.\)]*\s*                # separators after the header
    (?P<title>[^\n]{0,160})?         # optional title on same line
    \s*$
    """,
)

# Pattern 2: "Numero + Titolo" (OCR-tolerant: spaces inside numerals and title)
_NUMBER_TITLE = re.compile(
    r"""(?imx)
    ^\s*
    (?P<num>                         # number with optional inner spaces
        (?:(?:\d\s*){1,3}) |
        (?:(?:[IVXLCDMivxlcdm]\s*){1,8})
    )
    \s*[\-–:\.\)]\s*                 # separator
    (?-i:
        (?P<title>
            [A-ZÀ-ÖØ-Ý0-9]
            (?:\s?[A-ZÀ-ÖØ-Ý0-9'’\-&,\.\/]){2,}   # allow spaced-out capitals: C A P I T A L E
        )
    )
    \s*$
    """,
)




ARTICLE_PATTERNS: Dict[str, Pattern] = {
    "art_keyword": _ART_WITH_NUM,
    "number_title": _NUMBER_TITLE,
}


# Function to decide how are the headings/section separations using llm
async def _detect_article_style_llm(text: str, llm: Optional[OllamaLLM]) -> Optional[Pattern]:
    """
    Ask llm which stile is used for separting Articols/paragraphs.
    Returns a regex pattern to use to chunk the articles
    """
    if llm is None:
        return None

    # Extract a sample for checking
    def _make_sample() -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        sample = []
        for ln in lines:
            if len(ln) <= 80 or ln.isupper():
                sample.append(ln)
            if len(sample) >= 30: # 30 sentences/lines enough to detect pattern. 
                break
        return "\n".join(sample)[:2000] if sample else text[:1000]

    sample_text = await asyncio.to_thread(_make_sample)

    prompt = f"""
    Sei un riconoscitore di struttura di statuti.
    Dato lo SNIPPET di testo qui sotto, individua come sono marcati gli INIZI DEGLI ARTICOLI in QUESTO documento e restituisci UNA SOLA regex (stile Python/PCRE) per spezzare il testo in chunk per articolo.

    Regole:
    - Bada SOLO a ciò che vedi nello snippet (maiuscole/minuscole, spazi, punteggiatura, trattini, accenti, OCR); non generalizzare oltre lo snippet.
    - Riconosci solo l’inizio di ARTICOLI, non paragrafi o sottosezioni.
    - Includi solo le varianti effettivamente presenti (es. "Art.", "ART.", "Articolo", "ARTICOLO", numeri arabi/romani, suffissi -bis/-ter, separatori ":" "-" "—").
    - La regex deve funzionare con `re.split`, quindi usa un lookahead su riga nuova: es. `(?m)(?=...)` per non perdere l’intestazione dell’articolo.
    - NON aggiungere testo, spiegazioni o markdown. Se non puoi dedurre un pattern affidabile, restituisci esattamente: NO_PATTERN.

    SNIPPET:
    {sample_text}
    """

    try:
        ans = (
            await llm.ainvoke(
                output_format = "text",
                memory = prompt,
                cache = False
            )
        ).content.strip() # type:ignore
        pat_to_extract_pattern = r"```python\s*\r?\n([\s\S]*?)\r?\n```"
        suggested_pattern = re.findall(pat_to_extract_pattern, ans)[0]
        ans = re.compile(fr"""{suggested_pattern}""")
    except Exception:
        return None

    return ans

async def _choose_best_pattern(text: str, llm: Optional[OllamaLLM] = None) -> Tuple[str, Pattern]:
    """
    Sceglie il pattern che trova più match plausibili.
    Se tutti deboli, prova LLM per suggerire lo stile; se ancora nulla, ritorna il più 'robusto'.
    """
    # for async
    def _count() -> Dict[str, int]:
        return {k: len(list(p.finditer(text))) for k, p in ARTICLE_PATTERNS.items()}

    counts = await asyncio.to_thread(_count)
    # consider viable pattern if >= 3 match
    viable = {k: c for k, c in counts.items() if c >= 3}
    if viable:
        # choose more prominent
        k = max(viable, key=viable.get) # type: ignore
        return k, ARTICLE_PATTERNS[k]

    else:
        # Prova a chiedere all'LLM
        suggested_pattern = await _detect_article_style_llm(text, llm)
        if suggested_pattern:
            return "llm_pattern", suggested_pattern

    # Fallback to Recursive splitter
    return None, None # type:ignore



def _split_by_pattern(text: str, pattern: re.Pattern) -> Dict[str, Any]:
    """
    Returns list of chunks:
    {
      'header': str,           # la riga di intestazione completa
      'num': Optional[str],    # gruppo 'num' se presente nel pattern
      'title': Optional[str],  # gruppo 'title' se presente nel pattern
      'content': str,          # header + corpo del blocco
      'start': int, 'end': int # indici nel testo originale
    }
    If no chunks by article/header, return whole text in one block.
    """

    matches = list(pattern.finditer(text))

    if not matches:
        logger.info("NO MATCHES FOUND FOR THE SELECTED PATTERN!")
        return {
            "split_successful": False,
            "chunks": [{
            "header": None,
            "num": None,
            "title": None,
            "content": text.strip(),
            "start": 0,
            "end": len(text)
        }]
        }

    chunks: List[Dict] = []

    # Preamble for text before first header
    first_start = matches[0].start()
    if first_start > 0:
        pre = text[:first_start].strip()
        if pre:
            chunks.append({
                "header": None,
                "num": None,
                "title": None,
                "content": pre,
                "start": 0,
                "end": first_start
            })

    # Blocks for every header
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        num = m.groupdict().get("num")
        title = m.groupdict().get("title")
        header = m.group(0).strip()

        chunks.append({
            "header": header,
            "num": num.strip() if num else None,
            "title": title.strip() if title else None,
            "content": block,
            "start": start,
            "end": end
        })

    return {
        "split_successful": True,
        "chunks": chunks
    }



async def chunk_statuto_by_articoli(
    doc: Document,
    evaluator_llm: str,
    pages_joining_str: Optional[str],
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Spezza il testo per Articolo, restituendo { pattern_name, chunks }.
    """
    llm = OllamaLLM(llm_model=evaluator_llm, temperature=0) if use_llm else None
    
    text = doc.page_content.strip()

    pattern_name, pat = await _choose_best_pattern(text, llm) # 4 possibilitis: art_keyword, number_title, llm_pattern, None

    # log pattern used for splitting
    logger.info("Suggested pattern for splitting document",
            extra={"source": doc.metadata.get("source", "<unknown>"), "pattern": pattern_name})

    # If you want the detailed pattern text only when debugging:
    logger.debug(f'Pattern details for document {doc.metadata.get("source", "<unknown>")}: {pat}')
    
    if pat is None:
        # tutto il documento come unico chunk
        return {
            "pattern_name": None, 
            "chunks": [{
                "header": None, "num": None, "title": None,
                "content": text, "start": 0, "end": len(text),
            }],
        }

    # esegui lo split in thread (CPU-bound ma breve)
    output_split_by_pattern = await asyncio.to_thread(_split_by_pattern, text, pat)
    split_successful, chunks = output_split_by_pattern.get("split_successful", False), output_split_by_pattern.get("chunks", [])

    # Evaluate if split correct 
    if split_successful and pages_joining_str:
        
        # Check if too few (or too many) chunks
        num_pages = re.split(fr"{pages_joining_str}", text)
        too_few = len(chunks) <= 0.3 * len(num_pages) # num_chunks less than 30% of num pages
        too_many = len(chunks) > 4 *  len(num_pages) # num_chunks more than 400% of num pages

        # Check if start of each chunk matches with the header split used for splitting
        def _starts_with_header(text: str, header_re: Pattern) -> bool:
            # header at start of chunk, allow leading whitespace
            return bool(header_re.match(text.lstrip()))

        header_flags = [_starts_with_header(c.get("content"), pat) for c in chunks]
        header_rate = sum(header_flags) / len(chunks)

        # hard fails
        hard_fail = (
            header_rate < 0.7 or # less than 70% of chunks fail the pattern used to split by articles/sub-paragraphs
            too_few or
            too_many
        )

        if hard_fail: 
            split_successful = None
            pattern_name = None
            chunks = [{
                "header": None, "num": None, "title": None,
                "content": text, "start": 0, "end": len(text),
            }]

    return {
        "pattern_name": pattern_name, 
        "chunks": chunks
        }



class _Output__custom_chunking(TypedDict):
    whole_articles: List[Document]
    chunks: List[Document]
    docs_not_split: List[Document]
    last_parent_id: int
    last_child_id: int


async def custom_chunking(
        docs: List[Document],
        text_splitter,
        tokenizer,
        evaluator_llm: str,
        max_embed_tokens: int,
        pages_joining_str: Optional[str], 
        use_llm: bool = False,
        llm_concurrency: int = 3,
    ) -> _Output__custom_chunking:
    """
    Test chunking based on Articoli, if a chunk has size more than max_embed_tokens, split it using Recursive splitter. 
    """

    # Chunk by articoli if possible
    sem = asyncio.Semaphore(max(1, llm_concurrency))

    async def _chunk_one(doc: Document) -> Dict[str, List[Dict]]:
        async with sem:
            return await chunk_statuto_by_articoli(doc, evaluator_llm, use_llm=use_llm, pages_joining_str=pages_joining_str)

    per_doc_chunks = await asyncio.gather(*[_chunk_one(d) for d in docs]) 

    # If no pattern detected return None for fallback logic of fixed size chunking
    pattern_detected = False
    for c in per_doc_chunks:
        pattern_detected = pattern_detected or c['pattern_name'] is not None
    if not pattern_detected:
        logger.warning("WARNING! \t No Pattern Detected, falling to Fixed-size chunking")
        return _Output__custom_chunking(
            whole_articles = [],
            chunks = [],
            docs_not_split = [],
            last_parent_id = 0,
            last_child_id = 0
        )
            


    # Re-create Document object from chunks obtained
    index_chunks = []
    chunk_docs = []
    docs_not_split: List[Document] = []
    for doc, art in zip(docs, per_doc_chunks):
        base_meta = dict(doc.metadata or {})
        pattern_name = art.get("pattern_name")
        
        # if no pattern detected need to be returned for fixed size splitting
        if not pattern_name:
            docs_not_split.append(doc)
        
        else:
            for ch in art["chunks"]:
                content = ch.pop("content", "") or ""
                meta = {**base_meta, **ch}
                meta["pattern_name"] = pattern_name
            
                # Combine very small chunks (like indices, if present)
                if len(tokenizer.encode(content)) < 30: #roughly chosen considering the name of headers won't be long
                    index_chunks.append(Document(page_content=content, metadata=meta))
                else:
                    chunk_docs.append(Document(page_content=content, metadata=meta))
    
    # Add chunks in index_chunks as one chunk for each document
    index_groups: Dict[str, List[Document]] = defaultdict(list)
    for doc in index_chunks:
        key = doc.metadata.get("source")
        index_groups[key].append(doc)  #type: ignore

    for docs in index_groups.values():
        merged_text = "\n".join(d.page_content for d in docs)
        merged_metadata = {**docs[0].metadata, **{"chunk_type": "index"}}
        chunk_docs.insert(0, Document(page_content=merged_text, metadata=merged_metadata))

    # Split via Recursivesplitter if chunk size > max_seq_length of embedder
    out_docs: List[Document] = []

    _child_id: int = 0 # Add child id for collegare with parent/full articles
    _child_id_lock = asyncio.Lock()

    async def _reserve_child_ids(n: int) -> int:
        """Atomically reserve n child ids and return the starting id."""
        async with _child_id_lock:
            nonlocal _child_id
            start = _child_id
            _child_id += n
            return start
        
    async def _maybe_split(doc: Document, **kwargs) -> List[Document]:
        # skip counting child id and return [] if doc only contains page_splitter text (line used to distinguish the pages)
        page_splitter_text = pages_joining_str # joinging_str
        if (page_splitter_text) and (doc.page_content in page_splitter_text):
            return []

        # tokenize in thread (hf tokenizer is CPU-bound)
        ids = await asyncio.to_thread(tokenizer.encode, doc.page_content)

        if len(ids) > max_embed_tokens:
            # split in thread
            split_doc = await asyncio.to_thread(text_splitter.split_documents, [doc])
            
            # Reserve a global id range for these children
            base = await _reserve_child_ids(len(split_doc))
            i = 0
            for s in split_doc:
                if (page_splitter_text) and (s.page_content in page_splitter_text):
                    continue
                s.metadata.update({'child_id': base + i})
                i += 1
            return [d for d in split_doc if d.metadata.get("child_id")]
        
        # Reserve a global id range for these children
        base = await _reserve_child_ids(1)
        doc.metadata.update({"child_id": base})

        return [doc]

    split_lists = await asyncio.gather(*[_maybe_split(d) for d in chunk_docs if d])
    for lst in split_lists:
        out_docs.extend(lst)
    
    def merge_whole_articles(out_docs: List[Document]) -> List[Document]:
        """Merge docs sharing the same (azienda, header). Preserve input order."""
        # Group by (azienda, header)
        groups: Dict[Tuple[str, str], List[Document]] = defaultdict(list)
        for doc in out_docs:
            key = (doc.metadata.get("azienda"), doc.metadata.get("header"))
            groups[key].append(doc)  #type: ignore
        
        # Recreate large chunk docs with chunk ids
        merge_docs_by_key: List[Document] = []
        for i, docs in enumerate(groups.values()):
            merged_text = " ".join(d.page_content for d in docs)
            start_index = min([d.metadata.get("start") for d in docs]) # type: ignore ("start" is an int)
            end_index = max([d.metadata.get("end") for d in docs]) # type: ignore ("end" is an int)
            merged_metadata = {**docs[0].metadata, **{"start": start_index, "end": end_index, "chunk_id": i}}
            merge_docs_by_key.append(Document(page_content=merged_text, metadata=merged_metadata))
        
        return merge_docs_by_key

    async def merge_whole_articles_async(out_docs: List[Document]) -> List[Document]:
        """Non-blocking wrapper for merge_whole_articles."""
        return await asyncio.to_thread(merge_whole_articles, out_docs)
    
    out_docs_whole = await merge_whole_articles_async(out_docs)
    
    # Rebuild result aligned to the original order
    for doc in out_docs:
        child_header = doc.metadata.get("header") # header for child
        child_azienda = doc.metadata.get("azienda") # azienda for child

        rel_merged_doc_chunk_id = [d.metadata.get("chunk_id") for d in out_docs_whole if (d.metadata.get("header") == child_header) and (d.metadata.get("azienda") == child_azienda)][0] # parent_id for child 
        doc.metadata["chunk_id"] = doc.metadata.get("child_id")
        doc.metadata["parent_id"] = rel_merged_doc_chunk_id



    last_parent_id = sorted([d.metadata.get("chunk_id", 0) for d in out_docs_whole], reverse=True)[0] if out_docs_whole else 0
    last_child_id = sorted([d.metadata.get("chunk_id", 0) for d in out_docs], reverse=True)[0] if out_docs else 0
    return _Output__custom_chunking(
            whole_articles = out_docs_whole,
            chunks = out_docs,
            docs_not_split = docs_not_split,
            last_parent_id = last_parent_id,
            last_child_id = last_child_id
        )
    





if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    import fitz
    from pathlib import Path
    import time
    import yaml
    import argparse

    from rag_info_extractor.common_logging import configure_logging


    t0 = time.time()

    # Configure logging settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging") # For DEBUG level logging, run in cli: python .\ingest_docs.py --verbose or -v
    args = parser.parse_args()
    configure_logging(default_level=logging.DEBUG if args.verbose else logging.INFO)
    
    # CONFIG FILE SETTINGS:
    cfg_path = Path("D:/Users/yye7607/Documents/work/Stage Amjad Ali/RAG/rag_information_extractor/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    cfgs = configs.get("args", {})

    EMBEDDING_MODEL_NAME = cfgs.get("EMBEDDING_MODEL_NAME")
    EVALUATOR_LLM = cfgs.get("EVALUATOR_LLM") 
    DATASET_TYPE = cfgs.get("DATASET_TYPE")
    MAX_EMBED_TOKENS = cfgs.get("MAX_EMBED_TOKENS")
    READ_MODE = cfgs.get("READ_MODE", "single")
    PAGES_JOINING_STR = cfgs.get("PAGES_JOINING_STR", "\n")
    BASE_DIR = cfgs.get("BASE_DIR", "./")
    PDF_LOADER = cfgs.get("PDF_LOADER", "./")
    
    DATASET_DIR = os.path.join(BASE_DIR, "data", "documents", DATASET_TYPE) #f"../data/documents/{DATASET_TYPE}"
    
    # Load pdf
    docs: List[Document] = []
    if PDF_LOADER == "pymupdf":
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info(f'Document exists: {os.path.exists(path)}')
            doc = fitz.open(path)

            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            
            meta = {**meta, **{"source": Path(doc.name).name if doc.name else None, "total_pages": doc.page_count}}
            text = PAGES_JOINING_STR.join(page.get_text("text") or "" for page in doc ) # type: ignore (return type of get_text is not only str)
            docs.append(Document(page_content=text, metadata=meta))
            doc.close()
    else:
        for doc in os.listdir(DATASET_DIR):
            path = f"{DATASET_DIR}/{doc}"
            logger.info(f'Document exists: {os.path.exists(path)}')
            loader = PyPDFLoader(path,
                                mode=READ_MODE,
                                pages_delimiter=PAGES_JOINING_STR
                                )
            docs.extend(loader.load())
        

    
    logger.info("Docs Loaded")
    # Define text splitter and tokenizer
    chunk_size = 430
    chunk_overlap = 105
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, use_fast=True)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,  # chunk size (tokens)
            chunk_overlap=chunk_overlap,  # chunk overlap (tokens)
            add_start_index=True,  # track index in original document
        )

    logger.info("Creating Chunks...")
    chunks = asyncio.run(custom_chunking(
        docs,
        text_splitter,
        tokenizer,
        evaluator_llm=EVALUATOR_LLM,
        max_embed_tokens=MAX_EMBED_TOKENS,
        use_llm=True,
        pages_joining_str=PAGES_JOINING_STR
    ))
    
    logger.info("Chunks created. Saving...")
    with open("output_temp", "w", encoding="utf-8") as f:
        f.write("OUTPUT FOR custom_chunking.py\n\n")
        f.write("Whole Articles: \n")
        
        f.write("PARENT CHUNKS: \n\n")
        parent_chunks = chunks.get("whole_articles")
        for c in parent_chunks:
            f.write(f"\n{"-"*50} CHUNK ID: {c.metadata.get("chunk_id")} {"-"*50}\n")
            f.write(f"{c.page_content}\n\n")

        # f.writelines([f"CHUNK ID = {c.metadata.get("chunk_id")}\n{c.page_content}\n\n" for c in parent_chunks])
        f.write(f"{"x"*100}\n")

        f.write("DOCS NOT SPLIT: \n\n")
        f.write(f"last_parent_id =  {chunks.get("last_parent_id")}\n")
        f.write(f"last_child_id =  {chunks.get("last_child_id")}\n\n")
        docs_not_split = chunks.get("docs_not_split")
        for c in docs_not_split:
            f.write(f"\n{"-"*50} CHUNK ID: {c.metadata.get("chunk_id")} {"-"*50}\n")
            f.write(f"{c.page_content}\n\n")
    
    logger.info(f"Total time taken to run the script: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}")