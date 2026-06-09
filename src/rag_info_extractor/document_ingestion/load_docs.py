import asyncio

# Logging
import logging
import re
import textwrap

# Python native
from pathlib import Path
from typing import Callable, Dict, List, Literal, Union

import pymupdf  # for pymupdf
import regex
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Relative Modules
from rag_info_extractor.document_ingestion.custom_chunking import custom_chunking
from rag_info_extractor.document_ingestion.fixed_size_chunking import fixed_size_chunking_async
from rag_info_extractor.document_ingestion.semantic_chunking import semantic_chunking_async
from rag_info_extractor.utils.embedder import HFEmbedder
from rag_info_extractor.utils.llm_connector import OllamaLLM
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Informations to be extracted for Metadata


## TODO: Generalize the method of collection of metadata (create class for each data with method get_metadat(), relevant content regex pattern and adjust _get_metadatas
## function accordingly (generalizable such that more metadatas can be added easily)
# -------------------------------- Azienda Name ---------------------------------
class _AziendaNameJSON(BaseModel):
    azienda_name: str = Field(
        description="nome della società a cui si riferisce il documento."
    )


def _get_name_azienda(page_content: str, llm_model: str) -> _AziendaNameJSON:
    """Extract Azienda name from the doc using LLM"""

    llm = OllamaLLM(llm_model=llm_model, temperature=0)

    template = textwrap.dedent(
        """\
        Sei un estrattore di entità. Dal testo seguente, individua la DENOMINAZIONE SOCIALE (nome completo ufficiale) della società.
        Regole:
        - Includi la forma giuridica (S.r.l., S.p.A., ecc.) se presente.
        - Rimuovi virgolette esterne; non aggiungere spiegazioni.
        - Se più società sono citate, scegli il soggetto principale del documento; se ambiguo, la prima denominazione completa.
        - Se assente, restituisci esattamente 'NON HO TROVATO'. 
        - Restituisci SOLO la DENOMINAZIONE SOCIALE (nome completo ufficiale) della società trovata.

        TESTO:
        {text}
    """
    )

    prompt = template.replace("{text}", page_content)

    ai_msg: _AziendaNameJSON = llm.ainvoke(
        output_format="structured",
        memory=prompt,
        cache=False,
        info_schema=_AziendaNameJSON,
    )  # type: ignore

    return ai_msg


# -------------------------------- Sede ---------------------------------
class _AziendaSedeJSON(BaseModel):
    azienda_sede: str = Field(
        description="sede della società a cui si riferisce il documento."
    )


def _get_sede_azienda(page_content: str, llm_model: str) -> _AziendaSedeJSON:
    """Extract Azienda name from the doc using LLM"""

    llm = OllamaLLM(llm_model=llm_model, temperature=0)

    template = textwrap.dedent(
        """\
        Sei un estrattore di entità.  
        Dal testo seguente, individua la SEDE LEGALE della società.

        Regole:
        - Riporta l’indirizzo completo così come appare nel testo (via, numero civico, città, CAP, provincia se presenti).
        - Non aggiungere né modificare nulla; mantieni l’ordine e la forma originale.
        - Se nel testo sono presenti più sedi, scegli la SEDE LEGALE principale; se non è chiaro, scegli la prima menzionata.
        - Se la sede non è indicata, restituisci esattamente 'NON HO TROVATO'.
        - Restituisci SOLO la SEDE LEGALE (nessun altro testo o spiegazione).

        TESTO:
        {text}
    """
    )

    prompt = template.replace("{text}", page_content)

    ai_msg: _AziendaSedeJSON = llm.ainvoke(
        output_format="structured",
        memory=prompt,
        cache=False,
        info_schema=_AziendaSedeJSON,
    )  # type: ignore

    return ai_msg


# async function to extract azienda Metadata
async def _get_metadata_azienda(
    page_content: str,
    llm_model: str,
    extract_metadata_functions: List[Callable] = [_get_name_azienda, _get_sede_azienda],
):
    tasks = [f(page_content, llm_model) for f in extract_metadata_functions]
    results = await asyncio.gather(*tasks)

    return results


# ------------------------------- Loading Helpers ---------------------------
def _expand_pdfs(folder: Union[str, Path]) -> List[Path]:
    """Return all PDF files under a folder (recursive)."""
    folder = Path(folder)
    return sorted([p for p in folder.rglob("*.pdf") if p.is_file()])


def _load_pdf_sync(
    path: Path,
    pdf_loader: Literal["pypdf", "pymupdf"],
    llm_model: str,
    extract_azienda_name: bool = True,
    extract_azienda_sede: bool = False,
    **kwargs,
) -> List[Document]:
    """Blocking load of one PDF with PyPDFLoader."""

    mode = kwargs.get("read_mode", "pages")  # Load as a single doc or one doc per page
    joining_str = kwargs.get("pages_joining_str", " ")
    # Load pdf
    if pdf_loader == "pymupdf":
        with open(str(path), "rb") as fh:
            data = fh.read()
        doc = pymupdf.open(stream=data, filetype="pdf")

        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        meta = {**meta, **{"source": path.name, "total_pages": doc.page_count}}

        docs = []

        if mode == "single":
            text = joining_str.join(page.get_text("text") or "" for page in doc)
            docs.append(Document(page_content=text, metadata=meta))
        else:
            for page in doc:
                text: str = page.get_text("text") or ""  # type: ignore (return type of get_text is not only str)
                if page.number is not None:
                    metadata = {
                        **meta,
                        **{"page": page.number, "page_label": page.number + 1},
                    }
                else:
                    metadata = meta
                docs.append(Document(page_content=text, metadata=metadata))

    else:
        if mode == "single":
            loader = PyPDFLoader(
                path,
                mode="single",
                pages_delimiter=kwargs.get("pages_joining_str", " "),
            )
            docs = loader.load()
        else:
            loader = PyPDFLoader(path)
            docs = loader.load()

    # Add relevant metadatas
    for d in docs:
        d.metadata.setdefault("source", str(path))
        d.metadata["filename"] = path.name

        # Use LLM to extract name of the azienda
        first_page = re.split(rf"{joining_str}", d.page_content, 1)[0]
        extract_metadata_functions = []
        if extract_azienda_name:
            extract_metadata_functions.append(_get_name_azienda)
        else:
            d.metadata["azienda"] = ""

        if extract_azienda_sede:
            extract_metadata_functions.append(_get_sede_azienda)
        else:
            d.metadata["sede"] = ""

        if extract_metadata_functions:
            azienda_name_extracted = False
            azienda_sede_extracted = False

            output_get_azienda_metadatas = asyncio.run(
                _get_metadata_azienda(
                    page_content=first_page,
                    llm_model=llm_model,
                    extract_metadata_functions=extract_metadata_functions,
                )
            )

            metadatas = {}
            for m in output_get_azienda_metadatas:
                metadatas.update(m.model_dump())
            azienda_name = metadatas.get("azienda_name")
            azienda_sede = metadatas.get("azienda_sede")

            # check if azienda_name and azienda_sede extracted
            azienda_name_extracted = azienda_name and (
                azienda_name.lower() != "non ho trovato"
            )
            azienda_sede_extracted = azienda_sede and (
                azienda_sede.lower() != "non ho trovato"
            )

            # If not found on first page, search for relevant pieces ("Denominazione" for azienda_name, "Sede" for azienda_sede)
            if not azienda_name_extracted:
                matches = regex.findall(
                    r"(?is)(.{0,500})((?b)(?:denominazione|denominiazione){e<=2}(?:.{0,100}?e['’`´]?\s*c[o0]st[i1]t[uuv]i?t[àa]\s+la\s+societ[àa].{0,500})?)(.{0,500})",
                    text,
                )
                if matches:
                    content_to_search_name = "\n\n".join(
                        " ".join(match) for match in matches
                    )
                    output_get_azienda_name = asyncio.run(
                        _get_metadata_azienda(
                            page_content=content_to_search_name,
                            llm_model=llm_model,
                            extract_metadata_functions=[_get_name_azienda],
                        )
                    )

                    for m in output_get_azienda_name:
                        azienda_name = m.model_dump().get("azienda_name")
                    azienda_name_extracted = True  # set to true to not enter in infinite loop to search if can't find

            if not azienda_sede_extracted:
                matches = regex.findall(
                    r"(?is)(?:sede(?:\s+legale|\s+sociale)?|con\s+sede|con\s+la\s+sede|domicilio\s+legale|sede\s+in)[\s\:\-–—]*(.{5,250}?)?(?=[\.\r\n]|$)",
                    text,
                )
                if matches:
                    content_to_search_sede = "\n\n".join(
                        " ".join(match) for match in matches
                    )
                    output_get_azienda_sede = asyncio.run(
                        _get_metadata_azienda(
                            page_content=content_to_search_sede,
                            llm_model=llm_model,
                            extract_metadata_functions=[_get_sede_azienda],
                        )
                    )

                    for m in output_get_azienda_sede:
                        azienda_sede = m.model_dump().get("azienda_sede")
                    azienda_sede_extracted = True  # set to true to not enter in infinite loop to search if can't find

            # set azienda_name and azienda_sede
            d.metadata["azienda"] = (
                azienda_name.lower() if isinstance(azienda_name, str) else ""
            )
            d.metadata["sede"] = (
                azienda_sede.lower() if isinstance(azienda_sede, str) else ""
            )

            # # ==================== FOR QUICK TESTING =========================================
            # ext = path.suffix
            # filename, azienda_name = re.split(r"___", path.name)
            # filename, azienda_name = filename + ext, azienda_name.replace(ext, "") # assume: filename__aziendaname.pdf
            # d.metadata["filename"] = filename
            # d.metadata["azienda"] = azienda_name.lower()
            # d.metadata["sede"] = ""
            # # =========================   XXX   =========================================

    return docs


# ------------------------------- Main function -------------------------------
async def aload_pdfs(
    folder: Union[str, Path],
    HF_embedding_model_name: str,
    evaluator_llm: str,
    llm_model: str,
    max_embed_tokens: int,
    *,
    num_pdfs: int = 4,
    pdf_loader: Literal["pypdf", "pymupdf"] = "pymupdf",
    split: bool = True,
    chunk_size: int = 430,
    chunk_overlap: int = 105,
    chunks_type: Literal[
        "fixed_size_chunks", "custom_chunks", "semantic_chunks", "custom_chunks_2"
    ] = "custom_chunks",
    **kwargs,
) -> Dict[str, List[Document]]:
    """Asynchronously load and optionally split all PDFs from a local folder."""
    pdfs = _expand_pdfs(folder)
    if not pdfs:
        return {"parent_chunks": [], "children_chunks": []}

    sem = asyncio.Semaphore(max(1, num_pdfs))

    async def worker(p: Path) -> List[Document]:
        async with sem:
            return await asyncio.to_thread(
                _load_pdf_sync,
                p,
                pdf_loader,
                llm_model,
                read_mode=kwargs.get("read_mode"),
                pages_joining_str=kwargs.get("pages_joining_str"),
            )  # add False if filename already contain azienda name

    doc_lists = await asyncio.gather(*[worker(p) for p in pdfs])
    docs = [d for sub in doc_lists for d in sub]

    if not split:
        logger.warning("No Splitting of Documents into chunks")
        return {"parent_chunks": docs, "children_chunks": docs}

    tokenizer = AutoTokenizer.from_pretrained(HF_embedding_model_name, use_fast=True)
    child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,  # chunk size (tokens)
        chunk_overlap=chunk_overlap,  # chunk overlap (tokens)
        add_start_index=True,  # track index in original document
    )

    if (chunks_type == "custom_chunks") or (chunks_type == "custom_chunks_2"):
        logger.info("Trying to Split using 'Custom Chunking'...")
        output = await custom_chunking(
            docs,
            child_splitter,
            tokenizer,
            evaluator_llm=evaluator_llm,
            max_embed_tokens=max_embed_tokens,
            use_llm=True,
            llm_concurrency=3,
            pages_joining_str=kwargs.get("pages_joining_str"),
        )
        parent_chunks = output["whole_articles"]
        children_chunks = output["chunks"]
        docs_not_split = output["docs_not_split"]
        last_parent_id = output["last_parent_id"]
        last_child_id = output["last_child_id"]

        logger.info(
            f"Splitting by 'Custom Chunking' successful for docs: {set([c.metadata.get('source') for c in parent_chunks])}"
        )

        if docs_not_split:
            logger.info(
                f'Splitting UN-SPLIT DOCS: {[d.metadata.get("filename") for d in docs_not_split]} using Fixed-sized chunking ...'
            )
            output_fs = await fixed_size_chunking_async(
                child_splitter,
                docs_not_split,
                tokenizer,
                max_concurrency=8,
                last_parent_id=last_parent_id,
                last_child_id=last_child_id,
            )
            parent_chunks_fs, children_chunks_fs = (
                output_fs["parent_chunks"],
                output_fs["children_chunks"],
            )
            parent_chunks += parent_chunks_fs
            children_chunks += children_chunks_fs

        if (not parent_chunks) and (not children_chunks):
            # fallback: only RecursiveSplitter
            logger.warning("No custom pattern detected!")
            logging.info(f"Splitting docs using Fixed-sized chunking ...")
            output = await fixed_size_chunking_async(
                child_splitter, docs, tokenizer, max_concurrency=8
            )
            parent_chunks, children_chunks = (
                output["parent_chunks"],
                output["children_chunks"],
            )

    elif chunks_type == "semantic_chunks":
        BREAKPOINT_THRESHOLD = 0.8
        logger.info(
            f"Trying to Split using 'Semantic Chunking' with THRESHOLD={BREAKPOINT_THRESHOLD}..."
        )

        embedding_func = HFEmbedder(normalize_embeddings=True)

        output = await semantic_chunking_async(
            docs,
            embedding_func,
            tokenizer,
            child_splitter,
            max_embed_tokens=max_embed_tokens,
            pages_joining_str=kwargs.get("pages_joining_str"),
            max_concurrency=8,
            breakpoint_threshold_amount=BREAKPOINT_THRESHOLD,
        )
        parent_chunks, children_chunks = (
            output["parent_chunks"],
            output["children_chunks"],
        )

    else:
        output = await fixed_size_chunking_async(
            child_splitter, docs, tokenizer, max_concurrency=8
        )
        parent_chunks, children_chunks = (
            output["parent_chunks"],
            output["children_chunks"],
        )

    return {"parent_chunks": parent_chunks, "children_chunks": children_chunks}
