# scripts/llm_connector.py

# logging relative
import logging
from typing import ClassVar, Literal, Optional

import httpx
import requests
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from rag_info_extractor.utils.load_config import cfgs

logger = logging.getLogger(__name__)

# CONFIG SETTINGS FOR LLM
# cfg_path = Path("D:/Documents/Italy/UNIPD/University Acadamico/TESI/project/rag_information_extractor/config.yaml")
# with open(cfg_path, "r", encoding="utf-8") as f:
#     configs = yaml.safe_load(f)

cfgs = cfgs.get("args", {})
OLLAMA_MODEL = cfgs.get("LLM_MODEL", "")
OLLAMA_HOST = cfgs.get("OLLAMA_HOST", "")


class OllamaLLM:
    """Componente per l'interazione con l'API Ollama."""

    def __init__(
        self,
        llm_model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        temperature: float = 0,
    ):
        self.DEFAULT_MODEL = llm_model
        self.DEFAULT_HOST = host
        self.temperature = temperature

    def get_response_text(
        self,
        prompt_content: str,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        format: Literal["json", None] = None,
        cache: bool = True,
        num_predict: Optional[int] = None,
    ) -> str:
        """Esegue una singola chiamata API per un prompt e restituisce il testo della risposta."""
        host = host if host else self.DEFAULT_HOST
        model_name = model_name if model_name else self.DEFAULT_MODEL
        temperature = temperature if temperature else self.temperature

        try:
            r = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt_content,
                    "stream": False,
                    "temperature": self.temperature,
                    "num_predict": num_predict,
                    "format": format,
                    "cache": cache,
                },
                timeout=9000,
            )
            r.raise_for_status()
            # Estrae la risposta
            completion = r.json().get("response", "").strip()
            return completion

        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Errore di connessione a Ollama ({host}/{model_name}): {e}"
            )

    def get_structured_response(
        self,
        prompt_content: str,
        info_schema: BaseModel,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        cache: bool = True,
        num_predict: Optional[int] = None,
    ) -> BaseModel:
        """Esegue una singola chiamata API per un prompt e restituisce il JSON (str) della risposta."""
        host = host if host else self.DEFAULT_HOST
        model_name = model_name if model_name else self.DEFAULT_MODEL
        temperature = temperature if temperature else self.temperature

        try:
            r = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt_content,
                    "stream": False,
                    "temperature": self.temperature,
                    "num_predict": num_predict,
                    "format": info_schema.model_json_schema(),
                    "cache": cache,
                    # "think": True
                },
                timeout=9000,
            )
            r.raise_for_status()
            # Estrae la risposta
            completion = r.json().get("response", "").strip()
            structured_output = info_schema.model_validate_json(completion)
            return structured_output

        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Errore di connessione a Ollama ({host}/{model_name}): {e}"
            )

    def invoke(
        self, output_format: Literal["text", "structured", "json"] = "text", **kwargs
    ) -> AIMessage | BaseModel | str:
        """Metodo standard per la pipeline (solitamente per la generazione finale)."""
        # Gather relevant kwargs
        host = kwargs.get("llm", {}).get("host", self.DEFAULT_HOST)
        model_name = kwargs.get("llm", {}).get("model_name", self.DEFAULT_MODEL)
        prompt_content = kwargs.get("memory", "")
        use_cache = kwargs.get("cache", True)
        num_predict = kwargs.get("num_predict")
        temperature = kwargs.get("temperature")

        if not prompt_content:
            logger.warning("NO Prompt provided!")

        if output_format == "structured":
            schema = kwargs.get("info_schema")
            if not schema:
                raise Exception(
                    "Missing info schema (kwarg: 'info_schema')! Provide the BaseModel schema for the information to be extracted!"
                )
            try:
                completion = self.get_structured_response(
                    prompt_content=prompt_content,
                    info_schema=schema,
                    temperature=temperature,
                    model_name=model_name,
                    host=host,
                    cache=use_cache,
                    num_predict=num_predict,
                )
            except ConnectionError as e:
                completion = str(e) + ". Assicurati che Ollama sia in esecuzione."

            return completion

        elif output_format == "json":
            try:
                completion = self.get_response_text(
                    prompt_content=prompt_content,
                    temperature=temperature,
                    model_name=model_name,
                    host=host,
                    cache=use_cache,
                    format="json",
                    num_predict=num_predict,
                )
            except ConnectionError as e:
                completion = str(e) + ". Assicurati che Ollama sia in esecuzione."
        else:
            try:
                completion = self.get_response_text(
                    prompt_content=prompt_content,
                    temperature=temperature,
                    model_name=model_name,
                    host=host,
                    cache=use_cache,
                    num_predict=num_predict,
                )
            except ConnectionError as e:
                completion = str(e) + ". Assicurati che Ollama sia in esecuzione."
                logger.warning(completion)

        return AIMessage(content=completion)

    # ------------------------------------------------------------
    # ASYNC VERSIONS
    # ------------------------------------------------------------

    async def aget_response_text(
        self,
        prompt_content: str,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        format: Literal["json", None] = None,
        cache: bool = True,
        num_predict: Optional[int] = None,
    ) -> str:

        host = host or self.DEFAULT_HOST
        model_name = model_name or self.DEFAULT_MODEL

        try:
            async with httpx.AsyncClient(timeout=9000) as client:
                r = await client.post(
                    f"{host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt_content,
                        "stream": False,
                        "temperature": self.temperature,
                        "num_predict": num_predict,
                        "format": format,
                        "cache": cache,
                    },
                )
                r.raise_for_status()
                return r.json().get("response", "").strip()

        except httpx.RequestError as e:
            raise ConnectionError(
                f"(async) Errore di connessione Ollama ({host}/{model_name}): {e}"
            )

    async def aget_structured_response(
        self,
        prompt_content: str,
        info_schema: BaseModel,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        cache: bool = True,
        num_predict: Optional[int] = None,
    ):

        host = host or self.DEFAULT_HOST
        model_name = model_name or self.DEFAULT_MODEL

        try:
            async with httpx.AsyncClient(timeout=9000) as client:
                r = await client.post(
                    f"{host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt_content,
                        "stream": False,
                        "temperature": self.temperature,
                        "num_predict": num_predict,
                        "format": info_schema.model_json_schema(),
                        "cache": cache,
                    },
                )
                r.raise_for_status()
                completion = r.json().get("response", "").strip()
                return info_schema.model_validate_json(completion)

        except httpx.RequestError as e:
            raise ConnectionError(
                f"(async) Errore di connessione Ollama ({host}/{model_name}): {e}"
            )

    async def ainvoke(
        self, output_format: Literal["text", "structured", "json"] = "text", **kwargs
    ):
        # Gather relevant kwargs
        host = kwargs.get("llm", {}).get("host", self.DEFAULT_HOST)
        model_name = kwargs.get("llm", {}).get("model_name", self.DEFAULT_MODEL)
        prompt_content = kwargs.get("memory", "")
        use_cache = kwargs.get("cache", True)
        num_predict = kwargs.get("num_predict")

        if not prompt_content:
            logger.warning("NO Prompt provided!")

        if output_format == "structured":
            schema = kwargs.get("info_schema")
            if not schema:
                raise Exception(
                    "Missing info schema (kwarg: 'info_schema')! Provide the BaseModel schema for the information to be extracted!"
                )

            try:
                return await self.aget_structured_response(
                    prompt_content,
                    schema,
                    model_name,
                    host,
                    cache=use_cache,
                    num_predict=num_predict,
                )
            except ConnectionError as e:
                return str(e) + ". Assicurati che Ollama sia in esecuzione."

        elif output_format == "json":
            try:
                completion = await self.aget_response_text(
                    prompt_content,
                    model_name,
                    host,
                    format="json",
                    cache=use_cache,
                    num_predict=num_predict,
                )
            except ConnectionError as e:
                completion = str(e) + ". Assicurati che Ollama sia in esecuzione."

        else:
            try:
                completion = await self.aget_response_text(
                    prompt_content,
                    model_name,
                    host,
                    cache=use_cache,
                    num_predict=num_predict,
                )
            except ConnectionError as e:
                completion = str(e) + ". Assicurati che Ollama sia in esecuzione."

        return AIMessage(content=completion)


if __name__ == "__main__":

    import json
    import time

    t0 = time.time()

    llm = OllamaLLM(llm_model="gemma3:4b")  # "qwen3.5:4b"

    class Durata(BaseModel):
        """
        Sezione: Durata della società.
        Domanda:
        1) "Qual è la durata della società (fino a quale data)?"
        Chiavi: durata
        """

        # reasoning: str
        question: ClassVar[str] = "Qual è la durata della società (fino a quale data)?"
        durata_dell_azienda: str = Field(
            default="",
            description=(
                "Data di durata della società come appare nel documento (es. '30 dicembre 2050')"
                "Se non specificato → stringa vuota."
            ),
        )

    class CapitaleSociale(BaseModel):
        """
        Sezione: Capitale Sociale.
        Domanda:
        1) "Qual è l’ammontare del capitale sociale della società?"
        Chiavi: capitale_sociale_euro
        """

        question: ClassVar[str] = (
            "Qual è l’ammontare del capitale sociale della società?"
        )
        capitale_sociale_euro: str = Field(
            default="",
            description=(
                "Valore numerico in euro del capitale sociale. "
                "Se non specificato nella RISPOSTA → stringa vuota."
            ),
        )

    system_prompt = """
    Sei un normalizzatore di RISPOSTE per un sistema RAG su statuti societari.
    Compila i campi del modello esclusivamente usando la RISPOSTA fornita qui sotto (non usare il contesto); mantieni il testo nei campi il più breve possibile.
    Regole:
    - Se la RISPOSTA è esattamente "Non ho trovato la risposta nei documenti forniti" oppure non copre un campo → metti stringa vuota "".
    - Sì/No: usa "Sì" o "No".
    - Elenchi: elementi separati da virgole, senza punto finale.
    - Numeri e date: riporta esattamente il testo così come compare nella RISPOSTA, senza convertirli né modificarli. Non convertire/normalizzare (non trasformare “duecento” in “200”)
    - Output: SOLO il JSON del modello richiesto; nessun testo extra; 
    DOMANDA:
    {question}

    RISPOSTA:
    {answer}La società dura fino al 31 dicembre 2060, salvo proroga o scioglimento anticipato.

    Compila il modello  usando SOLO la RISPOSTA.
    """
    infos_to_extract = [
        {
            "schema": Durata,
            "question": "Qual è la durata della società (fino a quale data)?",
            "answer": "La società dura fino al 31 dicembre 2060, salvo proroga o scioglimento anticipato.",
        },
        {
            "schema": CapitaleSociale,
            "question": "Qual è l’ammontare del capitale sociale della società?",
            "answer": "Il capitale sociale è di euro 25.000 (venticinquemila) ed è suddiviso in quote ai sensi dell'articolo 2468 del codice civile.",
        },
    ]

    def extract_all(infos_to_extract):
        # define prompts
        infos_to_extract_processed = []
        for info in infos_to_extract:
            prompt = system_prompt.replace("{question}", info.get("question"))
            prompt = prompt.replace("{answer}", info.get("answer"))
            info.pop("question")
            info.pop("answer")
            info = {"schema": info.get("schema"), "prompt": prompt}
            infos_to_extract_processed.append(info)

        results = [
            llm.invoke(
                memory=info.get("prompt"),
                output_format="structured",
                info_schema=info.get("schema"),
            )
            for info in infos_to_extract_processed
        ]
        for result in results:
            print(result.model_dump())  # type: ignore

    async def aextract_all(infos_to_extract):
        # define prompts
        infos_to_extract_processed = []
        for info in infos_to_extract:
            prompt = system_prompt.replace("{question}", info.get("question"))
            prompt = prompt.replace("{answer}", info.get("answer"))
            info.pop("question")
            info.pop("answer")
            info = {"schema": info.get("schema"), "prompt": prompt}
            infos_to_extract_processed.append(info)

        results = [
            await llm.ainvoke(
                memory=info.get("prompt"),
                output_format="structured",
                info_schema=info.get("schema"),
            )
            for info in infos_to_extract_processed
        ]
        for result in results:
            print(result.model_dump())  # type: ignore

    # response = asyncio.run(llm.ainvoke(memory=prompt, output_format="structured", info_schema=Durata))

    # if isinstance(response, Durata):
    #     print(response.model_dump())

    # extract_all(infos_to_extract)
    # asyncio.run(aextract_all(infos_to_extract[:1]))
    # print(llm.invoke(output_format="text", memory="Ciao, come stai?"))

    prompt = '\n    Sei un normalizzatore di RISPOSTE per un sistema RAG su statuti societari.\n    Compila i campi del modello esclusivamente usando la RISPOSTA fornita qui sotto (non usare il contesto); mantieni il testo nei campi il più breve possibile.\n    Regole:\n    - Se la RISPOSTA è esattamente "Non ho trovato la risposta nei documenti forniti" oppure non copre un campo → metti stringa vuota "".\n    - Sì/No: usa "Sì" o "No".\n    - Elenchi: elementi separati da virgole, senza punto finale.\n    - Numeri e date: riporta esattamente il testo così come compare nella RISPOSTA, senza convertirli né modificarli. Non convertire/normalizzare (non trasformare “duecento” in “200”)\n    - Output: SOLO il JSON del modello richiesto; nessun testo extra; \n    DOMANDA:\n    Qual è la durata della società (fino a quale data)?\n\n    RISPOSTA:\n    La società dura fino al 31 dicembre 2060, salvo proroga o scioglimento anticipato.La società dura fino al 31 dicembre 2060, salvo proroga o scioglimento anticipato.\n\n    Compila il modello  usando SOLO la RISPOSTA.\n'
    # extra_prompt = f"JSON schema richiesto:\n    {json.dumps(Durata.model_json_schema(), indent=4, ensure_ascii=False)}\n\n"
    print(json.dumps(Durata.model_json_schema(), indent=4, ensure_ascii=False))
    print(
        llm.get_structured_response(
            prompt_content=prompt, info_schema=Durata  # type: ignore
        )
    )

    print(f"Total time taken to execute the script: {time.time() - t0:.3f} s")

    # TODO: Make the get_structured_response work with qwen3.5:4b (reasoning) model.
