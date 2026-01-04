# load frorm other modules
from rag_info_extractor.llm_connector import OllamaLLM


def simplify_text_with_llm(
    text: str, 
    model: str = "gemma3:4b",
    temperature: float = 0
) -> str:
    
    # Simplify the text using the LLM
    llm = OllamaLLM(
        llm_model = model,
        temperature = temperature
    )

    system_prompt = """
    System:
    Sei un semplificatore di testo. Ricevi una o più frasi in italiano e le riscrivi in modo più chiaro e semplice, come spiegassi a tre-dicenne, mantenendo esattamente lo stesso significato.

    Regole:
    - Non aggiungere, omettere o alterare informazioni.
    - Mantieni numeri, date, percentuali, importi e riferimenti normativi (es. “art. 12, comma 3”) invariati.
    - Conserva nomi propri, acronimi e termini giuridici necessari; sostituisci solo il burocratese con parole comuni.
    - Usa frasi brevi, voce attiva, lessico semplice; rimuovi ridondanze.
    - Se l’input contiene più frasi/righe, restituiscile nello stesso ordine, una per riga.

    """
    
    prompt_content = system_prompt + "\nHuman:" + text
    response = llm.get_response_text(prompt_content)

    return response



if __name__ == "__main__":
    
    text = """
    L'utile netto di bilancio distribuibile è soltanto quello realmente conseguito e risultante dal bilancio
    regolarmente approvato. L'assemblea che approva il bilancio decide sulla distribuzione degli uli ai soci con
    maggioranza semplice del capitale sociale.

    """

    print(simplify_text_with_llm(text, model="llama3.2:3b-instruct-q8_0"))
