import os
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "openai:gpt-5-nano")

SAMPLES_DIR = Path(__file__).parent / "code" / "samples"

class Resume(BaseModel):
    name: str = Field(description = "Full name of the candidate")
    email: str | None = Field(description = "Get the email, if present, else none")
    years_experience: float = Field(description = "Total years of professional experience")
    skills: list[str] = Field(description = "List of technical skills mentioned")


def read_sample(filename: str) -> str:
    """Read a fixed, shipped sample file safely (no user-controlled paths). """
    path = (SAMPLES_DIR / filename).resolve()
    if not str(path).startswith(str(SAMPLES_DIR.resolve())):
        raise ValueError("Invalid sample path")
    return path.read_text(encoding = "utf-8")

def extract_resume(text: str) -> Resume | None:
    llm = init_chat_model(MODEL, temperature=0)
    structured_llm = llm.with_structured_output(Resume, include_raw=True)
    system = (
        "You are a precise data extractor. Extract only what is explicitly present; never invent data. "
        "For skills, include ONLY named technologies, tools, languages, or frameworks (e.g. Python, FastAPI, Docker). "
        "Do NOT include vague phrases like 'AI agents' or 'backend dev' as skills."
    )
    result = structured_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Extract structured data from this resume: \n\n{text}"},

    ])

    raw = result.get("raw")
    usage = getattr(raw, "usage_metadata", None)
    if usage:
        print(f"[dim]tokens - in: {usage.get('input_tokens')}"
               f" out: {usage.get('output_tokens')} total: {usage.get('total_tokens')}[/dim]")

    if result.get("parsing_error"):
        print("[red]Model output failed validation; handle/retry instead of trusting it.[/red]")  
        return None
    return result.get("parsed")

def main() -> None:
    text = read_sample("resume_messy.txt")
    print("[bold cyan]Extracting structured data from a messy resume...[/bold cyan]\n")  
    resume = extract_resume(text)

    if resume:
        output_json = resume.model_dump_json(indent=2)
        print(output_json)
        print(f"\n[green]Validated![/green] {len(resume.skills)} skills,"
               f" {resume.years_experience} yrs experience.")

        # Save output to samples folder
        output_path = SAMPLES_DIR / "resume_output.json"
        output_path.write_text(output_json, encoding="utf-8")
        print(f"\n[bold green]Output saved to:[/bold green] {output_path}")

if __name__ == "__main__":
    main()                           
