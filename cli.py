import asyncio
import json

import typer
from dotenv import load_dotenv

load_dotenv()

from app.services.llm_services import LLMService  # noqa: E402
from app.schemas.chat import ChatRequest  # noqa: E402

app = typer.Typer(help="LLMHub — CLI для взаимодействия с LLM-провайдерами")


@app.command()
def chat(
    message: str = typer.Argument(..., help="Сообщение для LLM"),
    provider: str = typer.Option("auto", "--provider", "-p", help="gemini | ollama | openai | auto"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Вывод в JSON"),
):
    service = LLMService()
    request = ChatRequest(message=message, preferred_provider=provider)
    result = asyncio.run(service.get_response(request))

    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"\n{result['answer']}\n")
        suffix = " (fallback)" if result["fallback_used"] else ""
        typer.echo(f"[{result['provider']} / {result['model']} — {result['latency_ms']}ms{suffix}]", err=True)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Хост"),
    port: int = typer.Option(8000, help="Порт"),
    reload: bool = typer.Option(False, "--reload"),
):
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


@app.command()
def providers():
    service = LLMService()
    rows = [
        ("openai", service.openai is not None, service.openai_model),
        ("gemini", service.gemini is not None, service.gemini_model),
        ("ollama", service.ollama is not None, service.ollama_model),
    ]
    typer.echo(f"{'PROVIDER':<10} {'STATUS':<12} {'MODEL'}")
    typer.echo("-" * 40)
    for name, available, model in rows:
        status = typer.style("available", fg=typer.colors.GREEN) if available else typer.style("unavailable", fg=typer.colors.RED)
        typer.echo(f"{name:<10} {status:<21} {model}")
    typer.echo()
    typer.echo(f"Router mode: {service.router_mode}")



if __name__ == "__main__":
    app()