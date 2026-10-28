"""Monta o PDF entregue à FIAP (T35) a partir de ``results/relatorio_tecnico.md``.

O relatório continua sendo escrito em Markdown; este script só acrescenta a capa (projeto, POSTECH/FIAP, Fase 1,
integrantes e links), o sumário e a numeração das figuras, e imprime o resultado em PDF com o Edge ou o Chrome
sem interface (``--headless``), que já vêm instalados no Windows.

Uso (na raiz do projeto)::

    python entrega/gerar_pdf.py                              # vídeo ainda "em andamento"
    python entrega/gerar_pdf.py --video https://youtu.be/... # com o link do vídeo (T36)

Saída: ``entrega/Tech_Challenge_Fase1.pdf``. Depende só do ``mistune``, que já vem com o Jupyter do
``requirements.txt``.
"""

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import mistune

RAIZ = Path(__file__).resolve().parents[1]
RELATORIO = RAIZ / "results" / "relatorio_tecnico.md"
FIGURAS = RAIZ / "results" / "figures"
SAIDA = RAIZ / "entrega" / "Tech_Challenge_Fase1.pdf"

PROJETO = "Cardio IA"
SUBTITULO = "Apoio à triagem de doença cardiovascular com aprendizado de máquina"
REPOSITORIO = "https://github.com/team-tech-challenge-fiap/tech-challenge-2026-fiap"
INTEGRANTES = [
    ("Guilherme Saar Santos", "GuilhermeSaarSantos"),
    ("Wesley Berto", "WesleyBert"),
    ("Michel Zarif", "MichelZarif"),
    ("Luis Felipe Laudacio", "luislaudacio"),
    ("Sara Felix", "sarafelix25"),
]

NAVEGADORES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro"]

CSS = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm;
        @bottom-right { content: counter(page); font: 9pt "Segoe UI", Arial, sans-serif; color: #666; } }
@page :first { @bottom-right { content: none; } }
* { box-sizing: border-box; }
body { font: 10.5pt/1.5 "Segoe UI", Arial, sans-serif; color: #1d1d1f; margin: 0; }
.capa { height: 250mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }
.capa .instituicao { font-size: 11pt; letter-spacing: .08em; text-transform: uppercase; color: #8a1538; font-weight: 600; }
.capa h1 { font-size: 34pt; margin: 0 0 6mm; color: #1d1d1f; }
.capa .subtitulo { font-size: 15pt; color: #444; margin: 0 0 4mm; }
.capa .fase { font-size: 12pt; color: #666; }
.capa table { border: none; margin: 0; width: auto; }
.capa td, .capa th { border: none; padding: 1mm 8mm 1mm 0; background: none; font-size: 11pt; }
.capa .links p { margin: 1.5mm 0; font-size: 11pt; }
.capa .rodape { color: #666; font-size: 10pt; }
.pendente { color: #b45309; font-weight: 600; }
.sumario { break-after: page; }
.sumario ol { padding-left: 6mm; }
.sumario li { margin: 1.5mm 0; }
.sumario a { color: #1d1d1f; text-decoration: none; }
h1 { font-size: 20pt; }
h2 { font-size: 16pt; color: #8a1538; break-before: page; margin-top: 0; border-bottom: 1.5px solid #8a1538; padding-bottom: 2mm; }
h3 { font-size: 12.5pt; margin-top: 7mm; break-after: avoid; }
h2 + p, h3 + p { break-before: avoid; }
p, li { orphans: 3; widows: 3; text-align: justify; hyphens: auto; }
a { color: #0b5cad; overflow-wrap: anywhere; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f2f2f4; padding: 0 1mm; border-radius: 2px; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; margin: 4mm 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #d0d0d6; padding: 1.5mm 2mm; text-align: left; vertical-align: top; }
th { background: #f2f2f4; }
figure { margin: 5mm 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 115mm; }
figcaption { font-size: 9pt; color: #444; margin-top: 2mm; }
blockquote { margin: 4mm 0; padding: 2mm 4mm; border-left: 3px solid #8a1538; background: #faf5f7; }
"""


def ler_relatorio() -> str:
    """O Markdown do relatório a partir da seção 1: título e cabeçalho viram a capa."""
    texto = RELATORIO.read_text(encoding="utf-8")
    inicio = texto.find("\n## ")
    if inicio < 0:
        sys.exit(f"Nenhuma seção '## ' encontrada em {RELATORIO}")
    return texto[inicio + 1:]


def slug(titulo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", titulo.lower()).strip("-") or "secao"


def numerar_e_ancorar(corpo: str) -> tuple[str, list[tuple[str, str]], int]:
    """Dá id às seções (para o sumário) e troca cada imagem por uma figura numerada com legenda."""
    secoes = []

    def ancora(m: re.Match) -> str:
        titulo = re.sub(r"<[^>]+>", "", m.group(1))
        ident = f"s{len(secoes) + 1}-{slug(titulo)}"
        secoes.append((ident, html.unescape(titulo)))
        return f'<h2 id="{ident}">{m.group(1)}</h2>'

    corpo = re.sub(r"<h2>(.*?)</h2>", ancora, corpo)

    contador = 0

    def figura(m: re.Match) -> str:
        nonlocal contador
        src, alt = m.group("src"), m.group("alt")
        caminho = (RELATORIO.parent / html.unescape(src)).resolve()
        if not caminho.exists():
            sys.exit(f"Figura não encontrada: {caminho}")
        contador += 1
        return (f'<figure><img src="{caminho.as_uri()}" alt="{alt}">'
                f"<figcaption><strong>Figura {contador}.</strong> {alt}</figcaption></figure>")

    padrao = r'<p><img src="(?P<src>[^"]+)" alt="(?P<alt>[^"]*)"\s*/?>\s*</p>'
    corpo = re.sub(padrao, figura, corpo)
    return corpo, secoes, contador


def capa(video: str | None) -> str:
    hoje = date.today()
    linhas = "".join(
        f'<tr><td>{nome}</td><td><a href="https://github.com/{login}">github.com/{login}</a></td></tr>'
        for nome, login in INTEGRANTES
    )
    link_video = (f'<a href="{html.escape(video)}">{html.escape(video)}</a>' if video
                  else '<span class="pendente">em andamento</span>')
    return f"""
<section class="capa">
  <div>
    <div class="instituicao">POSTECH · FIAP</div>
    <p class="fase">Tech Challenge · Fase 1</p>
  </div>
  <div>
    <h1>{PROJETO}</h1>
    <p class="subtitulo">{SUBTITULO}</p>
    <p class="fase">Relatório técnico</p>
  </div>
  <div>
    <h3>Integrantes</h3>
    <table>{linhas}</table>
  </div>
  <div class="links">
    <p><strong>Repositório:</strong> <a href="{REPOSITORIO}">{REPOSITORIO}</a></p>
    <p><strong>Vídeo de demonstração:</strong> {link_video}</p>
  </div>
  <div class="rodape">{hoje.day} de {MESES[hoje.month - 1]} de {hoje.year}</div>
</section>"""


def sumario(secoes: list[tuple[str, str]]) -> str:
    sem_numero = [(ident, re.sub(r"^\d+\.\s*", "", titulo)) for ident, titulo in secoes]
    itens = "".join(f'<li><a href="#{ident}">{html.escape(titulo)}</a></li>' for ident, titulo in sem_numero)
    return f'<section class="sumario"><h1>Sumário</h1><ol>{itens}</ol></section>'


def montar_html(video: str | None) -> tuple[str, int]:
    markdown = mistune.create_markdown(escape=False, plugins=["table", "strikethrough", "url"])
    corpo, secoes, figuras = numerar_e_ancorar(markdown(ler_relatorio()))
    documento = (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
                 f"<title>{PROJETO} · Tech Challenge Fase 1</title><style>{CSS}</style></head>"
                 f"<body>{capa(video)}{sumario(secoes)}<main>{corpo}</main></body></html>")
    return documento, figuras


def achar_navegador(caminho: str | None) -> Path:
    candidatos = [Path(caminho)] if caminho else NAVEGADORES
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    for nome in ("msedge", "google-chrome", "chromium", "chrome"):
        encontrado = shutil.which(nome)
        if encontrado:
            return Path(encontrado)
    sys.exit("Edge ou Chrome não encontrado. Informe o caminho com --navegador.")


def imprimir_pdf(documento: str, navegador: Path, saida: Path) -> None:
    saida.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as pasta:
        pagina = Path(pasta) / "relatorio.html"
        pagina.write_text(documento, encoding="utf-8")
        comando = [
            str(navegador), "--headless=new", "--disable-gpu", "--no-first-run",
            f"--user-data-dir={Path(pasta) / 'perfil'}", "--allow-file-access-from-files",
            "--no-pdf-header-footer", f"--print-to-pdf={saida}", pagina.as_uri(),
        ]
        subprocess.run(comando, check=True, timeout=180, capture_output=True)
    if not saida.exists() or saida.stat().st_size == 0:
        sys.exit("O navegador não gerou o PDF.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", help="link do vídeo (YouTube ou Vimeo); sem ele, a capa diz 'em andamento'")
    parser.add_argument("--navegador", help="caminho do Edge ou do Chrome, se não estiver no lugar padrão")
    parser.add_argument("--html", type=Path, help="também salva o HTML intermediário nesse caminho")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()

    documento, figuras = montar_html(args.video)
    if args.html:
        args.html.write_text(documento, encoding="utf-8")
    imprimir_pdf(documento, achar_navegador(args.navegador), args.saida.resolve())
    print(f"PDF gerado: {args.saida} ({figuras} figuras, vídeo: {args.video or 'em andamento'})")


if __name__ == "__main__":
    main()
