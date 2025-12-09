# 🎭 Reconhecimento Facial - Detecção de Expressões em Vídeo

Sistema de análise de expressões faciais e emoções em vídeos utilizando Deep Learning. O projeto processa vídeos frame a frame, detecta faces e identifica as emoções dominantes, gerando um vídeo de saída com anotações visuais.

## 📋 Índice

- [Funcionalidades](#-funcionalidades)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Exemplos](#-exemplos)
- [Troubleshooting](#-troubleshooting)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

## ✨ Funcionalidades

- ✅ **Reconhecimento facial**: Detecção de múltiplas faces em vídeos
- ✅ **Análise de expressões emocionais**: Análise de emoções em tempo real (felicidade, tristeza, raiva, surpresa, medo, nojo, neutro)
- ✅ **Detecção de atividades**: Categorização automática de atividades baseada em movimento e padrões comportamentais
- ✅ **Detecção de anomalias**: Identificação de movimentos bruscos e comportamentos atípicos
- ✅ **Geração de relatório**: Criação automática de relatório com estatísticas completas da análise
- ✅ Geração de vídeo de saída com anotações visuais
- ✅ Barra de progresso para acompanhamento do processamento
- ✅ Suporte a visualização em tempo real (opcional)
- ✅ Processamento frame a frame com alta precisão

## 🔧 Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- **Python 3.8 até 3.12** (recomendado: Python 3.12 como versão máxima ideal)
- **[UV](https://github.com/astral-sh/uv)** (recomendado) - Gerenciador rápido de versões Python e pacotes
- **pip** (alternativa, se não usar UV)
- **Git** (opcional, para clonar o repositório)

## 📦 Instalação

### 1. Clone o repositório (ou baixe os arquivos)

```bash
git clone <url-do-repositorio>
cd reconhecimento-facial
```

### 2. Instale o UV (Recomendado)

O UV é uma ferramenta moderna e extremamente rápida para gerenciar versões do Python e instalar pacotes. É altamente recomendado para uma melhor experiência de desenvolvimento.

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Após a instalação, reinicie o terminal ou adicione o UV ao PATH.

### 3. Configure o ambiente com UV

O UV gerencia automaticamente a versão do Python e cria o ambiente virtual:

```bash
# Instala Python 3.12 (se necessário) e cria o ambiente virtual
uv venv

# Ativa o ambiente virtual
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Instala as dependências
uv pip install -r requirements.txt
```

Ou, de forma ainda mais simples, o UV pode instalar tudo de uma vez:

```bash
# Cria o ambiente e instala as dependências automaticamente
uv venv --python 3.12
uv pip install -r requirements.txt
```

### Instalação Alternativa (sem UV)

Se preferir usar o método tradicional:

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Nota:** A primeira execução pode demorar mais tempo, pois o DeepFace baixará os modelos de deep learning necessários automaticamente.

## 🚀 Como Usar

### Uso Básico

1. **Baixe o vídeo de entrada:**
   - Acesse o [Google Drive](https://drive.google.com/drive/folders/11NGeYVnSvDF0bo3NHS47fb4b7vqkpALp)
   - Baixe o vídeo: `Unlocking Facial Recognition_ Diverse Activities Analysis.mp4`
   - **Importante:** Renomeie o arquivo para `input_video.mp4`
   - Coloque o arquivo renomeado na pasta `videos/`

2. Execute o script:

```bash
python detect-expression-video.py
```

3. O vídeo processado será salvo em `videos/output_video.mp4`

4. O relatório de análise será salvo em `relatorio_analise.txt` na raiz do projeto

### Uso Programático

Você também pode usar a função diretamente no seu código:

```python
from detect-expression-video import detect_expressions_in_video

# Processar vídeo e salvar resultado
success, summary = detect_expressions_in_video(
    video_path="caminho/para/video.mp4",
    output_path="caminho/para/saida.mp4",
    display=False,  # True para visualizar em tempo real
    report_path="relatorio.txt"  # Caminho para salvar o relatório
)

if success:
    print(f"Frames analisados: {summary['total_frames_analisados']}")
    print(f"Anomalias detectadas: {summary['numero_anomalias_detectadas']}")
```

### Parâmetros da Função

- `video_path` (str): Caminho para o vídeo de entrada
- `output_path` (str, opcional): Caminho para salvar o vídeo processado. Se `None`, não salva o vídeo
- `display` (bool, opcional): Se `True`, exibe o vídeo em tempo real durante o processamento (pressione 'q' para sair)
- `report_path` (str, opcional): Caminho para salvar o relatório de análise. Se `None`, o relatório será exibido no console

### Retorno da Função

A função retorna uma tupla `(success, summary)` onde:

- `success` (bool): `True` se o processamento foi concluído com sucesso
- `summary` (dict): Dicionário com estatísticas completas da análise, incluindo:
  - `total_frames_analisados`: Total de frames processados
  - `total_faces_detectadas`: Total de faces detectadas
  - `numero_anomalias_detectadas`: Número de anomalias encontradas
  - `atividades_detectadas`: Número de atividades categorizadas
  - `emocoes_detectadas`: Distribuição de emoções detectadas
  - E muito mais...

## 📁 Estrutura do Projeto

```
reconhecimento-facial/
│
├── detect-expression-video.py  # Script principal
├── requirements.txt            # Dependências do projeto
├── README.md                   # Este arquivo
├── .gitignore                  # Arquivos ignorados pelo Git
├── relatorio_analise.txt       # Relatório gerado automaticamente (não versionado)
│
└── videos/                     # Pasta para vídeos
    ├── input_video.mp4         # Vídeo de entrada (não versionado)
    └── output_video.mp4        # Vídeo de saída (não versionado)
```

## 🛠 Tecnologias Utilizadas

- **[OpenCV](https://opencv.org/)** - Processamento de vídeo e imagens
- **[DeepFace](https://github.com/serengil/deepface)** - Reconhecimento facial e análise de emoções
- **[TensorFlow](https://www.tensorflow.org/)** - Framework de deep learning
- **[NumPy](https://numpy.org/)** - Computação numérica
- **[tqdm](https://github.com/tqdm/tqdm)** - Barra de progresso

## 📝 Exemplos

### Exemplo 1: Processamento básico

```python
python detect-expression-video.py
```

Isso irá:

- Processar o vídeo `videos/input_video.mp4`
- Gerar o vídeo processado em `videos/output_video.mp4`
- Criar o relatório em `relatorio_analise.txt`

### Exemplo 2: Visualização em tempo real

```python
from detect-expression-video import detect_expressions_in_video

success, summary = detect_expressions_in_video(
    video_path="videos/input_video.mp4",
    output_path=None,
    display=True,  # Visualiza o vídeo em tempo real
    report_path="meu_relatorio.txt"
)

print(f"Anomalias encontradas: {summary['numero_anomalias_detectadas']}")
```

### Exemplo 3: Processar vídeo customizado

```python
from detect-expression-video import detect_expressions_in_video

success, summary = detect_expressions_in_video(
    video_path="meu_video.mp4",
    output_path="resultado.mp4",
    display=False,
    report_path="analise_completa.txt"
)

# Acessar estatísticas detalhadas
print(f"Total de frames: {summary['total_frames_analisados']}")
print(f"Atividades detectadas: {summary['atividades_detectadas']}")
print(f"Emoção mais frequente: {summary['emocao_mais_frequente']}")
```

## 📊 Relatório de Análise

O sistema gera automaticamente um relatório completo contendo:

- **Resumo Geral**: Total de frames analisados, faces detectadas e anomalias
- **Análise de Emoções**: Distribuição percentual de todas as emoções detectadas
- **Análise de Atividades**: Categorização e contagem de atividades identificadas
- **Detecção de Anomalias**: Lista detalhada de movimentos bruscos e comportamentos atípicos

O relatório é salvo em formato de texto e pode ser facilmente compartilhado ou incluído na documentação do projeto.

## 🔍 Troubleshooting

### Erro: "Could not open video"

- Verifique se o caminho do vídeo está correto
- Certifique-se de que o arquivo de vídeo existe e não está corrompido
- Verifique se o formato do vídeo é suportado (MP4, AVI, MOV, etc.)

### Erro: "ModuleNotFoundError"

- Certifique-se de que todas as dependências foram instaladas:
  - Com UV: `uv pip install -r requirements.txt`
  - Sem UV: `pip install -r requirements.txt`
- Verifique se o ambiente virtual está ativado
- Se estiver usando UV, certifique-se de que o Python 3.12 (ou versão compatível) está instalado: `uv python install 3.12`

### Processamento muito lento

- O processamento depende do tamanho do vídeo e do hardware
- Para vídeos grandes, considere reduzir a resolução ou usar um backend de detecção mais rápido
- O primeiro uso é mais lento devido ao download dos modelos

### Erro relacionado ao TensorFlow

- Certifique-se de ter uma versão compatível do TensorFlow instalada
- Em alguns sistemas, pode ser necessário instalar dependências adicionais do sistema

### Modelos não são baixados

- Verifique sua conexão com a internet
- Os modelos são baixados automaticamente na primeira execução
- Os modelos são salvos em `.deepface/` na pasta do usuário

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abrir um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👤 Autor

Desenvolvido como parte do curso de Pós-Graduação em IA da FIAP.

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!
