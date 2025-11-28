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

- ✅ Detecção de múltiplas faces em vídeos
- ✅ Análise de emoções em tempo real (felicidade, tristeza, raiva, surpresa, medo, nojo, neutro)
- ✅ Geração de vídeo de saída com anotações visuais
- ✅ Barra de progresso para acompanhamento do processamento
- ✅ Suporte a visualização em tempo real (opcional)
- ✅ Processamento frame a frame com alta precisão

## 🔧 Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- **Python 3.8+** (recomendado: Python 3.10 ou superior)
- **pip** (gerenciador de pacotes Python)
- **Git** (opcional, para clonar o repositório)

## 📦 Instalação

### 1. Clone o repositório (ou baixe os arquivos)

```bash
git clone <url-do-repositorio>
cd reconhecimento-facial
```

### 2. Crie um ambiente virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> **Nota:** A primeira execução pode demorar mais tempo, pois o DeepFace baixará os modelos de deep learning necessários automaticamente.

## 🚀 Como Usar

### Uso Básico

1. Coloque seu vídeo de entrada na pasta `videos/` com o nome `input_video.mp4`

2. Execute o script:

```bash
python detect-expression-video.py
```

3. O vídeo processado será salvo em `videos/output_video.mp4`

### Uso Programático

Você também pode usar a função diretamente no seu código:

```python
from detect-expression-video import detect_expressions_in_video

# Processar vídeo e salvar resultado
detect_expressions_in_video(
    video_path="caminho/para/video.mp4",
    output_path="caminho/para/saida.mp4",
    display=False  # True para visualizar em tempo real
)
```

### Parâmetros da Função

- `video_path` (str): Caminho para o vídeo de entrada
- `output_path` (str, opcional): Caminho para salvar o vídeo processado. Se `None`, não salva o vídeo
- `display` (bool, opcional): Se `True`, exibe o vídeo em tempo real durante o processamento (pressione 'q' para sair)

## 📁 Estrutura do Projeto

```
reconhecimento-facial/
│
├── detect-expression-video.py  # Script principal
├── requirements.txt            # Dependências do projeto
├── README.md                   # Este arquivo
├── .gitignore                  # Arquivos ignorados pelo Git
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

### Exemplo 2: Visualização em tempo real

```python
from detect-expression-video import detect_expressions_in_video

detect_expressions_in_video(
    video_path="videos/input_video.mp4",
    output_path=None,
    display=True  # Visualiza o vídeo em tempo real
)
```

### Exemplo 3: Processar vídeo customizado

```python
from detect-expression-video import detect_expressions_in_video

detect_expressions_in_video(
    video_path="meu_video.mp4",
    output_path="resultado.mp4",
    display=False
)
```

## 🔍 Troubleshooting

### Erro: "Could not open video"
- Verifique se o caminho do vídeo está correto
- Certifique-se de que o arquivo de vídeo existe e não está corrompido
- Verifique se o formato do vídeo é suportado (MP4, AVI, MOV, etc.)

### Erro: "ModuleNotFoundError"
- Certifique-se de que todas as dependências foram instaladas: `pip install -r requirements.txt`
- Verifique se o ambiente virtual está ativado

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

